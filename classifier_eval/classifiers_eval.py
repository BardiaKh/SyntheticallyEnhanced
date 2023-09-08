import argparse
import os
import shutil
from glob import glob
import pandas as pd
import numpy as np
import copy
import sys
import monai as mn
import bkh_pytorch_utils as bpu
import torch
import pytorch_lightning as pl
import timm
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

def get_data_dict(df, base_path, pathology_list):
    data_dict = list()
    for i in tqdm(range(len(df)), desc="Loading data"):
        row = df.iloc[i]
        if base_path is None:
            path = f"{row['Path']}"
        else:
            path = f"{base_path}/{row['Path']}"
        data_dict.append({
            'idx': i,
            'img': path,
            ** {pathology: row[pathology] for pathology in pathology_list if pathology in row.keys()}
        })

    return data_dict

class ClassifierModule(bpu.BKhModule):
    def __init__(self, num_classes=14, collate_fn=None, val_collate_fn=None, train_sampler=None, val_sampler=None, train_ds=None, val_ds=None, dl_workers=-1, pin_memory=True, batch_size=2, val_batch_size=None, lr=1e-4, prefetch_factor=1, persistent_workers=False):
        super().__init__(collate_fn=collate_fn, val_collate_fn=val_collate_fn, train_sampler=train_sampler, val_sampler=val_sampler, train_ds=train_ds, val_ds=val_ds, dl_workers=dl_workers, batch_size=batch_size, val_batch_size=val_batch_size, pin_memory=pin_memory, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
        self.model = timm.create_model("convnext_base.fb_in22k_ft_in1k", pretrained=False, in_chans=1, num_classes=num_classes)
    
    def forward(self, img):
        return self.model(img)

def main():
    parser = argparse.ArgumentParser(description="Set up evaluation config.")
    parser.add_argument("input", type=str, default=None, help="Absolute path to CSV file containing formatted data.")
    parser.add_argument("-o", "--output", type=str, default="./results.csv", help="Name of the output CSV file containing inference results.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for dataloader creation.")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of CPU workers for dataloader creation.")
    parser.add_argument("-p", "--base_path", type=str, default=None, help="If the image paths in your input CSV are relative to a base path, specify it here.")
    parser.add_argument("-c", "--cuda_id", type=int, default=0, help="The device ID of the CUDA card on which you would like to run inference.")
    parser.add_argument("-d", "--cache_dir", type=str, default="./cache", help="The directory in which to store cached data.")
    parser.add_argument("-t", "--pathologies", type=str, default="all", help="Comma-separated list of pathologies to handle. Defaults to 'all'.")
    parser.add_argument("-a", "--auc_report", action="store_true", help="Flag to generate AUC reports. (output file will be Excel rather than CSV)")
    args = parser.parse_args()

    if args.pathologies.lower() == 'all':
        PATHOLOGIES = ['Pleural Other', 'Fracture', 'Support Devices','Pleural Effusion','Pneumothorax','Atelectasis','Pneumonia','Consolidation', 'Edema','Lung Lesion', 'Lung Opacity', 'Cardiomegaly','Enlarged Cardiomediastinum', 'No Finding']
    else:
        PATHOLOGIES = [path.strip() for path in args.pathologies.split(',')]
    WEIGHTS_PATH = './weights'
    IMG_SIZE = 256
    CACHE_DIR = args.cache_dir
    BATCH_SIZE = args.batch_size
    NUM_DL_WORKERS = args.workers
    BASE_PATH = args.base_path
    CSV_SAVE_PATH = args.output
    CUDA_ID = args.cuda_id

    print(
        "\n",
        "*"*100, "\n",
        f"Running inference on {args.input} with batch size {BATCH_SIZE} and {NUM_DL_WORKERS} workers on CUDA:{CUDA_ID}.", "\n",
        "*"*100, "\n",
        "\n"
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{CUDA_ID}"
    os.makedirs(CACHE_DIR, exist_ok=True)

    df = pd.read_csv(args.input)
    data_dict = get_data_dict(df, base_path=BASE_PATH, pathology_list=PATHOLOGIES)

    transforms=mn.transforms.Compose([
        mn.transforms.LoadImageD(keys="img", ensure_channel_first=True),
        mn.transforms.ResizeD(keys='img', size_mode="longest", mode="bilinear", spatial_size=IMG_SIZE, align_corners=False),
        mn.transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0, upper=100, b_min=-1, b_max=1, clip=True),
        mn.transforms.SpatialPadD(keys='img', spatial_size=(IMG_SIZE, IMG_SIZE), mode="constant", constant_values=-1),
        mn.transforms.ToTensorD(keys=[*[pathology for pathology in PATHOLOGIES if pathology in df.columns]], dtype=torch.float),
        mn.transforms.AddChannelD(keys=[*[pathology for pathology in PATHOLOGIES if pathology in df.columns]]),
        mn.transforms.ConcatItemsD(keys=[*[pathology for pathology in PATHOLOGIES if pathology in df.columns]], name='cls'),
        mn.transforms.SelectItemsD(keys=["img","cls","idx"]),
        mn.transforms.ToTensorD(keys=["img","cls"], dtype=torch.float, track_meta=False)
    ])

    dataset = mn.data.PersistentDataset(data=data_dict, transform=transforms, cache_dir=f"{CACHE_DIR}/val")  

    model=ClassifierModule(
        val_ds=dataset,
        num_classes=len(PATHOLOGIES),
        batch_size=BATCH_SIZE,
        dl_workers = NUM_DL_WORKERS,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=True
    )
    model.eval()
    model.cuda()
    model.half()
    model.stats()

    results_dict = []

    weights = glob(f"{WEIGHTS_PATH}/*/*.ckpt")
    for i, weight in enumerate(weights):
        model_name = "-".join(weight.split('/')[-2:]).split('_')[0]
        model.load_ckpt(weight, ema=True)
        dl = model.val_dataloader()

        with torch.inference_mode():
            for batch in tqdm(dl, desc=f"Inferring model {i+1}/{len(weights)} ({model_name})"):
                img, cls, ids = batch['img'], batch['cls'], batch['idx']
                img = img.cuda().half()
                logits = model(img)
                probs = torch.sigmoid(logits)
                probs = probs.float().cpu().numpy()
                for k, idx in enumerate(ids):
                    labels = [0] * len(PATHOLOGIES)
                    sigmoid_outputs = [0] * len(PATHOLOGIES)
                    for pidx, pathology in enumerate(PATHOLOGIES):
                        if pathology not in df.columns:
                            labels[pidx] = -1
                        else: 
                            labels[pidx] = df.loc[idx.item(), pathology]
                        sigmoid_outputs[pidx] = probs[k][pidx]

                    results_dict.append({
                        'Path': df.loc[idx.item(), 'Path'],
                        'Labels': tuple(labels),
                        'Probs': sigmoid_outputs,
                        'Logits': logits[k].cpu().numpy().tolist(),
                        'Model Name': copy.copy(model_name),
                    })

    results_df = pd.DataFrame(results_dict)

    if args.auc_report:
        auc_dict = {}
        
        for model_name in results_df['Model Name'].unique():
            model_results = results_df[results_df['Model Name'] == model_name]
            auc_dict[model_name] = {}
            
            macro_auc_list = []
            
            for pidx, pathology in enumerate(PATHOLOGIES):
                if pathology in df.columns:
                    true_labels = model_results['Labels'].apply(lambda x: x[pidx])
                    pred_scores = model_results['Probs'].apply(lambda x: x[pidx])
                    auc_score = roc_auc_score(true_labels, pred_scores)
                    auc_dict[model_name][pathology] = auc_score
                    macro_auc_list.append(auc_score)
            
            # Calculate macro-averaged AUC for this model
            macro_auc = sum(macro_auc_list) / len(macro_auc_list)
            auc_dict[model_name]['Overall (Macro)'] = macro_auc
        
        # Convert the nested dictionary to a DataFrame
        auc_df = pd.DataFrame(auc_dict).transpose().reset_index().rename(columns={"index": "Model Name"})
        
        with pd.ExcelWriter(CSV_SAVE_PATH.replace('.csv', '.xlsx')) as writer:
            results_df.to_excel(writer, sheet_name='Inference Results', index=False)
            auc_df.to_excel(writer, sheet_name='AUC Report', index=False)
    else:
        results_df.to_csv(CSV_SAVE_PATH, index=False)

    print("\n","Cleaning  cache directory...", "\n")

    shutil.rmtree(CACHE_DIR)

    print(
        "\n",
        "*"*100, "\n",
        f"Inference complete and outputs saved to {CSV_SAVE_PATH}.", "\n",
        "*"*100, "\n",
        "\n"
    )
    sys.exit(0)

if __name__ == "__main__":
    main()
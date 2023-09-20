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
import yaml
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

DEFAULT_CONFIG = {
    "batch_size": 32,
    "workers": 4,
    "base_path": None,
    "output": "./results.csv",
    "cuda_id": 0,
    "pathologies": "all",
    "report_auc": True,
    "equalize_hist": False,
    "weight_folders": [],
    "cache_dir": "./cache",
    "input": None
}

def load_config(file_path, default_config):
    """
    Load a YAML configuration file and merge it with the default configuration.

    Args:
    - file_path (str): The path to the YAML configuration file.
    - default_config (dict): The default configuration.

    Returns:
    - dict: The merged configuration.
    """
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Merge the loaded configuration with the default values
    for key, default_value in default_config.items():
        config[key] = config.get(key, default_value)
    
    return config

def parse_path(path_to_parse, config_file_path):
    """
    Parse a given path and return it. If the path is relative, return it relative to the config file's location.

    Args:
    - path_to_parse (str): The path to be parsed.
    - config_file_path (str): The path to the config.yaml file.

    Returns:
    - str: The absolute path.
    """
    if os.path.isabs(path_to_parse):
        return path_to_parse

    config_dir = os.path.dirname(config_file_path)
    return os.path.join(config_dir, path_to_parse)
    
def get_weights_from_folders(weight_folders, config_path):
    """
    Get a list of weights from a list of folders.

    Args:
    - weight_folders (list): A list of folders containing weights.
    - config_path (str): The path to the config.yaml file.

    Returns:
    - list: A list of weights.
    """
    weights = []
    for folder in weight_folders:
        folder = parse_path(folder, config_path)
        weights.extend(glob(f"{folder}/*.ckpt"))
    
    return weights

def get_data_dict(df, base_path, pathology_list):
    """
    Get a list of data dictionaries from a dataframe.

    Args:
    - df (pandas.DataFrame): The dataframe containing the data.
    - base_path (str): The base path to the data.
    - pathology_list (list): A list of pathologies to include in the data dictionary.

    Returns:
    - list: A list of data dictionaries.
    """
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
    parser.add_argument("yaml_config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config_path = args.yaml_config
    config = load_config(config_path, DEFAULT_CONFIG)

    IMG_SIZE = 256
    INPUT_FILE = parse_path(config["input"], config_path)
    CACHE_DIR = parse_path(config["cache_dir"], config_path)
    BATCH_SIZE = config["batch_size"]
    NUM_DL_WORKERS = config["workers"]
    BASE_PATH = parse_path(config["base_path"], config_path)
    CSV_SAVE_PATH = parse_path(config["output"], config_path)
    CUDA_ID = config["cuda_id"]
    PATHOLOGIES = config["pathologies"] if config["pathologies"] != 'all' else ['Pleural Other', 'Fracture', 'Support Devices','Pleural Effusion','Pneumothorax','Atelectasis','Pneumonia','Consolidation', 'Edema','Lung Lesion', 'Lung Opacity', 'Cardiomegaly','Enlarged Cardiomediastinum', 'No Finding']
    REPORT_AUC = config["report_auc"]
    EQUALIZE_HIST = config["equalize_hist"]
    WEIGHT_FOLDERS = config["weight_folders"]

    print(
        "\n",
        "*"*100, "\n",
        f"Running inference on {INPUT_FILE} with batch size {BATCH_SIZE} and {NUM_DL_WORKERS} workers on CUDA:{CUDA_ID}.", "\n",
        "*"*100, "\n",
        "\n"
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{CUDA_ID}"
    os.makedirs(CACHE_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    data_dict = get_data_dict(df, base_path=BASE_PATH, pathology_list=PATHOLOGIES)

    transforms=mn.transforms.Compose([
        mn.transforms.LoadImageD(keys="img", ensure_channel_first=False),
        bpu.EnsureGrayscaleD(keys="img"),
        mn.transforms.HistogramNormalized(keys="img") if EQUALIZE_HIST else mn.transforms.IdentityD(keys="img"),
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

    weights = get_weights_from_folders(WEIGHT_FOLDERS, config_path)
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

    if REPORT_AUC:
        auc_dict = {}
        
        for model_name in results_df['Model Name'].unique():
            model_results = results_df[results_df['Model Name'] == model_name]
            auc_dict[model_name] = {}
            
            macro_auc_list = []
            
            for pidx, pathology in enumerate(PATHOLOGIES):
                true_labels_list = model_results['Labels'].apply(lambda x: x[pidx]).tolist()
                pred_scores_list = model_results['Probs'].apply(lambda x: x[pidx]).tolist()

                # Removing instances where the true label is -1 (uncertain cases)
                valid_idx = [idx for idx, label in enumerate(true_labels_list) if label != -1]
                true_labels = [true_labels_list[i] for i in valid_idx]
                pred_scores = [pred_scores_list[i] for i in valid_idx]

                auc_score = roc_auc_score(true_labels, pred_scores)
                auc_dict[model_name][pathology] = auc_score
                macro_auc_list.append(auc_score)
            
            # Calculate macro-averaged AUC for this model
            macro_auc = sum(macro_auc_list) / len(macro_auc_list)
            auc_dict[model_name]['Overall (Macro)'] = macro_auc
        
        # Convert the nested dictionary to a DataFrame
        auc_df = pd.DataFrame(auc_dict).transpose().reset_index().rename(columns={"index": "Model Name"})
        
        results_df.to_csv(CSV_SAVE_PATH, index=False)
        auc_df.to_csv(CSV_SAVE_PATH.replace('.csv', '_auc.csv'), index=False)
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
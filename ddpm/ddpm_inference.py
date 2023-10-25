import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from mediffusion import DiffusionModule
from mediffusion.utils import to_uint8
import monai as mn
import pandas as pd
import skimage
import argparse
import math

DATA_SAVE_PATH = '/PATH/TO/SAVE/IMAGES'
IMG_SIZE = 256
BATCH_SIZE = 64
PRECISION = 16
GUIDANCE_SCALE = 0
NUM_DDIM_STEPS = 200
START_IDX = 0 # For resuming inference

PATHOLOGIES = ['Pleural Other', 'Fracture', 'Support Devices','Pleural Effusion','Pneumothorax','Atelectasis','Pneumonia','Consolidation', 'Edema','Lung Lesion', 'Lung Opacity', 'Cardiomegaly','Enlarged Cardiomediastinum', 'No Finding']

def get_data_dict(df):
    conversion_schema = {
        "sex": {
            "Female": 0,
            "Male": 1,
        },
        "race": {
            "American Indian/Alaskan Native": 0,
            "African American": 1,
            "White": 2,
            "Pacific Islander": 3,
            "Asian": 4,
        },
    }
    data_dict = list()
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        data_dict.append({
            'filename': f"{row['Save Path']}",
            'seed': row['Seed'],
            'sex': conversion_schema['sex'][row['Sex']],
            'race': conversion_schema['race'][row['Race']],
            'age': row['Age']//10,
            ** {pathology: row[pathology] for pathology in PATHOLOGIES}
        })
    
    return data_dict

def infer(rank, in_queue, out_queue, model_path, config_path):
    device = torch.device(f"cuda:{rank}")
    model = DiffusionModule(config_path)
    model.load_ckpt(model_path, ema=True)
    model = model.to(device).half()
    model.eval()
    
    while True:
        batch = in_queue.get()
        if batch is None:
            break

        filenames = batch['filename']
        seeds = batch['seed']
        
        noise = torch.empty(len(filenames), 1, IMG_SIZE, IMG_SIZE, device=device)

        # Iterate over each image in the batch, and generate specific noise using the corresponding seed
        for i, seed in enumerate(seeds):
            generator = torch.Generator(device=device)
            generator.manual_seed(seed.item())
            noise[i] = torch.randn(1, IMG_SIZE, IMG_SIZE, device=device, generator=generator)
        
        model_kwargs = {"cls": batch['cls'].to(device)}
        
        imgs = model.predict(noise, model_kwargs=model_kwargs, generator=generator, classifier_cond_scale=GUIDANCE_SCALE,
                           start_denoise_step=None, inference_protocol=f"DDIM{NUM_DDIM_STEPS}", 
                           post_process_fn=to_uint8, clip_denoised=True)

        for i, img in enumerate(imgs):
            img = img[0].cpu().numpy().transpose(1, 0)
            skimage.io.imsave(f"{DATA_SAVE_PATH}/{filenames[i]}.png", img)

        out_queue.put(len(batch))
        
        
def main():
    parser = argparse.ArgumentParser(description="Inference Script")
    parser.add_argument("--num-splits", type=int, default=1, help="Total number of splits to break the dataframe into")
    parser.add_argument("--idx", type=int, default=0, help="Index for breaking the dataframe into sections")

    args = parser.parse_args()
    num_splits = args.num_splits
    split_idx = args.idx
    assert split_idx < 5, "Index should be less than 5"

    # Set the number of GPUs to use
    world_size = torch.cuda.device_count()

    print("*"*50)
    print(f"Starting Inference with {world_size} GPUs")
    print("*"*50)
    
    # Load the data and create a DataLoader with your dataset
    df = pd.read_csv("./sample_seeded_data.csv").iloc[START_IDX:]
    df = df.reset_index(drop=True)
    split_size = math.ceil(len(df)/num_splits) 
    selectd_df = df.iloc[split_idx*split_size:min(len(df),(split_idx+1)*split_size)]
    selectd_df = selectd_df.reset_index(drop=True)
    data = get_data_dict(selectd_df)
    
    label_transforms=mn.transforms.Compose([
        mn.transforms.ToTensorD(keys=["age", "sex", "race", *PATHOLOGIES], dtype=torch.float),
        mn.transforms.AddChannelD(keys=["age", *PATHOLOGIES]),
        mn.transforms.AsDiscreteD(keys=["sex", "race"], to_onehot=[2, 5]),
        mn.transforms.ConcatItemsD(keys=["age", "sex", "race", *PATHOLOGIES], name='cls'),
        mn.transforms.SelectItemsD(keys=["filename","cls", "seed"]),
    ])
    
    dataset = mn.data.Dataset(data, transform=label_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    
    # Create Queues and start the subprocesses
    in_queue = mp.Queue()
    out_queue = mp.Queue()
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=infer, args=(rank, in_queue, out_queue, "PATH/TO/DDPM/CHECKPOINT.ckpt", "./mediffusion_config.yaml"))
        p.start()
        processes.append(p)
    
    # Create a progress bar to show the progress of the inference
    pbar = tqdm(total=len(dataloader))
    
    # Iterate over the data in the dataloader and add it to the queue
    for i, batch in enumerate(dataloader):
        # Add the index of the item to the item dictionary
        in_queue.put(batch)
        
        # Update the progress bar based on the number of items processed by the subprocesses
        while not out_queue.empty():
            out_queue.get()
            pbar.update(1)
    
    # Add sentinel values to the queue to signal the subprocesses to exit
    for _ in range(world_size):
        in_queue.put(None)
    
    # Wait for all subprocesses to finish
    for p in processes:
        p.join()
    
    # Update the progress bar to show that all items have been processed
    pbar.update(pbar.total - pbar.n)
    pbar.close()

if __name__ == '__main__':
    main()
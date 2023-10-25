# %%
import os
import math
import monai as mn
from tqdm import tqdm
import pandas as pd
import bkh_pytorch_utils as bpu
from mediffusion import DiffusionModule, Trainer
import torch
# %%
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" if bpu.is_notebook_running() else "0,1,2,3"
os.environ['WANDB_API_KEY']='WANDB-API-KEY'
os.environ['WANDB_SILENT']='true'

# %%
SEED=7654321
bpu.seed_all(SEED)

MONAI_CACHE_DIR='./cache'
DATA_PATH = '/PATH/TO/CheXpert/chexpertchestxrays-u20210408'

FOLD = 0

IMG_SIZE = 256
BATCH_SIZE = 36
TOTAL_IMAGE_SEEN = 40e6
NUM_DEVICES = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
TRAIN_ITERATIONS = int(TOTAL_IMAGE_SEEN / (BATCH_SIZE * NUM_DEVICES))

PATHOLOGIES = ['Pleural Other', 'Fracture', 'Support Devices','Pleural Effusion','Pneumothorax','Atelectasis','Pneumonia','Consolidation', 'Edema','Lung Lesion', 'Lung Opacity', 'Cardiomegaly','Enlarged Cardiomediastinum', 'No Finding']

# %%
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
            'img': f"{DATA_PATH}/{row['Path']}",
            'sex': conversion_schema['sex'][row['Sex']],
            'race': conversion_schema['race'][row['Race']],
            'age': row['Age']//10,
            ** {pathology: row[pathology] for pathology in PATHOLOGIES}
        })
    
    return data_dict

# %%
df = pd.read_csv("/sample_train_data.csv")

val_df = df.iloc[-10:].copy()
val_df.reset_index(inplace=True, drop=True)
train_df = df.iloc[:-10].copy()
train_df.reset_index(inplace=True, drop=True)

# %%
val_dict = get_data_dict(val_df)
train_dict = get_data_dict(train_df)

# %%
train_transforms=mn.transforms.Compose([
    mn.transforms.LoadImageD(keys="img"),
    bpu.EnsureGrayscaleD(keys="img"),
    mn.transforms.ResizeD(keys='img', size_mode="longest", mode="bilinear", spatial_size=IMG_SIZE, align_corners=False),
    mn.transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0, upper=100, b_min=-1, b_max=1, clip=True),
    mn.transforms.SpatialPadD(keys='img', spatial_size=(IMG_SIZE, IMG_SIZE), mode="constant", constant_values=-1),
    mn.transforms.ToTensorD(keys=["age", "sex", "race", *PATHOLOGIES], dtype=torch.float),
    mn.transforms.AddChannelD(keys=["age", *PATHOLOGIES]),
    mn.transforms.AsDiscreteD(keys=["sex", "race"], to_onehot=[2, 5]),
    mn.transforms.ConcatItemsD(keys=["age", "sex", "race", *PATHOLOGIES], name='cls'),
    mn.transforms.SelectItemsD(keys=["img","cls"]),
    mn.transforms.ToTensorD(keys=["img","cls"], dtype=torch.float, track_meta=False),
])
val_transforms=mn.transforms.Compose([
    mn.transforms.LoadImageD(keys="img"),
    bpu.EnsureGrayscaleD(keys="img"),
    mn.transforms.ResizeD(keys='img', size_mode="longest", mode="bilinear", spatial_size=IMG_SIZE, align_corners=False),
    mn.transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0, upper=100, b_min=-1, b_max=1, clip=True),
    mn.transforms.SpatialPadD(keys='img', spatial_size=(IMG_SIZE, IMG_SIZE), mode="constant", constant_values=-1),
    mn.transforms.ToTensorD(keys=["age", "sex", "race", *PATHOLOGIES], dtype=torch.float),
    mn.transforms.AddChannelD(keys=["age", *PATHOLOGIES]),
    mn.transforms.AsDiscreteD(keys=["sex", "race"], to_onehot=[2, 5]),
    mn.transforms.ConcatItemsD(keys=["age", "sex", "race", *PATHOLOGIES], name='cls'),
    mn.transforms.SelectItemsD(keys=["img","cls"]),
    mn.transforms.ToTensorD(keys=["img","cls"], dtype=torch.float, track_meta=False),
])

# %%
bpu.empty_monai_cache(MONAI_CACHE_DIR)

train_ds=mn.data.PersistentDataset(data=train_dict, transform=train_transforms, cache_dir=f"{MONAI_CACHE_DIR}/train")
val_ds=mn.data.PersistentDataset(data=val_dict*NUM_DEVICES, transform=val_transforms, cache_dir=f"{MONAI_CACHE_DIR}/val")
train_sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=int(TOTAL_IMAGE_SEEN))

# %%
trainer = Trainer(
    max_steps=TRAIN_ITERATIONS,
    val_check_interval=5000,
    root_directory="./outputs",
    precision="16-mixed",       
    devices=-1,
    nodes=1,
    wandb_project="Your_Project_Name",
    logger_instance="Your_Logger_Instance",
)

# %%
model=ddpm.DiffusionPLModule("./mediffusion_config.yaml",
        train_ds=train_ds, val_ds=val_ds, dl_workers=2, train_sampler=train_sampler,
        batch_size=BATCH_SIZE, val_batch_size=BATCH_SIZE//2)

model.stats()
model.compile()
# %%
trainer.fit(model)
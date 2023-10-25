# %%
import os
import sys
import pandas as pd
import numpy as np
import monai as mn
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
import bkh_pytorch_utils as bpu
import torch
import torchmetrics as tm
import timm
import argparse

# %%
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" if bpu.is_notebook_running() else "0,1,2,3"
os.environ['WANDB_API_KEY']='WANDB-API-KEY'
os.environ['WANDB_SILENT']='true'

# %%
parser = argparse.ArgumentParser(description="Set up augmentation ratio.")
parser.add_argument("--aug-ratio", type=int, default=2, help="Augmentation ratio for data.")
args = parser.parse_args()

# %%
SEED=7654321
bpu.seed_all(SEED)

MONAI_CACHE_DIR='./cls_cache'
REAL_DATA_PATH = '/PATH/TO/CheXpert/chexpertchestxrays-u20210408'
SYNTH_DATA_PATH = '/PATH/TO/SYNTHETIC_DATA'

FOLD = 0

MODEL_TYPE="convnext_base_in22ft1k"
IMG_SIZE = 256
BATCH_SIZE = 32
PRECISION = 16
LEARNING_RATE = 1e-5
EPOCHS = 400
AUGMENT_RATIO = args.aug_ratio

PATHOLOGIES = ['Pleural Other', 'Fracture', 'Support Devices','Pleural Effusion','Pneumothorax','Atelectasis','Pneumonia','Consolidation', 'Edema','Lung Lesion', 'Lung Opacity', 'Cardiomegaly','Enlarged Cardiomediastinum', 'No Finding'] # , 

# %%
real_df = pd.read_csv("./sample_train_data.csv")

# %%
real_train_df = real_df[(real_df['Fold']!=FOLD)]
real_val_df = real_df[real_df['Fold']==FOLD]

real_train_df = real_train_df.reset_index(drop=True)
real_val_df = real_val_df.reset_index(drop=True)

# %%
def get_data_dict(df_part, add_synth, include_real):
    synth_df = pd.read_csv("./sample_seeded_data.csv")
    data_dict = list()
    if include_real:
        for i in tqdm(range(len(df_part)), desc="Processing Real Data"):
            row = df_part.iloc[i]
            data_dict.append({
                'img': f"{REAL_DATA_PATH}/{row['Path']}",
                ** {pathology: row[pathology] for pathology in PATHOLOGIES}
            })

    # synth_data
    if add_synth and AUGMENT_RATIO > 0:
        synth_rows = synth_df[(synth_df['ReplicaID'] < AUGMENT_RATIO) & (synth_df['Fold'] != FOLD)].reset_index(drop=True)
        for row in tqdm(range(len(synth_rows)), desc="Processing Synth Data"):
            synth_row = synth_rows.iloc[row]
            data_dict.append({
                'img': f"{SYNTH_DATA_PATH}/{synth_row['Save Path']}.png",
                ** {pathology: synth_row[pathology] for pathology in PATHOLOGIES}
            })
    
    return data_dict

# %%
train_dict = get_data_dict(real_train_df, add_synth=True, include_real=True)
val_dict = get_data_dict(real_val_df, add_synth=False, include_real=True)

# %%
train_transforms=mn.transforms.Compose([
    mn.transforms.LoadImageD(keys="img", ensure_channel_first=False),
    bpu.EnsureGrayscaleD(keys="img"),
    mn.transforms.ResizeD(keys='img', size_mode="longest", mode="bilinear", spatial_size=IMG_SIZE, align_corners=False),
    mn.transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0, upper=100, b_min=-1, b_max=1, clip=True),
    mn.transforms.SpatialPadD(keys='img', spatial_size=(IMG_SIZE, IMG_SIZE), mode="constant", constant_values=-1),
    mn.transforms.RandFlipD(keys="img", spatial_axis=0, prob=0.5),
    mn.transforms.RandFlipD(keys="img", spatial_axis=1, prob=0.5),
    mn.transforms.RandGaussianNoiseD(keys="img", mean=0.0, std=0.3, prob=0.5),
    mn.transforms.RandAffineD(keys="img", mode="bilinear", prob=0.5, rotate_range=0.4, scale_range=0.1, translate_range=IMG_SIZE//20, padding_mode="border"),
    mn.transforms.ToTensorD(keys=[*PATHOLOGIES], dtype=torch.float),
    mn.transforms.AddChannelD(keys=[*PATHOLOGIES]),
    mn.transforms.ConcatItemsD(keys=[*PATHOLOGIES], name='cls'),
    mn.transforms.SelectItemsD(keys=["img","cls"]),
    mn.transforms.ToTensorD(keys=["img","cls"], dtype=torch.float, track_meta=False),
])
val_transforms=mn.transforms.Compose([
    mn.transforms.LoadImageD(keys="img", ensure_channel_first=False),
    bpu.EnsureGrayscaleD(keys="img"),
    mn.transforms.ResizeD(keys='img', size_mode="longest", mode="bilinear", spatial_size=IMG_SIZE, align_corners=False),
    mn.transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0, upper=100, b_min=-1, b_max=1, clip=True),
    mn.transforms.SpatialPadD(keys='img', spatial_size=(IMG_SIZE, IMG_SIZE), mode="constant", constant_values=-1),
    mn.transforms.ToTensorD(keys=[*PATHOLOGIES], dtype=torch.float),
    mn.transforms.AddChannelD(keys=[*PATHOLOGIES]),
    mn.transforms.ConcatItemsD(keys=[*PATHOLOGIES], name='cls'),
    mn.transforms.SelectItemsD(keys=["img","cls"]),
    mn.transforms.ToTensorD(keys=["img","cls"], dtype=torch.float, track_meta=False)
])

# %%
bpu.empty_monai_cache(MONAI_CACHE_DIR)

train_ds=mn.data.PersistentDataset(data=train_dict, transform=train_transforms, cache_dir=f"{MONAI_CACHE_DIR}/train")
val_ds=mn.data.PersistentDataset(data=val_dict, transform=val_transforms, cache_dir=f"{MONAI_CACHE_DIR}/val")

# %%
class PLModule(bpu.BKhModule):
    def __init__(self, network_type="b4", num_classes=2, in_chans=3, collate_fn=None, val_collate_fn=None, train_sampler=None, val_sampler=None, train_ds=None, val_ds=None, dl_workers=-1, pin_memory=True, batch_size=2, val_batch_size=None, lr=1e-4, prefetch_factor=1, persistent_workers=False):
        super().__init__(collate_fn=collate_fn, val_collate_fn=val_collate_fn, train_sampler=train_sampler, val_sampler=val_sampler, train_ds=train_ds, val_ds=val_ds, dl_workers=dl_workers, batch_size=batch_size, val_batch_size=val_batch_size, pin_memory=pin_memory, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
        
        self.lr=lr
        self.num_classes=num_classes

        self.model = timm.create_model(network_type, pretrained=True, in_chans=in_chans, num_classes=self.num_classes)

        self.mixup = bpu.Mixup(
            mixup_alpha=1.0,
            cutmix_alpha=1.0,
            prob=0.5,
            mode="batch",
            num_classes=self.num_classes,
            label_smoothing=0.1,
            one_hot_encode=False,
        )

        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
        if self.num_classes == 1:
            self.train_auc = tm.AUROC(task="binary")
            self.val_auc = tm.AUROC(task="binary")
        else:
            self.train_auc = tm.AUROC(num_labels=num_classes, task="multilabel", average="macro")
            self.val_auc = tm.AUROC(num_labels=num_classes, task="multilabel", average="macro")

    def forward(self, img):
        return self.model(img)
            
    def training_step(self, batch, batch_idx):
        imgs, target = self.mixup(batch['img'], batch['cls'])
        
        out = self(imgs)
        loss = self.loss_fn(out,target)
        
        probs = torch.sigmoid(out)

        self.train_auc(probs,torch.round(target).long())

        self.log('train_loss', loss.item(), on_epoch=True, on_step=False, prog_bar=False)
        self.log('train_auc', self.train_auc, on_epoch=True, on_step=False, prog_bar=False)

        return loss
        
    def validation_step(self, batch, batch_idx):
        imgs, target = batch['img'], batch['cls']
        
        out = self(imgs)
        loss = self.loss_fn(out,target)
        
        probs = torch.sigmoid(out)

        self.val_auc(probs,target.long())

        self.log('valid_loss', loss.item(), on_epoch=True, on_step=False, prog_bar=False)
        self.log('valid_auc', self.val_auc, on_epoch=True, on_step=False, prog_bar=False)

        return loss

    def configure_optimizers(self):
        if self.total_steps is None:
            max_epochs=self.trainer.max_epochs
            grad_acc=self.trainer.accumulate_grad_batches
            self.set_total_steps(steps=len(self.train_dataloader())*max_epochs//grad_acc)

        params = bpu.add_weight_decay(self.model,3e-4)
        optimizer = bpu.Lion(params, lr=self.lr)

        return optimizer

# %%
def get_trainer(logger_instance, distributed=True):
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback1= ModelCheckpoint(
        dirpath="../outputs/pl",
        filename=f'Aug:{AUGMENT_RATIO}_F{FOLD}_{{epoch}}_{{valid_auc:0.4F}}_{{valid_loss:0.4F}}_SR',
        monitor="valid_loss",
        mode="min",
        save_last=False,
        save_top_k=1,
    )
    early_stop_callback = EarlyStopping(
        monitor='valid_loss',
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='min'
    )
    
    wandb_logger = WandbLogger(
        save_dir="../outputs/wandb",
        name=logger_instance,
        project='Chexpert_Classification',
        offline=False,
        log_model=False,
    )
    
    progress_bar = RichProgressBar()
    ema = bpu.EMA(decay=0.9999, ema_interval_steps=1, ema_device="cuda")

    return wandb_logger, pl.Trainer(
        gradient_clip_val=1.0,
        deterministic=False,
        callbacks=[progress_bar, lr_monitor, checkpoint_callback1, early_stop_callback, ema],
        profiler='simple',
        logger=wandb_logger,
        precision=PRECISION,
        accelerator="gpu",
        devices=1 if bpu.is_notebook_running() or not distributed else -1,
        strategy='auto' if bpu.is_notebook_running() else DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=1,
        default_root_dir= "../outputs/pl",
        num_sanity_val_steps=0,
        fast_dev_run=False,
        max_epochs=EPOCHS,
        use_distributed_sampler = False if bpu.is_notebook_running() else True,
)

# %%
model=PLModule(
    network_type=MODEL_TYPE,
    num_classes=len(PATHOLOGIES),
    in_chans=1,
    train_ds=train_ds,
    val_ds=val_ds,
    train_sampler=None,
    collate_fn=torch.utils.data._utils.collate.default_collate,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    dl_workers = 2,
    pin_memory=True,
    prefetch_factor=1,
    persistent_workers=True
)
model.stats()

# %%
_,trainer=get_trainer(f'Classifier_{MODEL_TYPE}_F{FOLD}_AUG:{AUGMENT_RATIO}')
trainer.fit(model)

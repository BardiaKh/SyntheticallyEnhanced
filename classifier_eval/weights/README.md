# Model Weights

This folder contains all the classifier weights that were trained using different data augmentation ratios.

| Weight Path                                        | Real Data | Synthetic Data | AUC   |
|----------------------------------------------------|-----------|----------------|-------|
| ./Real+Synth/Aug:0_F0_epoch=28_valid_auc=0.8053.ckpt  | ✔         | ✖                      | 0.8053|
| ./Real+Synth/Aug:1_F0_epoch=18_valid_auc=0.8183.ckpt  | ✔         | ✔ (+100%)              | 0.8183|
| ./Real+Synth/Aug:2_F0_epoch=14_valid_auc=0.8242.ckpt  | ✔         | ✔ (+200%)              | 0.8242|
| ./Real+Synth/Aug:3_F0_epoch=12_valid_auc=0.8284.ckpt  | ✔         | ✔ (+300%)              | 0.8284|
| ./Real+Synth/Aug:4_F0_epoch=11_valid_auc=0.8307.ckpt  | ✔         | ✔ (+400%)              | 0.8307|
| ./Real+Synth/Aug:5_F0_epoch=10_valid_auc=0.8322.ckpt  | ✔         | ✔ (+500%)              | 0.8322|
| ./Real+Synth/Aug:6_F0_epoch=10_valid_auc=0.8326.ckpt  | ✔         | ✔ (+600%)              | 0.8326|
| ./Real+Synth/Aug:7_F0_epoch=9_valid_auc=0.8361.ckpt   | ✔         | ✔ (+700%)              | 0.8361|
| ./Real+Synth/Aug:8_F0_epoch=9_valid_auc=0.8363.ckpt   | ✔         | ✔ (+800%)              | 0.8363|
| ./Real+Synth/Aug:9_F0_epoch=8_valid_auc=0.8370.ckpt   | ✔         | ✔ (+900%)              | 0.8370|
| ./Real+Synth/Aug:10_F0_epoch=8_valid_auc=0.8371.ckpt  | ✔         | ✔ (+1000%)             | 0.8371|
| ./Synth/Aug:1_F0_epoch=24_valid_auc=0.7930.ckpt  | ✖         | ✔ (+100%)              | 0.7930|
| ./Synth/Aug:2_F0_epoch=17_valid_auc=0.8107.ckpt  | ✖         | ✔ (+200%)              | 0.8107|
| ./Synth/Aug:3_F0_epoch=13_valid_auc=0.8177.ckpt  | ✖         | ✔ (+300%)              | 0.8177|
| ./Synth/Aug:4_F0_epoch=10_valid_auc=0.8219.ckpt  | ✖         | ✔ (+400%)              | 0.8219|
| ./Synth/Aug:5_F0_epoch=9_valid_auc=0.8229.ckpt   | ✖         | ✔ (+500%)              | 0.8229|
| ./Synth/Aug:6_F0_epoch=8_valid_auc=0.8235.ckpt   | ✖         | ✔ (+600%)              | 0.8235|
| ./Synth/Aug:7_F0_epoch=8_valid_auc=0.8280.ckpt   | ✖         | ✔ (+700%)              | 0.8280|
| ./Synth/Aug:8_F0_epoch=7_valid_auc=0.8282.ckpt   | ✖         | ✔ (+800%)              | 0.8282|
| ./Synth/Aug:9_F0_epoch=8_valid_auc=0.8306.ckpt   | ✖         | ✔ (+900%)              | 0.8306|
| ./Synth/Aug:10_F0_epoch=6_valid_auc=0.8288.ckpt  | ✖         | ✔ (+1000%)             | 0.8288|

The `weight_compression.py` is responsible for removing unnecessary keys from the weight files to compress them by 60%.
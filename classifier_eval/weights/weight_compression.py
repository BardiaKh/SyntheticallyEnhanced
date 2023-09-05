import torch
from glob import glob
from tqdm import tqdm

WEIGHTS_PATH = '.'

def main():
    weights = glob(f"{WEIGHTS_PATH}/*/*.ckpt")
    for i, weight in enumerate(tqdm(weights, desc=f"Compressing weights")):
        weight_file = torch.load(weight)
        keys = list(weight_file.keys())
        for key in keys:
            if key not in ["ema_state_dict","pytorch-lightning_version"]:
                del weight_file[key]

        torch.save(weight_file, weight)

if __name__ == "__main__":
    main()
# Pathology Classification

## Setup and Installation

### Step 1: Create a Conda Environment

If you haven't installed Conda yet, you can download it from [here](https://docs.anaconda.com/anaconda/install/). After installing, create a new Conda environment by running:

```bash
conda create --name synth-eval python=3.10
```

Activate the environment:

```bash
conda activate synth-eval
```

### Step 2: Install PyTorch

Install PyTorch specifically for CUDA 11.8 by running:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install Other Dependencies

Navigate to the project directory where the `requirements.txt` file is located, and run:

```bash
pip install -r requirements.txt
```

This will install all the necessary packages listed in the `requirements.txt` file.

## Running the Code

A sample python file for training the model is provided in `classifier_training.py`. To structure your real data you can use `sample_train_data.csv`.  To use synthetic data augmentation, you can use the `sample_seeded_data.csv` as a template (the same file is used for DDPM-based image generation).

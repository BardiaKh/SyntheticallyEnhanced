# Model Evaluation

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

## Running the Project

### Step 1: Format the Data
Our program accepts a `.csv` file as input. A sample schema for this file can be found in `sample_schema.csv`. Note that you are allowed to have as many or as few of the pathologies in the list of following pathologies for which your dataset allows:
* `Pleural Other`
* `Fracture`
* `Support Devices`
* `Pleural Effusion`
* `Pneumothorax`
* `Atelectasis`
* `Pneumonia`
* `Consolidation`
* `Edema`
* `Lung Lesion`
* `Lung Opacity`
* `Cardiomegaly`
* `Enlarged Cardiomediastinum`
* `No Finding`

**Note**: The file location column should be named `Path`. Also, make sure the column names are *case sensitive*.

### Step 2: Run Inference
To run inference with our model, you will need to execute the `classifiers_eval.py` file. To do so, you must provide the required positional argument `input`, as well as any number of optional arguments. Below is an example of a terminal command you could run to get inference results:
```bash
python classifiers_eval.py /absolute/path/to/input.csv --output results.csv --workers 8 --cuda_id 3
```
Here is a list of the possible arguments you can provide and their effect on the inference script:
* `input` is a **required** positional argument containing the absolute path your formatted `.csv` file.
* `--output` is an optional argument containing the name of the output `.csv` file in which to save inference results. Defaults to `./results.csv`.
* `--bs` is an optional containing the batch size for inference. Defaults to `32`.
* `--workers` is an optional argument contaning the number of CPU workers in the dataloader. Defaults to `4`.
* `--base_path` is the absolute path of the location to which your image paths are relative. Defaults to `None`.
* `--cuda_id` is the ID of the GPU on which you would like to run inference. Defaults to `0`.
* `--cache_dir` is the directory in which to store cached data (will be removed after inference). Defaults to `./cache`.

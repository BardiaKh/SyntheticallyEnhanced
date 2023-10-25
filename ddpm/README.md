# Image Genration

## Model Training

We used [Mediffusion](https://github.com/BardiaKh/Mediffusion) for training our diffusion model. Please refere to the package to learn more about its installation.

Model configs are present in the `mediffusion_config.yaml` file. Additionally, a sample python file for training the model is provided in `ddpm_training.py`. To structure your data, you can use `sample_train_data.csv` as a template.

## Image Generation

Sample generation script is located at `ddpm_inference.py`. To structure your data, you can use `sample_seeded_data.csv` as a template. 
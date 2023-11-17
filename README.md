# Synthetically Enhanced: Unveiling Synthetic Data’s Potential in Medical Imaging Research
[arxiv pre-print]([https://arxiv.com](https://arxiv.org/abs/2311.09402))

<img src="https://i.ibb.co/cQtRz8T/SE-GA.jpg"/>

Chest X-rays (CXR) are the most common medical imaging study and are used to diagnose multiple medical conditions. This study examines the impact of synthetic data supplementation, using diffusion models, on the performance of deep learning (DL) classifiers for CXR analysis. We employed three datasets: CheXpert, MIMIC-CXR, and Emory Chest X-ray, training conditional denoising diffusion probabilistic models (DDPMs) to generate synthetic frontal radiographs. Our approach ensured that synthetic images mirrored the demographic and pathological traits of the original data. Evaluating the classifiers' performance on internal and external datasets revealed that synthetic data supplementation enhances model accuracy, particularly in detecting less prevalent pathologies. Furthermore, models trained on synthetic data alone approached the performance of those trained on real data. This suggests that synthetic data can potentially compensate for real data shortages in training robust DL models. However, despite promising outcomes, the superiority of real data persists.

---

This repository contains the following information that we believe will improve our work's reproducibility:

- Training the diffusion model (`/ddpm`) 
- Data splits for public datasets (`/data_splits`)
- Training code for pathology classifiers (`/classifier_train`)
- Validation scripts for pathology classifiers (`/classifier_eval`)

---

*Last update: 11/15/2023*

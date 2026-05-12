# GalaxEye Satellite Change Detection

## Overview

This project performs building damage detection using multi-modal satellite imagery:
- EO (Electro Optical) imagery
- SAR (Synthetic Aperture Radar) imagery

A UNet-based deep learning segmentation model was trained to detect damaged building regions.

---

## Project Structure

```text
data/
    train/
    val/
    test/

outputs/
    unet_model.pth
    prediction.png

dataset.py
model.py
train.py
evaluate.py
inference.py
requirements.txt

Markdown
Installation

Install required libraries:pip install -r requirements.txt

Training

Run the training pipeline:python train.py

Evaluation

Run evaluation metrics:python evaluate.py

This calculates:
IoU
Precision
Recall

Inference

Generate prediction visualization:python inference.py

This produces:
EO image
SAR image
Ground truth mask
Predicted mask

Output image is saved in:outputs/prediction.png

Model Details
Model Architecture: UNet
Encoder Backbone: ResNet34
Framework: PyTorch
Input Modalities:
EO imagery
SAR imagery

## Evaluation Metrics

| Metric    | Value  |
| ----------| ------ |
| IoU       | 0.1636 |
| Precision | 0.4238 |
| Recall    | 0.1753 |
| F1 Score  | 0.1901 |

Observations

The model successfully learned building damage regions from EO and SAR satellite imagery. Predicted masks show partial overlap with ground truth damage regions. Performance can be improved further with:

longer training
more epochs
hyperparameter tuning
advanced augmentation techniques

Environment Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Author

S M Puneeth Krishna
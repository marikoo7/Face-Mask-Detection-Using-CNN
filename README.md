# Face-Mask-Detection-Using-CNN
This project implements an automated Face Mask Detection system using deep learning techniques. The goal is to classify facial images into two categories: with mask and without mask, using custom CNN architectures and a transfer learning approach.

## Project Overview
The project explores multiple deep learning models for face mask detection, including a baseline CNN, an improved CNN with regularization techniques, and a transfer learning model based on EfficientNetB0.
All models are trained, evaluated, and compared using a unified pipeline to identify the best-performing approach.

## Dataset
Source: Kaggle Face Mask Detection Dataset

Classes:
with_mask
without_mask

Image Size: 128 × 128

Data Split:
Training: 70%
Validation: 15%
Test: 15%

Preprocessing and augmentation are applied to improve model generalization. EfficientNetB0 uses its official preprocessing function.

## Project Structure
FACE-MASK-DETECTION-USING-CNN/

│

├── Project_7_7/

│   │

│   ├── .idea/                     # IDE configuration files

│   │   ├── inspectionProfiles/

│   │   ├── profiles_settings.xml

│   │   ├── misc.xml

│   │   └── vcs.xml

│   │

│   ├── code/

│   │   ├── __pycache__/            # Cached Python bytecode files

│   │   │   ├── model.cpython-313.pyc

│   │   │   └── utils.cpython-313-pytest-8.3.4.pyc

│   │   │

│   │   ├── dataset_split/          # Dataset after splitting

│   │   │   ├── train/

│   │   │   │   ├── with_mask/

│   │   │   │   └── without_mask/

│   │   │   │

│   │   │   ├── val/

│   │   │   │   ├── with_mask/

│   │   │   │   └── without_mask/

│   │   │   │

│   │   │   └── test/

│   │   │       ├── with_mask/

│   │   │       └── without_mask/

│   │   │

│   │   ├── raw_images_combined/    # Raw dataset before splitting

│   │   │   ├── with_mask/

│   │   │   └── without_mask/

│   │   │

│   │   ├── dataset.py              # Dataset preparation and generators

│   │   ├── model.py                # Model architectures

│   │   ├── train.py                # Training pipeline

│   │   ├── evaluate.py             # Evaluation and visualization

│   │   └── utils.py                # Model summaries and statistics

│   │

│   ├── results/                    # Evaluation outputs and plots

│   │

│   ├── saved_model/                # Saved trained models

│   │

│   ├── Project by CNN.pdf         # Project documentation (PDF format)

│   │

│   └── README.md                   # Project README file



## Models Implemented

Baseline CNN:
A lightweight CNN used as a performance benchmark.

Improved CNN:
A deeper CNN with batch normalization and dropout to improve stability and generalization.

EfficientNetB0 (Transfer Learning):
A pre-trained model used as a feature extractor, providing the best overall performance.

## Training
Optimizer: Adam

Loss Function: Binary Cross-Entropy

Epochs: 20

Batch Size: 32

Callbacks:
EarlyStopping
ModelCheckpoint
ReduceLROnPlateau

Each model is trained independently, and the best checkpoint is saved based on validation accuracy.

## Evaluation
Models are evaluated using a separate test set. The evaluation includes:

1.Test Accuracy

2.Precision, Recall, and F1-score

3.Confusion Matrix

3.ROC Curve and AUC

4.Training Accuracy and Loss Curves

5.Sample Prediction Visualizations

A final comparison is performed to rank models based on accuracy and AUC.

## Results Summary
| Model          | Accuracy (%) | AUC    |
| -------------- | ------------ | ------ |
| Baseline CNN   | 78.05        | 0.8524 |
| Improved CNN   | 88.98        | 0.9871 |
| EfficientNetB0 | 98.31        | 0.9984 |

Best Model: EfficientNetB0

## How to Run
1.Install dependencies:
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow

2.Train models:
python train.py

3.Evaluate models:
python evaluate.py


## Conclusion
The results demonstrate that transfer learning using EfficientNetB0 significantly outperforms custom CNN architectures for face mask detection. While simpler CNN models offer lower computational cost, EfficientNetB0 provides superior accuracy and robustness, making it the preferred choice for real-world deployment.

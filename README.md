# DFUC_CAM

**Multi-stage Segmentation of Diabetic Foot Ulcers Using Self-Supervised Learning**

[![License](https://img.shields.io/github/license/RyersonMultimediaLab/DFUC_CAM)](LICENSE)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Methodology](#methodology)
  - [1. CAM generation](#1-pseudo-label-generation)
  - [2. Self-Supervised Learning](#2-self-supervised-learning)
  - [3. Fine-tuning for Segmentation](#3-fine-tuning-for-segmentation)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

Diabetic Foot Ulcers (DFUs) are a significant healthcare challenge, often leading to severe complications such as infections and ischemia, which may necessitate limb amputation or result in mortality. Accurate and prompt diagnosis of ulcer regions is crucial for preventing these outcomes. Manual examination of DFUs is labor-intensive and subject to variability. To address these challenges, we present **DFUC_CAM**, a novel method for instance segmentation of DFUs leveraging self-supervised learning.

**DFUC_CAM** employs a multi-stage methodology:
1. **Class Activation Maps (CAMs)**: Utilizes a convolutional neural network (CNN) trained for classification to generate CAMs that localize and highlight discriminative regions within DFU images.
2. **Self-Supervised Learning Framework**: Uses CAMs as pseudo labels to train a segmentation network.
3. **Segmentation Heads**: Fine-tunes three different segmentation heads to optimize DFU segmentation performance.

Our approach demonstrates significant potential in enhancing diabetic foot ulcer detection and segmentation in clinical settings.

## Features

- **Self-Supervised Learning**: Reduces reliance on large labeled datasets by learning from unlabeled data.
- **Class Activation Maps (CAMs)**: Provides interpretable localization of ulcer regions.
- **Multi-Stage Segmentation**: Combines classification, self-supervised pretraining, and fine-tuning for robust segmentation.
- **Flexible Segmentation Heads**: Supports Linear, CNN, and UNet segmentation heads for diverse requirements.
- **Dataset Augmentation**: Expands the training dataset through intelligent cropping and extensive augmentation.

## Methodology

### 1. CAM generation

- **Objective**: Convert a global classification task into a weakly supervised localization problem.
- **Approach**:
  - Train a binary classification network using **RepLKNet** to differentiate between ulcer and non-ulcer images.
  - Generate Class Activation Maps (CAMs) to highlight discriminative ulcer regions.
  - Utilize CAMs as pseudo labels for subsequent training stages.

### 2. Self-Supervised Learning

- **Objective**: Enhance the network’s feature representations without additional manual annotations.
- **Approach**:
  - Adapt the **Masked AutoEncoders (MAE)** architecture for ulcer detection.
  - Expand the training dataset from 4,000 to 360,000 images using CAM-based intelligent cropping.
  - Apply extensive augmentations (e.g., motion blur, color adjustments) to increase dataset variety and robustness.

### 3. Fine-tuning for Segmentation

- **Objective**: Achieve precise DFU segmentation using specialized segmentation heads.
- **Approach**:
  - **Linear Head**: Predicts low-resolution segmentation maps.
  - **CNN Head**: Recovers detailed spatial information with convolutional layers.
  - **UNet Head**: Utilizes a U-shaped architecture for multi-scale feature processing and precise pixel-wise segmentation.

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA (optional, for GPU support)

### Clone the Repository

```bash
git clone https://github.com/RyersonMultimediaLab/DFUC_CAM.git
cd DFUC_CAM
```

*Note: Ensure you have the appropriate versions of CUDA and PyTorch installed for GPU support.*

## Usage

Please use the Readme.md for subfolders

## Datasets

- **DFUC2024**: Used for training the classification and self-supervised learning stages.
- **FUSeg**: Utilized for quantitative evaluation and fine-tuning the segmentation heads.

*Note: Ensure you have the appropriate licenses and permissions to use these datasets.*

## Results

Our approach was evaluated in the MICCAI 2024 Diabetic Foot Ulcer Challenge (DFUC2024).

- **Validation Phase**: Achieved a mean Dice score of 0.342.
- **Final Testing Phase**: Improved the mean Dice score to 0.385.

Among the segmentation heads, the **UNet Head** demonstrated superior performance with a Dice score of 0.385, outperforming CNN and Linear heads.

## Acknowledgements

- **Funding**: Mitacs Globalink and the Natural Sciences and Engineering Research Council of Canada Discovery program.
- **Datasets**: DFUC2024 and FUSeg datasets for providing essential data for training and evaluation.

## License

This project is licensed under the [MIT License](LICENSE).

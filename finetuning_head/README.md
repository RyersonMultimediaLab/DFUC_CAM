
# Finetuning MAE for Ulcer Detection

This project implements semantic segmentation models based on Masked Autoencoders (MAE) for detecting ulcers in medical images.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Configuration](#configuration)
- [Key Components](#key-components)
- [Customization](#customization)
- [Acknowledgments](#acknowledgments)

## Prerequisites

- Docker
- NVIDIA GPU (recommended)
- NVIDIA Container Toolkit (for GPU support)

## Installation

1. Pull the Docker image:
   ```
   docker pull ayushnangia16/finetune_dice:v1
   ```

2. Run a container:
   ```
   docker run -it --gpus all -v /path/to/your/local/directory:/workspace ayushnangia16/finetune_dice:v1
   ```

3. Inside the container, install additional dependencies:
   ```
   pip install datasets
   pip install --upgrade albumentations
   ```

## Usage

There are three main notebooks:

- `CNN_finetune.ipynb`
- `Linear_finetune.ipynb`
- `UNet_finetune.ipynb`

To run a notebook:

1. Start Jupyter Notebook in the container:
   ```
   jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
   ```

2. Open the notebook in your browser.

3. Update paths to pretrained MAE weights and dataset.

4. Run all cells to train the model.

5. Use the inference code at the end to make predictions.

## Models

1. **Improved CNN Classifier**: Uses a small CNN on top of MAE features.
2. **Linear Classifier**: Uses a simple linear layer on MAE features.
3. **U-Net Decoder**: Uses a U-Net style decoder on MAE features.

## Configuration

Each notebook contains configurations for:

- Dataset and data loading
- Model architecture
- Optimization and learning rate schedule
- Training parameters

## Key Components

1. **Dataset**: Uses `SegmentationDataset` with `albumentations` for augmentations
2. **Optimizer**: AdamW
3. **Learning Rate Schedule**: CosineAnnealingLR
4. **Loss Functions**: Combination of Dice Loss and Focal Loss
5. **Evaluation**: Dice score

## Customization

You can customize the training by modifying the following:

1. **Data Augmentations**: Modify `train_transform` and `val_transform`
2. **Model Architecture**: Adjust the model definitions in each notebook
3. **Learning Rate**: Change `learning_rate` in the `train_model` function
4. **Training Duration**: Modify `num_epochs` in the `train_model` function
5. **Batch Size**: Adjust `batch_size` in the DataLoader initialization

## Acknowledgments

- This project uses pretrained MAE models for feature extraction
- The segmentation models are inspired by various semantic segmentation architectures
- The FUGseg_dilation dataset is used for training and evaluation


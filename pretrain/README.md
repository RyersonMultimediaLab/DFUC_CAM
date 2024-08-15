# Masked Autoencoder (MAE) Pre Training 

## Description

Masked Autoencoder (MAE) training setup using the MMSelfSup library. It's designed for pre-training Vision Transformer (ViT) models on custom datasets using self-supervised learning techniques.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Key Components](#key-components)
- [Customization](#customization)
- [Acknowledgments](#acknowledgments)
- [Additional Resources](#additional-resources)

## Installation

This project uses a Docker container for easy setup and consistent environments. Follow these steps to get started:

1. Ensure you have Docker installed on your system. If not, you can download it from [Docker's official website](https://www.docker.com/get-started).

2. Pull the Docker image:
   ```
   docker pull ayushnangia16/mmselfsup:v3
   ```

3. Run the Docker container, mounting your dataset:
   ```
   docker run -it --gpus all -v /path/to/your/dataset:/workspace/data ayushnangia16/mmselfsup:v3
   ```
   Replace `/path/to/your/dataset` with the actual path to your dataset on your host machine.

4. Once inside the container, clone this repository:
   ```
   git clone https://github.com/RyersonMultimediaLab/DFUC_CAM
   cd DFUC_CAM/pretrain
   ```
    For more detailed installation instructions, refer to the [MMSelfSup documentation](https://mmselfsup.readthedocs.io/).

## Usage

1. Prepare your dataset and update the `data_root` in the configuration file.

2. Run the training script:
   ```
   bash tools/dist_train.sh ${CONFIG} ${GPUS} --cfg-options model.pretrained=${PRETRAIN}
   ```
    For more detailed installation instructions, refer to the [MMSelfSup documentation](https://mmselfsup.readthedocs.io/).

## Configuration

The main configuration file is `configs/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k.py`. It includes settings for:

- Dataset and data loading
- Model architecture
- Optimization and learning rate schedule
- Training parameters

## Key Components

1. **Base Configurations**: 
   - Model: `mae_vit-base-p16.py`
   - Training schedule: `adamw_coslr-200e_in1k.py`
   - Runtime defaults: `default_runtime.py`

2. **Dataset**: Uses `mmcls.CustomDataset`

3. **Optimizer**: AdamW with custom learning rate scaling

4. **Learning Rate Schedule**: Linear warm-up followed by cosine annealing

5. **Training**: 400 epochs with checkpoint saving

For more details, see the [configuration file](configs/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k.py).

## Customization

You can customize the training by modifying the following in the configuration file:

1. **Data Root**: Change `data_root = '/workspace/Dest'` to your dataset location.
2. **Batch Size and Workers**: Adjust `batch_size` and `num_workers` based on your hardware.
3. **Learning Rate**: Modify the base learning rate in `optimizer = dict(lr=1.5e-4 * 4096 / 256, ...)`.
4. **Training Duration**: Change `train_cfg = dict(max_epochs=400)` for different training lengths.
5. **Model Architecture**: Update the base model configuration in `_base_`.
6. **Dataset Type**: Modify `dataset_type` and related configurations for different datasets.

## Acknowledgments

- This project uses the [MMSelfSup](https://github.com/open-mmlab/mmselfsup) library.
- The MAE implementation is based on the paper ["Masked Autoencoders Are Scalable Vision Learners"](https://arxiv.org/abs/2111.06377).

## Additional Resources

For more detailed information about the MMSelfSup framework and the MAE algorithm, please refer to the following resources:

- [MMSelfSup Overview](https://mmselfsup.readthedocs.io/en/latest/overview.html)
- [MAE Paper and Implementation Details](https://mmselfsup.readthedocs.io/en/latest/papers/mae.html)

These links provide comprehensive information about the framework's design, supported algorithms, and specific details about the MAE implementation in MMSelfSup.


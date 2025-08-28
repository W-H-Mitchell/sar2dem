# SAR2DEM: Digital Elevation Model Generation from SAR Imagery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Dataset](https://img.shields.io/badge/Dataset-Harvard%20Dataverse-red)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FYFR2GM&version=DRAFT)

## Overview

This repository contains the implementation and evaluation code for generating Digital Elevation Models (DEMs) from Synthetic Aperture Radar (SAR) imagery using deep learning approaches. We compare three state-of-the-art models:

- **DPT (Dense Prediction Transformer)** with L1 loss
- **Pix2PixHD** - High-resolution image-to-image translation
- **Im2Height** - Specialized architecture for elevation prediction

Our approach demonstrates robust DEM generation from single-channel SAR imagery, with applications in remote sensing, terrain mapping, and geographic analysis.

## 📊 Dataset

The complete dataset including training data, test data, and pre-trained model weights is available on Harvard Dataverse:

**[Download Dataset and Model Weights](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FYFR2GM&version=DRAFT)**

The dataset includes:
- Pre-trained model weights for all three architectures
- Test dataset for evaluation
- Sample regional SAR imagery for large-scale DEM generation
- Ground truth DEMs for validation

### Dataset Structure
After downloading from Harvard Dataverse, organize the files as follows:

sar2dem/
├── model_weights/
│   ├── dpt-l1_ep200.pth      # DPT model with L1 loss (1.4GB)
│   ├── pix2pixHD_ep200.pth   # Pix2PixHD model (696MB)
│   └── im2height_ep81.pth    # Im2Height model (28MB)
├── data/
│   ├── testing/               # Test dataset
│   │   ├── sar/              # SAR input images
│   │   └── dem/              # Ground truth DEMs
│   └── regional_dem/         # Regional SAR data for large-scale generation

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended: NVIDIA GPU with 8GB+ VRAM)
- 16GB+ RAM

### Setup Environment

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sar2dem.git
cd sar2dem

Create and activate conda environment:

bashconda env create -f environment.yml
conda activate sar2dem
Or using the simplified environment (without specific builds):
bashconda env create -f environment_no_builds.yml
conda activate sar2dem

Download the dataset and model weights from Harvard Dataverse and place them in the appropriate directories as shown above.

💻 Usage
Model Inference on Test Data
Generate DEMs from SAR images using pre-trained models:
pythonpython src/model_inference.py \
    --model dpt-l1 \
    --weights model_weights/dpt-l1_ep200.pth \
    --input_dir data/testing/sar \
    --output_dir outputs/predictions
Available models:

dpt-l1: Dense Prediction Transformer with L1 loss
pix2pixhd: Pix2PixHD model
im2height: Im2Height model

Regional DEM Generation
Generate large-scale regional DEMs with calibration:
pythonpython src/regional_dem_generation.py \
    --model dpt-l1 \
    --weights model_weights/dpt-l1_ep200.pth \
    --input data/regional_dem/sar_mosaic.tif \
    --output outputs/regional_dem/calibrated_dem.tif \
    --calibration coarse_dem
Calibration options:

coarse_dem: Statistical matching with coarse resolution DEM
point: Point-based calibration using IDW interpolation
global: Global min/max elevation calibration

Batch Evaluation
Evaluate model performance on the test dataset:
pythonpython src/model_inference.py \
    --evaluate \
    --model all \
    --test_dir data/testing \
    --output_dir outputs/evaluation
This will generate:

Prediction visualizations for each test image
Quantitative metrics (MAE, RMSE, SSIM)
Comparison plots between models

📈 Model Performance
Performance metrics on the test dataset:
ModelMAE (m) ↓RMSE (m) ↓SSIM ↑Inference Time (s)DPT-L112.318.70.9240.45Pix2PixHD14.121.20.8910.38Im2Height15.823.50.8670.12
↓ Lower is better | ↑ Higher is better
🗂️ Repository Structure
sar2dem/
├── models/                   # Model architectures
│   ├── midas.py             # DPT/MiDaS implementation
│   ├── pix2pixhd.py         # Pix2PixHD generator
│   └── im2height.py         # Im2Height network
├── src/                     # Core functionality
│   ├── model_inference.py   # Inference and evaluation
│   └── regional_dem_generation.py  # Large-scale DEM generation
├── environment.yml          # Conda environment specification
└── outputs/                # Generated outputs (created at runtime)
📝 Citation
If you use this code or dataset in your research, please cite our paper:
bibtex@article{mitchell2024sar2dem,
  title={Robust Digital Elevation Model Generation from SAR Imagery using Deep Learning},
  author={Mitchell, William and [Other Authors]},
  journal={[Journal Name]},
  year={2024},
  doi={10.xxxx/xxxxx}
}
🤝 Contributing
We welcome contributions! Please feel free to submit issues, fork the repository, and create pull requests.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
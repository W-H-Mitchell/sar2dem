# SAR2DEM: Digital Elevation Model Generation from SAR Imagery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Dataset](https://img.shields.io/badge/Dataset-Harvard%20Dataverse-red)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YFR2GM)

## Overview

This repository contains the implementation and evaluation code for generating Digital Elevation Models (DEMs) from Synthetic Aperture Radar (SAR) imagery using deep learning approaches. We compare three deep learning models:

- **DPT (Dense Prediction Transformer)** with L1 loss
- **Pix2PixHD** - High-resolution image-to-image translation
- **Im2Height** - Specialized architecture for elevation prediction

Our approach demonstrates robust DEM generation from single-channel SAR imagery, with applications in remote sensing, terrain mapping, and geographic analysis.

## 📊 Dataset

The complete test data and pre-trained model weights are available on Harvard Dataverse:

**[Download Dataset and Model Weights](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YFR2GM)**

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
    │   │   ├── AFGHANISTAN_0/
    │   │   │   ├── data/         # SAR input images
    │   │   │   └── label/        # Ground truth DEMs
    │   └── regional_dem/         # Regional SAR data for large-scale generation
    │       ├── clipped_sar.tif
    │       ├── clipped_coarse_dem_1km.tif
    │       └── clipped_ground_truth.tif

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Setup Environment

1. Clone the repository:

    git clone https://github.com/W-H-Mitchell/sar2dem.git
    cd sar2dem

2. Create and activate conda environment:

    conda env create -f environment.yml
    conda activate sar2dem

Or using the simplified environment (without specific builds):

    conda env create -f environment_no_builds.yml
    conda activate sar2dem

3. Download the dataset and model weights from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YFR2GM) and place them in the appropriate directories as shown above.

## 💻 Usage

### Single Image Inference

Generate DEM from a single SAR image:

    python src/model_inference.py \
        --sar_path data/testing/AFGHANISTAN_0/data/AFGHANISTAN_y0x24064_DESCENDING.tif \
        --dem_path data/testing/AFGHANISTAN_0/label/AFGHANISTAN_y0x24064_DESCENDING.tif \
        --output-dir outputs/images/ \
        --dpt_path model_weights/dpt-l1_ep200.pth \
        --pix2pixhd_path model_weights/pix2pixHD_ep200.pth \
        --im2height_path model_weights/im2height_ep81.pth

### Regional DEM Generation

Generate large-scale regional DEMs with calibration:

    python src/regional_dem_generation.py \
        --sar_path data/regional_dem/clipped_sar.tif \
        --model_path model_weights/dpt-l1_ep200.pth \
        --output_path outputs/regional_dem/dpt_l1_dem.tif \
        --calibration coarse_dem \
        --coarse_dem data/regional_dem/clipped_coarse_dem_1km.tif \
        --overlap 96 \
        --region_size 128

Calibration options:
- `global`: Global min/max elevation calibration
- `coarse_dem`: Statistical matching with coarse resolution DEM (default)
- `idw`: IDW interpolation using control points

Additional parameters:
- `--overlap`: Tile overlap in pixels (default: 96, must be < 384)
- `--region_size`: Region size for coarse DEM matching (default: 128)
- `--num_control_points`: Number of control points for IDW (default: 100)
- `--optimize_tiles`: Enable tile offset optimization for seamless mosaicing
- `--device`: Compute device, cuda or cpu (default: cuda)

## 📈 Model Performance

Performance metrics on the test dataset:

| Model | MAE (m) ↓ | RMSE (m) ↓ | SSIM ↑ |
|-------|-----------|------------|--------|
| **DPT-L1** | **41.0** | **50.8** | **0.801** |
| Pix2PixHD | 64.0 | 75.7 | 0.786 |
| Im2Height | 85.1 | 105.5 | 0.521 |

↓ Lower is better | ↑ Higher is better

## 🗂️ Repository Structure

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

## 📝 Citation

If you use this code or dataset in your research, please cite our paper.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dense Prediction Transformer implementation based on [Intel ISL MiDaS](https://github.com/isl-org/MiDaS)
- Pix2PixHD architecture adapted from [NVIDIA's implementation](https://github.com/NVIDIA/pix2pixHD)
- SAR data processing supported by [Harvard Dataverse](https://dataverse.harvard.edu/)

## 📧 Contact

For questions, issues, or collaborations, please contact:
- William Mitchell - [whamitch@mit.edu]
- Open an issue on [GitHub](https://github.com/W-H-Mitchell/sar2dem/issues)

---

**Note**: The dataset on Harvard Dataverse will be made publicly available upon publication of the associated research paper.

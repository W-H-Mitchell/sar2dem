import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from skimage.metrics import structural_similarity as ssim
import warnings

from models.pix2pixhd import GlobalGenerator
from models.im2height import Im2Height
warnings.filterwarnings('ignore')


# CONFIGURATION
class Config:
    # Model paths - update these to the downloaded model weights
    MODEL_PATHS = {
        "DPT": "./model_weights/DPT-L1_ep200.pth",
        "pix2pixHD": "./model_weights/pix2pixHD_ep200.pth",
        "Im2Height": "./model_weights/im2height_sar2dem.pth"
    }
    
    # SAR normalization parameters (from training)
    SAR_MIN = 0
    SAR_MAX = 3000
    SAR_MEAN = 0.5
    SAR_STD = 0.5
    
    # DEM normalization parameters (from training dataset)
    DEM_MIN = 0.17894675
    DEM_MAX = 7695.1045
    
    # Input/Output paths
    INPUT_SAR_PATH = None
    GT_DEM_PATH = None  # Optional ground truth
    OUTPUT_DIR = "./outputs/"


# MODEL DEFINITIONS
def load_dpt_model(checkpoint_path, device):
    """Load DPT-Hybrid (MiDaS) model"""
    model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid', 
                          pretrained=False, trust_repo=True)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model

def load_pix2pixhd_model(checkpoint_path, device='cuda'):
    """Load Pix2PixHD model"""
    model = GlobalGenerator(input_nc=3, output_nc=1, ngf=64, 
                           n_downsampling=4, n_blocks=9)
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    if 'model' in state_dict:
        state_dict = state_dict['model']
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device).eval()
    return model

def load_im2height_model(checkpoint_path, device):
    """Load Im2Height model"""
    model = Im2Height() 
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


# DATA PREPROCESSING
def load_and_preprocess_sar(sar_path, config):
    """Load and preprocess SAR image"""
    with rasterio.open(sar_path) as src:
        sar_img = src.read(1)
        profile = src.profile
        transform = src.transform
    
    # Normalize SAR using log transform
    sar_img = np.clip(sar_img.astype(np.float32), config.SAR_MIN, config.SAR_MAX)
    eps = 1e-6
    shifted = sar_img - config.SAR_MIN + eps
    shifted_max = config.SAR_MAX - config.SAR_MIN + eps
    sar_norm = np.log(shifted) / np.log(shifted_max)
    sar_norm = (sar_norm - config.SAR_MEAN) / config.SAR_STD
    
    # Convert to 3-channel tensor [1, 3, H, W]
    sar_tensor = torch.from_numpy(sar_norm).float()
    if sar_tensor.dim() == 2:
        sar_tensor = sar_tensor.unsqueeze(0)
    sar_tensor = sar_tensor.repeat(3, 1, 1).unsqueeze(0)
    
    return sar_tensor, sar_img, profile, transform

def load_ground_truth_dem(gt_path):
    """Load ground truth DEM"""
    with rasterio.open(gt_path) as src:
        gt_dem = src.read(1).astype(np.float32)
    return gt_dem


# MODEL INFERENCE
def run_inference(model, sar_tensor, device, model_type="dpt"):
    """Run model inference"""
    with torch.no_grad():
        sar_tensor = sar_tensor.to(device)
        output = model(sar_tensor)
        
        output = output.cpu().squeeze().numpy()
        
        if output.ndim == 3 and output.shape[0] in [1, 3]:
            output = output.mean(axis=0)
    
    return output

def denormalize_dem(dem_norm, config):
    """Convert normalized DEM back to meters"""
    return dem_norm * (config.DEM_MAX - config.DEM_MIN) + config.DEM_MIN


# EVALUATION METRICS
def calculate_metrics(pred_dem, gt_dem):
    """Calculate evaluation metrics"""
    mae = np.mean(np.abs(pred_dem - gt_dem))
    rmse = np.sqrt(np.mean((pred_dem - gt_dem) ** 2))
    
    # Calculate SSIM
    data_range = np.max(gt_dem) - np.min(gt_dem)
    ssim_value = ssim(gt_dem, pred_dem, data_range=data_range)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'ssim': ssim_value
    }


# VISUALIZATION
def create_comparison_figure(sar_img, predictions, save_path, gt_dem=None):
    """Create comparison visualization with optional ground truth"""
    n_models = len(predictions)
    n_cols = n_models + 2 if gt_dem is not None else n_models + 1
    
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 8))
    
    # Create hillshade effect
    ls = LightSource(azdeg=315, altdeg=45)
    
    # Find common colormap range
    all_dems = list(predictions.values())
    if gt_dem is not None:
        all_dems.append(gt_dem)
    vmin = min([np.min(dem) for dem in all_dems])
    vmax = max([np.max(dem) for dem in all_dems])
    
    # Row 1: SAR, Ground Truth (if available), and predictions
    col_idx = 0
    
    # SAR image
    axes[0, col_idx].imshow(sar_img, cmap='gray')
    axes[0, col_idx].set_title('Input SAR')
    axes[0, col_idx].axis('off')
    axes[1, col_idx].axis('off')  # Empty cell below SAR
    col_idx += 1
    
    # Ground Truth (if available)
    if gt_dem is not None:
        im = axes[0, col_idx].imshow(gt_dem, cmap='terrain', vmin=vmin, vmax=vmax)
        axes[0, col_idx].set_title('Ground Truth')
        axes[0, col_idx].axis('off')
        
        # Hillshade for GT
        hillshade = ls.hillshade(gt_dem, vert_exag=2)
        axes[1, col_idx].imshow(hillshade, cmap='gray')
        axes[1, col_idx].set_title('GT Hillshade')
        axes[1, col_idx].axis('off')
        col_idx += 1
    
    # Model predictions
    for model_name, dem in predictions.items():
        axes[0, col_idx].imshow(dem, cmap='terrain', vmin=vmin, vmax=vmax)
        axes[0, col_idx].set_title(f'{model_name}')
        axes[0, col_idx].axis('off')
        
        # Hillshade
        hillshade = ls.hillshade(dem, vert_exag=2)
        axes[1, col_idx].imshow(hillshade, cmap='gray')
        axes[1, col_idx].set_title(f'{model_name} Hillshade')
        axes[1, col_idx].axis('off')
        col_idx += 1
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap='terrain'), 
        cax=cbar_ax)
    cbar.set_label('Elevation (m)')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")

def save_geotiff(dem, profile, output_path):
    """Save DEM as GeoTIFF"""
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw'
    )
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(dem.astype(np.float32), 1)
    print(f"Saved DEM to {output_path}")

def save_metrics(metrics_dict, output_path):
    """Save metrics to text file"""
    with open(output_path, 'w') as f:
        f.write("Model Evaluation Metrics\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, metrics in metrics_dict.items():
            f.write(f"{model_name}:\n")
            f.write(f"  SSIM: {metrics['ssim']:.4f}\n")
            f.write(f"  MAE:  {metrics['mae']:.2f} m\n")
            f.write(f"  RMSE: {metrics['rmse']:.2f} m\n")
            f.write("\n")
    
    print(f"Saved metrics to {output_path}")


# MAIN PIPELINE
def process_sar_image(sar_path, config):
    """Main processing pipeline"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess SAR
    print(f"Loading SAR image: {sar_path}")
    sar_tensor, sar_img, profile, transform = load_and_preprocess_sar(sar_path, config)
    
    # Load ground truth if provided
    gt_dem = None
    if config.GT_DEM_PATH and os.path.exists(config.GT_DEM_PATH):
        print(f"Loading ground truth DEM: {config.GT_DEM_PATH}")
        gt_dem = load_ground_truth_dem(config.GT_DEM_PATH)
    
    # Load models
    models = {}
    model_types = {}
    
    for model_name, model_path in config.MODEL_PATHS.items():
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found: {model_path}")
            continue
            
        print(f"Loading {model_name} model...")
        try:
            if model_name == "DPT":
                models[model_name] = load_dpt_model(model_path, device)
                model_types[model_name] = "dpt"
            elif model_name == "pix2pixHD":
                models[model_name] = load_pix2pixhd_model(model_path, device)
                model_types[model_name] = "pix2pixhd"
            elif model_name == "Im2Height":
                models[model_name] = load_im2height_model(model_path, device)
                model_types[model_name] = "im2height"
            print(f"  {model_name} loaded successfully")
        except Exception as e:
            print(f"  Error loading {model_name}: {e}")
    
    if not models:
        print("Error: No models loaded successfully")
        return
    
    # Run inference
    predictions = {}
    predictions_meters = {}
    metrics_dict = {}
    
    for model_name, model in models.items():
        print(f"Running {model_name} inference...")
        
        # Get normalized prediction
        pred_norm = run_inference(model, sar_tensor, device, 
                                 model_types[model_name])
        
        # Convert to meters
        pred_meters = denormalize_dem(pred_norm, config)
        
        predictions[model_name] = pred_norm
        predictions_meters[model_name] = pred_meters
        
        # Calculate metrics if ground truth available
        if gt_dem is not None:
            metrics = calculate_metrics(pred_meters, gt_dem)
            metrics_dict[model_name] = metrics
            print(f"  {model_name} - SSIM: {metrics['ssim']:.4f}, MAE: {metrics['mae']:.2f} m")
        
        # Save as GeoTIFF
        output_path = os.path.join(config.OUTPUT_DIR, f"{model_name}_output.tif")
        save_geotiff(pred_meters, profile, output_path)
    
    # Save metrics if available
    if metrics_dict:
        metrics_path = os.path.join(config.OUTPUT_DIR, "metrics.txt")
        save_metrics(metrics_dict, metrics_path)
    
    # Create visualization
    viz_path = os.path.join(config.OUTPUT_DIR, "comparison.png")
    create_comparison_figure(sar_img, predictions_meters, viz_path, gt_dem)
    
    print("\nProcessing complete!")
    return predictions_meters


def main():
    parser = argparse.ArgumentParser(description='SAR to DEM prediction')
    parser.add_argument('--sar_path', help='Path to input SAR image')
    parser.add_argument('--gt-dem', help='Path to ground truth DEM (optional)')
    parser.add_argument('--output-dir', default='./outputs/', help='Output directory')
    parser.add_argument('--dpt-model', help='Path to DPT model weights')
    parser.add_argument('--pix2pixhd-model', help='Path to Pix2PixHD weights')
    parser.add_argument('--im2height-model', help='Path to Im2Height weights')
    
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.INPUT_SAR_PATH = args.sar_path
    config.GT_DEM_PATH = args.gt_dem
    config.OUTPUT_DIR = args.output_dir
    
    if args.dpt_model:
        config.MODEL_PATHS["DPT"] = args.dpt_model
    if args.pix2pixhd_model:
        config.MODEL_PATHS["pix2pixHD"] = args.pix2pixhd_model
    if args.im2height_model:
        config.MODEL_PATHS["Im2Height"] = args.im2height_model
    
    # Process
    process_sar_image(config.INPUT_SAR_PATH, config)

if __name__ == "__main__":
    main()
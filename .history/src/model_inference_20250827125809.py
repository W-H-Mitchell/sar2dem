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

# Import model definitions
from models.midas import setupMidasFabric, loadModelFabric
from models.pix2pixhd import GlobalGenerator
from models.im2height import Im2Height

warnings.filterwarnings('ignore')


class Config:
    """Configuration for SAR to DEM inference"""
    # Model paths - update these to downloaded model weights
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
    
    # DEM normalization parameters (from training dataset) - used as fallback
    DEM_MIN_FALLBACK = 0.17894675
    DEM_MAX_FALLBACK = 7695.1045
    
    # Input/Output paths
    INPUT_SAR_PATH = None
    GT_DEM_PATH = None  # Optional ground truth
    OUTPUT_DIR = "./outputs/"


def load_dpt_model(checkpoint_path, device):
    """Load DPT model using the Fabric loader"""
    try:
        # Use the loadModelFabric function from midas.py
        model = loadModelFabric(checkpoint_path)
        model.to(device).eval()
        return model
    except Exception as e:
        print(f"Error loading DPT model: {e}")
        # Fallback to direct loading
        model = setupMidasFabric()
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


def load_pix2pixhd_model(checkpoint_path, device):
    """Load Pix2PixHD model"""
    model = GlobalGenerator(input_nc=3, output_nc=1, ngf=64, 
                           n_downsampling=4, n_blocks=9)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(state_dict, dict):
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device).eval()
    return model


def load_im2height_model(checkpoint_path, device):
    """Load Im2Height model"""
    model = Im2Height()
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(state_dict, dict):
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


def load_and_preprocess_sar(sar_path, config):
    """Load and preprocess SAR image"""
    with rasterio.open(sar_path) as src:
        sar_img = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
    
    # Normalize SAR using log transform (matching training)
    sar_img_clipped = np.clip(sar_img, config.SAR_MIN, config.SAR_MAX)
    eps = 1e-6
    shifted = sar_img_clipped - config.SAR_MIN + eps
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
    """Load ground truth DEM and calculate its min/max"""
    with rasterio.open(gt_path) as src:
        gt_dem = src.read(1).astype(np.float32)
    
    # Calculate min and max, ignoring NaN values if present
    valid_mask = ~np.isnan(gt_dem)
    if valid_mask.any():
        dem_min = np.min(gt_dem[valid_mask])
        dem_max = np.max(gt_dem[valid_mask])
    else:
        dem_min = np.min(gt_dem)
        dem_max = np.max(gt_dem)
    
    return gt_dem, dem_min, dem_max


def run_inference(model, sar_tensor, device, model_type="dpt"):
    """Run model inference"""
    with torch.no_grad():
        sar_tensor = sar_tensor.to(device)
        output = model(sar_tensor)
        
        # Handle different output formats
        if isinstance(output, dict) and 'prediction' in output:
            output = output['prediction']
        
        output = output.cpu().squeeze().numpy()
        
        # Handle multi-channel outputs
        if output.ndim == 3 and output.shape[0] in [1, 3]:
            output = output.mean(axis=0)
    
    return output


def denormalize_dem(dem_norm, dem_min, dem_max):
    """
    Convert normalized DEM back to meters using provided min/max values
    
    Args:
        dem_norm: Normalized DEM values (typically 0-1 range)
        dem_min: Minimum elevation value to use for denormalization
        dem_max: Maximum elevation value to use for denormalization
    """
    return dem_norm * (dem_max - dem_min) + dem_min


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


def create_comparison_figure(sar_img, predictions, save_path, gt_dem=None):
    """Create comparison visualization"""
    n_models = len(predictions)
    n_cols = n_models + 2 if gt_dem is not None else n_models + 1
    
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 8))
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)
    
    # Create hillshade effect
    ls = LightSource(azdeg=315, altdeg=45)
    all_dems = list(predictions.values())
    if gt_dem is not None:
        all_dems.append(gt_dem)
    vmin = min([np.min(dem) for dem in all_dems])
    vmax = max([np.max(dem) for dem in all_dems])
    
    col_idx = 0
    
    # SAR image
    axes[0, col_idx].imshow(sar_img, cmap='gray')
    axes[0, col_idx].set_title('Input SAR')
    axes[0, col_idx].axis('off')
    axes[1, col_idx].axis('off')
    col_idx += 1
    
    # Ground Truth (if available)
    if gt_dem is not None:
        axes[0, col_idx].imshow(gt_dem, cmap='terrain', vmin=vmin, vmax=vmax)
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
    cbar.set_label('Elevation (m)', fontsize=10)
    
    plt.suptitle('SAR to DEM Model Comparison', fontsize=14, y=1.02)
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
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(dem.astype(np.float32), 1)
    print(f"Saved DEM to {output_path}")


def save_metrics(metrics_dict, output_path, dem_min, dem_max):
    """Save metrics to text file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("Model Evaluation Metrics\n")
        f.write("=" * 50 + "\n\n")
        
        # Write denormalization range used
        f.write(f"Denormalization range: [{dem_min:.2f}, {dem_max:.2f}] m\n")
        f.write("=" * 50 + "\n\n")
        
        # SSIM score
        sorted_metrics = sorted(metrics_dict.items(), 
                              key=lambda x: x[1]['ssim'], reverse=True)
        
        for model_name, metrics in sorted_metrics:
            f.write(f"{model_name}:\n")
            f.write(f"  SSIM: {metrics['ssim']:.4f}\n")
            f.write(f"  MAE:  {metrics['mae']:.2f} m\n")
            f.write(f"  RMSE: {metrics['rmse']:.2f} m\n")
            f.write("\n")
        
        # best model
        if sorted_metrics:
            best_model = sorted_metrics[0][0]
            f.write(f"Best Model (by SSIM): {best_model}\n")
    
    print(f"Saved metrics to {output_path}")


def process_sar_image(sar_path, config):
    """Main processing pipeline"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # load and preprocess SAR
    print(f"Loading SAR image: {sar_path}")
    sar_tensor, sar_img, profile, transform = load_and_preprocess_sar(sar_path, config)
    print(f"  SAR shape: {sar_img.shape}, range: [{sar_img.min():.1f}, {sar_img.max():.1f}]")
    
    # load ground truth if provided and get its min/max
    gt_dem = None
    dem_min = config.DEM_MIN_FALLBACK
    dem_max = config.DEM_MAX_FALLBACK
    
    if config.GT_DEM_PATH and os.path.exists(config.GT_DEM_PATH):
        print(f"Loading ground truth DEM: {config.GT_DEM_PATH}")
        gt_dem, gt_min, gt_max = load_ground_truth_dem(config.GT_DEM_PATH)
        print(f"  GT DEM range: [{gt_min:.1f}, {gt_max:.1f}] m")
        print(f"  Using GT range for denormalization")
        dem_min = gt_min
        dem_max = gt_max
    else:
        print(f"  No ground truth provided, using fallback denormalization range: [{dem_min:.1f}, {dem_max:.1f}] m")
    
    # load models
    models = {}
    model_types = {}
    
    for model_name, model_path in config.MODEL_PATHS.items():
        if not model_path or not os.path.exists(model_path):
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
            print(f"  ✓ {model_name} loaded successfully")
        except Exception as e:
            print(f"  ✗ Error loading {model_name}: {e}")
    
    if not models:
        print("Error: No models loaded successfully")
        return None
    
    # run inference
    predictions = {}
    predictions_meters = {}
    metrics_dict = {}
    
    for model_name, model in models.items():
        print(f"Running {model_name} inference...")
        
        # get normalized prediction
        pred_norm = run_inference(model, sar_tensor, device, 
                                 model_types[model_name])
        
        # denormalize using GT min/max if available, otherwise use fallback
        pred_meters = denormalize_dem(pred_norm, dem_min, dem_max)
        
        predictions[model_name] = pred_norm
        predictions_meters[model_name] = pred_meters
        
        print(f"  Prediction range: [{pred_meters.min():.1f}, {pred_meters.max():.1f}] m")
        
        # calculate metrics if ground truth available
        if gt_dem is not None:
            metrics = calculate_metrics(pred_meters, gt_dem)
            metrics_dict[model_name] = metrics
            print(f"  SSIM: {metrics['ssim']:.4f}, MAE: {metrics['mae']:.2f} m, RMSE: {metrics['rmse']:.2f} m")
        
        # save as GeoTIFF
        output_path = os.path.join(config.OUTPUT_DIR, f"{model_name}_prediction.tif")
        save_geotiff(pred_meters, profile, output_path)
    
    # save metrics if available
    if metrics_dict:
        metrics_path = os.path.join(config.OUTPUT_DIR, "metrics.txt")
        save_metrics(metrics_dict, metrics_path, dem_min, dem_max)
    
    # create visualization
    viz_path = os.path.join(config.OUTPUT_DIR, "comparison.png")
    create_comparison_figure(sar_img, predictions_meters, viz_path, gt_dem)
    
    print("\n✓ Processing complete!")
    print(f"Results saved to: {config.OUTPUT_DIR}")
    print(f"Denormalization range used: [{dem_min:.1f}, {dem_max:.1f}] m")
    
    return predictions_meters


def main():
    parser = argparse.ArgumentParser(description='SAR to DEM prediction using trained models')
    parser.add_argument('--sar-path', type=str, required=True,
                        help='Path to input SAR image (.tif)')
    parser.add_argument('--dem-path', type=str, default=None,
                        help='Path to ground truth DEM for evaluation (optional)')
    parser.add_argument('--output-dir', type=str, default='./outputs/',
                        help='Output directory for results')
    parser.add_argument('--dpt-model', type=str, 
                        default='./model_weights/DPT-L1_ep200.pth',
                        help='Path to DPT model weights')
    parser.add_argument('--pix2pixhd-model', type=str,
                        default='./model_weights/pix2pixHD_ep200.pth',
                        help='Path to Pix2PixHD weights')
    parser.add_argument('--im2height-model', type=str, 
                        default='./model_weights/pix2pixHD_ep81.pth',
                        help='Path to Im2Height weights (optional)')
    args = parser.parse_args()

    config = Config()
    config.INPUT_SAR_PATH = args.sar_path
    config.GT_DEM_PATH = args.dem_path
    config.OUTPUT_DIR = args.output_dir
    
    # Update model paths
    config.MODEL_PATHS = {}
    if args.dpt_model and os.path.exists(args.dpt_model):
        config.MODEL_PATHS["DPT"] = args.dpt_model
    if args.pix2pixhd_model and os.path.exists(args.pix2pixhd_model):
        config.MODEL_PATHS["pix2pixHD"] = args.pix2pixhd_model
    if args.im2height_model and os.path.exists(args.im2height_model):
        config.MODEL_PATHS["Im2Height"] = args.im2height_model
    
    # Process
    process_sar_image(config.INPUT_SAR_PATH, config)


if __name__ == "__main__":
    main()
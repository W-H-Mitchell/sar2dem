import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import rasterio
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pix2pixhd import load_pix2pixhd_model
from models.im2height import Im2Height, load_im2height_model as load_im2height_weights

class Config:
    """configuration for SAR to DEM inference"""
    # model paths - update these to downloaded model weights
    MODEL_PATHS = {
        "DPT": "./model_weights/dpt-l1_ep200.pth",
        "Pix2PixHD": "./model_weights/pix2pixHD_ep200.pth", 
        "IM2HEIGHT": "./model_weights/im2height_ep81.pth"
    }
    
    # sar normalization parameters (from training)
    SAR_MIN = 0
    SAR_MAX = 3000
    SAR_MEAN = 0.5
    SAR_STD = 0.5

def setup_device():
    """set up computation device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def normalize_sar(sar_img):
    """normalize SAR image using log transform"""
    # clip to valid range
    sar_clipped = np.clip(sar_img, Config.SAR_MIN, Config.SAR_MAX)
    
    # log transform
    eps = 1e-6
    shifted = sar_clipped - Config.SAR_MIN + eps
    shifted_max = Config.SAR_MAX - Config.SAR_MIN + eps
    sar_norm = np.log(shifted) / np.log(shifted_max)
    
    # standardize
    sar_norm = (sar_norm - Config.SAR_MEAN) / Config.SAR_STD
    
    return sar_norm




def load_im2height_model(checkpoint_path, device):
    """load im2height model"""
    model = Im2Height()
    model = load_im2height_weights(model, checkpoint_path, device)
    
    model.to(device).eval()
    return model


def predict_dem(model, sar_tensor, device):
    """run model inference"""
    with torch.no_grad():
        sar_tensor = sar_tensor.to(device)
        output = model(sar_tensor)
        
        # handle different output formats
        if isinstance(output, dict) and 'prediction' in output:
            output = output['prediction']
        
        # convert to numpy
        output = output.cpu().squeeze().numpy()
        
        # handle multi-channel outputs
        if output.ndim == 3 and output.shape[0] == 3:
            output = output.mean(axis=0)
    
    return output


def denormalize_dem(pred_raw, gt_min, gt_max):
    """
    Normalize prediction to [0,1] using its own range, scale to ground truth range
    """
    # first normalize to [0,1] 
    pred_min = np.min(pred_raw)
    pred_max = np.max(pred_raw)

    if pred_max - pred_min < 1e-6:
        pred_norm = np.zeros_like(pred_raw)
    else:
        pred_norm = (pred_raw - pred_min) / (pred_max - pred_min)
    pred_scaled = pred_norm * (gt_max - gt_min) + gt_min
    
    return pred_scaled


def calculate_metrics(pred, gt):
    """calculate SSIM and MAE"""
    # mae
    mae = np.mean(np.abs(pred - gt))
    
    # ssim
    data_range = np.max(gt) - np.min(gt)
    ssim_value = ssim(gt, pred, data_range=data_range)
    
    return ssim_value, mae


def create_figure(sar_img, gt_dem, predictions, save_path):
    """create visualization matching the paper figures"""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # get common colormap range from ground truth
    vmin, vmax = np.min(gt_dem), np.max(gt_dem)
    
    # plot SAR
    axes[0].imshow(sar_img, cmap='gray')
    axes[0].set_title('SAR', fontsize=14)
    axes[0].axis('off')
    
    # plot ground truth
    im = axes[1].imshow(gt_dem, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[1].set_title('Ground Truth', fontsize=14)
    axes[1].axis('off')
    
    # plot predictions
    model_names = ['DPT', 'Pix2PixHD', 'IM2HEIGHT']
    for i, model_name in enumerate(model_names):
        if model_name in predictions:
            pred_dem = predictions[model_name]['dem']
            ssim_val = predictions[model_name]['ssim']
            mae_val = predictions[model_name]['mae']
            
            axes[i+2].imshow(pred_dem, cmap='terrain', vmin=vmin, vmax=vmax)
            axes[i+2].set_title(f'{model_name}\nSSIM: {ssim_val:.3f}; MAE: {mae_val:.1f} m', 
                              fontsize=12)
            axes[i+2].axis('off')
    
    # add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Elevation (m)', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved figure to {save_path}")


def process_sar_to_dem(sar_path, gt_path, output_dir='./outputs/'):
    """main processing pipeline"""
    device = setup_device()
    print(f"using device: {device}")
    
    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # load SAR image
    print(f"loading SAR image: {sar_path}")
    with rasterio.open(sar_path) as src:
        sar_img = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    
    # normalize SAR
    sar_norm = normalize_sar(sar_img)
    
    # convert to tensor [1, 3, H, W]
    sar_tensor = torch.from_numpy(sar_norm).float()
    if sar_tensor.dim() == 2:
        sar_tensor = sar_tensor.unsqueeze(0)
    sar_tensor = sar_tensor.repeat(3, 1, 1).unsqueeze(0)
    
    # load ground truth
    print(f"loading ground truth: {gt_path}")
    with rasterio.open(gt_path) as src:
        gt_dem = src.read(1).astype(np.float32)
    
    gt_min, gt_max = np.min(gt_dem), np.max(gt_dem)
    print(f"ground truth range: [{gt_min:.1f}, {gt_max:.1f}] m")
    
    # load models and predict
    predictions = {}
    
    for model_name, model_path in Config.MODEL_PATHS.items():
        if not os.path.exists(model_path):
            print(f"warning: {model_name} model not found at {model_path}")
            continue
            
        print(f"loading {model_name}...")
        try:
            # load model
            if model_name == "DPT":
                model = load_dpt_model(model_path, device)
            elif model_name == "Pix2PixHD":
                model = load_pix2pixhd_model(model_path, device)
            elif model_name == "IM2HEIGHT":
                model = load_im2height_model(model_path, device)
            
            # predict
            pred_norm = predict_dem(model, sar_tensor, device)
            
            # denormalize using ground truth range
            pred_dem = denormalize_dem(pred_norm, gt_min, gt_max)
            
            # calculate metrics
            ssim_val, mae_val = calculate_metrics(pred_dem, gt_dem)
            
            predictions[model_name] = {
                'dem': pred_dem,
                'ssim': ssim_val,
                'mae': mae_val
            }
            
            print(f"  {model_name} - SSIM: {ssim_val:.3f}, MAE: {mae_val:.1f} m")
            
            # save geotiff
            profile.update(dtype=rasterio.float32, count=1, compress='lzw')
            with rasterio.open(os.path.join(output_dir, f'{model_name}_prediction.tif'), 
                             'w', **profile) as dst:
                dst.write(pred_dem.astype(np.float32), 1)
                
        except Exception as e:
            print(f"  error loading {model_name}: {e}")
    
    # create visualization
    if predictions:
        fig_path = os.path.join(output_dir, 'comparison.png')
        create_figure(sar_img, gt_dem, predictions, fig_path)
    
    print("\nprocessing complete!")
    return predictions


def main():
    parser = argparse.ArgumentParser(description='SAR to DEM prediction using trained models')
    parser.add_argument('--sar_path', default='./data/testing/AFGHANISTAN_0/data/AFGHANISTAN_y0x24064_DESCENDING.tif',
                        help='Path to input SAR image')
    parser.add_argument('--dem_path', default='./data/testing/AFGHANISTAN_0/label/AFGHANISTAN_y0x24064_DESCENDING.tif',
                        help='Path to ground truth DEM (optional)')
    parser.add_argument('--output-dir', default='./outputs/', 
                        help='Output directory')
    parser.add_argument('--dpt_path', type=str, 
                        default='./model_weights/dpt-l1_ep200.pth',
                        help='Path to DPT model weights')
    parser.add_argument('--pix2pixhd_path', type=str,
                        default='./model_weights/pix2pixHD_ep200.pth',
                        help='Path to Pix2PixHD weights')
    parser.add_argument('--im2height_path', type=str, 
                        default='./model_weights/im2height_ep81.pth',
                        help='Path to Im2Height weights (optional)')
    args = parser.parse_args()

    config = Config()
    config.INPUT_SAR_PATH = args.sar_path
    config.GT_DEM_PATH = args.dem_path
    config.OUTPUT_DIR = args.output_dir
    
    # update model paths
    config.MODEL_PATHS = {}
    if args.dpt_path and os.path.exists(args.dpt_path):
        config.MODEL_PATHS["DPT"] = args.dpt_path
    if args.pix2pixhd_path and os.path.exists(args.pix2pixhd_path):
        config.MODEL_PATHS["pix2pixHD"] = args.pix2pixhd_path
    if args.im2height_path and os.path.exists(args.im2height_path):
        config.MODEL_PATHS["Im2Height"] = args.im2height_path
    
    # process
    process_sar_to_dem(config.INPUT_SAR_PATH, config.GT_DEM_PATH, config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
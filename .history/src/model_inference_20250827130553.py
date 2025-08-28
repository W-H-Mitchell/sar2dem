import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import LightSource
from skimage.metrics import structural_similarity as ssim
from pathlib import Path


def load_data(sar_path, gt_path, pred_paths):
    """load SAR, ground truth, and prediction files"""
    # load SAR
    with rasterio.open(sar_path) as src:
        sar = src.read(1).astype(np.float32)
    
    # load ground truth
    with rasterio.open(gt_path) as src:
        gt_dem = src.read(1).astype(np.float32)
    
    # load predictions
    predictions = {}
    for model_name, pred_path in pred_paths.items():
        with rasterio.open(pred_path) as src:
            predictions[model_name] = src.read(1).astype(np.float32)
    
    return sar, gt_dem, predictions


def calculate_metrics(pred, gt):
    """calculate SSIM and MAE between prediction and ground truth"""
    # mae
    mae = np.mean(np.abs(pred - gt))
    
    # ssim
    data_range = np.max(gt) - np.min(gt)
    ssim_value = ssim(gt, pred, data_range=data_range)
    
    return ssim_value, mae


def create_comparison_figure(sar, gt_dem, predictions, save_path):
    """create the comparison figure matching the paper style"""
    
    # setup figure - 1 row, 5 columns
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # get elevation range from all DEMs for consistent colormap
    all_dems = [gt_dem] + list(predictions.values())
    vmin = min([np.min(dem) for dem in all_dems])
    vmax = max([np.max(dem) for dem in all_dems])
    
    # column 1: SAR
    axes[0].imshow(sar, cmap='gray')
    axes[0].set_title('SAR', fontsize=12)
    axes[0].axis('off')
    
    # column 2: ground truth
    im = axes[1].imshow(gt_dem, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[1].set_title('Ground Truth', fontsize=12)
    axes[1].axis('off')
    
    # columns 3-5: model predictions
    model_names = ['DPT', 'Pix2PixHD', 'IM2HEIGHT']
    for idx, model_name in enumerate(model_names):
        if model_name in predictions:
            pred = predictions[model_name]
            
            # calculate metrics
            ssim_val, mae_val = calculate_metrics(pred, gt_dem)
            
            # display prediction
            axes[idx+2].imshow(pred, cmap='terrain', vmin=vmin, vmax=vmax)
            axes[idx+2].set_title(model_name, fontsize=12)
            axes[idx+2].axis('off')
            
            # add metrics below image
            metric_text = f'SSIM: {ssim_val:.2f}; MAE: {mae_val:.1f} m'
            axes[idx+2].text(0.5, -0.1, metric_text, 
                           transform=axes[idx+2].transAxes,
                           ha='center', fontsize=10)
    
    # add colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    cbar = fig.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=vmin, vmax=vmax), 
        cmap='terrain'), cax=cbar_ax)
    cbar.set_label('Elevation (m)', fontsize=10)
    
    # adjust layout
    plt.subplots_adjust(left=0.02, right=0.91, top=0.95, bottom=0.05, wspace=0.05)
    
    # save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_multi_row_figure(data_list, save_path):
    """create multi-row comparison figure like in the paper"""
    
    n_rows = len(data_list)
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4*n_rows))
    
    # process each row
    for row_idx, (sar, gt_dem, predictions, row_label) in enumerate(data_list):
        
        # get elevation range for this row
        all_dems = [gt_dem] + list(predictions.values())
        vmin = min([np.min(dem) for dem in all_dems])
        vmax = max([np.max(dem) for dem in all_dems])
        
        # add row label (a, b, c, d)
        axes[row_idx, 0].text(-0.15, 0.5, row_label, 
                             transform=axes[row_idx, 0].transAxes,
                             fontsize=14, fontweight='bold', va='center')
        
        # sar
        axes[row_idx, 0].imshow(sar, cmap='gray')
        if row_idx == 0:
            axes[row_idx, 0].set_title('SAR', fontsize=12)
        axes[row_idx, 0].axis('off')
        
        # ground truth
        axes[row_idx, 1].imshow(gt_dem, cmap='terrain', vmin=vmin, vmax=vmax)
        if row_idx == 0:
            axes[row_idx, 1].set_title('Ground Truth', fontsize=12)
        axes[row_idx, 1].axis('off')
        
        # model predictions
        model_names = ['DPT', 'Pix2PixHD', 'IM2HEIGHT']
        for idx, model_name in enumerate(model_names):
            if model_name in predictions:
                pred = predictions[model_name]
                
                # calculate metrics
                ssim_val, mae_val = calculate_metrics(pred, gt_dem)
                
                # display
                axes[row_idx, idx+2].imshow(pred, cmap='terrain', vmin=vmin, vmax=vmax)
                if row_idx == 0:
                    axes[row_idx, idx+2].set_title(model_name, fontsize=12)
                axes[row_idx, idx+2].axis('off')
                
                # add metrics
                metric_text = f'SSIM: {ssim_val:.3f}; MAE: {mae_val:.1f} m'
                axes[row_idx, idx+2].text(0.5, -0.05, metric_text,
                                         transform=axes[row_idx, idx+2].transAxes,
                                         ha='center', fontsize=9)
        
        # add individual colorbar for each row
        cbar_ax = fig.add_axes([0.92, 0.77 - row_idx*0.235 + 0.04, 0.015, 0.15])
        cbar = fig.colorbar(plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=vmin, vmax=vmax), 
            cmap='terrain'), cax=cbar_ax)
        cbar.set_label('Elevation (m)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    
    # adjust layout
    plt.subplots_adjust(left=0.02, right=0.91, top=0.97, bottom=0.03, 
                       hspace=0.15, wspace=0.05)
    
    # save
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():

    
    # define file paths for each test case
    test_cases = [
        {
            'sar': './data/test1_sar.tif',
            'gt': './data/test1_gt.tif',
            'predictions': {
                'DPT': './predictions/test1_dpt.tif',
                'Pix2PixHD': './predictions/test1_pix2pixhd.tif',
                'IM2HEIGHT': './predictions/test1_im2height.tif'
            },
            'label': 'a'
        },
        {
            'sar': './data/test2_sar.tif',
            'gt': './data/test2_gt.tif',
            'predictions': {
                'DPT': './predictions/test2_dpt.tif',
                'Pix2PixHD': './predictions/test2_pix2pixhd.tif',
                'IM2HEIGHT': './predictions/test2_im2height.tif'
            },
            'label': 'b'
        },

    ]
    
    # load data for all test cases
    data_list = []
    for test_case in test_cases:
        sar, gt_dem, predictions = load_data(
            test_case['sar'],
            test_case['gt'],
            test_case['predictions']
        )
        data_list.append((sar, gt_dem, predictions, test_case['label']))
    
    # create multi-row figure
    create_multi_row_figure(data_list, './figures/model_comparison.png')
    print("Figure saved to ./figures/model_comparison.png")


if __name__ == "__main__":
    main()
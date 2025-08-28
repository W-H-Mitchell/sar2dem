import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import rasterio
from tqdm import tqdm
import torch.nn.functional as F
from scipy.interpolate import griddata
from scipy.optimize import minimize
from typing import Optional, Tuple, Dict, List
from scipy.ndimage import zoom
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.midas import loadModelFabric
from models.pix2pixhd import load_pix2pixhd_model
from models.im2height import Im2Height, load_im2height_model as load_im2height_weights
@dataclass
class TileInfo:
    """Store information about a processed tile."""
    data: np.ndarray
    row: int
    col: int
    height: int
    width: int


class DEMMosaicPredictor:
    """Simplified DEM mosaic prediction with calibration and tile optimization."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize predictor with model.
        
        Args:
            model_path: Path to trained model weights
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        
        # SAR normalization parameters (from training)
        self.sar_min = 0
        self.sar_max = 3000
        self.sar_mean = 0.5
        self.sar_std = 0.5
        
        # DEM denormalization parameters (from training)
        self.dem_min = 0.17894675
        self.dem_max = 7695.1045
        
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup compute device."""
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model."""
        print(f"Loading model from {model_path}")
        model = loadModelFabric(model_path)
        model.to(self.device)
        model.eval()
        return model
    
    def predict_dem(self, 
                    sar_path: str,
                    output_path: str,
                    tile_size: int = 384,
                    overlap: float = 0.25,
                    calibration_method: str = "none",
                    coarse_dem_path: Optional[str] = None,
                    control_points: Optional[Dict] = None,
                    region_size: int = 128,
                    optimize_tiles: bool = False) -> np.ndarray:
        """
        Predict DEM from SAR image with optional calibration.
        
        Args:
            sar_path: Path to SAR image
            output_path: Path to save output DEM
            tile_size: Size of tiles for processing
            overlap: Overlap between tiles (0-1)
            calibration_method: 'none', 'coarse_dem', 'idw', or 'global'
            coarse_dem_path: Path to coarse DEM for calibration
            control_points: Dict with 'x', 'y', 'z' arrays for IDW calibration
            region_size: Size of regions for coarse DEM matching
            optimize_tiles: Whether to optimize tile offsets for seamless mosaicing
            
        Returns:
            Predicted DEM array
        """
        # Load SAR image
        sar_data, sar_meta = self._load_sar(sar_path)
        
        # Process through tiles
        if optimize_tiles:
            print("Processing with tile optimization...")
            raw_dem = self._process_tiles_with_optimization(sar_data, tile_size, overlap)
        else:
            raw_dem = self._process_tiles(sar_data, tile_size, overlap)
        
        # Apply calibration
        if calibration_method == "global":
            print("Applying global min/max calibration...")
            calibrated_dem = self._calibrate_global(raw_dem)
        elif calibration_method == "coarse_dem" and coarse_dem_path:
            print("Applying coarse DEM calibration...")
            coarse_dem = self._load_coarse_dem(coarse_dem_path, sar_meta)
            calibrated_dem = self._calibrate_with_coarse_dem(raw_dem, coarse_dem, region_size)
        elif calibration_method == "idw" and control_points:
            print("Applying IDW calibration...")
            calibrated_dem = self._calibrate_with_idw(raw_dem, control_points)
        else:
            print("Using raw model output (no calibration)")
            calibrated_dem = raw_dem
        
        # Save output
        self._save_dem(calibrated_dem, output_path, sar_meta)
        
        return calibrated_dem
    
    def _load_sar(self, sar_path: str) -> Tuple[np.ndarray, dict]:
        """Load SAR image."""
        with rasterio.open(sar_path) as src:
            sar_data = src.read()
            meta = src.meta.copy()
        return sar_data, meta
    
    def _load_coarse_dem(self, dem_path: str, sar_meta: dict) -> np.ndarray:
        """Load and resample coarse DEM to match SAR."""
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            
            # Simple resampling to match SAR dimensions
            if dem_data.shape != (sar_meta['height'], sar_meta['width']):
                zoom_factors = (sar_meta['height'] / dem_data.shape[0],
                              sar_meta['width'] / dem_data.shape[1])
                dem_data = zoom(dem_data, zoom_factors, order=1)
        
        return dem_data
    
    def _normalize_sar(self, sar_tile: np.ndarray) -> torch.Tensor:
        """Normalize SAR tile for model input."""
        # Apply log normalization
        eps = 1e-6
        sar_clipped = np.clip(sar_tile, self.sar_min, self.sar_max)
        sar_log = np.log(sar_clipped - self.sar_min + eps) / np.log(self.sar_max - self.sar_min + eps)
        sar_normalized = (sar_log - self.sar_mean) / self.sar_std
        
        # Convert to 3-channel input
        if sar_normalized.ndim == 2:
            sar_normalized = np.stack([sar_normalized] * 3, axis=0)
        elif sar_normalized.shape[0] == 1:
            sar_normalized = np.repeat(sar_normalized, 3, axis=0)
        
        return torch.from_numpy(sar_normalized).float()
    
    def _denormalize_dem(self, dem_normalized: np.ndarray) -> np.ndarray:
        """Convert model output to elevation values."""
        return dem_normalized * (self.dem_max - self.dem_min) + self.dem_min
    
    def _process_tiles(self, sar_data: np.ndarray, tile_size: int, overlap: float) -> np.ndarray:
        """Process SAR image through tiles and stitch results."""
        if sar_data.ndim == 3:
            sar_data = sar_data[0]  # Use first channel
        
        height, width = sar_data.shape
        stride = int(tile_size * (1 - overlap))
        
        # Initialize output and weights for blending
        output = np.zeros((height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        
        # Create weight map for smooth blending
        weight_map = self._create_weight_map(tile_size)
        
        # Process tiles
        tiles_processed = 0
        for y in range(0, height - tile_size + 1, stride):
            for x in range(0, width - tile_size + 1, stride):
                # Extract tile
                tile = sar_data[y:y+tile_size, x:x+tile_size]
                
                # Normalize and prepare for model
                tile_tensor = self._normalize_sar(tile).unsqueeze(0).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    pred = self.model(tile_tensor).squeeze().cpu().numpy()
                
                # Denormalize
                pred_dem = self._denormalize_dem(pred)
                
                # Add to output with blending
                output[y:y+tile_size, x:x+tile_size] += pred_dem * weight_map
                weights[y:y+tile_size, x:x+tile_size] += weight_map
                
                tiles_processed += 1
        
        print(f"Processed {tiles_processed} tiles")
        
        # Normalize by weights
        mask = weights > 0
        output[mask] /= weights[mask]
        
        return output
    
    def _process_tiles_with_optimization(self, sar_data: np.ndarray, 
                                        tile_size: int, overlap: float) -> np.ndarray:
        """
        Process tiles with optimization to minimize offsets in overlapping regions.
        """
        if sar_data.ndim == 3:
            sar_data = sar_data[0]
        
        height, width = sar_data.shape
        stride = int(tile_size * (1 - overlap))
        
        # First pass: process all tiles and store them
        tiles = []
        print("Processing tiles...")
        
        for y in tqdm(range(0, height - tile_size + 1, stride)):
            for x in range(0, width - tile_size + 1, stride):
                # Extract tile
                tile = sar_data[y:y+tile_size, x:x+tile_size]
                
                # Normalize and prepare for model
                tile_tensor = self._normalize_sar(tile).unsqueeze(0).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    pred = self.model(tile_tensor).squeeze().cpu().numpy()
                
                # Denormalize
                pred_dem = self._denormalize_dem(pred)
                
                # Store tile info
                tiles.append(TileInfo(
                    data=pred_dem,
                    row=y,
                    col=x,
                    height=tile_size,
                    width=tile_size
                ))
        
        print(f"Optimizing {len(tiles)} tiles...")
        
        # Optimize tile offsets
        optimized_tiles = self._optimize_tile_offsets(tiles, height, width)
        
        # Stitch optimized tiles
        output = self._stitch_optimized_tiles(optimized_tiles, height, width)
        
        return output
    
    def _optimize_tile_offsets(self, tiles: List[TileInfo], 
                               full_height: int, full_width: int) -> List[TileInfo]:
        """
        Optimize vertical offsets for tiles to minimize differences in overlapping regions.
        """
        n_tiles = len(tiles)
        
        # Initialize offsets (one per tile)
        initial_offsets = np.zeros(n_tiles)
        
        def objective(offsets):
            """Compute total difference in overlapping regions."""
            total_error = 0
            n_overlaps = 0
            
            for i, tile_i in enumerate(tiles):
                for j, tile_j in enumerate(tiles):
                    if i >= j:
                        continue
                    
                    # Check for overlap
                    overlap = self._get_tile_overlap(tile_i, tile_j)
                    if overlap is not None:
                        # Extract overlapping regions
                        i_start_y, i_end_y, i_start_x, i_end_x = overlap['tile1_coords']
                        j_start_y, j_end_y, j_start_x, j_end_x = overlap['tile2_coords']
                        
                        region_i = tile_i.data[i_start_y:i_end_y, i_start_x:i_end_x] + offsets[i]
                        region_j = tile_j.data[j_start_y:j_end_y, j_start_x:j_end_x] + offsets[j]
                        
                        # Compute difference
                        diff = np.mean((region_i - region_j) ** 2)
                        total_error += diff
                        n_overlaps += 1
            
            # Add regularization to keep offsets small
            reg_term = 0.01 * np.sum(offsets ** 2)
            
            return total_error / max(n_overlaps, 1) + reg_term
        
        # Optimize
        result = minimize(objective, initial_offsets, method='L-BFGS-B')
        
        if result.success:
            print(f"Optimization converged. Final error: {result.fun:.6f}")
        else:
            print(f"Optimization did not converge: {result.message}")
        
        # Apply optimized offsets
        optimized_tiles = []
        for i, tile in enumerate(tiles):
            new_tile = TileInfo(
                data=tile.data + result.x[i],
                row=tile.row,
                col=tile.col,
                height=tile.height,
                width=tile.width
            )
            optimized_tiles.append(new_tile)
        
        return optimized_tiles
    
    def _get_tile_overlap(self, tile1: TileInfo, tile2: TileInfo) -> Optional[Dict]:
        """Find overlapping region between two tiles."""
        # Calculate overlap bounds
        y1_start, y1_end = tile1.row, tile1.row + tile1.height
        x1_start, x1_end = tile1.col, tile1.col + tile1.width
        
        y2_start, y2_end = tile2.row, tile2.row + tile2.height
        x2_start, x2_end = tile2.col, tile2.col + tile2.width
        
        # Check for overlap
        overlap_y_start = max(y1_start, y2_start)
        overlap_y_end = min(y1_end, y2_end)
        overlap_x_start = max(x1_start, x2_start)
        overlap_x_end = min(x1_end, x2_end)
        
        if overlap_y_start >= overlap_y_end or overlap_x_start >= overlap_x_end:
            return None  # No overlap
        
        # Calculate indices within each tile
        tile1_y_start = overlap_y_start - y1_start
        tile1_y_end = overlap_y_end - y1_start
        tile1_x_start = overlap_x_start - x1_start
        tile1_x_end = overlap_x_end - x1_start
        
        tile2_y_start = overlap_y_start - y2_start
        tile2_y_end = overlap_y_end - y2_start
        tile2_x_start = overlap_x_start - x2_start
        tile2_x_end = overlap_x_end - x2_start
        
        return {
            'tile1_coords': (tile1_y_start, tile1_y_end, tile1_x_start, tile1_x_end),
            'tile2_coords': (tile2_y_start, tile2_y_end, tile2_x_start, tile2_x_end),
            'global_coords': (overlap_y_start, overlap_y_end, overlap_x_start, overlap_x_end)
        }
    
    def _stitch_optimized_tiles(self, tiles: List[TileInfo], 
                                height: int, width: int) -> np.ndarray:
        """Stitch optimized tiles with smooth blending."""
        output = np.zeros((height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        
        for tile in tiles:
            # Create weight map
            weight_map = self._create_weight_map(tile.height)
            
            # Add tile to output
            y_end = min(tile.row + tile.height, height)
            x_end = min(tile.col + tile.width, width)
            
            actual_height = y_end - tile.row
            actual_width = x_end - tile.col
            
            output[tile.row:y_end, tile.col:x_end] += (
                tile.data[:actual_height, :actual_width] * 
                weight_map[:actual_height, :actual_width]
            )
            weights[tile.row:y_end, tile.col:x_end] += (
                weight_map[:actual_height, :actual_width]
            )
        
        # Normalize by weights
        mask = weights > 0
        output[mask] /= weights[mask]
        
        return output
    
    def _create_weight_map(self, size: int) -> np.ndarray:
        """Create Gaussian weight map for tile blending."""
        y, x = np.mgrid[0:size, 0:size]
        center = size // 2
        sigma = size / 6
        dist_sq = ((x - center) ** 2 + (y - center) ** 2)
        weight = np.exp(-dist_sq / (2 * sigma ** 2))
        return weight
    
    def _calibrate_global(self, dem: np.ndarray) -> np.ndarray:
        """Apply global min/max calibration using known elevation range."""
        # Use percentiles to avoid outliers
        p2 = np.percentile(dem[~np.isnan(dem)], 2)
        p98 = np.percentile(dem[~np.isnan(dem)], 98)
        
        # Clip to percentile range
        dem_clipped = np.clip(dem, p2, p98)
        
        # Scale to expected global range (you can adjust these based on your region)
        # These are example values - replace with your actual elevation range
        global_min = 0  # Sea level
        global_max = 8000  # Max elevation in meters
        
        dem_scaled = (dem_clipped - p2) / (p98 - p2)
        dem_calibrated = dem_scaled * (global_max - global_min) + global_min
        
        return dem_calibrated
    
    def _calibrate_with_coarse_dem(self, 
                                   fine_dem: np.ndarray, 
                                   coarse_dem: np.ndarray,
                                   region_size: int) -> np.ndarray:
        """
        Calibrate fine DEM using coarse DEM statistics.
        
        Regional adaptive scaling: matches local statistics of fine DEM
        to coarse DEM in overlapping regions.
        """
        height, width = fine_dem.shape
        output = np.zeros_like(fine_dem)
        weights = np.zeros_like(fine_dem)
        
        # Weight map for smooth blending
        weight_map = self._create_weight_map(region_size)
        stride = region_size // 2
        
        for y in range(0, height - region_size + 1, stride):
            for x in range(0, width - region_size + 1, stride):
                # Extract regions
                fine_region = fine_dem[y:y+region_size, x:x+region_size]
                coarse_region = coarse_dem[y:y+region_size, x:x+region_size]
                
                # Skip if too many invalid values
                if np.isnan(fine_region).sum() > 0.5 * fine_region.size:
                    continue
                
                # Match statistics
                fine_mean = np.nanmean(fine_region)
                fine_std = np.nanstd(fine_region)
                coarse_mean = np.nanmean(coarse_region)
                coarse_std = np.nanstd(coarse_region)
                
                if fine_std > 0:
                    # Normalize and rescale
                    normalized = (fine_region - fine_mean) / fine_std
                    calibrated = normalized * coarse_std + coarse_mean
                else:
                    calibrated = np.full_like(fine_region, coarse_mean)
                
                # Add to output with blending
                output[y:y+region_size, x:x+region_size] += calibrated * weight_map
                weights[y:y+region_size, x:x+region_size] += weight_map
        
        # Normalize by weights
        mask = weights > 0
        output[mask] /= weights[mask]
        output[~mask] = fine_dem[~mask]  # Use original where not processed
        
        return output
    
    def _calibrate_with_idw(self, 
                           dem: np.ndarray,
                           control_points: Dict,
                           power: float = 2.0) -> np.ndarray:
        """
        Calibrate DEM using Inverse Distance Weighting from control points.
        
        Args:
            dem: Input DEM
            control_points: Dict with 'x', 'y', 'z' arrays
            power: Power parameter for IDW
            
        Returns:
            Calibrated DEM
        """
        height, width = dem.shape
        
        # Create grid of coordinates
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        points = np.column_stack((control_points['y'], control_points['x']))
        values = control_points['z']
        
        # Get control point elevations from DEM
        dem_values = []
        for y, x in points:
            if 0 <= int(y) < height and 0 <= int(x) < width:
                dem_values.append(dem[int(y), int(x)])
            else:
                dem_values.append(np.nan)
        
        dem_values = np.array(dem_values)
        valid = ~np.isnan(dem_values)
        
        if valid.sum() < 3:
            print("Warning: Not enough valid control points for IDW")
            return dem
        
        # Calculate differences at control points
        differences = values[valid] - dem_values[valid]
        
        # Interpolate correction surface
        correction = griddata(points[valid], differences, 
                            (y_grid, x_grid), method='cubic', fill_value=0)
        
        return dem + correction
    
    def _save_dem(self, dem: np.ndarray, output_path: str, meta: dict):
        """Save DEM to GeoTIFF."""
        output_meta = meta.copy()
        output_meta.update({
            'count': 1,
            'dtype': 'float32',
            'nodata': None
        })
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with rasterio.open(output_path, 'w', **output_meta) as dst:
            dst.write(dem.astype(np.float32), 1)
        
        print(f"Saved DEM to {output_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate DEM from SAR with calibration")
    parser.add_argument("--sar_path", default="./data/regional/clipped_sar.tif",
                        help="Path to SAR image")
    parser.add_argument("--model_path", 
                        help="Path to trained model")
    parser.add_argument("--output_path", help="Path for output DEM")
    parser.add_argument("--calibration", choices=["global", "coarse_dem", "idw"], 
                       default="global", help="Calibration method")
    parser.add_argument("--coarse_dem", help="Path to coarse DEM for calibration")
    parser.add_argument("--control_points", help="Path to control points file (CSV with x,y,z)")
    parser.add_argument("--tile_size", type=int, default=384, help="Tile size for processing")
    parser.add_argument("--overlap", type=float, default=0.25, help="Tile overlap (0-1)")
    parser.add_argument("--region_size", type=int, default=128, help="Region size for coarse DEM matching")
    parser.add_argument("--optimize_tiles", action="store_true", 
                       help="Optimize tile offsets for seamless mosaicing")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Compute device")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = DEMMosaicPredictor(args.model_path, args.device)
    
    # Load control points if provided
    control_points = None
    if args.control_points:
        df = pd.read_csv(args.control_points)
        control_points = {
            'x': df['x'].values,
            'y': df['y'].values,
            'z': df['z'].values
        }
    
    # Generate DEM
    dem = predictor.predict_dem(
        sar_path=args.sar_path,
        output_path=args.output_path,
        tile_size=args.tile_size,
        overlap=args.overlap,
        calibration_method=args.calibration,
        coarse_dem_path=args.coarse_dem,
        control_points=control_points,
        region_size=args.region_size,
        optimize_tiles=args.optimize_tiles
    )
    
    print(f"DEM shape: {dem.shape}")
    print(f"Elevation range: {np.nanmin(dem):.2f} - {np.nanmax(dem):.2f} m")


if __name__ == "__main__":
    main()
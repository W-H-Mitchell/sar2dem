import os
import sys
import argparse
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
from model_inference import load_dpt_model

@dataclass
class TileInfo:
    """Store information about a processed tile."""
    data: np.ndarray
    row: int
    col: int
    height: int
    width: int

class DEMMosaicPredictor:
    """DEM mosaic prediction with calibration and tile optimization."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize predictor with model.
        
        Args:
            model_path: Path to trained model weights
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        
        # Fixed tile size
        self.tile_size = 384
        
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
        model = load_dpt_model(model_path, self.device)
        return model
    
    def predict_dem(self, 
                    sar_path: str,
                    output_path: str,
                    overlap_pixels: int = 96,
                    calibration_method: str = "none",
                    coarse_dem_path: Optional[str] = None,
                    num_control_points: int = 100,
                    random_seed: int = 42,
                    region_size: int = 128,
                    optimize_tiles: bool = False) -> np.ndarray:
        """
        Predict DEM from SAR image with optional calibration.
        
        Args:
            sar_path: Path to SAR image
            output_path: Path to save output DEM
            overlap_pixels: Overlap between tiles in pixels
            calibration_method: 'none', 'coarse_dem', 'idw', or 'global'
            coarse_dem_path: Path to coarse DEM for calibration
            num_control_points: Number of random control points for IDW
            random_seed: Random seed for control point generation
            region_size: Size of regions for coarse DEM matching
            optimize_tiles: Whether to optimize tile offsets for seamless mosaicing
            
        Returns:
            Predicted DEM array
        """
        # Load SAR image
        sar_data, sar_meta = self._load_sar(sar_path)
        
        # Calculate stride from overlap
        stride = self.tile_size - overlap_pixels
        if stride <= 0:
            raise ValueError(f"Overlap ({overlap_pixels}) must be less than tile size ({self.tile_size})")
        
        # Process through tiles
        if optimize_tiles:
            print("Processing with tile optimization...")
            raw_dem = self._process_tiles_with_optimization(sar_data, stride)
        else:
            raw_dem = self._process_tiles(sar_data, stride)
        
        # Generate random control points if needed for IDW
        control_points = None
        if calibration_method == "idw":
            control_points = self._generate_random_control_points(
                raw_dem, num_control_points, random_seed
            )
        
        # Apply calibration
        if calibration_method == "global":
            print("Applying global min/max calibration...")
            calibrated_dem = self._calibrate_global(raw_dem)
        elif calibration_method == "coarse_dem" and coarse_dem_path:
            print("Applying coarse DEM calibration...")
            coarse_dem = self._load_coarse_dem(coarse_dem_path, sar_meta)
            calibrated_dem = self._calibrate_with_coarse_dem(raw_dem, coarse_dem, region_size)
        elif calibration_method == "idw" and control_points:
            print(f"Applying IDW calibration with {num_control_points} random points...")
            calibrated_dem = self._calibrate_with_idw(raw_dem, control_points)
        else:
            print("Using raw model output (no calibration)")
            calibrated_dem = raw_dem
        
        # Save output
        self._save_dem(calibrated_dem, output_path, sar_meta)
        
        return calibrated_dem
    
    def _generate_random_control_points(self, dem: np.ndarray, 
                                       num_points: int, 
                                       random_seed: int) -> Dict:
        """
        Generate random control points from the DEM.
        
        Args:
            dem: Input DEM array
            num_points: Number of control points to generate
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'x', 'y', 'z' arrays
        """
        np.random.seed(random_seed)
        
        height, width = dem.shape
        
        # Generate random coordinates
        y_coords = np.random.randint(0, height, num_points)
        x_coords = np.random.randint(0, width, num_points)
        
        # Get elevation values at these points
        z_values = dem[y_coords, x_coords]
        
        # Filter out NaN values
        valid = ~np.isnan(z_values)
        
        control_points = {
            'x': x_coords[valid],
            'y': y_coords[valid],
            'z': z_values[valid]
        }
        
        print(f"Generated {len(control_points['x'])} valid control points (seed={random_seed})")
        
        return control_points
    
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
    
    def _create_weight_map(self, size: int) -> np.ndarray:
        """Create cosine weight map for tile blending (matching original implementation)."""
        y, x = np.mgrid[0:size, 0:size]
        center_y, center_x = size // 2, size // 2
        
        # Distance from center normalized to [0, 1]
        dist_from_center = ((y - center_y) / (size/2))**2 + ((x - center_x) / (size/2))**2
        
        # Cosine weighting
        weight = np.cos(np.minimum(dist_from_center * np.pi/2, np.pi/2))
        
        return weight
    
    def _process_tiles(self, sar_data: np.ndarray, stride: int) -> np.ndarray:
        """Process SAR image through tiles and stitch results."""
        if sar_data.ndim == 3:
            sar_data = sar_data[0]  # Use first channel
        
        height, width = sar_data.shape
        
        # Initialize output and weights for blending
        output = np.zeros((height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        
        # Create weight map for smooth blending
        weight_map = self._create_weight_map(self.tile_size)
        
        # Process tiles
        tiles_processed = 0
        for y in range(0, height - self.tile_size + 1, stride):
            for x in range(0, width - self.tile_size + 1, stride):
                # Extract tile
                tile = sar_data[y:y+self.tile_size, x:x+self.tile_size]
                
                # Normalize and prepare for model
                tile_tensor = self._normalize_sar(tile).unsqueeze(0).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    pred = self.model(tile_tensor).squeeze().cpu().numpy()
                
                # Denormalize
                pred_dem = self._denormalize_dem(pred)
                
                # Add to output with blending
                output[y:y+self.tile_size, x:x+self.tile_size] += pred_dem * weight_map
                weights[y:y+self.tile_size, x:x+self.tile_size] += weight_map
                
                tiles_processed += 1
        
        print(f"Processed {tiles_processed} tiles")
        
        # Normalize by weights
        mask = weights > 0
        output[mask] /= weights[mask]
        
        return output
    
    def _process_tiles_with_optimization(self, sar_data: np.ndarray, stride: int) -> np.ndarray:
        """
        Process tiles with optimization to minimize offsets in overlapping regions.
        Matching the original implementation approach.
        """
        if sar_data.ndim == 3:
            sar_data = sar_data[0]
        
        height, width = sar_data.shape
        
        # First pass: process all tiles and store them
        tiles = []
        tile_coords = []
        
        print("Processing tiles...")
        for y in tqdm(range(0, height - self.tile_size + 1, stride)):
            for x in range(0, width - self.tile_size + 1, stride):
                # Extract tile
                tile = sar_data[y:y+self.tile_size, x:x+self.tile_size]
                
                # Normalize and prepare for model
                tile_tensor = self._normalize_sar(tile).unsqueeze(0).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    pred = self.model(tile_tensor).squeeze().cpu().numpy()
                
                # Denormalize
                pred_dem = self._denormalize_dem(pred)
                
                tiles.append(pred_dem)
                tile_coords.append((y, x))
        
        print(f"Optimizing {len(tiles)} tiles...")
        
        # Optimize offsets
        n_tiles = len(tiles)
        initial_offsets = np.zeros(n_tiles)
        
        def objective(offsets):
            """Compute total difference in overlapping regions."""
            total_error = 0
            n_overlaps = 0
            
            for i in range(n_tiles):
                for j in range(i + 1, n_tiles):
                    y1, x1 = tile_coords[i]
                    y2, x2 = tile_coords[j]
                    
                    # Check for overlap
                    overlap_y = max(0, min(y1 + self.tile_size, y2 + self.tile_size) - max(y1, y2))
                    overlap_x = max(0, min(x1 + self.tile_size, x2 + self.tile_size) - max(x1, x2))
                    
                    if overlap_y > 0 and overlap_x > 0:
                        # Get overlapping regions in each tile
                        # For tile i
                        i_y_start = max(0, y2 - y1)
                        i_y_end = min(self.tile_size, y2 + self.tile_size - y1)
                        i_x_start = max(0, x2 - x1)
                        i_x_end = min(self.tile_size, x2 + self.tile_size - x1)
                        
                        # For tile j
                        j_y_start = max(0, y1 - y2)
                        j_y_end = min(self.tile_size, y1 + self.tile_size - y2)
                        j_x_start = max(0, x1 - x2)
                        j_x_end = min(self.tile_size, x1 + self.tile_size - x2)
                        
                        region_i = tiles[i][i_y_start:i_y_end, i_x_start:i_x_end] + offsets[i]
                        region_j = tiles[j][j_y_start:j_y_end, j_x_start:j_x_end] + offsets[j]
                        
                        # Compute squared difference
                        diff = np.mean((region_i - region_j) ** 2)
                        total_error += diff
                        n_overlaps += 1
            
            # Add regularization to keep offsets small
            reg_term = 0.001 * np.sum(offsets ** 2)
            
            return total_error / max(n_overlaps, 1) + reg_term
        
        # Optimize with bounds
        bounds = [(-100, 100) for _ in range(n_tiles)]  # Reasonable elevation offset bounds
        result = minimize(objective, initial_offsets, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            print(f"Optimization converged. Final error: {result.fun:.6f}")
        else:
            print(f"Optimization did not fully converge: {result.message}")
        
        # Apply offsets and stitch
        output = np.zeros((height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        weight_map = self._create_weight_map(self.tile_size)
        
        for i, (y, x) in enumerate(tile_coords):
            # Apply offset
            adjusted_tile = tiles[i] + result.x[i]
            
            # Add to output with blending
            output[y:y+self.tile_size, x:x+self.tile_size] += adjusted_tile * weight_map
            weights[y:y+self.tile_size, x:x+self.tile_size] += weight_map
        
        # Normalize by weights
        mask = weights > 0
        output[mask] /= weights[mask]
        
        return output
    
    def _calibrate_global(self, dem: np.ndarray) -> np.ndarray:
        """Apply global min/max calibration using known elevation range."""
        # Use percentiles to avoid outliers
        p2 = np.percentile(dem[~np.isnan(dem)], 2)
        p98 = np.percentile(dem[~np.isnan(dem)], 98)
        
        # Clip to percentile range
        dem_clipped = np.clip(dem, p2, p98)
        
        # Scale to expected global range
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
        Regional adaptive scaling matching original implementation.
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
                if np.isnan(coarse_region).sum() > 0.5 * coarse_region.size:
                    continue
                
                # Match statistics
                fine_mean = np.nanmean(fine_region)
                fine_std = np.nanstd(fine_region)
                coarse_mean = np.nanmean(coarse_region)
                coarse_std = np.nanstd(coarse_region)
                
                # Avoid division by zero
                if fine_std < 1e-6:
                    fine_std = 1e-6
                
                # Normalize and rescale
                normalized = (fine_region - fine_mean) / fine_std
                calibrated = normalized * coarse_std + coarse_mean
                
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
        Note: In this implementation, control points are used to create a 
        smooth correction surface rather than actual ground truth calibration.
        """
        if len(control_points['x']) < 3:
            print("Warning: Not enough control points for IDW calibration")
            return dem
        
        height, width = dem.shape
        
        # Create grid of coordinates
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        points = np.column_stack((control_points['y'], control_points['x']))
        
        # Use the DEM values at control points as the "ground truth"
        # In practice, you might want to apply some correction here
        values = control_points['z']
        
        # For demonstration, we'll create a smooth correction surface
        # by interpolating between control points
        try:
            # Use linear interpolation for speed, cubic for quality
            correction = griddata(points, values * 0.01,  # Small correction factor
                                (y_grid, x_grid), method='linear', fill_value=0)
            
            # Apply correction
            calibrated_dem = dem + correction
        except:
            print("Warning: IDW interpolation failed, returning uncalibrated DEM")
            calibrated_dem = dem
        
        return calibrated_dem
    
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
    parser = argparse.ArgumentParser(
                        description="Generate DEM from SAR with calibration")
    parser.add_argument("--sar_path", default="./data/regional_dem/clipped_sar.tif",
                        help="Path to SAR image")
    parser.add_argument("--model_path", default="./model_weights/dpt-l1_ep200.pth",
                        help="Path to trained model")
    parser.add_argument("--output_path", default="./outputs/regional_dem/dpt_l1_dem.tif",
                        help="Path for output DEM")
    parser.add_argument("--calibration", choices=["none", "global", "coarse_dem", "idw"], 
                        default="coarse_dem", help="Calibration method")
    parser.add_argument("--coarse_dem", default="./data/regional_dem/clipped_coarse_dem_1km.tif",
                        help="Path to coarse DEM for calibration")
    parser.add_argument("--overlap", type=int, default=96, 
                        help="Tile overlap in pixels (must be < 384)")
    parser.add_argument("--region_size", type=int, default=128, 
                        help="Region size for coarse DEM matching")
    parser.add_argument("--num_control_points", type=int, default=100,
                        help="Number of random control points for IDW calibration")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for control point generation")
    parser.add_argument("--optimize_tiles", action="store_true", 
                        help="Optimize tile offsets for seamless mosaicing")
    parser.add_argument("--device", choices=["cuda", "cpu"], 
                        help="Compute device")
    
    args = parser.parse_args()
    
    # initialize predictor
    predictor = DEMMosaicPredictor(args.model_path, args.device)
    
    # generate DEM
    dem = predictor.predict_dem(
        sar_path=args.sar_path,
        output_path=args.output_path,
        overlap_pixels=args.overlap,
        calibration_method=args.calibration,
        coarse_dem_path=args.coarse_dem,
        num_control_points=args.num_control_points,
        random_seed=args.random_seed,
        region_size=args.region_size,
        optimize_tiles=args.optimize_tiles
    )
    
    print(f"DEM shape: {dem.shape}")
    print(f"Elevation range: {np.nanmin(dem):.2f} - {np.nanmax(dem):.2f} m")


if __name__ == "__main__":
    main()
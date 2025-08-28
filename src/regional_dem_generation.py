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
        
        # Training dataset elevation bounds for global calibration
        self.training_elev_min = 0.0  # m
        self.training_elev_max = 7650.0  # m
        
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
                    calibration_method: str = "coarse_dem",
                    coarse_dem_path: Optional[str] = None,
                    num_control_points: int = 100,
                    random_seed: int = 42,
                    region_size: int = 128,
                    optimize_tiles: bool = False,
                    reference_dem_path: Optional[str] = None) -> np.ndarray:
        """
        Predict DEM from SAR image with optional calibration.
        
        Args:
            sar_path: Path to SAR image
            output_path: Path to save output DEM
            overlap_pixels: Overlap between tiles in pixels
            calibration_method: 'none', 'global', 'coarse_dem', or 'idw'
            coarse_dem_path: Path to coarse DEM for calibration
            num_control_points: Number of random control points for IDW
            random_seed: Random seed for control point generation
            region_size: Size of regions for coarse DEM matching
            optimize_tiles: Whether to optimize tile offsets for seamless mosaicing
            reference_dem_path: Path to reference DEM for generating control points
            
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
        
        # Apply calibration
        if calibration_method == "global":
            print("Applying global min/max calibration...")
            calibrated_dem = self._calibrate_global(raw_dem)
            
        elif calibration_method == "coarse_dem" and coarse_dem_path:
            print("Applying regional adaptive scaling...")
            coarse_dem = self._load_coarse_dem(coarse_dem_path, sar_meta)
            calibrated_dem = self._calibrate_with_coarse_dem(raw_dem, coarse_dem, region_size)
            
        elif calibration_method == "idw":
            print(f"Applying sparse point calibration (IDW) with {num_control_points} points...")
            # Load reference DEM if provided for control points
            if reference_dem_path:
                reference_dem = self._load_reference_dem(reference_dem_path, sar_meta)
            else:
                # Use coarse DEM as reference if available
                if coarse_dem_path:
                    reference_dem = self._load_coarse_dem(coarse_dem_path, sar_meta)
                else:
                    print("Warning: No reference DEM for IDW calibration, using raw output")
                    calibrated_dem = raw_dem
                    self._save_dem(calibrated_dem, output_path, sar_meta)
                    return calibrated_dem
                    
            control_points = self._generate_control_points_from_reference(
                raw_dem, reference_dem, num_control_points, random_seed
            )
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
    
    def _load_reference_dem(self, dem_path: str, sar_meta: dict) -> np.ndarray:
        """Load reference DEM for control points."""
        return self._load_coarse_dem(dem_path, sar_meta)
    
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
        """
        Create cosine weight map for tile blending.
        W(i,j) = cos(min(π/2 * d(i,j)/(s/2)^2, π/2))
        """
        i, j = np.mgrid[0:size, 0:size]
        center_i, center_j = size // 2, size // 2
        
        # Normalized squared distance from center
        d = ((j - center_j) / (size/2))**2 + ((i - center_i) / (size/2))**2
        
        # Cosine weighting
        weight = np.cos(np.minimum(d * np.pi/2, np.pi/2))
        
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
                
                # Add to output with blending: DEM_composite = Σ(DEM_t * W_t) / Σ(W_t)
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
        bounds = [(-100, 100) for _ in range(n_tiles)]
        result = minimize(objective, initial_offsets, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            print(f"Optimization converged. Final error: {result.fun:.6f}")
        else:
            print(f"Optimization did not fully converge: {result.message}")
        
        # Apply offsets and stitch with weighted blending
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
        """
        Apply global min/max calibration using training dataset bounds.
        DEM_scaled = E_min + (E_max - E_min) * (DEM_model - min(DEM_model)) / (max(DEM_model) - min(DEM_model))
        """
        dem_min = np.nanmin(dem)
        dem_max = np.nanmax(dem)
        
        if dem_max - dem_min < 1e-6:
            print("Warning: DEM has very small range, returning uncalibrated")
            return dem
        
        # Scale to training dataset elevation range
        dem_scaled = self.training_elev_min + (self.training_elev_max - self.training_elev_min) * \
                     (dem - dem_min) / (dem_max - dem_min)
        
        return dem_scaled
    
    def _calibrate_with_coarse_dem(self, 
                                   fine_dem: np.ndarray, 
                                   coarse_dem: np.ndarray,
                                   region_size: int) -> np.ndarray:
        """
        Regional adaptive scaling: statistical matching with coarse DEM.
        DEM_scaled = μ_coarseDEM + σ_coarseDEM * ((DEM_model - μ_model) / σ_model)
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
                
                # Calculate statistics
                μ_model = np.nanmean(fine_region)
                σ_model = np.nanstd(fine_region)
                μ_coarseDEM = np.nanmean(coarse_region)
                σ_coarseDEM = np.nanstd(coarse_region)
                
                # Avoid division by zero
                if σ_model < 1e-6:
                    σ_model = 1e-6
                
                # Apply statistical matching
                calibrated = μ_coarseDEM + σ_coarseDEM * ((fine_region - μ_model) / σ_model)
                
                # Add to output with blending
                output[y:y+region_size, x:x+region_size] += calibrated * weight_map
                weights[y:y+region_size, x:x+region_size] += weight_map
        
        # Normalize by weights
        mask = weights > 0
        output[mask] /= weights[mask]
        output[~mask] = fine_dem[~mask]  # Use original where not processed
        
        return output
    
    def _generate_control_points_from_reference(self, 
                                               model_dem: np.ndarray,
                                               reference_dem: np.ndarray,
                                               num_points: int, 
                                               random_seed: int) -> Dict:
        """
        Generate control points with reference/model ratios for IDW calibration.
        """
        np.random.seed(random_seed)
        
        height, width = model_dem.shape
        
        # Generate random coordinates
        y_coords = np.random.randint(0, height, num_points)
        x_coords = np.random.randint(0, width, num_points)
        
        # Get values from both DEMs
        model_values = model_dem[y_coords, x_coords]
        reference_values = reference_dem[y_coords, x_coords]
        
        # Filter out NaN values
        valid = ~(np.isnan(model_values) | np.isnan(reference_values))
        valid = valid & (np.abs(model_values) > 1e-6)  # Avoid division by zero
        
        control_points = {
            'x': x_coords[valid],
            'y': y_coords[valid], 
            'model_values': model_values[valid],
            'reference_values': reference_values[valid],
            'ratios': reference_values[valid] / model_values[valid]
        }
        
        print(f"Generated {len(control_points['x'])} valid control points (seed={random_seed})")
        
        return control_points
    
    def _calibrate_with_idw(self, 
                           dem: np.ndarray,
                           control_points: Dict,
                           power: float = 2.0) -> np.ndarray:
        """
        sparse point calibration using IDW with ratio-based correction.
        DEM_scaled(x,y) = DEM_model(x,y) * Σ(w_i(x,y) * r_i) / Σ(w_i(x,y))
        where w_i(x,y) = 1/d(x,y,i)^2 and r_i = reference/model ratio at point i
        """
        if len(control_points['x']) < 3:
            print("Warning: Not enough control points for IDW calibration")
            return dem
        
        height, width = dem.shape
        
        # Control point coordinates and ratios
        cp_x = control_points['x']
        cp_y = control_points['y']
        ratios = control_points['ratios']
        
        # Create output array
        calibrated_dem = np.zeros_like(dem)
        
        # Process in chunks to manage memory
        chunk_size = 100
        for y_start in range(0, height, chunk_size):
            y_end = min(y_start + chunk_size, height)
            for x_start in range(0, width, chunk_size):
                x_end = min(x_start + chunk_size, width)
                
                # Create coordinate grids for this chunk
                y_grid, x_grid = np.mgrid[y_start:y_end, x_start:x_end]
                
                # Initialize weighted sum and weight sum
                weighted_ratio_sum = np.zeros((y_end - y_start, x_end - x_start))
                weight_sum = np.zeros((y_end - y_start, x_end - x_start))
                
                # Calculate IDW for each control point
                for i in range(len(cp_x)):
                    # Calculate distances
                    distances = np.sqrt((x_grid - cp_x[i])**2 + (y_grid - cp_y[i])**2)
                    
                    # Avoid division by zero at control point locations
                    distances[distances < 1e-10] = 1e-10
                    
                    # Calculate weights: w_i = 1/d^power
                    weights = 1.0 / (distances ** power)
                    
                    # Add to weighted sums
                    weighted_ratio_sum += weights * ratios[i]
                    weight_sum += weights
                
                # Calculate interpolated ratio field
                ratio_field = weighted_ratio_sum / weight_sum
                
                # Apply ratio-based calibration
                calibrated_dem[y_start:y_end, x_start:x_end] = \
                    dem[y_start:y_end, x_start:x_end] * ratio_field
        
        return calibrated_dem
    
    def _save_dem(self, dem: np.ndarray, output_path: str, meta: dict):
        """save DEM to GeoTIFF."""
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
    parser.add_argument("--calibration", choices=["global", "coarse_dem", "idw"], 
                        default="coarse_dem", help="Calibration method")
    parser.add_argument("--coarse_dem", default="./data/regional_dem/clipped_coarse_dem_1km.tif",
                        help="Path to coarse DEM for calibration")
    parser.add_argument("--reference_dem", default="./data/regional_dem/clipped_ground_truth.tif",
                        help="Path to reference DEM for IDW control points")
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
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Compute device")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = DEMMosaicPredictor(args.model_path, args.device)
    
    # Generate DEM
    dem = predictor.predict_dem(
        sar_path=args.sar_path,
        output_path=args.output_path,
        overlap_pixels=args.overlap,
        calibration_method=args.calibration,
        coarse_dem_path=args.coarse_dem,
        num_control_points=args.num_control_points,
        random_seed=args.random_seed,
        region_size=args.region_size,
        optimize_tiles=args.optimize_tiles,
        reference_dem_path=args.reference_dem
    )
    
    print(f"DEM shape: {dem.shape}")
    print(f"Elevation range: {np.nanmin(dem):.2f} - {np.nanmax(dem):.2f} m")


if __name__ == "__main__":
    main()
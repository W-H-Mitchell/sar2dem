"""
Preprocessing pipeline for SAR-to-DEM training data.

Converts Sentinel-1 GRD SAFE archives and Copernicus GLO-30 DEMs into aligned
SAR/DEM tile pairs suitable for training deep learning models to reconstruct
topography from single SAR images.

Pipeline:
    1. Extract VV-polarization GeoTIFF from Sentinel-1 GRD SAFE archive
    2. Resample SAR onto DEM grid using rioxarray reproject_match
    3. Generate filtered SAR/DEM tile pairs using land cover and data quality checks

Data sources:
    SAR: Sentinel-1 GRD, IW mode, VV polarization
         (https://dataspace.copernicus.eu/)
         Downloaded via SentinelSat (Note: endpoints have since expired)
    DEM: Copernicus GLO-30 30 m GeoTIFF
         (https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model)
         Downloaded via AWS Open Data Registry (https://registry.opendata.aws/copernicus-dem/)
    LUC: ESA WorldCover 10 m classification, reprojected to DEM grid
         (https://esa-worldcover.org/)
"""

import os
import sys
import argparse
import zipfile
import shutil
import tempfile
from itertools import product
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np
import rasterio
import rioxarray
from rasterio.warp import reproject, Resampling


# ESA WorldCover class codes (10 m product)
LUC_TREE_COVER = 10
LUC_SHRUBLAND = 20
LUC_GRASSLAND = 30
LUC_CROPLAND = 40
LUC_BUILT_UP = 50
LUC_WATER = 80
LUC_NODATA = 0


@dataclass
class FilterLimits:
    """Data class to store filter threshold values."""
    max_veg_percent: float = 10.0       # tree cover, grassland, or cropland
    max_builtup_percent: float = 2.0
    max_water_percent: float = 10.0
    max_nodata_percent: float = 10.0
    max_invalid_sar_percent: float = 2.0
    min_sar_value: float = -1.0
    max_sar_value: float = 40000.0
    dem_z_score_limit: float = 10.0


class SARProcessor:
    """Extract VV polarization from Sentinel-1 SAFE and align to DEM grid."""

    def __init__(self, cleanup_temp: bool = True):
        """
        Initialize SAR processor.

        Args:
            cleanup_temp: Whether to remove extracted SAFE contents after use
        """
        self.cleanup_temp = cleanup_temp

    def extract_vv(self, safe_path: str, work_dir: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """
        Return path to the VV-polarization GeoTIFF in a Sentinel-1 SAFE product.

        Accepts either a zipped SAFE archive (.zip) or an unzipped .SAFE directory.
        For zipped inputs the archive is extracted to work_dir (a temporary
        directory if not provided).

        Args:
            safe_path: Path to S1 GRD SAFE archive (.zip) or unzipped .SAFE directory
            work_dir: Directory to extract zipped archives into

        Returns:
            Tuple of (path_to_vv_tiff, path_to_temp_dir_for_cleanup)
        """
        temp_dir = None

        if safe_path.endswith(".zip"):
            if work_dir is None:
                work_dir = tempfile.mkdtemp(prefix="s1_")
                temp_dir = work_dir
            else:
                os.makedirs(work_dir, exist_ok=True)

            print(f"Extracting {safe_path} to {work_dir}")
            with zipfile.ZipFile(safe_path, "r") as zf:
                zf.extractall(work_dir)

            safe_dirs = [d for d in os.listdir(work_dir) if d.endswith(".SAFE")]
            if not safe_dirs:
                raise FileNotFoundError(f"No .SAFE directory found inside {safe_path}")
            safe_dir = os.path.join(work_dir, safe_dirs[0])

        elif safe_path.endswith(".SAFE") or os.path.isdir(safe_path):
            safe_dir = safe_path

        else:
            raise ValueError(f"Expected .zip or .SAFE input, got: {safe_path}")

        measurement_dir = os.path.join(safe_dir, "measurement")
        if not os.path.isdir(measurement_dir):
            raise FileNotFoundError(
                f"No 'measurement' subdirectory in {safe_dir}. "
                "Is this a Sentinel-1 GRD SAFE?"
            )

        vv_files = sorted([
            os.path.join(measurement_dir, f)
            for f in os.listdir(measurement_dir)
            if "-vv-" in f and f.endswith(".tiff")
        ])

        if not vv_files:
            raise FileNotFoundError(
                f"No VV-polarization GeoTIFF found in {measurement_dir}"
            )
        if len(vv_files) > 1:
            print(f"Warning: multiple VV files found, using {os.path.basename(vv_files[0])}")

        return vv_files[0], temp_dir

    def align_to_dem(self, sar_path: str, dem_path: str, output_path: str) -> None:
        """
        Resample SAR raster onto the grid of a reference DEM.

        Detects whether the SAR is georeferenced via Ground Control Points (raw
        Sentinel-1 GRD products) or an affine transform (terrain-corrected
        products) and uses the appropriate reprojection path.

        The SAR input is the detected-amplitude GeoTIFF from a GRD product; no
        radiometric calibration is applied here. Calibration to sigma-nought
        should be performed upstream if required.

        Args:
            sar_path: Path to SAR GeoTIFF (typically the VV measurement from SAFE)
            dem_path: Path to reference DEM GeoTIFF
            output_path: Path to save aligned SAR GeoTIFF
        """
        print(f"Loading SAR: {sar_path}")

        with rasterio.open(sar_path) as src:
            gcps, _ = src.gcps
            has_gcps = bool(gcps)

        if has_gcps:
            print(f"SAR is GCP-georeferenced ({len(gcps)} GCPs); warping via GCPs")
            self._align_via_gcps(sar_path, dem_path, output_path)
        else:
            print("SAR has affine transform; using reproject_match")
            self._align_via_transform(sar_path, dem_path, output_path)

        print(f"Saved aligned SAR to {output_path}")

    def _align_via_gcps(self, sar_path: str, dem_path: str, output_path: str) -> None:
        """Warp a GCP-georeferenced SAR raster onto the DEM grid.

        For raw Sentinel-1 GRD products, pixels outside the swath are DN = 0
        and the file typically has no nodata tag set. We pass src_nodata = 0
        to the warp and write NaN on the destination. Note: this treats
        genuinely-zero DN pixels (radar shadow, still water) as nodata, which
        is acceptable because the tile filter rejects these tiles downstream.
        """
        with rasterio.open(sar_path) as src:
            sar_data = src.read(1).astype(np.float32)
            gcps, gcp_crs = src.gcps
            src_nodata = src.nodata
            if src_nodata is None:
                src_nodata = 0
                print("Source nodata not set; assuming DN=0 for pixels outside swath")

        with rasterio.open(dem_path) as dem_src:
            dst_transform = dem_src.transform
            dst_crs = dem_src.crs
            dst_height = dem_src.height
            dst_width = dem_src.width
            dst_profile = dem_src.profile.copy()

        print(f"Resampling {sar_data.shape} SAR onto {dst_height}x{dst_width} DEM grid...")
        dst_arr = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

        reproject(
            source=sar_data,
            destination=dst_arr,
            src_crs=gcp_crs,
            gcps=gcps,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            resampling=Resampling.bilinear,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )

        dst_profile.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with rasterio.open(output_path, "w", **dst_profile) as dst:
            dst.write(dst_arr, 1)

    def _align_via_transform(self, sar_path: str, dem_path: str, output_path: str) -> None:
        """Reproject an affine-georeferenced SAR raster onto the DEM grid."""
        sar = rioxarray.open_rasterio(sar_path, masked=True)
        dem = rioxarray.open_rasterio(dem_path, masked=True)

        print("Resampling SAR onto DEM grid...")
        aligned = sar.rio.reproject_match(dem)
        aligned = aligned.rio.write_nodata(np.nan, encoded=False)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        aligned.rio.to_raster(output_path)

    def process(self, safe_path: str, dem_path: str, output_path: str) -> None:
        """
        Run the full SAR extraction and alignment pipeline.

        Args:
            safe_path: Path to S1 GRD SAFE archive or directory
            dem_path: Path to reference DEM GeoTIFF
            output_path: Path to save aligned SAR GeoTIFF
        """
        vv_tiff, temp_dir = self.extract_vv(safe_path)
        try:
            self.align_to_dem(vv_tiff, dem_path, output_path)
        finally:
            if self.cleanup_temp and temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)


class TileGenerator:
    """Generate filtered SAR/DEM tile pairs from an aligned scene."""

    def __init__(
        self,
        tile_size: int = 384,
        stride: int = 256,
        limits: FilterLimits = FilterLimits(),
    ):
        """
        Initialize tile generator.

        Args:
            tile_size: Side length of output tiles in pixels
            stride: Step between tile origins in pixels
            limits: FilterLimits object with land-cover and data-quality thresholds
        """
        self.tile_size = tile_size
        self.stride = stride
        self.limits = limits

    def _iter_tile_indices(self, height: int, width: int) -> List[Tuple[int, int]]:
        """Return (row, col) origins for every tile that fits within the raster."""
        rows = list(range(0, height, self.stride))
        cols = list(range(0, width, self.stride))
        indices = []
        for row, col in product(rows, cols):
            if row + self.tile_size <= height and col + self.tile_size <= width:
                indices.append((row, col))
        return indices

    def _check_sar(self, sar_tile: np.ndarray, total_cells: int) -> bool:
        """Check SAR tile against valid-value thresholds (treats NaN as invalid)."""
        invalid = np.count_nonzero(
            np.isnan(sar_tile) |
            (sar_tile <= self.limits.min_sar_value) |
            (sar_tile >= self.limits.max_sar_value)
        )
        invalid_percent = (invalid / total_cells) * 100
        return invalid_percent <= self.limits.max_invalid_sar_percent

    def _check_dem(self, dem_tile: np.ndarray) -> bool:
        """Check DEM tile for invalid values and outliers."""
        if np.any(np.isnan(dem_tile)) or np.any(dem_tile <= 0):
            return False

        dem_std = dem_tile.std()
        if dem_std > 0:
            z_scores = np.abs((dem_tile - dem_tile.mean()) / dem_std)
            if np.any(z_scores > self.limits.dem_z_score_limit):
                return False
        return True

    def _check_landcover(self, luc_tile: np.ndarray, total_cells: int) -> bool:
        """Check land cover percentages against limits."""
        percentages = {
            "tree_cover": (np.count_nonzero(luc_tile == LUC_TREE_COVER) / total_cells) * 100,
            "grassland": (np.count_nonzero(luc_tile == LUC_GRASSLAND) / total_cells) * 100,
            "cropland": (np.count_nonzero(luc_tile == LUC_CROPLAND) / total_cells) * 100,
            "built_up": (np.count_nonzero(luc_tile == LUC_BUILT_UP) / total_cells) * 100,
            "water": (np.count_nonzero(luc_tile == LUC_WATER) / total_cells) * 100,
            "nodata": (np.count_nonzero(luc_tile == LUC_NODATA) / total_cells) * 100,
        }

        fails = (
            percentages["tree_cover"] > self.limits.max_veg_percent or
            percentages["grassland"] > self.limits.max_veg_percent or
            percentages["cropland"] > self.limits.max_veg_percent or
            percentages["built_up"] > self.limits.max_builtup_percent or
            percentages["water"] > self.limits.max_water_percent or
            percentages["nodata"] > self.limits.max_nodata_percent
        )
        return not fails

    def _tile_passes(
        self,
        sar_tile: np.ndarray,
        dem_tile: np.ndarray,
        luc_tile: np.ndarray,
    ) -> bool:
        """Check if a tile passes all land-cover and data-quality filters."""
        total_cells = sar_tile.size

        if not self._check_sar(sar_tile, total_cells):
            return False
        if not self._check_dem(dem_tile):
            return False
        if not self._check_landcover(luc_tile, total_cells):
            return False
        return True

    def generate(
        self,
        sar_path: str,
        dem_path: str,
        luc_path: str,
        output_dir: str,
        name_prefix: str = "tile",
    ) -> None:
        """
        Generate SAR/DEM tile pairs that pass all filters.

        Output layout:
            output_dir/
                data/   <prefix>_y<row>x<col>.tif   SAR tiles (model input)
                label/  <prefix>_y<row>x<col>.tif   DEM tiles (training target)

        The SAR, DEM, and LUC rasters must share the same grid (CRS, extent,
        resolution). Run SARProcessor.align_to_dem first if SAR is not already
        matched to the DEM.

        Args:
            sar_path: Path to aligned SAR GeoTIFF
            dem_path: Path to reference DEM GeoTIFF
            luc_path: Path to ESA WorldCover LUC GeoTIFF on the DEM grid
            output_dir: Directory for output data/ and label/ subdirectories
            name_prefix: Filename prefix for output tiles
        """
        data_dir = os.path.join(output_dir, "data")
        label_dir = os.path.join(output_dir, "label")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        print(f"Loading rasters...")
        sar = rioxarray.open_rasterio(sar_path, lock=True)
        dem = rioxarray.open_rasterio(dem_path, lock=True)
        luc = rioxarray.open_rasterio(luc_path, lock=True)

        if sar.shape[-2:] != dem.shape[-2:] or sar.shape[-2:] != luc.shape[-2:]:
            raise ValueError(
                f"SAR, DEM, and LUC must share the same grid. "
                f"Got SAR {sar.shape}, DEM {dem.shape}, LUC {luc.shape}. "
                f"Run SARProcessor.align_to_dem (and reproject the LUC to the DEM) first."
            )

        _, height, width = dem.shape
        tile_indices = self._iter_tile_indices(height, width)
        print(f"Evaluating {len(tile_indices)} candidate tiles...")

        n_kept = 0
        for row, col in tile_indices:
            y_slice = slice(row, row + self.tile_size)
            x_slice = slice(col, col + self.tile_size)

            sar_tile = sar.isel(y=y_slice, x=x_slice).values[0]
            dem_tile = dem.isel(y=y_slice, x=x_slice).values[0]
            luc_tile = luc.isel(y=y_slice, x=x_slice).values[0]

            if sar_tile.shape != (self.tile_size, self.tile_size):
                continue
            if not self._tile_passes(sar_tile, dem_tile, luc_tile):
                continue

            suffix = f"y{row}x{col}"
            data_filename = os.path.join(data_dir, f"{name_prefix}_{suffix}.tif")
            label_filename = os.path.join(label_dir, f"{name_prefix}_{suffix}.tif")

            sar.isel(y=y_slice, x=x_slice).rio.to_raster(data_filename)
            dem.isel(y=y_slice, x=x_slice).rio.to_raster(label_filename)
            n_kept += 1

        print(f"Kept {n_kept}/{len(tile_indices)} tiles -> {output_dir}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="SAR-to-DEM preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # align-sar subcommand
    p_align = subparsers.add_parser(
        "align-sar",
        help="Extract VV from S1 SAFE and resample onto DEM grid",
    )
    p_align.add_argument("--safe_path", required=True,
                         help="Path to S1 GRD SAFE .zip or unzipped .SAFE directory")
    p_align.add_argument("--dem_path", required=True,
                         help="Path to reference DEM GeoTIFF")
    p_align.add_argument("--output_path", required=True,
                         help="Output aligned SAR GeoTIFF")

    # tile subcommand
    p_tile = subparsers.add_parser(
        "tile",
        help="Generate filtered SAR/DEM tile pairs",
    )
    p_tile.add_argument("--sar_path", required=True,
                        help="Path to aligned SAR GeoTIFF (output of align-sar)")
    p_tile.add_argument("--dem_path", required=True,
                        help="Path to reference DEM GeoTIFF")
    p_tile.add_argument("--luc_path", required=True,
                        help="Path to ESA WorldCover LUC GeoTIFF on DEM grid")
    p_tile.add_argument("--output_dir", required=True,
                        help="Directory to write data/ and label/ subdirectories")
    p_tile.add_argument("--tile_size", type=int, default=384,
                        help="Tile side length in pixels")
    p_tile.add_argument("--stride", type=int, default=256,
                        help="Stride between tile origins in pixels")
    p_tile.add_argument("--name_prefix", default="tile",
                        help="Filename prefix for output tiles")

    args = parser.parse_args()

    if args.command == "align-sar":
        processor = SARProcessor()
        processor.process(args.safe_path, args.dem_path, args.output_path)

    elif args.command == "tile":
        generator = TileGenerator(
            tile_size=args.tile_size,
            stride=args.stride,
        )
        generator.generate(
            sar_path=args.sar_path,
            dem_path=args.dem_path,
            luc_path=args.luc_path,
            output_dir=args.output_dir,
            name_prefix=args.name_prefix,
        )


if __name__ == "__main__":
    main()
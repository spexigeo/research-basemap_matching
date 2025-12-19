"""
Preprocessing module for orthomosaic registration.
Handles image loading, downsampling, overlap computation, and coordinate system alignment.
"""

import numpy as np
import cv2
import rasterio
from rasterio.warp import Resampling
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging
import pyproj

logging.basicConfig(level=logging.INFO)


class ImagePreprocessor:
    """Handles preprocessing of source and target images for registration."""
    
    def __init__(self, source_path: str, target_path: str, output_dir: Path):
        """
        Initialize preprocessor.
        
        Args:
            source_path: Path to source orthomosaic
            target_path: Path to target basemap
            output_dir: Output directory for processed files
        """
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy-load metadata (will be loaded when first accessed)
        self._source_metadata_loaded = False
        self._target_metadata_loaded = False
    
    def _load_source_metadata(self):
        """Lazy-load source metadata when needed."""
        if not self._source_metadata_loaded:
            with rasterio.open(self.source_path) as src:
                self._source_transform = src.transform
                self._source_crs = src.crs
                self._source_shape = (src.height, src.width)
                self._source_bounds = src.bounds
                self._source_count = src.count
            self._source_metadata_loaded = True
    
    def _load_target_metadata(self):
        """Lazy-load target metadata when needed."""
        if not self._target_metadata_loaded:
            with rasterio.open(self.target_path) as tgt:
                self._target_transform = tgt.transform
                self._target_crs = tgt.crs
                self._target_shape = (tgt.height, tgt.width)
                self._target_bounds = tgt.bounds
            self._target_metadata_loaded = True
    
    @property
    def source_crs(self):
        """Lazy-load source CRS."""
        self._load_source_metadata()
        return self._source_crs
    
    @source_crs.setter
    def source_crs(self, value):
        self._source_crs = value
    
    @property
    def source_transform(self):
        """Lazy-load source transform."""
        self._load_source_metadata()
        return self._source_transform
    
    @source_transform.setter
    def source_transform(self, value):
        self._source_transform = value
    
    @property
    def source_shape(self):
        """Lazy-load source shape."""
        self._load_source_metadata()
        return self._source_shape
    
    @source_shape.setter
    def source_shape(self, value):
        self._source_shape = value
    
    @property
    def source_bounds(self):
        """Lazy-load source bounds."""
        self._load_source_metadata()
        return self._source_bounds
    
    @source_bounds.setter
    def source_bounds(self, value):
        self._source_bounds = value
    
    @property
    def source_count(self):
        """Lazy-load source count."""
        self._load_source_metadata()
        return self._source_count
    
    @source_count.setter
    def source_count(self, value):
        self._source_count = value
    
    @property
    def target_crs(self):
        """Lazy-load target CRS."""
        self._load_target_metadata()
        return self._target_crs
    
    @target_crs.setter
    def target_crs(self, value):
        self._target_crs = value
    
    @property
    def target_transform(self):
        """Lazy-load target transform."""
        self._load_target_metadata()
        return self._target_transform
    
    @target_transform.setter
    def target_transform(self, value):
        self._target_transform = value
    
    @property
    def target_shape(self):
        """Lazy-load target shape."""
        self._load_target_metadata()
        return self._target_shape
    
    @target_shape.setter
    def target_shape(self, value):
        self._target_shape = value
    
    @property
    def target_bounds(self):
        """Lazy-load target bounds."""
        self._load_target_metadata()
        return self._target_bounds
    
    @target_bounds.setter
    def target_bounds(self, value):
        self._target_bounds = value
    
    @property
    def source_res(self):
        """Lazy-load source resolution."""
        self._load_source_metadata()
        if not hasattr(self, '_source_res'):
            self._source_res = self._calculate_resolution(
                self.source_crs, self.source_bounds, self.source_shape
            )
        return self._source_res
    
    @property
    def target_res(self):
        """Lazy-load target resolution."""
        self._load_target_metadata()
        if not hasattr(self, '_target_res'):
            self._target_res = self._calculate_resolution(
                self.target_crs, self.target_bounds, self.target_shape
            )
        return self._target_res
    
    def log_metadata(self):
        """Log metadata (call when needed for debugging)."""
        self._load_source_metadata()
        self._load_target_metadata()
        logging.info(f"Source: {self.source_shape[1]}x{self.source_shape[0]} @ {self.source_res:.4f} m/px")
        logging.info(f"Target: {self.target_shape[1]}x{self.target_shape[0]} @ {self.target_res:.4f} m/px")
        logging.info(f"Source CRS: {self.source_crs}")
        logging.info(f"Target CRS: {self.target_crs}")
    
    def _calculate_resolution(self, crs, bounds, shape) -> float:
        """Calculate resolution in meters per pixel."""
        if crs and crs.is_geographic:
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.top + bounds.bottom) / 2
            geod = pyproj.Geod(ellps='WGS84')
            _, _, dist_x = geod.inv(bounds.left, center_lat, bounds.right, center_lat)
            res_x = dist_x / shape[1]
            _, _, dist_y = geod.inv(center_lon, bounds.bottom, center_lon, bounds.top)
            res_y = dist_y / shape[0]
            return (res_x + res_y) / 2
        else:
            # Use transform to get resolution
            if crs == self.source_crs:
                return abs(self.source_transform.a)
            else:
                return abs(self.target_transform.a)
    
    def compute_overlap_region(self, scale_factor: float) -> Optional[Dict]:
        """
        Compute the geographic overlap region between source and target.
        Returns pixel coordinates for cropping both images at the given scale.
        """
        # Get bounds in geographic coordinates (meters for UTM)
        src_bounds = self.source_bounds
        tgt_bounds = self.target_bounds
        
        # Find overlap region
        overlap_left = max(src_bounds.left, tgt_bounds.left)
        overlap_bottom = max(src_bounds.bottom, tgt_bounds.bottom)
        overlap_right = min(src_bounds.right, tgt_bounds.right)
        overlap_top = min(src_bounds.top, tgt_bounds.top)
        
        # Check if there's any overlap
        if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
            logging.error("No geographic overlap between source and target!")
            return None
        
        overlap_width_m = overlap_right - overlap_left
        overlap_height_m = overlap_top - overlap_bottom
        
        logging.info(f"  Geographic overlap: {overlap_width_m:.1f}m × {overlap_height_m:.1f}m")
        
        # Convert overlap bounds to pixel coordinates
        src_transform_inv = ~self.source_transform
        tgt_transform_inv = ~self.target_transform
        
        src_col_left, src_row_top = src_transform_inv * (overlap_left, overlap_top)
        src_col_right, src_row_bottom = src_transform_inv * (overlap_right, overlap_bottom)
        
        tgt_col_left, tgt_row_top = tgt_transform_inv * (overlap_left, overlap_top)
        tgt_col_right, tgt_row_bottom = tgt_transform_inv * (overlap_right, overlap_bottom)
        
        # Ensure valid pixel coordinates
        src_x1_raw = min(src_col_left, src_col_right)
        src_x2_raw = max(src_col_left, src_col_right)
        src_y1_raw = min(src_row_top, src_row_bottom)
        src_y2_raw = max(src_row_top, src_row_bottom)
        
        tgt_x1_raw = min(tgt_col_left, tgt_col_right)
        tgt_x2_raw = max(tgt_col_left, tgt_col_right)
        tgt_y1_raw = min(tgt_row_top, tgt_row_bottom)
        tgt_y2_raw = max(tgt_row_top, tgt_row_bottom)
        
        # Convert to integer pixel coordinates and scale
        src_x1 = int(src_x1_raw * scale_factor)
        src_y1 = int(src_y1_raw * scale_factor)
        src_x2 = int(src_x2_raw * scale_factor)
        src_y2 = int(src_y2_raw * scale_factor)
        
        tgt_x1 = int(tgt_x1_raw * scale_factor)
        tgt_y1 = int(tgt_y1_raw * scale_factor)
        tgt_x2 = int(tgt_x2_raw * scale_factor)
        tgt_y2 = int(tgt_y2_raw * scale_factor)
        
        # Ensure coordinates are within image bounds
        src_height = int(self.source_shape[0] * scale_factor)
        src_width = int(self.source_shape[1] * scale_factor)
        tgt_height = int(self.target_shape[0] * scale_factor)
        tgt_width = int(self.target_shape[1] * scale_factor)
        
        # Clamp to valid ranges
        src_x1 = max(0, min(src_x1, src_width - 1))
        src_x2 = max(src_x1 + 1, min(src_x2, src_width))
        src_y1 = max(0, min(src_y1, src_height - 1))
        src_y2 = max(src_y1 + 1, min(src_y2, src_height))
        
        tgt_x1 = max(0, min(tgt_x1, tgt_width - 1))
        tgt_x2 = max(tgt_x1 + 1, min(tgt_x2, tgt_width))
        tgt_y1 = max(0, min(tgt_y1, tgt_height - 1))
        tgt_y2 = max(tgt_y1 + 1, min(tgt_y2, tgt_height))
        
        overlap_info = {
            'source': {'x1': src_x1, 'y1': src_y1, 'x2': src_x2, 'y2': src_y2},
            'target': {'x1': tgt_x1, 'y1': tgt_y1, 'x2': tgt_x2, 'y2': tgt_y2},
            'geographic': {
                'left': overlap_left,
                'right': overlap_right,
                'top': overlap_top,
                'bottom': overlap_bottom,
                'width_m': overlap_width_m,
                'height_m': overlap_height_m
            }
        }
        
        return overlap_info
    
    def load_downsampled(self, scale_factor: float) -> Tuple[np.ndarray, np.ndarray]:
        """Load downsampled versions of source and target images."""
        
        def load_and_downsample(path: Path, scale: float) -> np.ndarray:
            with rasterio.open(path) as src:
                out_height = max(int(src.height * scale), 100)
                out_width = max(int(src.width * scale), 100)
                
                logging.info(f"  Loading {path.name} -> {out_width}x{out_height}")
                
                # Read RGB bands if available, otherwise first band
                if src.count >= 3:
                    indexes = [1, 2, 3]
                    num_bands = 3
                else:
                    indexes = [1]
                    num_bands = 1
                
                data = src.read(
                    indexes=indexes,
                    out_shape=(num_bands, out_height, out_width),
                    resampling=Resampling.bilinear
                )
                
                # Convert to 8-bit grayscale
                if data.shape[0] >= 3:
                    rgb = np.moveaxis(data[:3], 0, -1)
                    rgb_min, rgb_max = rgb.min(), rgb.max()
                    if rgb_max > rgb_min:
                        rgb_normalized = ((rgb - rgb_min) / (rgb_max - rgb_min) * 255).astype(np.uint8)
                    else:
                        rgb_normalized = np.zeros_like(rgb, dtype=np.uint8)
                    gray = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2GRAY)
                else:
                    single = data[0]
                    single_min, single_max = single.min(), single.max()
                    if single_max > single_min:
                        gray = ((single - single_min) / (single_max - single_min) * 255).astype(np.uint8)
                    else:
                        gray = np.zeros((out_height, out_width), dtype=np.uint8)
                
                return gray
        
        # Load and downsample directly (no cache files - overlap images are saved separately)
        source_img = load_and_downsample(self.source_path, scale_factor)
        target_img = load_and_downsample(self.target_path, scale_factor)
        
        return source_img, target_img
    
    def crop_to_overlap(self, source: np.ndarray, target: np.ndarray, 
                       overlap_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Crop source and target images to overlap region."""
        src_crop = overlap_info['source']
        tgt_crop = overlap_info['target']
        
        source_overlap = source[src_crop['y1']:src_crop['y2'], 
                               src_crop['x1']:src_crop['x2']]
        target_overlap = target[tgt_crop['y1']:tgt_crop['y2'],
                               tgt_crop['x1']:tgt_crop['x2']]
        
        logging.info(f"  Cropped source: {source_overlap.shape}")
        logging.info(f"  Cropped target: {target_overlap.shape}")
        
        return source_overlap, target_overlap


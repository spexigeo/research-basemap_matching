"""
Core registration library for orthomosaic alignment - IMPROVED VERSION
Contains all registration algorithms with hierarchical refinement.

KEY IMPROVEMENTS:
1. Pre-aligns source at each scale using cumulative transform
2. Adaptive search regions based on expected remaining error
3. Support for full affine and homography transforms
4. Iterative refinement at finest scale
5. Better outlier filtering before RANSAC
"""

import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import ColorInterp
from rasterio.crs import CRS
from affine import Affine
from shapely.geometry import box
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
from scipy.fft import fft2, ifft2
import json
import logging
from datetime import datetime
import pyproj

try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class RegistrationConfig:
    """Configuration for registration process."""
    source_path: str
    target_path: str
    method: str = 'patch_ncc'
    output_dir: str = 'outputs'
    verbose: bool = False
    hierarchical_scales: List[float] = field(default_factory=lambda: [0.1, 0.2])
    preprocess_method: str = 'gradient'
    visualization: Dict = field(default_factory=lambda: {
        'create_match_visualizations': True,
        'create_difference_maps': True,
        'create_png_overviews': True
    })
    output_format: Dict = field(default_factory=lambda: {
        'compression': 'JPEG',
        'jpeg_quality': 90,
        'tiled': True,
        'blocksize': 512
    })

    # Method-specific parameters
    ransac_threshold: float = 5.0
    max_features: int = 5000
    patch_size: int = 256
    patch_grid_spacing: int = 128
    ncc_threshold: float = 0.5
    reproject_to_metric: bool = True

    # NEW: Transform type
    transform_type: str = 'similarity'  # 'similarity', 'affine', 'homography'

    # NEW: Iterative refinement
    enable_refinement: bool = True
    refinement_iterations: int = 2

    # NEW: Adaptive search
    adaptive_search: bool = True
    min_search_margin: int = 50
    max_search_margin: int = 200

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


class OrthoRegistration:
    """Main registration class with multiple algorithms."""

    def __init__(self, config: RegistrationConfig, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.transform_matrix = None
        self.matches_info = {}
        self.registration_stats = {}

        # Create subdirectories
        self.viz_dir = self.output_dir / 'visualizations'
        self.intermediate_dir = self.output_dir / 'intermediate'
        self.logs_dir = self.output_dir / 'logs'

        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # GPU support
        self.use_gpu = False  # Disabled for Mac
        logging.info("Running on CPU (optimized for M4)")

        # Read metadata
        with rasterio.open(config.source_path) as src:
            self.source_transform = src.transform
            self.source_crs = src.crs
            self.source_shape = (src.height, src.width)
            self.source_bounds = src.bounds
            self.source_count = src.count

        with rasterio.open(config.target_path) as tgt:
            self.target_transform = tgt.transform
            self.target_crs = tgt.crs
            self.target_shape = (tgt.height, tgt.width)
            self.target_bounds = tgt.bounds

        # Calculate resolutions
        self.source_res = self._calculate_resolution(self.source_crs, self.source_bounds, self.source_shape)
        self.target_res = self._calculate_resolution(self.target_crs, self.target_bounds, self.target_shape)

        logging.info(f"Source: {self.source_shape[1]}x{self.source_shape[0]} @ {self.source_res:.4f} m/px")
        logging.info(f"Target: {self.target_shape[1]}x{self.target_shape[0]} @ {self.target_res:.4f} m/px")
        logging.info(f"Source CRS: {self.source_crs}")
        logging.info(f"Target CRS: {self.target_crs}")
        logging.info(f"Transform type: {self.config.transform_type}")

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
            if hasattr(self, 'source_transform'):
                return abs(self.source_transform.a) if crs == self.source_crs else abs(self.target_transform.a)
            return 1.0

    def _compute_overlap_region(self, scale_factor: float) -> Dict:
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

        # Convert overlap bounds to pixel coordinates in each image
        # Using the transform: transform * (col, row) = (x, y) in geographic coords
        # Inverse: ~transform * (x, y) = (col, row) in pixel coords

        src_transform_inv = ~self.source_transform
        tgt_transform_inv = ~self.target_transform

        # Get pixel coordinates of overlap region in source
        # Note: rasterio transform maps (col, row) to (x, y), so inverse maps (x, y) to (col, row)
        src_col_left, src_row_top = src_transform_inv * (overlap_left, overlap_top)
        src_col_right, src_row_bottom = src_transform_inv * (overlap_right, overlap_bottom)

        # Get pixel coordinates of overlap region in target
        tgt_col_left, tgt_row_top = tgt_transform_inv * (overlap_left, overlap_top)
        tgt_col_right, tgt_row_bottom = tgt_transform_inv * (overlap_right, overlap_bottom)

        # IMPORTANT: Ensure we have valid pixel coordinates (left < right, top < bottom)
        # Also need to handle the fact that row coordinates increase downward in images
        # but upward in geographic coordinates
        src_x1_raw = min(src_col_left, src_col_right)
        src_x2_raw = max(src_col_left, src_col_right)
        src_y1_raw = min(src_row_top, src_row_bottom)  # Top row (smaller row number)
        src_y2_raw = max(src_row_top, src_row_bottom)  # Bottom row (larger row number)

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

        # Debug: log the raw computed coordinates
        logging.info(f"  Raw source coords (before scaling): x=[{src_x1_raw:.1f}, {src_x2_raw:.1f}], y=[{src_y1_raw:.1f}, {src_y2_raw:.1f}]")
        logging.info(f"  Raw target coords (before scaling): x=[{tgt_x1_raw:.1f}, {tgt_x2_raw:.1f}], y=[{tgt_y1_raw:.1f}, {tgt_y2_raw:.1f}]")

        # Ensure coordinates are within image bounds
        src_height = int(self.source_shape[0] * scale_factor)
        src_width = int(self.source_shape[1] * scale_factor)
        tgt_height = int(self.target_shape[0] * scale_factor)
        tgt_width = int(self.target_shape[1] * scale_factor)

        # Clamp to valid ranges and ensure x1 < x2, y1 < y2
        src_x1 = max(0, min(src_x1, src_width - 1))
        src_x2 = max(src_x1 + 1, min(src_x2, src_width))
        src_y1 = max(0, min(src_y1, src_height - 1))
        src_y2 = max(src_y1 + 1, min(src_y2, src_height))

        tgt_x1 = max(0, min(tgt_x1, tgt_width - 1))
        tgt_x2 = max(tgt_x1 + 1, min(tgt_x2, tgt_width))
        tgt_y1 = max(0, min(tgt_y1, tgt_height - 1))
        tgt_y2 = max(tgt_y1 + 1, min(tgt_y2, tgt_height))

        # Validate that we actually have a reasonable crop region
        src_crop_width = src_x2 - src_x1
        src_crop_height = src_y2 - src_y1
        tgt_crop_width = tgt_x2 - tgt_x1
        tgt_crop_height = tgt_y2 - tgt_y1

        # Check if source crop is suspiciously large (likely wrong)
        # BUT: If source is completely within target, this is expected and OK
        src_is_within_target = (overlap_left == src_bounds.left and 
                               overlap_right == src_bounds.right and
                               overlap_bottom == src_bounds.bottom and
                               overlap_top == src_bounds.top)
        
        if src_crop_width > src_width * 0.9 or src_crop_height > src_height * 0.9:
            if src_is_within_target:
                logging.info(f"  Source image is completely within target bounds - using full source (expected)")
                logging.info(f"  Source crop: {src_crop_width}x{src_crop_height} (full image)")
                logging.info(f"  Target crop: {tgt_crop_width}x{tgt_crop_height} (overlap region)")
            else:
                logging.warning(f"  WARNING: Source crop region is very large ({src_crop_width}x{src_crop_height} vs image {src_width}x{src_height})")
                logging.warning(f"  This suggests the overlap computation may be incorrect!")
                logging.warning(f"  Source bounds: {src_bounds}")
                logging.warning(f"  Target bounds: {tgt_bounds}")
                logging.warning(f"  Overlap bounds: left={overlap_left:.2f}, bottom={overlap_bottom:.2f}, right={overlap_right:.2f}, top={overlap_top:.2f}")

        logging.info(f"  Scaled source coords: x=[{src_x1}, {src_x2}], y=[{src_y1}, {src_y2}] (size: {src_crop_width}x{src_crop_height})")
        logging.info(f"  Scaled target coords: x=[{tgt_x1}, {tgt_x2}], y=[{tgt_y1}, {tgt_y2}] (size: {tgt_crop_width}x{tgt_crop_height})")

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

        logging.info(f"  Source crop region: [{src_y1}:{src_y2}, {src_x1}:{src_x2}]")
        logging.info(f"  Target crop region: [{tgt_y1}:{tgt_y2}, {tgt_x1}:{tgt_x2}]")

        return overlap_info

    def load_downsampled(self, scale_factor: float) -> Tuple[np.ndarray, np.ndarray]:
        """Load downsampled versions - returns ORIGINAL grayscale, not preprocessed."""

        def load_and_downsample(path, scale):
            import time
            start_time = time.time()
            
            with rasterio.open(path) as src:
                out_height = max(int(src.height * scale), 100)
                out_width = max(int(src.width * scale), 100)

                logging.info(f"  Loading {Path(path).name} -> {out_width}x{out_height}")

                # Use out_shape parameter for efficient downsampling during read
                # This avoids loading full resolution into memory
                # Read RGB bands (first 3) if available, otherwise read first band
                if src.count >= 3:
                    # Read first 3 bands (RGB)
                    indexes = [1, 2, 3]
                    num_bands = 3
                else:
                    # Read first band only
                    indexes = [1]
                    num_bands = 1
                
                data = src.read(
                    indexes=indexes,
                    out_shape=(num_bands, out_height, out_width),
                    resampling=Resampling.bilinear
                )
                
                load_time = time.time() - start_time
                logging.debug(f"  Loaded in {load_time:.2f}s")

                # Convert to 8-bit grayscale with proper normalization
                if data.shape[0] >= 3:
                    # Use RGB bands
                    rgb = np.moveaxis(data[:3], 0, -1)

                    # Normalize to 0-255 based on actual data range
                    rgb_min = rgb.min()
                    rgb_max = rgb.max()

                    if rgb_max > rgb_min:
                        rgb_normalized = ((rgb - rgb_min) / (rgb_max - rgb_min) * 255).astype(np.uint8)
                    else:
                        rgb_normalized = np.zeros_like(rgb, dtype=np.uint8)

                    gray = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2GRAY)
                else:
                    # Single band
                    single = data[0]

                    # Normalize to 0-255 based on actual data range
                    single_min = single.min()
                    single_max = single.max()

                    if single_max > single_min:
                        gray = ((single - single_min) / (single_max - single_min) * 255).astype(np.uint8)
                    else:
                        gray = np.zeros((out_height, out_width), dtype=np.uint8)

                logging.info(f"  Pixel value range: {gray.min()} - {gray.max()}")

                return gray

        # Check for cached original images in shared preprocessing directory (parent of run directories)
        preprocessing_dir = self.output_dir.parent / 'preprocessing'
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        
        source_cache_path = preprocessing_dir / f'source_original_scale{scale_factor:.3f}.png'
        target_cache_path = preprocessing_dir / f'target_original_scale{scale_factor:.3f}.png'
        
        if source_cache_path.exists() and target_cache_path.exists():
            logging.info(f"  Loading cached original images from preprocessing/")
            source_img = cv2.imread(str(source_cache_path), cv2.IMREAD_GRAYSCALE)
            target_img = cv2.imread(str(target_cache_path), cv2.IMREAD_GRAYSCALE)
            if source_img is not None and target_img is not None:
                logging.info(f"  ✓ Loaded cached original images")
                return source_img, target_img
            else:
                logging.warning(f"  Failed to load cached images, recomputing...")

        source_img = load_and_downsample(self.config.source_path, scale_factor)
        target_img = load_and_downsample(self.config.target_path, scale_factor)

        # Save ORIGINAL (non-preprocessed) images to cache ONLY if they don't exist
        if not source_cache_path.exists():
            cv2.imwrite(str(source_cache_path), source_img)
            logging.info(f"  Saved source_original_scale{scale_factor:.3f}.png to preprocessing/")
        else:
            logging.debug(f"  source_original_scale{scale_factor:.3f}.png already exists, skipping save")
            
        if not target_cache_path.exists():
            cv2.imwrite(str(target_cache_path), target_img)
            logging.info(f"  Saved target_original_scale{scale_factor:.3f}.png to preprocessing/")
        else:
            logging.debug(f"  target_original_scale{scale_factor:.3f}.png already exists, skipping save")

        return source_img, target_img

    def preprocess_images(self, source: np.ndarray, target: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess images and save both versions. Check cache first."""
        method = self.config.preprocess_method

        if method == 'none':
            return source, target

        # Check for cached preprocessed images in shared preprocessing directory (parent of run directories)
        preprocessing_dir = self.output_dir.parent / 'preprocessing'
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        
        source_cache_path = preprocessing_dir / f'source_preprocessed_{method}_scale{scale:.3f}.png'
        target_cache_path = preprocessing_dir / f'target_preprocessed_{method}_scale{scale:.3f}.png'
        
        if source_cache_path.exists() and target_cache_path.exists():
            logging.info(f"  Loading cached preprocessed images from preprocessing/")
            source_prep = cv2.imread(str(source_cache_path), cv2.IMREAD_GRAYSCALE)
            target_prep = cv2.imread(str(target_cache_path), cv2.IMREAD_GRAYSCALE)
            if source_prep is not None and target_prep is not None:
                logging.info(f"  ✓ Loaded cached images")
                return source_prep, target_prep
            else:
                logging.warning(f"  Failed to load cached images, recomputing...")

        logging.info(f"  Preprocessing: {method}")

        if method == 'histogram':
            source_prep = source.copy()
            target_prep = self._match_histograms(source, target)
        elif method == 'edges':
            source_prep = cv2.Canny(source, 50, 150)
            target_prep = cv2.Canny(target, 50, 150)
        elif method == 'gradient':
            source_prep = self._compute_gradient_magnitude(source)
            target_prep = self._compute_gradient_magnitude(target)
        elif method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            source_prep = clahe.apply(source)
            target_prep = clahe.apply(target)
        else:
            source_prep = source
            target_prep = target

        # Save preprocessed images to cache directory ONLY if they don't exist
        if not source_cache_path.exists():
            cv2.imwrite(str(source_cache_path), source_prep)
            logging.info(f"  Saved source_preprocessed_{method}_scale{scale:.3f}.png to preprocessing/")
        else:
            logging.debug(f"  source_preprocessed_{method}_scale{scale:.3f}.png already exists, skipping save")
            
        if not target_cache_path.exists():
            cv2.imwrite(str(target_cache_path), target_prep)
            logging.info(f"  Saved target_preprocessed_{method}_scale{scale:.3f}.png to preprocessing/")
        else:
            logging.debug(f"  target_preprocessed_{method}_scale{scale:.3f}.png already exists, skipping save")

        return source_prep, target_prep

    def _match_histograms(self, source: np.ndarray, template: np.ndarray) -> np.ndarray:
        """Match histogram of source to template."""
        src_values, bin_idx, counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
        tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)
        src_quantiles = np.cumsum(counts).astype(np.float64) / source.size
        tmpl_quantiles = np.cumsum(tmpl_counts).astype(np.float64) / template.size
        interp_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
        return interp_values[bin_idx].reshape(source.shape).astype(source.dtype)

    def _compute_gradient_magnitude(self, img: np.ndarray) -> np.ndarray:
        """Compute gradient magnitude."""
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _apply_transform_to_image(self, image: np.ndarray, M: np.ndarray,
                                  target_shape: Tuple[int, int],
                                  enlarge_canvas: bool = False) -> np.ndarray:
        """
        Apply transformation matrix to image.

        Args:
            image: Source image
            M: Transform matrix (2x3 for affine, 3x3 for homography)
            target_shape: Output shape (height, width)
            enlarge_canvas: If True, enlarge canvas to prevent cropping

        Returns:
            Transformed image
        """
        # If enlarging canvas, add padding based on translation
        if enlarge_canvas:
            # Get translation magnitude
            if M.shape[0] == 3 and M.shape[1] == 3:
                # Homography - approximate with translation part
                tx, ty = M[0, 2], M[1, 2]
            else:
                # Affine
                tx, ty = M[0, 2], M[1, 2]

            # CRITICAL: Understand warpAffine translation direction
            # Positive tx means output is shifted RIGHT, so need padding on RIGHT to see it
            # Negative tx means output is shifted LEFT, so need padding on LEFT to see it
            margin = 100

            # If tx > 0: image shifts RIGHT, need padding on RIGHT
            # If tx < 0: image shifts LEFT, need padding on LEFT
            pad_left = int(max(0, -tx) + margin)  # Only if tx < 0 (shift left)
            pad_right = int(max(0, tx) + margin)  # Only if tx > 0 (shift right)
            pad_top = int(max(0, -ty) + margin)  # Only if ty < 0 (shift up)
            pad_bottom = int(max(0, ty) + margin)  # Only if ty > 0 (shift down)

            logging.debug(f"    Canvas enlargement:")
            logging.debug(f"      Translation: ({tx:.1f}, {ty:.1f})")
            logging.debug(f"      Padding: left={pad_left}, right={pad_right}, top={pad_top}, bottom={pad_bottom}")

            # Create larger output shape
            out_width = target_shape[1] + pad_left + pad_right
            out_height = target_shape[0] + pad_top + pad_bottom

            # Adjust transform to account for left/top padding
            M_padded = M.copy()
            M_padded[0, 2] += pad_left  # Add left padding to x translation
            M_padded[1, 2] += pad_top  # Add top padding to y translation

            logging.debug(
                f"      Enlarged canvas: {out_width}x{out_height} (original: {target_shape[1]}x{target_shape[0]})")
            logging.debug(f"      Adjusted transform: tx={M_padded[0, 2]:.1f}, ty={M_padded[1, 2]:.1f}")

            if M.shape[0] == 3 and M.shape[1] == 3:
                # Homography
                result = cv2.warpPerspective(image, M_padded, (out_width, out_height),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=0)
            else:
                # Affine (2x3)
                result = cv2.warpAffine(image, M_padded, (out_width, out_height),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)

            # Crop back to target shape from the padded position
            y1 = pad_top
            y2 = pad_top + target_shape[0]
            x1 = pad_left
            x2 = pad_left + target_shape[1]

            logging.debug(f"      Cropping to target: [{y1}:{y2}, {x1}:{x2}]")

            result = result[y1:y2, x1:x2]

            return result
        else:
            # Standard warp to exact target shape
            if M.shape[0] == 3 and M.shape[1] == 3:
                # Homography
                return cv2.warpPerspective(image, M, (target_shape[1], target_shape[0]),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=0)
            else:
                # Affine (2x3)
                return cv2.warpAffine(image, M, (target_shape[1], target_shape[0]),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)

    def _estimate_transform_robust(self, src_pts: np.ndarray, dst_pts: np.ndarray,
                                   transform_type: str = 'similarity') -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate transformation with specified type using RANSAC.

        Args:
            src_pts: Source points (N, 2)
            dst_pts: Destination points (N, 2)
            transform_type: 'similarity', 'affine', or 'homography'

        Returns:
            (transform_matrix, inliers)
        """
        if transform_type == 'similarity':
            # Rigid + uniform scale (4 DOF)
            M, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_threshold,
                maxIters=2000,
                confidence=0.99
            )

        elif transform_type == 'affine':
            # Full affine (6 DOF)
            M, inliers = cv2.estimateAffine2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_threshold,
                maxIters=2000,
                confidence=0.99
            )

        elif transform_type == 'homography':
            # Homography (8 DOF)
            M, inliers = cv2.findHomography(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_threshold,
                maxIters=2000,
                confidence=0.99
            )
            # Convert inliers from (N, 1) to (N,) for consistency
            if inliers is not None:
                inliers = inliers.ravel()[:, np.newaxis]
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

        return M, inliers

    def register_phase_correlation(self, source: np.ndarray, target: np.ndarray, scale: float) -> np.ndarray:
        """Phase correlation for translation estimation."""
        min_h = min(source.shape[0], target.shape[0])
        min_w = min(source.shape[1], target.shape[1])
        source = source[:min_h, :min_w]
        target = target[:min_h, :min_w]

        f_src = fft2(source)
        f_tgt = fft2(target)
        cross_power = (f_src * np.conj(f_tgt)) / (np.abs(f_src * np.conj(f_tgt)) + 1e-10)
        correlation = np.real(ifft2(cross_power))

        y, x = np.unravel_index(np.argmax(correlation), correlation.shape)

        if y > min_h // 2:
            y -= min_h
        if x > min_w // 2:
            x -= min_w

        x_m = x * self.target_res / scale
        y_m = y * self.target_res / scale

        M = np.float32([[1, 0, x], [0, 1, y]])

        self.matches_info = {
            'shift_pixels': (float(x), float(y)),
            'shift_meters': (float(x_m), float(y_m)),
            'confidence': float(np.max(correlation))
        }

        logging.info(f"  Shift: ({x:.1f}, {y:.1f}) px = ({x_m:.2f}, {y_m:.2f}) m, conf={np.max(correlation):.3f}")

        return M

    def register_patch_ncc(self, source: np.ndarray, target: np.ndarray, scale: float,
                           source_original: np.ndarray, target_original: np.ndarray,
                           expected_error_m: float = None,
                           crop_offset_source: np.ndarray = None,
                           crop_offset_target: np.ndarray = None,
                           overlap_info: Dict = None,
                           force_ncc_threshold: float = None) -> np.ndarray:
        """
        Patch-based NCC matching with adaptive search region.

        Args:
            source: Preprocessed source image (CROPPED to overlap region)
            target: Preprocessed target image (CROPPED to overlap region)
            scale: Current scale factor
            source_original: Original (non-preprocessed) source for visualization (CROPPED)
            target_original: Original (non-preprocessed) target for visualization (CROPPED)
            expected_error_m: Expected remaining error in meters (for adaptive search)
            crop_offset_source: Pixel offset of source crop in full image
            crop_offset_target: Pixel offset of target crop in full image
            overlap_info: Geographic overlap information for coordinate conversion
        """

        min_dim = min(source.shape[0], source.shape[1], target.shape[0], target.shape[1])
        
        # Scale-adaptive patch size: larger patches at coarser scales for better matching
        # At scale 0.15, use larger patches; at scale 0.5, use smaller patches
        scale_factor = scale
        if scale_factor < 0.2:
            patch_size_multiplier = 1.5  # Larger patches at coarse scales
        elif scale_factor < 0.4:
            patch_size_multiplier = 1.2
        else:
            patch_size_multiplier = 1.0
        
        patch_size = max(min(int(self.config.patch_size * patch_size_multiplier), min_dim // 4), 32)
        grid_spacing = max(min(int(self.config.patch_grid_spacing * patch_size_multiplier), patch_size), 16)

        # Scale-adaptive NCC threshold: lower at coarser scales
        # At very coarse scales, features are less distinct, so lower threshold significantly
        if force_ncc_threshold is not None:
            ncc_threshold = force_ncc_threshold
        elif scale_factor < 0.2:
            ncc_threshold = max(0.15, self.config.ncc_threshold * 0.5)  # 50% lower, min 0.15 at very coarse scales
        elif scale_factor < 0.4:
            ncc_threshold = max(0.20, self.config.ncc_threshold * 0.65)  # 35% lower, min 0.20 at medium scales
        else:
            ncc_threshold = self.config.ncc_threshold
        
        # Scale-adaptive search margin: larger at coarser scales
        if self.config.adaptive_search and expected_error_m is not None:
            # Convert expected error to pixels at current scale
            expected_error_px = int(expected_error_m / (self.target_res / scale))
            # Add 50% margin
            base_search_margin = int(expected_error_px * 1.5)
        else:
            # Base search margin scales with image size and scale
            # At coarse scales, need larger search margin
            base_search_margin = int(150 / scale_factor) if scale_factor > 0 else 150
        
        # Scale up search margin for coarser scales
        if scale_factor < 0.2:
            search_margin = int(base_search_margin * 2.0)  # 2x at very coarse scales
        elif scale_factor < 0.4:
            search_margin = int(base_search_margin * 1.5)  # 1.5x at medium scales
        else:
            search_margin = base_search_margin
        
        search_margin = np.clip(search_margin,
                                self.config.min_search_margin,
                                max(self.config.max_search_margin, int(500 / scale_factor)))  # Allow larger margins at coarse scales
        
        logging.info(f"  Scale-adaptive parameters: patch_size={patch_size}, grid_spacing={grid_spacing}, ncc_threshold={ncc_threshold:.3f}, search_margin={search_margin}")

        h, w = target.shape
        matches = []
        all_attempts = []

        logging.info(f"  Patch matching: {patch_size}px patches, {grid_spacing}px spacing, {search_margin}px search")
        logging.info(f"  Images: source={source.shape}, target={target.shape}")
        logging.info(f"  CRITICAL: Using CROPPED overlap images for matching")
        logging.info(f"  Source is cropped overlap region, target is cropped overlap region")
        
        source_h, source_w = source.shape
        
        # CRITICAL VALIDATION: Check if source crop is the full image
        # This is OK if source is completely within target, but we need to adjust matching strategy
        source_orig_full_h = int(self.source_shape[0] * scale)
        source_orig_full_w = int(self.source_shape[1] * scale)
        source_crop_ratio_h = source_h / source_orig_full_h if source_orig_full_h > 0 else 1.0
        source_crop_ratio_w = source_w / source_orig_full_w if source_orig_full_w > 0 else 1.0
        
        source_is_full_image = (source_crop_ratio_h > 0.95 and source_crop_ratio_w > 0.95)
        
        if source_is_full_image:
            logging.warning(f"  WARNING: Source crop is the full image ({source_crop_ratio_w:.1%} x {source_crop_ratio_h:.1%})")
            logging.warning(f"  This means source is completely within target bounds")
            logging.warning(f"  Source full size: {source_orig_full_w}x{source_orig_full_h}, crop size: {source_w}x{source_h}")
            logging.warning(f"  Target crop size: {w}x{h}")
            logging.warning(f"  We need to match target patches to the FULL source image")
            logging.warning(f"  This requires converting target pixel coords to source geographic coords, then to source pixel coords")
            logging.warning(f"  For now, we'll search in the source using a larger search region")
            # Increase search margin significantly since we're searching in full source
            search_margin = max(search_margin * 3, 500)  # At least 500px or 3x original
            logging.info(f"  Increased search margin to {search_margin}px to account for full source image")
        
        # Check size ratio between cropped images
        size_ratio = max(source_h / h, source_w / w) if h > 0 and w > 0 else 1.0
        
        if size_ratio > 3.0:
            logging.warning(f"  Source crop is {size_ratio:.1f}x larger than target crop - patch matching may be inefficient")
            logging.warning(f"  Consider using phase correlation or feature-based matching instead")
        
        logging.info(f"  Source crop is {source_crop_ratio_w:.1%} x {source_crop_ratio_h:.1%} of full image - OK")
        logging.info(f"  Searching at same pixel coordinates in both cropped images (same geographic region)")

        tested = 0
        for y in range(patch_size // 2, h - patch_size // 2, grid_spacing):
            for x in range(patch_size // 2, w - patch_size // 2, grid_spacing):
                tested += 1
                ty1, ty2 = y - patch_size // 2, y + patch_size // 2
                tx1, tx2 = x - patch_size // 2, x + patch_size // 2
                target_patch = target[ty1:ty2, tx1:tx2]

                # CRITICAL FIX: Handle two cases:
                # 1. Both images are cropped to overlap -> pixel (x,y) in target = pixel (x,y) in source
                # 2. Source is full image, target is cropped -> need geographic coordinate conversion
                
                if source_is_full_image and overlap_info is not None:
                    # Convert target pixel (x, y) in cropped target to geographic coords
                    # Then convert to source pixel coords
                    tgt_crop = overlap_info['target']
                    geo_info = overlap_info['geographic']
                    
                    # Target pixel (x, y) in cropped target -> full target pixel at current scale
                    # The cropped target starts at (tgt_crop['x1'], tgt_crop['y1']) in scaled target
                    # So full target pixel at current scale is: (tgt_crop['x1'] + x, tgt_crop['y1'] + y)
                    # NOTE: tgt_crop coordinates are already at current scale (from _compute_overlap_region)
                    tgt_scaled_x = tgt_crop['x1'] + x
                    tgt_scaled_y = tgt_crop['y1'] + y
                    
                    # Convert scaled target pixel to full-resolution target pixel
                    # tgt_crop['x1'] is already scaled, so we need to convert back to full-res first
                    # Actually, looking at _compute_overlap_region, tgt_crop['x1'] is computed as int(tgt_x1_raw * scale_factor)
                    # So to get full-res: (tgt_crop['x1'] / scale) + (x / scale) = (tgt_crop['x1'] + x) / scale
                    tgt_full_x = tgt_scaled_x / scale
                    tgt_full_y = tgt_scaled_y / scale
                    
                    # Convert full-resolution target pixel to geographic coords using target transform
                    # Transform maps (col, row) -> (x, y) in geographic coords
                    geo_x, geo_y = self.target_transform * (tgt_full_x, tgt_full_y)
                    
                    # Convert geographic coords to full-resolution source pixel coords
                    # Inverse transform maps (x, y) -> (col, row) in pixel coords
                    src_full_x, src_full_y = ~self.source_transform * (geo_x, geo_y)
                    
                    # Scale to current scale
                    src_scaled_x = src_full_x * scale
                    src_scaled_y = src_full_y * scale
                    
                    # This is our search center in the full source image at current scale
                    center_x = int(np.clip(src_scaled_x, 0, source_w - 1))
                    center_y = int(np.clip(src_scaled_y, 0, source_h - 1))
                    
                    if tested <= 5:  # Log first few for debugging
                        logging.info(f"  Target pixel ({x}, {y}) in crop -> target full ({tgt_full_x:.1f}, {tgt_full_y:.1f}) -> geo ({geo_x:.1f}, {geo_y:.1f}) -> source ({src_full_x:.1f}, {src_full_y:.1f}) -> source scaled ({center_x}, {center_y})")
                else:
                    # Both images are cropped to same overlap region
                    # Pixel (x, y) in target_overlap corresponds to pixel (x, y) in source_overlap
                    center_x = min(max(x, patch_size // 2), source_w - patch_size // 2)
                    center_y = min(max(y, patch_size // 2), source_h - patch_size // 2)
                
                # Calculate search margins (ensure we don't go out of bounds)
                margin_x = min(search_margin, 
                              max(0, center_x - patch_size // 2),
                              source_w - center_x - patch_size // 2)
                margin_y = min(search_margin,
                              max(0, center_y - patch_size // 2),
                              source_h - center_y - patch_size // 2)
                
                logging.debug(f"  Target patch at ({x}, {y}) -> searching source around ({center_x}, {center_y}) with margin ({margin_x}, {margin_y})")
                
                sx1 = max(0, center_x - margin_x)
                sx2 = min(source_w, center_x + margin_x + patch_size)
                sy1 = max(0, center_y - margin_y)
                sy2 = min(source_h, center_y + margin_y + patch_size)
                
                search_region = source[sy1:sy2, sx1:sx2]

                if search_region.shape[0] >= patch_size and search_region.shape[1] >= patch_size:
                    try:
                        result = cv2.matchTemplate(search_region, target_patch, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)

                        src_x = sx1 + max_loc[0] + patch_size // 2
                        src_y = sy1 + max_loc[1] + patch_size // 2

                        # Calculate offset: difference between found position and expected position
                        if source_is_full_image and overlap_info is not None:
                            # Expected position is center_x, center_y (from geographic conversion)
                            expected_x = center_x
                            expected_y = center_y
                        else:
                            # Expected position is same pixel (x, y) in cropped images
                            expected_x = x
                            expected_y = y
                        
                        offset_x = src_x - expected_x
                        offset_y = src_y - expected_y
                        offset_m = np.sqrt(offset_x ** 2 + offset_y ** 2) * self.target_res / scale

                        all_attempts.append({
                            'target': (int(x), int(y)),
                            'source': (int(src_x), int(src_y)),
                            'confidence': float(max_val),
                            'offset_px': (int(offset_x), int(offset_y)),
                            'offset_m': float(offset_m)
                        })

                        if max_val > ncc_threshold:
                            # Additional validation: check if match is reasonable
                            if source_is_full_image and overlap_info is not None:
                                # Expected position is from geographic conversion
                                expected_src_x = center_x
                                expected_src_y = center_y
                            else:
                                # Expected position is same pixel in cropped images
                                expected_src_x = x
                                expected_src_y = y
                            
                            distance_from_expected = np.sqrt((src_x - expected_src_x)**2 + (src_y - expected_src_y)**2)
                            
                            # Allow matches within 3x the search margin
                            max_reasonable_distance = search_margin * 3
                            if distance_from_expected > max_reasonable_distance:
                                logging.debug(f"  Rejecting match at ({src_x}, {src_y}) - too far from expected ({expected_src_x:.0f}, {expected_src_y:.0f}), distance={distance_from_expected:.0f}px")
                                continue
                            
                            matches.append(([src_x, src_y], [x, y], max_val))

                    except cv2.error as e:
                        logging.debug(f"  Template matching error at ({x}, {y}): {e}")
                        continue

        logging.info(f"  Tested {tested} patches, found {len(matches)} matches (threshold={ncc_threshold:.3f}, configured={self.config.ncc_threshold:.3f})")

        # Save detailed match log
        match_log_file = self.intermediate_dir / f'patch_attempts_scale{scale:.3f}.json'
        with open(match_log_file, 'w') as f:
            json.dump({
                'scale': scale,
                'threshold': ncc_threshold,
                'configured_threshold': self.config.ncc_threshold,
                'tested': tested,
                'matches_found': len(matches),
                'search_margin': int(search_margin),  # Convert to Python int for JSON
                'all_attempts': all_attempts[:100]  # Save first 100 for inspection
            }, f, indent=2)
        logging.info(f"  Saved patch attempts to {match_log_file.name}")

        if len(matches) < 10:
            logging.warning(f"  INSUFFICIENT MATCHES: {len(matches)} < 10")
            if len(all_attempts) > 0:
                best_attempts = sorted(all_attempts, key=lambda x: x['confidence'], reverse=True)[:5]
                best_conf_str = [f"{a['confidence']:.3f}" for a in best_attempts]
                logging.warning(f"  Best confidences found: {best_conf_str}")
                logging.warning(f"  Try lowering ncc_threshold to {max(0.2, ncc_threshold - 0.1):.2f} (currently using {ncc_threshold:.3f})")

            # Set empty match info and return identity
            self.matches_info = {
                'total_patches': 0,
                'inliers': 0,
                'inlier_ratio': 0.0,
                'mean_offset_m': 0.0,
                'std_offset_m': 0.0,
                'median_offset_m': 0.0,
                'min_offset_m': 0.0,
                'max_offset_m': 0.0
            }
            return np.eye(2, 3, dtype=np.float32)

        src_pts = np.float32([m[0] for m in matches])
        dst_pts = np.float32([m[1] for m in matches])

        # NEW: Filter outliers by offset distance before RANSAC
        offsets = src_pts - dst_pts
        offset_norms = np.linalg.norm(offsets, axis=1)
        median_offset = np.median(offset_norms)
        mad = np.median(np.abs(offset_norms - median_offset))
        # Keep points within 3 MAD of median
        outlier_mask = np.abs(offset_norms - median_offset) < (3 * mad + 1e-6)

        if np.sum(outlier_mask) >= 10:
            src_pts = src_pts[outlier_mask]
            dst_pts = dst_pts[outlier_mask]
            logging.info(f"  Pre-RANSAC filtering: {np.sum(outlier_mask)}/{len(matches)} points kept")

        # RANSAC with specified transform type
        M, inliers = self._estimate_transform_robust(
            src_pts, dst_pts,
            transform_type=self.config.transform_type
        )

        if M is None:
            logging.warning("  RANSAC failed")
            self.matches_info = {
                'total_patches': 0,
                'inliers': 0,
                'inlier_ratio': 0.0,
                'mean_offset_m': 0.0,
                'std_offset_m': 0.0,
                'median_offset_m': 0.0,
                'min_offset_m': 0.0,
                'max_offset_m': 0.0
            }
            return np.eye(2, 3, dtype=np.float32)

        inlier_count = int(np.sum(inliers))
        offsets_px = src_pts[inliers.ravel().astype(bool)] - dst_pts[inliers.ravel().astype(bool)]
        offsets_m = offsets_px * self.target_res / scale
        offset_norms = np.linalg.norm(offsets_m, axis=1)

        self.matches_info = {
            'total_patches': len(matches),
            'inliers': inlier_count,
            'inlier_ratio': inlier_count / len(matches),
            'mean_offset_m': float(np.mean(offset_norms)),
            'std_offset_m': float(np.std(offset_norms)),
            'median_offset_m': float(np.median(offset_norms)),
            'min_offset_m': float(np.min(offset_norms)),
            'max_offset_m': float(np.max(offset_norms))
        }

        logging.info(f"  RANSAC: {inlier_count}/{len(matches)} inliers ({100 * inlier_count / len(matches):.1f}%)")
        logging.info(f"  Offsets: mean={self.matches_info['mean_offset_m']:.2f}m, "
                     f"std={self.matches_info['std_offset_m']:.2f}m, "
                     f"median={self.matches_info['median_offset_m']:.2f}m")

        # Create visualization with ORIGINAL images (not preprocessed)
        self._visualize_matches_detailed(src_pts, dst_pts, inliers, source_original, target_original,
                                         scale, 'patch_ncc', all_attempts[:100])

        # Save match coordinates
        match_data = {
            'scale': scale,
            'source_points': src_pts.tolist(),
            'target_points': dst_pts.tolist(),
            'inliers': inliers.ravel().tolist(),
            'offsets_meters': offsets_m.tolist(),
            'statistics': self.matches_info
        }
        match_file = self.intermediate_dir / f'matches_data_scale{scale:.3f}.json'
        with open(match_file, 'w') as f:
            json.dump(match_data, f, indent=2)

        return M

    def _visualize_matches_detailed(self, src_pts, dst_pts, inliers, source_img, target_img,
                                    scale, method_name, all_attempts):
        """Create detailed match visualization."""
        fig = plt.figure(figsize=(24, 12))

        # Create grid for 4 subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)

        ax1 = fig.add_subplot(gs[0, 0])  # Target with matches
        ax2 = fig.add_subplot(gs[0, 1])  # Source with matches
        ax3 = fig.add_subplot(gs[1, :])  # Side-by-side with flow lines

        inlier_mask = inliers.ravel().astype(bool)

        # Top left: Target with all features
        ax1.imshow(target_img, cmap='gray')
        ax1.plot(dst_pts[:, 0], dst_pts[:, 1], 'yo', markersize=6, alpha=0.3, label='All matches')
        ax1.plot(dst_pts[inlier_mask, 0], dst_pts[inlier_mask, 1], 'go', markersize=8, label='Inliers')
        ax1.set_title(f'Target (Basemap)\nScale: {scale}, {np.sum(inlier_mask)} inliers',
                      fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.axis('off')

        # Top right: Source with all features
        ax2.imshow(source_img, cmap='gray')
        ax2.plot(src_pts[:, 0], src_pts[:, 1], 'yo', markersize=6, alpha=0.3, label='All matches')
        ax2.plot(src_pts[inlier_mask, 0], src_pts[inlier_mask, 1], 'go', markersize=8, label='Inliers')
        ax2.set_title(f'Source (Orthomosaic)\nScale: {scale}, {len(src_pts)} total matches',
                      fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.axis('off')

        # Bottom: Side-by-side with flow lines (create manually)
        ax3_left = fig.add_axes([0.05, 0.05, 0.45, 0.4])
        ax3_right = fig.add_axes([0.52, 0.05, 0.45, 0.4])

        ax3_left.imshow(target_img, cmap='gray')
        ax3_left.set_title('Target with Match Flow', fontsize=12, fontweight='bold')
        ax3_left.axis('off')

        ax3_right.imshow(source_img, cmap='gray')
        ax3_right.set_title('Source with Match Flow', fontsize=12, fontweight='bold')
        ax3_right.axis('off')

        # Draw flow lines
        for i, (sx, sy), (tx, ty) in zip(range(len(src_pts)), src_pts, dst_pts):
            is_inlier = inlier_mask[i]
            color = 'green' if is_inlier else 'red'
            alpha = 0.6 if is_inlier else 0.2

            ax3_left.plot(tx, ty, 'o', color=color, markersize=4, alpha=alpha)
            ax3_right.plot(sx, sy, 'o', color=color, markersize=4, alpha=alpha)

            con = mpatches.ConnectionPatch(
                xyA=(sx, sy), xyB=(tx, ty),
                coordsA=ax3_right.transData, coordsB=ax3_left.transData,
                color=color, linewidth=0.5, alpha=alpha
            )
            fig.add_artist(con)

        # Save
        vis_file = self.viz_dir / f'matches_detailed_scale{scale:.3f}.png'
        plt.savefig(vis_file, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"  Saved visualization to {vis_file.name}")

    def hierarchical_registration(self) -> Optional[np.ndarray]:
        """
        Hierarchical coarse-to-fine registration with progressive refinement.

        KEY IMPROVEMENT: At each scale, pre-align the source using cumulative transform
        before matching. This allows each scale to refine the alignment rather than
        starting from scratch.

        TRANSFORM MANAGEMENT:
        - M_cumulative: Always stored in FULL RESOLUTION pixels
        - M at each scale: Found in CURRENT SCALE pixels
        - When pre-aligning: Scale M_cumulative down to current scale
        - When accumulating: Scale M up to full resolution first
        """
        M_cumulative = np.eye(2, 3, dtype=np.float32)  # In full-resolution pixels
        scales = self.config.hierarchical_scales

        for i, scale in enumerate(scales):
            logging.info("\n" + "=" * 70)
            logging.info(f"Level {i + 1}/{len(scales)}: Scale {scale}")
            logging.info("=" * 70)

            # Load ORIGINAL images at current scale
            source_orig, target_orig = self.load_downsampled(scale)

            # CRITICAL FIX: Compute geographic overlap and crop both images
            logging.info(f"  Computing geographic overlap region...")
            overlap_info = self._compute_overlap_region(scale)

            if overlap_info is None:
                logging.error(f"  No overlap at scale {scale}, skipping level")
                continue

            # Crop both images to overlap region
            src_crop = overlap_info['source']
            tgt_crop = overlap_info['target']

            source_overlap = source_orig[src_crop['y1']:src_crop['y2'],
            src_crop['x1']:src_crop['x2']]
            target_overlap = target_orig[tgt_crop['y1']:tgt_crop['y2'],
            tgt_crop['x1']:tgt_crop['x2']]

            logging.info(f"  Cropped source: {source_overlap.shape}")
            logging.info(f"  Cropped target: {target_overlap.shape}")

            # Check if overlap images already exist in preprocessing cache
            preprocessing_dir = self.output_dir.parent / 'preprocessing'
            source_overlap_cache = preprocessing_dir / f'source_overlap_scale{scale:.3f}.png'
            target_overlap_cache = preprocessing_dir / f'target_overlap_scale{scale:.3f}.png'
            
            if not (source_overlap_cache.exists() and target_overlap_cache.exists()):
                # Save cropped images for verification (only if not cached)
                cv2.imwrite(
                    str(self.intermediate_dir / f'source_overlap_scale{scale:.3f}.png'),
                    source_overlap
                )
                cv2.imwrite(
                    str(self.intermediate_dir / f'target_overlap_scale{scale:.3f}.png'),
                    target_overlap
                )
                logging.info(f"  Saved overlap images to intermediate/")
            else:
                logging.info(f"  Overlap images already exist in preprocessing/, skipping save")

            # Store the offset from cropping (needed to convert back to full image coords)
            crop_offset_source = np.array([src_crop['x1'], src_crop['y1']])
            crop_offset_target = np.array([tgt_crop['x1'], tgt_crop['y1']])

            # CRITICAL: Apply cumulative transform to source before matching
            # This is the key to hierarchical refinement - each scale refines the previous
            # At scale 0, M_cumulative is identity, so no transform needed
            # At scale 1+, we apply the cumulative transform from previous scales
            if i > 0 and M_cumulative is not None and not (np.allclose(M_cumulative[:2, :2], np.eye(2)) and 
                                                             np.allclose(M_cumulative[:2, 2], [0, 0], atol=1e-3)):
                # Scale cumulative transform down to current scale
                # M_cumulative is in full-resolution pixels, we need it at current scale
                M_cumulative_at_scale = M_cumulative.copy()
                M_cumulative_at_scale[:, 2] *= scale  # Scale translation to current scale
                
                # Apply transform to source overlap region
                # Transform maps: target_coords = M @ source_coords
                # We need to warp source_overlap: for each pixel in source, find where it maps in target space
                # Then sample from that location in source (inverse mapping)
                h_src, w_src = source_overlap.shape
                
                # Create coordinate grid for target space (where we want to sample)
                # We'll create a grid in source space and transform it to see where to sample
                y_coords, x_coords = np.meshgrid(np.arange(h_src), np.arange(w_src), indexing='ij')
                coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=0)
                
                # Transform coordinates: target_coords = M @ source_coords
                # For warping, we need: where in source to sample for each target pixel
                # So: source_coords = M^-1 @ target_coords
                M_3x3 = np.vstack([M_cumulative_at_scale, [0, 0, 1]])
                M_inv = np.linalg.inv(M_3x3)
                
                # Transform each source coordinate to see where it maps
                coords_homogeneous = np.vstack([coords, np.ones((1, coords.shape[1]))])
                coords_transformed = M_inv[:2, :] @ coords_homogeneous
                
                # Reshape for remap
                map_x = coords_transformed[0, :].reshape(h_src, w_src).astype(np.float32)
                map_y = coords_transformed[1, :].reshape(h_src, w_src).astype(np.float32)
                
                # Warp source image using inverse mapping
                source_overlap = cv2.remap(source_overlap, map_x, map_y, cv2.INTER_LINEAR, 
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                
                logging.info(f"  Applied cumulative transform to source before matching (scale {scale})")
            
            # Preprocess for matching (use cropped overlap regions, source may be transformed)
            source_prep, target_prep = self.preprocess_images(source_overlap, target_overlap, scale)

            # Estimate expected remaining error for adaptive search
            expected_error_m = None
            if i > 0 and 'mean_offset_m' in self.matches_info:
                expected_error_m = self.matches_info['mean_offset_m']

            # Register (pass overlap regions for visualization)
            # CRITICAL: Pass crop offsets so matching can account for geographic offset
            if self.config.method == 'phase_correlation':
                M = self.register_phase_correlation(source_prep, target_prep, scale)
            elif self.config.method == 'patch_ncc':
                M = self.register_patch_ncc(
                    source_prep, target_prep, scale,
                    source_overlap, target_overlap,
                    expected_error_m=expected_error_m,
                    crop_offset_source=crop_offset_source,
                    crop_offset_target=crop_offset_target,
                    overlap_info=overlap_info
                )
            else:
                logging.error(f"Method {self.config.method} not implemented in this version")
                return None

            # IMPORTANT: The transform M is in overlap region coordinates at current scale
            # We need to adjust it to full image coordinates by adding the crop offsets
            # BUT: Only do this if we actually found matches! If M is identity (no matches),
            # we should skip this scale or use a fallback method.
            
            if M is not None:
                # Check if this is just an identity matrix (no matches found)
                is_identity = (np.allclose(M[:2, :2], np.eye(2)) and 
                              np.allclose(M[:2, 2], [0, 0], atol=1e-3))
                
                if is_identity:
                    logging.warning(f"  No matches found at scale {scale} - transform is identity")
                    logging.warning(f"  Skipping this scale to avoid incorrect crop offset accumulation")
                    logging.warning(f"  Consider: lowering ncc_threshold, using phase correlation, or checking image overlap")
                    
                    # Try phase correlation as fallback
                    logging.info(f"  Attempting phase correlation as fallback...")
                    try:
                        M_phase = self.register_phase_correlation(source_prep, target_prep, scale)
                        if M_phase is not None and not (np.allclose(M_phase[:2, :2], np.eye(2)) and 
                                                       np.allclose(M_phase[:2, 2], [0, 0], atol=1e-3)):
                            logging.info(f"  Phase correlation found a transform, using it")
                            M = M_phase
                            is_identity = False
                        else:
                            logging.warning(f"  Phase correlation also failed")
                            continue  # Skip this scale entirely
                    except Exception as e:
                        logging.warning(f"  Phase correlation failed: {e}")
                        continue  # Skip this scale entirely
                
                if not is_identity:
                    # The transform found is: target_overlap_coords = M @ source_overlap_coords
                    # To convert to full coords: target_full = M @ source_full + offset_adjustment
                    # offset_adjustment = crop_offset_target - M @ crop_offset_source

                    # For affine transforms, we need to apply the rotation/scale to the source offset
                    # then add the target offset
                    source_offset_transformed = M[:2, :2] @ crop_offset_source
                    offset_adjustment = crop_offset_target - source_offset_transformed
                    
                    M[0, 2] += offset_adjustment[0]
                    M[1, 2] += offset_adjustment[1]

                    logging.info(f"  Adjusted transform to full image coordinates:")
                    logging.info(f"    Source crop offset: {crop_offset_source}")
                    logging.info(f"    Target crop offset: {crop_offset_target}")
                    logging.info(f"    Offset adjustment: {offset_adjustment}")
                    logging.info(f"    [{M[0, 0]:.6f}  {M[0, 1]:.6f}  {M[0, 2]:.2f}]")
                    logging.info(f"    [{M[1, 0]:.6f}  {M[1, 1]:.6f}  {M[1, 2]:.2f}]")
            else:
                logging.warning(f"  Transform is None - skipping this scale")
                continue

            # Store statistics
            level_stats = self.matches_info.copy()
            level_stats['transform_type'] = self.config.transform_type
            self.registration_stats[f'level_{i + 1}_scale_{scale}'] = level_stats

            # Save intermediate transform
            # IMPORTANT: Save M_fullres (at full resolution) not M (at current scale)
            # This is the transform that will be accumulated
            if M is not None:
                transform_file = self.intermediate_dir / f'transform_level{i + 1}_scale{scale:.3f}.txt'
                try:
                    # Save M_fullres (full resolution) so it can be properly accumulated
                    np.savetxt(transform_file, M_fullres, fmt='%.6f')
                    logging.info(f"  Saved transform to {transform_file.name} (full resolution)")
                except Exception as e:
                    logging.error(f"  Failed to save transform file: {e}")

                logging.info(f"  Transform at this scale (full resolution):")
                logging.info(f"    {M_fullres[0]}")
                logging.info(f"    {M_fullres[1]}")
                logging.info(f"  Transform at this scale (current scale {scale}):")
                logging.info(f"    {M[0]}")
                logging.info(f"    {M[1]}")
            else:
                logging.warning(f"  Cannot save transform - M is None")

            # CRITICAL: M is in "overlap region coordinates at current scale"
            # After offset adjustment, M is in "full image coordinates at current scale"
            # 
            # To convert to full resolution:
            # - The rotation/scale part (M[:2, :2]) stays the same (it's a ratio, not pixel-based)
            # - The translation part (M[:2, 2]) is in pixels at current scale, so divide by scale
            M_fullres = M.copy()
            M_fullres[:, 2] /= scale  # Scale translation from scale-s pixels to full-res pixels

            logging.info(f"  Transform scaled to full resolution:")
            logging.info(f"    {M_fullres[0]}")
            logging.info(f"    {M_fullres[1]}")

            # Accumulate transforms (both now in full resolution)
            # M_cumulative: transforms original full-res source -> full-res target (from previous scales)
            # M_fullres: transforms transformed full-res source -> full-res target (refinement from this scale)
            # Since we applied M_cumulative to source before matching, M_fullres refines the already-transformed source
            # We want: M_cumulative_new = M_fullres @ M_cumulative
            # This means: apply cumulative first (original -> transformed), then refinement (transformed -> target)
            # 
            # CRITICAL: M_fullres should be near-identity (diagonal ~1.0) since it's a refinement
            # If it's not near-identity, something is wrong with the matching or coordinate conversion
            if not np.allclose(M_fullres[:2, :2], np.eye(2), atol=0.1):
                logging.warning(f"  WARNING: Transform diagonal is {M_fullres[0, 0]:.3f}, {M_fullres[1, 1]:.3f} (expected ~1.0)")
                logging.warning(f"  This suggests the transform may be incorrectly scaled or computed")
            
            if self.config.transform_type == 'homography':
                # Homography: 3x3 matrix
                if M_fullres.shape == (3, 3):
                    M_full = M_fullres
                else:
                    M_full = np.vstack([M_fullres, [0, 0, 1]])
                M_cum_full = np.vstack([M_cumulative, [0, 0, 1]])
                # Apply this scale's transform after cumulative
                M_cumulative_new = M_full @ M_cum_full
                M_cumulative = M_cumulative_new[:2, :]
            else:
                # Affine: convert to 3x3, multiply, extract 2x3
                M_full = np.vstack([M_fullres, [0, 0, 1]])
                M_cum_full = np.vstack([M_cumulative, [0, 0, 1]])
                # Apply this scale's transform after cumulative
                M_cumulative_new = M_full @ M_cum_full
                M_cumulative = M_cumulative_new[:2, :]
            
            logging.info(f"  Cumulative transform after this scale:")
            logging.info(f"    {M_cumulative[0]}")
            logging.info(f"    {M_cumulative[1]}")

        self.transform_matrix = M_cumulative

        # NEW: Optional iterative refinement at finest scale
        if self.config.enable_refinement and len(scales) > 0:
            logging.info("\n" + "=" * 70)
            logging.info("Iterative Refinement")
            logging.info("=" * 70)

            finest_scale = scales[-1]

            for iteration in range(self.config.refinement_iterations):
                logging.info(f"\nRefinement iteration {iteration + 1}/{self.config.refinement_iterations}")

                # Load and pre-align at finest scale
                source_orig, target_orig = self.load_downsampled(finest_scale)
                M_scaled = self.transform_matrix.copy()
                M_scaled[:, 2] *= finest_scale  # Scale to current resolution

                source_aligned = self._apply_transform_to_image(
                    source_orig, M_scaled, source_orig.shape, enlarge_canvas=True
                )

                # Preprocess
                source_prep, target_prep = self.preprocess_images(
                    source_aligned, target_orig, finest_scale
                )

                # Match with small search region
                expected_error_m = self.matches_info.get('mean_offset_m', 5.0)

                if self.config.method == 'patch_ncc':
                    M_refine = self.register_patch_ncc(
                        source_prep, target_prep, finest_scale,
                        source_aligned, target_orig,
                        expected_error_m=expected_error_m
                    )
                else:
                    M_refine = self.register_phase_correlation(
                        source_prep, target_prep, finest_scale
                    )

                # Check if refinement is making progress
                if M_refine is not None:
                    refine_shift = np.sqrt(M_refine[0, 2] ** 2 + M_refine[1, 2] ** 2)
                    logging.info(f"  Refinement shift: {refine_shift:.2f} px at scale {finest_scale}")

                    if refine_shift < 1.0:  # Less than 1 pixel improvement
                        logging.info("  Refinement converged (< 1px change)")
                        break

                    # Scale refinement to full resolution and accumulate
                    M_refine_fullres = M_refine.copy()
                    M_refine_fullres[:, 2] /= finest_scale

                    M_refine_full = np.vstack([M_refine_fullres, [0, 0, 1]])
                    M_cum_full = np.vstack([self.transform_matrix, [0, 0, 1]])
                    self.transform_matrix = (M_cum_full @ M_refine_full)[:2, :]

        # Save final
        final_file = self.output_dir / 'transform_final.txt'
        np.savetxt(final_file, M_cumulative, fmt='%.6f')
        logging.info(f"\nFinal cumulative transform (in full-resolution pixels):")
        logging.info(f"  {M_cumulative[0]}")
        logging.info(f"  {M_cumulative[1]}")

        return M_cumulative

    def apply_transform(self) -> Path:
        """Apply transformation with proper compression."""
        import time
        start_time = time.time()
        
        if self.transform_matrix is None:
            raise ValueError("No transform computed")

        output_path = self.output_dir / 'orthomosaic_registered.tif'

        logging.info("Applying transformation...")
        logging.info(f"Using geotransform update method")

        with rasterio.open(self.config.source_path) as src:
            src_transform = src.transform
            M = self.transform_matrix

            # Update geotransform
            new_transform = Affine(
                src_transform.a,
                src_transform.b,
                src_transform.c + M[0, 2] * src_transform.a,
                src_transform.d,
                src_transform.e,
                src_transform.f + M[1, 2] * src_transform.e
            )

            # Profile with compression
            profile = src.profile.copy()
            
            # JPEG compression only works with uint8 data
            # If source is not uint8, we need to either convert or use different compression
            compression = self.config.output_format['compression']
            if compression == 'JPEG' and profile['dtype'] != 'uint8':
                logging.warning(f"Source dtype is {profile['dtype']}, JPEG compression requires uint8")
                logging.warning("Converting to uint8 and using JPEG compression")
                # Convert to uint8 by scaling
                profile['dtype'] = 'uint8'
            
            profile.update(
                transform=new_transform,
                compress=compression,
                tiled=self.config.output_format['tiled'],
                blockxsize=self.config.output_format['blocksize'],
                blockysize=self.config.output_format['blocksize']
            )
            
            # JPEG quality only applies to JPEG compression
            if compression == 'JPEG':
                profile['jpeg_quality'] = self.config.output_format['jpeg_quality']
            
            logging.info(f"Output format: {profile['compress']} dtype={profile['dtype']} quality={profile.get('jpeg_quality', 'N/A')}")

            # Copy data with proper dtype conversion if needed
            # Use windowed reading for large images to be memory efficient
            src_dtype = src.dtypes[0]  # Get dtype from first band
            dtype_itemsize = np.dtype(src_dtype).itemsize
            image_size_mb = (src.width * src.height * src.count * dtype_itemsize) / (1024 ** 2)
            use_windowed = image_size_mb > 500  # Use windowed reading for images > 500MB
            
            if use_windowed:
                logging.info(f"Large image detected ({image_size_mb:.1f} MB), using windowed reading")
                block_size = self.config.output_format['blocksize']
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    for band_idx in range(1, src.count + 1):
                        for ji, window in dst.block_windows(band_idx):
                            data = src.read(band_idx, window=window)
                            
                            # Convert to uint8 if needed for JPEG
                            if compression == 'JPEG' and data.dtype != np.uint8:
                                # Normalize to 0-255 (compute min/max per block for efficiency)
                                data_min = data.min()
                                data_max = data.max()
                                if data_max > data_min:
                                    data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                                else:
                                    data = np.zeros_like(data, dtype=np.uint8)
                            
                            dst.write(data, band_idx, window=window)
                    if src.colorinterp:
                        dst.colorinterp = src.colorinterp
            else:
                # For smaller images, read all at once (faster)
                with rasterio.open(output_path, 'w', **profile) as dst:
                    for band_idx in range(1, src.count + 1):
                        data = src.read(band_idx)
                        
                        # Convert to uint8 if needed for JPEG
                        if compression == 'JPEG' and data.dtype != np.uint8:
                            # Normalize to 0-255
                            data_min = data.min()
                            data_max = data.max()
                            if data_max > data_min:
                                data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                            else:
                                data = np.zeros_like(data, dtype=np.uint8)
                        
                        dst.write(data, band_idx)
                    if src.colorinterp:
                        dst.colorinterp = src.colorinterp

        file_size_mb = output_path.stat().st_size / (1024 ** 2)
        elapsed = time.time() - start_time
        logging.info(f"Saved: {output_path.name} ({file_size_mb:.1f} MB) in {elapsed:.1f}s")

        return output_path

    def create_png_overviews(self):
        """Create PNG overviews."""
        logging.info("Creating PNG overviews...")

        scale = 0.1
        source_orig, target_orig = self.load_downsampled(scale)

        M_scaled = self.transform_matrix.copy()
        M_scaled[:, 2] *= scale

        source_aligned = self._apply_transform_to_image(
            source_orig, M_scaled, target_orig.shape
        )

        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        axes[0, 0].imshow(source_orig, cmap='gray')
        axes[0, 0].set_title('Original Orthomosaic', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(target_orig, cmap='gray')
        axes[0, 1].set_title('Target Basemap', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(source_aligned, cmap='gray')
        axes[1, 0].set_title('Registered Orthomosaic', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        if source_aligned.shape == target_orig.shape:
            diff = np.abs(target_orig.astype(float) - source_aligned.astype(float))
            im = axes[1, 1].imshow(diff, cmap='hot')
            axes[1, 1].set_title('Absolute Difference', fontsize=14, fontweight='bold')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        plt.tight_layout()

        overview_file = self.viz_dir / 'registration_overview.png'
        plt.savefig(overview_file, dpi=150, bbox_inches='tight')
        plt.close()

        logging.info(f"Saved overview to {overview_file.name}")

    def create_difference_map(self):
        """Create difference map."""
        logging.info("Creating difference map...")

        scale = 0.2
        source_orig, target_orig = self.load_downsampled(scale)

        M_scaled = self.transform_matrix.copy()
        M_scaled[:, 2] *= scale

        source_aligned = self._apply_transform_to_image(
            source_orig, M_scaled, target_orig.shape
        )

        if source_aligned.shape == target_orig.shape:
            diff = target_orig.astype(float) - source_aligned.astype(float)

            fig, axes = plt.subplots(1, 3, figsize=(20, 7))

            axes[0].imshow(source_aligned, cmap='gray')
            axes[0].set_title('Registered', fontsize=12)
            axes[0].axis('off')

            axes[1].imshow(target_orig, cmap='gray')
            axes[1].set_title('Target', fontsize=12)
            axes[1].axis('off')

            im = axes[2].imshow(diff, cmap='RdBu_r', vmin=-50, vmax=50)
            axes[2].set_title('Difference', fontsize=12)
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2])

            plt.tight_layout()

            diff_file = self.viz_dir / 'difference_map.png'
            plt.savefig(diff_file, dpi=150, bbox_inches='tight')
            plt.close()

            logging.info(f"Saved difference map to {diff_file.name}")

    def generate_report(self):
        """Generate final report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_files': {
                'source': str(self.config.source_path),
                'target': str(self.config.target_path)
            },
            'configuration': {
                'method': self.config.method,
                'transform_type': self.config.transform_type,
                'hierarchical_scales': self.config.hierarchical_scales,
                'preprocess_method': self.config.preprocess_method,
                'ransac_threshold': self.config.ransac_threshold,
                'ncc_threshold': self.config.ncc_threshold,
                'enable_refinement': self.config.enable_refinement,
                'adaptive_search': self.config.adaptive_search
            },
            'source_metadata': {
                'resolution_m_per_px': self.source_res,
                'shape': self.source_shape,
                'crs': str(self.source_crs)
            },
            'target_metadata': {
                'resolution_m_per_px': self.target_res,
                'shape': self.target_shape,
                'crs': str(self.target_crs)
            },
            'transform_matrix': self.transform_matrix.tolist() if self.transform_matrix is not None else None,
            'registration_statistics': self.registration_stats
        }

        report_file = self.output_dir / 'registration_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Text report
        text_file = self.output_dir / 'registration_report.txt'
        with open(text_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ORTHOMOSAIC REGISTRATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write(f"Method: {self.config.method}\n")
            f.write(f"Transform type: {self.config.transform_type}\n\n")

            f.write("Resolutions:\n")
            f.write(f"  Source: {self.source_res:.4f} m/pixel\n")
            f.write(f"  Target: {self.target_res:.4f} m/pixel\n\n")

            f.write("Transform Matrix:\n")
            if self.transform_matrix is not None:
                for row in self.transform_matrix:
                    f.write(f"  [{row[0]:8.6f} {row[1]:8.6f} {row[2]:10.2f}]\n")
            f.write("\n")

            f.write("Statistics by Level:\n")
            for level, stats in self.registration_stats.items():
                f.write(f"\n{level}:\n")
                for key, value in stats.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")

        logging.info(f"Reports saved to {report_file.name} and {text_file.name}")
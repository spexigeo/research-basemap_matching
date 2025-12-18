"""
Core registration library for orthomosaic alignment.
Contains all registration algorithms and utilities.
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

    def load_downsampled(self, scale_factor: float) -> Tuple[np.ndarray, np.ndarray]:
        """Load downsampled versions - returns ORIGINAL grayscale, not preprocessed."""

        def load_and_downsample(path, scale):
            with rasterio.open(path) as src:
                out_height = max(int(src.height * scale), 100)
                out_width = max(int(src.width * scale), 100)

                logging.info(f"  Loading {Path(path).name} -> {out_width}x{out_height}")

                data = src.read(
                    out_shape=(src.count, out_height, out_width),
                    resampling=Resampling.bilinear
                )

                # Convert to 8-bit grayscale with proper normalization
                if data.shape[0] >= 3:
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

        source_img = load_and_downsample(self.config.source_path, scale_factor)
        target_img = load_and_downsample(self.config.target_path, scale_factor)

        # Save ORIGINAL (non-preprocessed) images
        cv2.imwrite(str(self.intermediate_dir / f'source_original_scale{scale_factor:.3f}.png'), source_img)
        cv2.imwrite(str(self.intermediate_dir / f'target_original_scale{scale_factor:.3f}.png'), target_img)
        logging.info(f"  Saved original images to intermediate/")

        return source_img, target_img

    def preprocess_images(self, source: np.ndarray, target: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess images and save both versions."""
        method = self.config.preprocess_method

        if method == 'none':
            return source, target

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

        # Save preprocessed images
        cv2.imwrite(str(self.intermediate_dir / f'source_preprocessed_{method}_scale{scale:.3f}.png'), source_prep)
        cv2.imwrite(str(self.intermediate_dir / f'target_preprocessed_{method}_scale{scale:.3f}.png'), target_prep)
        logging.info(f"  Saved preprocessed images to intermediate/")

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
                           source_original: np.ndarray, target_original: np.ndarray) -> np.ndarray:
        """Patch-based NCC matching with detailed logging."""

        min_dim = min(source.shape[0], source.shape[1], target.shape[0], target.shape[1])
        patch_size = max(min(self.config.patch_size, min_dim // 4), 32)
        grid_spacing = max(min(self.config.patch_grid_spacing, patch_size), 16)

        h, w = target.shape
        matches = []
        all_attempts = []

        logging.info(f"  Patch matching: {patch_size}px patches, {grid_spacing}px spacing")
        logging.info(f"  Images: source={source.shape}, target={target.shape}")

        tested = 0
        for y in range(patch_size // 2, h - patch_size // 2, grid_spacing):
            for x in range(patch_size // 2, w - patch_size // 2, grid_spacing):
                tested += 1
                ty1, ty2 = y - patch_size // 2, y + patch_size // 2
                tx1, tx2 = x - patch_size // 2, x + patch_size // 2
                target_patch = target[ty1:ty2, tx1:tx2]

                if ty2 <= source.shape[0] and tx2 <= source.shape[1]:
                    search_margin = min(150, ty1, tx1, source.shape[0] - ty2, source.shape[1] - tx2)
                    sy1 = ty1 - search_margin
                    sy2 = ty2 + search_margin
                    sx1 = tx1 - search_margin
                    sx2 = tx2 + search_margin
                    search_region = source[sy1:sy2, sx1:sx2]

                    if search_region.shape[0] > patch_size and search_region.shape[1] > patch_size:
                        try:
                            result = cv2.matchTemplate(search_region, target_patch, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, max_loc = cv2.minMaxLoc(result)

                            src_x = sx1 + max_loc[0] + patch_size // 2
                            src_y = sy1 + max_loc[1] + patch_size // 2

                            offset_x = src_x - x
                            offset_y = src_y - y
                            offset_m = np.sqrt(offset_x ** 2 + offset_y ** 2) * self.target_res / scale

                            all_attempts.append({
                                'target': (x, y),
                                'source': (src_x, src_y),
                                'confidence': float(max_val),
                                'offset_px': (offset_x, offset_y),
                                'offset_m': float(offset_m)
                            })

                            if max_val > self.config.ncc_threshold:
                                matches.append(([src_x, src_y], [x, y], max_val))

                        except cv2.error:
                            continue

        logging.info(f"  Tested {tested} patches, found {len(matches)} matches (threshold={self.config.ncc_threshold})")

        # Save detailed match log
        match_log_file = self.intermediate_dir / f'patch_attempts_scale{scale:.3f}.json'
        with open(match_log_file, 'w') as f:
            json.dump({
                'scale': scale,
                'threshold': self.config.ncc_threshold,
                'tested': tested,
                'matches_found': len(matches),
                'all_attempts': all_attempts[:100]  # Save first 100 for inspection
            }, f, indent=2)
        logging.info(f"  Saved patch attempts to {match_log_file.name}")

        if len(matches) < 10:
            logging.warning(f"  INSUFFICIENT MATCHES: {len(matches)} < 10")
            if len(all_attempts) > 0:
                best_attempts = sorted(all_attempts, key=lambda x: x['confidence'], reverse=True)[:5]
                best_conf_str = [f"{a['confidence']:.3f}" for a in best_attempts]
                logging.warning(f"  Best confidences found: {best_conf_str}")
                logging.warning(f"  Try lowering ncc_threshold to {max(0.3, self.config.ncc_threshold - 0.1):.2f}")

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

        # RANSAC
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_threshold,
            maxIters=2000,
            confidence=0.99
        )

        if M is None:
            logging.warning("  RANSAC failed")
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
        """Hierarchical coarse-to-fine registration."""
        M_cumulative = np.eye(2, 3, dtype=np.float32)
        scales = self.config.hierarchical_scales

        for i, scale in enumerate(scales):
            logging.info("\n" + "=" * 70)
            logging.info(f"Level {i + 1}/{len(scales)}: Scale {scale}")
            logging.info("=" * 70)

            # Load ORIGINAL images
            source_orig, target_orig = self.load_downsampled(scale)

            # Preprocess for matching
            source_prep, target_prep = self.preprocess_images(source_orig, target_orig, scale)

            # Register (pass originals for visualization)
            if self.config.method == 'phase_correlation':
                M = self.register_phase_correlation(source_prep, target_prep, scale)
            elif self.config.method == 'patch_ncc':
                M = self.register_patch_ncc(source_prep, target_prep, scale,
                                            source_orig, target_orig)
            else:
                logging.error(f"Method {self.config.method} not implemented in this version")
                return None

            # Store statistics
            self.registration_stats[f'level_{i + 1}_scale_{scale}'] = self.matches_info.copy()

            # Save intermediate transform
            transform_file = self.intermediate_dir / f'transform_level{i + 1}_scale{scale:.3f}.txt'
            np.savetxt(transform_file, M, fmt='%.6f')

            # Scale for next level
            if i < len(scales) - 1:
                scale_ratio = scales[i + 1] / scale
                M[:, 2] *= scale_ratio

            # Accumulate
            M_full = np.vstack([M, [0, 0, 1]])
            M_cum_full = np.vstack([M_cumulative, [0, 0, 1]])
            M_cumulative = (M_cum_full @ M_full)[:2, :]

        self.transform_matrix = M_cumulative

        # Save final
        final_file = self.output_dir / 'transform_final.txt'
        np.savetxt(final_file, M_cumulative, fmt='%.6f')
        logging.info(f"\nFinal transform:\n{M_cumulative}")

        return M_cumulative

    def apply_transform(self) -> Path:
        """Apply transformation with proper compression."""
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
            profile.update(
                transform=new_transform,
                compress=self.config.output_format['compression'],
                jpeg_quality=self.config.output_format['jpeg_quality'],
                tiled=self.config.output_format['tiled'],
                blockxsize=self.config.output_format['blocksize'],
                blockysize=self.config.output_format['blocksize']
            )

            logging.info(f"Output format: {profile['compress']} quality={profile.get('jpeg_quality', 'N/A')}")

            # Copy data
            with rasterio.open(output_path, 'w', **profile) as dst:
                for band_idx in range(1, src.count + 1):
                    data = src.read(band_idx)
                    dst.write(data, band_idx)
                if src.colorinterp:
                    dst.colorinterp = src.colorinterp

        file_size_mb = output_path.stat().st_size / (1024 ** 2)
        logging.info(f"Saved: {output_path.name} ({file_size_mb:.1f} MB)")

        return output_path

    def create_png_overviews(self):
        """Create PNG overviews."""
        logging.info("Creating PNG overviews...")

        scale = 0.1
        source_orig, target_orig = self.load_downsampled(scale)

        M_scaled = self.transform_matrix.copy()
        M_scaled[:, 2] *= scale

        source_aligned = cv2.warpAffine(
            source_orig, M_scaled,
            (target_orig.shape[1], target_orig.shape[0]),
            flags=cv2.INTER_LINEAR
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

        source_aligned = cv2.warpAffine(
            source_orig, M_scaled,
            (target_orig.shape[1], target_orig.shape[0]),
            flags=cv2.INTER_LINEAR
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
                'hierarchical_scales': self.config.hierarchical_scales,
                'preprocess_method': self.config.preprocess_method,
                'ransac_threshold': self.config.ransac_threshold,
                'ncc_threshold': self.config.ncc_threshold
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
            f.write(f"Method: {self.config.method}\n\n")

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
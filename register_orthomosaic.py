#!/usr/bin/env python3
"""
Main registration pipeline for orthomosaic alignment.
Hierarchical registration using resolution pyramid with cumulative transformations.

Default scales: [0.125, 0.25, 0.5, 1.0]
Default matcher: LightGlue
Default transforms: shift (0.125, 0.25), homography (0.5, 1.0)
"""

import numpy as np
import cv2
import rasterio
from rasterio.warp import Resampling
from rasterio.windows import Window
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import time
import logging
import argparse

from defaults import DEFAULT_SCALES, DEFAULT_ALGORITHMS, DEFAULT_MATCHER, DEFAULT_DEBUG_LEVEL, DEFAULT_OUTPUT_DIR
from preprocessing import ImagePreprocessor
from matching import match_lightglue, match_sift, match_orb, match_patch_ncc, create_mask, LIGHTGLUE_AVAILABLE, visualize_matches
from transformations import (
    load_matches, remove_gross_outliers, compute_2d_shift, compute_similarity_transform,
    compute_affine_transform, compute_homography, compute_polynomial_transform,
    compute_spline_transform, compute_rubber_sheeting_transform, choose_best_transform,
    apply_transform_to_image, apply_polynomial_transform_to_image,
    apply_spline_transform_to_image, apply_rubber_sheeting_transform_to_image
)

# Set up logging to both console and file
log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG for verbose logging
    format=log_format,
    handlers=[
        logging.StreamHandler(),  # Console output
    ]
)

# Add file handler for verbose logging
def setup_file_logging(output_dir: Path, debug_level: str = 'none'):
    """Set up file logging based on debug level.
    
    Args:
        output_dir: Output directory
        debug_level: 'none', 'intermediate', or 'high'
            - 'none': registration_verbose_low.log with INFO level
            - 'intermediate': registration_verbose_intermediate.log with INFO level
            - 'high': registration_verbose_high.log with DEBUG level
    """
    if debug_level == 'high':
        log_file = output_dir / 'registration_verbose_high.log'
        log_level = logging.DEBUG
    elif debug_level == 'intermediate':
        log_file = output_dir / 'registration_verbose_intermediate.log'
        log_level = logging.INFO
    else:
        log_file = output_dir / 'registration_verbose_low.log'
        log_level = logging.INFO
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)
    return log_file

LOG_PATH = Path("/Users/mauriciohessflores/Documents/Code/MyCode/.cursor/debug.log")
SESSION_ID = "debug-session"
RUN_ID_DEFAULT = "run1"


def _dbg_log(hypothesis_id: str, location: str, message: str, data: Dict, run_id: str = RUN_ID_DEFAULT):
    """Append a single NDJSON debug log line."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sessionId": SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000)
        }
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


class OrthomosaicRegistration:
    """Main registration class for hierarchical orthomosaic alignment."""
    
    def __init__(self, source_path: str, target_path: str, output_dir: str,
                 scales: Optional[List[float]] = None,
                 matcher: str = 'lightglue',
                 transform_types: Optional[Dict[float, str]] = None,
                 debug_level: str = 'none',
                 gcp_evaluation_path: Optional[str] = None):
        """
        Initialize registration.
        
        Args:
            source_path: Path to source orthomosaic
            target_path: Path to target basemap
            output_dir: Output directory
            scales: List of scales (default: [0.125, 0.25, 0.5, 1.0])
            matcher: Matching method ('lightglue', 'sift', 'orb', 'patch_ncc')
            transform_types: Dict mapping scale to transform type (default: shift for 0.125/0.25, homography for 0.5/1.0)
            debug_level: Debug level ('none', 'intermediate', 'high')
                - 'none': Only log file and final orthomosaic
                - 'intermediate': 'none' + intermediate/ directory files
                - 'high': 'intermediate' + matching_and_transformations/ directory files + verbose log
            gcp_evaluation_path: Optional path to GCP file (CSV or KMZ) for evaluation at second-to-last scale
        """
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.output_dir = Path(output_dir)
        self.debug_level = debug_level
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file logging based on debug level
        self.log_file = setup_file_logging(self.output_dir, debug_level)
        if debug_level == 'high':
            logging.info(f"Verbose logging to: {self.log_file}")
        else:
            logging.info(f"Logging to: {self.log_file}")
        
        # Create subdirectories conditionally based on debug level
        self.preprocessing_dir = None
        self.matching_dir = None
        self.intermediate_dir = None
        self._temp_intermediate_dir = None
        
        # Always need intermediate_dir for processing, but use temp location at 'none' level
        if debug_level in ['intermediate', 'high']:
            self.intermediate_dir = self.output_dir / 'intermediate'
            self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        else:
            # At 'none' level, use temp directory (will be cleaned up)
            import tempfile
            self._temp_intermediate_dir = Path(tempfile.mkdtemp(prefix='ortho_reg_'))
            self.intermediate_dir = self._temp_intermediate_dir
        
        # Only 'high' level creates matching_and_transformations/ directory
        if debug_level == 'high':
            self.preprocessing_dir = self.output_dir / 'preprocessing'
            self.matching_dir = self.output_dir / 'matching_and_transformations'
            self.preprocessing_dir.mkdir(parents=True, exist_ok=True)
            self.matching_dir.mkdir(parents=True, exist_ok=True)
        
        # Use defaults from constants module
        self.scales = scales if scales is not None else DEFAULT_SCALES.copy()
        
        # Default transform types by scale
        if transform_types is None:
            # Build from default scales and algorithms
            self.transform_types = {scale: algo for scale, algo in zip(DEFAULT_SCALES, DEFAULT_ALGORITHMS)}
            # If custom scales provided, use default algorithm for new scales (shift)
            for scale in self.scales:
                if scale not in self.transform_types:
                    self.transform_types[scale] = 'shift'
        else:
            self.transform_types = transform_types
        
        self.matcher = matcher.lower() if matcher else DEFAULT_MATCHER
        self.gcp_evaluation_path = gcp_evaluation_path
        
        # Initialize preprocessor (will lazy-load metadata when needed)
        self.preprocessor = ImagePreprocessor(source_path, target_path, self.output_dir)
        
        # Lazy-load source CRS/transform (will be loaded when needed)
        self._source_crs = None
        self._source_transform = None
        
        logging.info(f"Initialized registration:")
        logging.info(f"  Scales: {self.scales}")
        logging.info(f"  Matcher: {self.matcher}")
        logging.info(f"  Transform types: {self.transform_types}")
    
    @property
    def source_crs(self):
        """Lazy-load source CRS when needed."""
        if self._source_crs is None:
            with rasterio.open(self.source_path) as src:
                self._source_crs = src.crs
                self._source_transform = src.transform
        return self._source_crs
    
    @property
    def source_transform(self):
        """Lazy-load source transform when needed."""
        if self._source_transform is None:
            with rasterio.open(self.source_path) as src:
                self._source_crs = src.crs
                self._source_transform = src.transform
        return self._source_transform
    
    def create_resolution_pyramid(self):
        """Create resolution pyramid versions of source and target."""
        logging.info("\n" + "=" * 80)
        logging.info("STEP 1: Creating Resolution Pyramid")
        logging.info("=" * 80)
        
        # Load and log metadata now (when actually needed)
        self.preprocessor.log_metadata()
        
        for scale in self.scales:
            logging.info(f"\nProcessing scale {scale:.3f}...")
            
            # Load downsampled images
            source_img, target_img = self.preprocessor.load_downsampled(scale)
            
            # Compute overlap region
            overlap_info = self.preprocessor.compute_overlap_region(scale)
            if overlap_info is None:
                logging.error(f"No overlap at scale {scale}, skipping")
                continue
            
            # Crop to overlap
            source_overlap, target_overlap = self.preprocessor.crop_to_overlap(
                source_img, target_img, overlap_info
            )
            
            # Always save overlap images (needed for matching), but use temp location if not 'high' level
            if self.debug_level == 'high' and self.preprocessing_dir:
                overlap_dir = self.preprocessing_dir
            else:
                # Use intermediate_dir (which exists as temp at 'none' level)
                overlap_dir = self.intermediate_dir
            
            source_overlap_path = overlap_dir / f'source_overlap_scale{scale:.3f}.png'
            target_overlap_path = overlap_dir / f'target_overlap_scale{scale:.3f}.png'
            
            if not source_overlap_path.exists():
                cv2.imwrite(str(source_overlap_path), source_overlap)
            if not target_overlap_path.exists():
                cv2.imwrite(str(target_overlap_path), target_overlap)
            
            if self.debug_level == 'high':
                logging.info(f"  Saved overlap images for scale {scale:.3f}")
    
    def register(self) -> Optional[Path]:
        """
        Run hierarchical registration pipeline following explicit steps (a)-(g).
        """
        logging.info("\n" + "=" * 80)
        logging.info("HIERARCHICAL ORTHOMOSAIC REGISTRATION")
        logging.info("=" * 80)
        
        # Step 1: Create resolution pyramid (only loads downsampled versions, not full res)
        self.create_resolution_pyramid()
        
        # Base orthos - create lazily when needed (don't create scale 1.0 until actually needed)
        base_orthos = {}
        
        def get_base_ortho(scale: float) -> Path:
            """Lazy-load base ortho for a scale."""
            if scale not in base_orthos:
                logging.debug(f"  Creating base ortho for scale {scale:.3f}...")
                base_orthos[scale] = self._get_or_create_downsampled_orthomosaic(scale)
            return base_orthos[scale]

        def _safe_read_gray(img_path: Path) -> Optional[np.ndarray]:
            """Read grayscale image; fall back to rasterio/PIL if OpenCV hits size limits."""
            # Check file size first - if very large, skip OpenCV entirely
            skip_opencv = False
            try:
                file_size_mb = img_path.stat().st_size / (1024 * 1024)
                # If file is > 500MB, likely too large for OpenCV, use rasterio/PIL directly
                if file_size_mb > 500:
                    logging.debug(f"    Large image ({file_size_mb:.1f}MB), skipping OpenCV for {img_path.name}")
                    skip_opencv = True
            except OSError:
                pass  # File size check failed, try OpenCV anyway
            
            # Try OpenCV first (fast for small images), but skip if file is too large
            if not skip_opencv:
                try:
                    arr = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if arr is not None and arr.size > 0:
                        return arr
                except Exception as e:
                    logging.debug(f"    OpenCV read failed for {img_path.name}: {e}")
            
            # Fallback to rasterio (handles large images better)
            try:
                import warnings
                with warnings.catch_warnings():
                    # Suppress georeferencing warnings for PNG files
                    warnings.filterwarnings('ignore', message='.*geotransform.*')
                    warnings.filterwarnings('ignore', message='.*georeferenced.*')
                    with rasterio.open(img_path) as src:
                        data = src.read(1)
                        dmin, dmax = data.min(), data.max()
                        if dmax > dmin:
                            data = ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
                        else:
                            data = np.zeros_like(data, dtype=np.uint8)
                        return data
            except Exception as e:
                # Final fallback: try PIL
                try:
                    from PIL import Image
                    img = Image.open(img_path).convert('L')
                    return np.array(img, dtype=np.uint8)
                except Exception as e2:
                    logging.error(f"    All read methods failed for {img_path.name}: {e}, {e2}")
                    return None
        
        # Helper to run matching, save JSON/viz/hist, return transform_result
        def run_matching(source_img_path: Path, target_img_path: Path, scale: float,
                         transform_type: str, transform_json_name: str) -> Tuple[Optional[Dict], Optional[Path]]:
            if not source_img_path.exists() or not target_img_path.exists():
                logging.error(f"Missing overlap images for scale {scale:.3f}")
                return None, None
            source_img = _safe_read_gray(source_img_path)
            target_img = _safe_read_gray(target_img_path)
            if source_img is None or target_img is None:
                logging.error(f"Failed to read overlap images for scale {scale:.3f}")
                return None, None
            source_mask = create_mask(source_img)
            target_mask = create_mask(target_img)
            logging.info(f"\nComputing matches using {self.matcher} at scale {scale:.3f}...")
            matches_result = self._compute_matches(source_img, target_img, source_mask, target_mask, scale)
            if not matches_result or 'matches' not in matches_result or len(matches_result['matches']) == 0:
                logging.error(f"No matches found at scale {scale:.3f}")
                return None, None
            # Extract match points for transformation computation
            # Always save matches JSON temporarily (needed for load_matches), but only keep if high debug
            import tempfile
            temp_dir = Path(tempfile.gettempdir())
            temp_matches_json = temp_dir / f'matches_scale{scale:.3f}_temp_{int(time.time())}.json'
            self._save_matches_json(matches_result, temp_matches_json, scale,
                                   source_image=str(source_img_path),
                                   target_image=str(target_img_path))
            src_pts, dst_pts = load_matches(temp_matches_json)
            
            # Save matches files only at 'high' debug level
            if self.debug_level == 'high' and self.matching_dir:
                matches_json = self.matching_dir / f'matches_scale{scale:.3f}.json'
                import shutil
                shutil.copy(temp_matches_json, matches_json)
                matches_viz_path = self.matching_dir / f'matches_scale{scale:.3f}.png'
                matches_result['scale'] = scale
                visualize_matches(
                    source_img, target_img, matches_result, matches_viz_path,
                    source_name=source_img_path.name, target_name=target_img_path.name,
                    skip_json=True  # We write JSON separately with summary statistics
                )
            else:
                matches_json = temp_matches_json  # Use temp for histogram if needed
            
            src_pts_clean, dst_pts_clean, _ = remove_gross_outliers(src_pts, dst_pts)
            transform_result = self._compute_transformation(src_pts_clean, dst_pts_clean, transform_type, scale)
            
            # Save transform JSON and histogram only at 'high' debug level
            if self.debug_level == 'high' and self.matching_dir:
                transform_json = self.matching_dir / transform_json_name
                self._save_transform_json(transform_result, transform_json, scale)
                inlier_mask_for_hist = transform_result.get('inliers')
                if inlier_mask_for_hist is not None:
                    inlier_mask_for_hist = np.array(inlier_mask_for_hist, dtype=bool)
                    if len(inlier_mask_for_hist) != len(src_pts_clean):
                        inlier_mask_for_hist = None
                if inlier_mask_for_hist is None:
                    inlier_mask_for_hist = np.ones(len(src_pts_clean), dtype=bool)
                self._create_error_histogram(
                    transform_result, scale,
                    src_pts=src_pts_clean, dst_pts=dst_pts_clean,
                    inlier_mask=inlier_mask_for_hist,
                    json_name=transform_json.name,
                    matches_path=matches_json
                )
            else:
                transform_json = None
            
            # Clean up temp file if not using it
            if temp_matches_json != matches_json and temp_matches_json.exists():
                temp_matches_json.unlink()
            return transform_result, transform_json
        
        # Helper to apply transform and extract overlap png with given basename
        def apply_and_overlap(input_ortho: Path, transform_result: Dict,
                              source_scale: float, target_scale: float,
                              output_basename: str) -> Tuple[Optional[Path], Optional[Path]]:
            # Always save transformed orthomosaic (needed for processing pipeline)
            # At 'none' level, it goes to temp directory which gets cleaned up
            transformed_path = self._apply_transform_to_orthomosaic(
                input_ortho, transform_result, source_scale, target_scale
            )
            if transformed_path is None:
                return None, None
            
            # Save overlap PNG to intermediate_dir (always exists, either real or temp)
            overlap_png = self._extract_overlap_from_orthomosaic(
                transformed_path, target_scale, self.intermediate_dir,
                output_name=output_basename
            )
            return transformed_path, overlap_png
        
        # a) scale 0.125 shift
        scale_a = 0.125
        t_type_a = 'shift'
        # Use intermediate_dir for overlap images (works for all debug levels)
        overlap_dir = self.preprocessing_dir if (self.debug_level == 'high' and self.preprocessing_dir) else self.intermediate_dir
        source_overlap_a = overlap_dir / f'source_overlap_scale{scale_a:.3f}.png'
        target_overlap_a = overlap_dir / f'target_overlap_scale{scale_a:.3f}.png'
        transform_a, _ = run_matching(
            source_overlap_a, target_overlap_a, scale_a, t_type_a,
            transform_json_name=f'transform_scale{scale_a:.3f}.json'
        )
        if transform_a is None:
            return None
        ortho_a_base = get_base_ortho(scale_a)
        ortho_a_shift, overlap_a_shift_png = apply_and_overlap(
            ortho_a_base, transform_a, scale_a, scale_a,
            output_basename=f'orthomosaic_scale{scale_a:.3f}_shift.png'
        )
        
        # b) apply transform_scale0.125 to scale 0.25 -> shift0.125
        scale_b = 0.25
        ortho_b_base = get_base_ortho(scale_b)
        ortho_b_shift0125, overlap_b_shift0125_png = apply_and_overlap(
            ortho_b_base, transform_a, scale_a, scale_b,
            output_basename=f'orthomosaic_scale{scale_b:.3f}_shift0.125.png'
        )
        if ortho_b_shift0125 is None:
            return None
        
        # c) matches at 0.25 using shifted source -> use transform type from config
        transform_type_b = self.transform_types.get(scale_b, 'shift')
        transform_b, _ = run_matching(
            overlap_b_shift0125_png, overlap_dir / f'target_overlap_scale{scale_b:.3f}.png',
            scale_b, transform_type_b, transform_json_name=f'transform_scale{scale_b:.3f}.json'
        )
        if transform_b is None:
            return None
        ortho_b_shift0250, overlap_b_shift0250_png = apply_and_overlap(
            ortho_b_shift0125, transform_b, scale_b, scale_b,
            output_basename=f'orthomosaic_scale{scale_b:.3f}_shift0.250.png'
        )
        
        # Run GCP evaluation at second-to-last scale if requested
        if self.gcp_evaluation_path and len(self.scales) >= 2:
            second_to_last_scale = self.scales[-2]  # Second-to-last scale
            if scale_b == second_to_last_scale:
                logging.info(f"\n{'='*80}")
                logging.info(f"Running GCP Evaluation at scale {second_to_last_scale:.3f} (second-to-last)")
                logging.info(f"{'='*80}")
                try:
                    from gcp_analysis import analyze_gcps
                    # Use the transformed orthomosaic at this scale for evaluation
                    # Create patches like gcp_analysis does
                    gcp_evaluation_output_dir = self.output_dir / 'gcp_analysis'
                    analyze_gcps(
                        registered_orthomosaic_path=str(ortho_b_shift0250),
                        gcp_file_path=self.gcp_evaluation_path,
                        output_dir=str(gcp_evaluation_output_dir),
                        patch_size=300
                    )
                    logging.info(f"✓ GCP evaluation complete for scale {second_to_last_scale:.3f}")
                except Exception as e:
                    logging.warning(f"GCP evaluation failed at scale {second_to_last_scale:.3f}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # d) apply transform_scale0.250 to scale 0.5 -> shift0.250 (if 0.5 is in scales)
        scale_c = 0.5
        transform_c = transform_b  # Default: use transform_b if scale_c is skipped
        ortho_c_shift0500 = ortho_b_shift0250  # Default: use previous ortho if scale_c is skipped
        
        if scale_c in self.scales:
            ortho_c_base = get_base_ortho(scale_c)
            ortho_c_shift0250, overlap_c_shift0250_png = apply_and_overlap(
                ortho_c_base, transform_b, scale_b, scale_c,
                output_basename=f'orthomosaic_scale{scale_c:.3f}_shift0.250.png'
            )
            if ortho_c_shift0250 is None:
                return None
            
            # e) matches at 0.5 using shifted source -> use transform type from config
            transform_type_c = self.transform_types.get(scale_c, 'homography')
            transform_c, _ = run_matching(
                overlap_c_shift0250_png, overlap_dir / f'target_overlap_scale{scale_c:.3f}.png',
                scale_c, transform_type_c, transform_json_name=f'transform_scale{scale_c:.3f}.json'
            )
            if transform_c is None:
                return None
            # Use transform type in filename for clarity
            transform_type_c = self.transform_types.get(scale_c, 'homography')
            ortho_c_shift0500, overlap_c_shift0500_png = apply_and_overlap(
                ortho_c_shift0250, transform_c, scale_c, scale_c,
                output_basename=f'orthomosaic_scale{scale_c:.3f}_{transform_type_c}.png'
            )
            
            # Run GCP evaluation at second-to-last scale if requested
            if self.gcp_evaluation_path and len(self.scales) >= 2:
                second_to_last_scale = self.scales[-2]  # Second-to-last scale
                if scale_c == second_to_last_scale:
                    logging.info(f"\n{'='*80}")
                    logging.info(f"Running GCP Evaluation at scale {second_to_last_scale:.3f} (second-to-last)")
                    logging.info(f"{'='*80}")
                    try:
                        from gcp_analysis import analyze_gcps
                        # Use the transformed orthomosaic at this scale for evaluation
                        # Create patches like gcp_analysis does
                        gcp_evaluation_output_dir = self.output_dir / 'gcp_analysis'
                        analyze_gcps(
                            registered_orthomosaic_path=str(ortho_c_shift0500),
                            gcp_file_path=self.gcp_evaluation_path,
                            output_dir=str(gcp_evaluation_output_dir),
                            patch_size=300
                        )
                        logging.info(f"✓ GCP evaluation complete for scale {second_to_last_scale:.3f}")
                    except Exception as e:
                        logging.warning(f"GCP evaluation failed at scale {second_to_last_scale:.3f}: {e}")
                        import traceback
                        traceback.print_exc()
        
        # f) Always create final output at full resolution (scale 1.0)
        scale_d = 1.0
        final_output = self.output_dir / 'orthomosaic_registered.tif'
        
        if scale_d in self.scales:
            # If 1.0 is in scales, process it normally (compute matches and transform at 1.0)
            ortho_d_base = get_base_ortho(scale_d)
            # Use transform_c (which may be transform_b if scale_c was skipped)
            prev_scale = scale_c if scale_c in self.scales else scale_b
            ortho_d_shift_prev, overlap_d_shift_prev_png = apply_and_overlap(
                ortho_d_base, transform_c, prev_scale, scale_d,
                output_basename=f'orthomosaic_scale{scale_d:.3f}_shift{prev_scale:.3f}.png'
            )
            if ortho_d_shift_prev is None:
                return None
            
            # g) matches at 1.0 using shifted source -> use transform type from config
            transform_type_d = self.transform_types.get(scale_d, 'shift')
            transform_d, _ = run_matching(
                overlap_d_shift_prev_png, overlap_dir / f'target_overlap_scale{scale_d:.3f}.png',
                scale_d, transform_type_d, transform_json_name=f'transform_scale{scale_d:.3f}.json'
            )
            if transform_d is None:
                return None
            # Use transform type in filename for clarity
            transform_type_d = self.transform_types.get(scale_d, 'shift')
            ortho_d_final, _ = apply_and_overlap(
                ortho_d_shift_prev, transform_d, scale_d, scale_d,
                output_basename=f'orthomosaic_scale{scale_d:.3f}_{transform_type_d}.png'
            )
            
            if ortho_d_final and ortho_d_final.exists():
                import shutil
                shutil.copy(ortho_d_final, final_output)
                logging.info(f"Final output: {final_output}")
                
                # Clean up temp intermediate directory if used (at 'none' debug level)
                if self._temp_intermediate_dir and self._temp_intermediate_dir.exists():
                    import shutil
                    shutil.rmtree(self._temp_intermediate_dir)
                    logging.debug(f"Cleaned up temporary directory: {self._temp_intermediate_dir}")
                
                return final_output
            else:
                logging.error("Final output was not generated.")
                return None
        else:
            # If 1.0 is NOT in scales, apply the last transform (from highest scale) to full resolution
            # Get the last transform (from the highest scale processed)
            last_scale = max(self.scales)
            last_transform = transform_c if scale_c in self.scales else transform_b
            
            logging.info(f"\n{'='*80}")
            logging.info(f"Applying transform from scale {last_scale:.3f} to full resolution (1.0)")
            logging.info(f"{'='*80}")
            
            # Get full resolution orthomosaic
            ortho_fullres = get_base_ortho(scale_d)
            
            # Apply the last transform, scaling it from last_scale to 1.0
            # Transformations are stored in meters, so we need to scale translation components
            # by the ratio of resolutions: 1.0 / last_scale
            ortho_final = self._apply_transform_to_orthomosaic(
                ortho_fullres, last_transform, last_scale, scale_d
            )
            
            if ortho_final and ortho_final.exists():
                import shutil
                shutil.copy(ortho_final, final_output)
                logging.info(f"Final output (full resolution): {final_output}")
                
                # Clean up temp intermediate directory if used (at 'none' debug level)
                if self._temp_intermediate_dir and self._temp_intermediate_dir.exists():
                    import shutil
                    shutil.rmtree(self._temp_intermediate_dir)
                    logging.debug(f"Cleaned up temporary directory: {self._temp_intermediate_dir}")
                
                return final_output
            else:
                logging.error("Failed to create final output at full resolution.")
                return None
        
        # Clean up temp directory even on failure
        if self._temp_intermediate_dir and self._temp_intermediate_dir.exists():
            import shutil
            shutil.rmtree(self._temp_intermediate_dir)
        
        return None
    
    def _compute_matches(self, source: np.ndarray, target: np.ndarray,
                        source_mask: np.ndarray, target_mask: np.ndarray,
                        scale: float) -> Optional[Dict]:
        """Compute matches using specified matcher."""
        pixel_resolution = 0.02 / scale  # meters per pixel
        
        if self.matcher == 'lightglue':
            if not LIGHTGLUE_AVAILABLE:
                logging.warning("LightGlue not available, falling back to SIFT")
                return match_sift(source, target, source_mask, target_mask)
            return match_lightglue(
                source, target, source_mask, target_mask,
                use_tiles=True,
                pixel_resolution_meters=pixel_resolution
            )
        elif self.matcher == 'sift':
            return match_sift(source, target, source_mask, target_mask)
        elif self.matcher == 'orb':
            return match_orb(source, target, source_mask, target_mask)
        elif self.matcher == 'patch_ncc':
            return match_patch_ncc(source, target, source_mask, target_mask, scale=scale)
        else:
            raise ValueError(f"Unknown matcher: {self.matcher}")
    
    def _compute_transformation(self, src_pts: np.ndarray, dst_pts: np.ndarray,
                               transform_type: str, scale: float) -> Dict:
        """Compute transformation of specified type and convert to meters."""
        pixel_resolution = 0.02 / scale
        ransac_threshold = 5.0 * (scale / 0.15)  # Scale threshold with resolution
        
        # Compute transformation (in pixels at this scale)
        if transform_type == 'shift':
            result = compute_2d_shift(src_pts, dst_pts, ransac_threshold)
        elif transform_type == 'similarity':
            result = compute_similarity_transform(src_pts, dst_pts, ransac_threshold)
        elif transform_type == 'affine':
            result = compute_affine_transform(src_pts, dst_pts, ransac_threshold)
        elif transform_type == 'homography':
            result = compute_homography(src_pts, dst_pts, ransac_threshold)
        elif transform_type == 'polynomial_2':
            result = compute_polynomial_transform(src_pts, dst_pts, degree=2, ransac_threshold=ransac_threshold)
        elif transform_type == 'polynomial_3':
            result = compute_polynomial_transform(src_pts, dst_pts, degree=3, ransac_threshold=ransac_threshold)
        elif transform_type == 'spline':
            result = compute_spline_transform(src_pts, dst_pts, ransac_threshold)
        elif transform_type == 'rubber_sheeting':
            result = compute_rubber_sheeting_transform(src_pts, dst_pts, ransac_threshold)
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
        
        # Convert transformation parameters to meters
        if result.get('matrix') is not None:
            M = np.array(result['matrix'])
            # Convert translation components (M[0,2] and M[1,2]) from pixels to meters
            M_meters = M.copy()
            M_meters[0, 2] = M[0, 2] * pixel_resolution
            M_meters[1, 2] = M[1, 2] * pixel_resolution
            result['matrix_meters'] = M_meters.tolist()
            result['matrix'] = M.tolist()  # Keep pixels for backward compatibility during transition
        
        if result.get('coefficients_x') is not None:
            # For polynomial transforms, coefficients need special handling
            # The constant term (coefficients[0]) represents translation in pixels
            # Higher order terms are in pixels^2, pixels^3, etc.
            # We'll convert the constant terms to meters, but keep higher order terms
            # in a normalized form (meters per unit of normalized coordinates)
            coeffs_x = np.array(result['coefficients_x'])
            coeffs_y = np.array(result['coefficients_y'])
            degree = result.get('degree', 2)
            
            # For polynomial transforms, conversion to meters is complex.
            # For now, we'll store the coefficients as-is (in pixels) and handle scaling
            # during application. The meters version will be the same for now.
            # TODO: Implement proper polynomial coefficient conversion to meters
            coeffs_x_meters = coeffs_x.copy()
            coeffs_y_meters = coeffs_y.copy()
            # Store as pixels for now - proper conversion requires coordinate space transformation
            
            result['coefficients_x_meters'] = coeffs_x_meters.tolist()
            result['coefficients_y_meters'] = coeffs_y_meters.tolist()
            result['coefficients_x'] = coeffs_x.tolist()  # Keep pixels for backward compatibility
            result['coefficients_y'] = coeffs_y.tolist()
        
        # Store pixel resolution used for conversion
        result['pixel_resolution_at_computation'] = pixel_resolution
        
        return result
    
    def _apply_cumulative_transformation_to_pyramid(self, new_transform_result: Dict,
                                                   current_scale: float,
                                                   cumulative_orthos: Dict,
                                                   transform_type: str) -> Dict:
        """
        Apply cumulative transformation to source orthomosaics.
        
        For each target scale >= current_scale:
        1. Get the input orthomosaic (with all previous transforms applied cumulatively)
        2. Apply the new transformation
        3. Save with the current transform type in the filename
        
        The key is that transformations are CUMULATIVE:
        - At 0.125: apply shift_1 to original -> orthomosaic_scale0.125_shift.tif
        - At 0.25: apply shift_2 to orthomosaic_scale0.250_shift.tif (from 0.125) -> orthomosaic_scale0.250_shift.tif (overwrites with cumulative)
        - At 0.5: apply homography to orthomosaic_scale0.500_shift.tif (from 0.25) -> orthomosaic_scale0.500_homography.tif
        
        Args:
            new_transform_result: New transformation result dictionary
            current_scale: Current scale where transformation was computed
            cumulative_orthos: Dict mapping scale -> path to ortho with previous transforms
            transform_type: Type of transformation being applied (for filename)
        
        Returns:
            Updated cumulative_orthos dict
        """
        # Get index of current scale
        current_idx = self.scales.index(current_scale)
        
        # Apply to current scale and all remaining scales
        for i in range(current_idx, len(self.scales)):
            target_scale = self.scales[i]
            
            # Determine input orthomosaic path
            # For the first scale (0.125), use original downsampled orthomosaic
            # For subsequent scales, use the orthomosaic from the PREVIOUS scale's transformation
            if i == 0:
                # First scale: use original downsampled orthomosaic
                input_ortho_path = self._get_or_create_downsampled_orthomosaic(target_scale)
            else:
                # Subsequent scales: use the transformed orthomosaic from the previous scale
                prev_scale = self.scales[i - 1]
                prev_transform_type = self.transform_types.get(prev_scale, 'shift')
                prev_ortho_path = self.intermediate_dir / f'orthomosaic_scale{target_scale:.3f}_{prev_transform_type}.tif'
                
                if prev_ortho_path.exists():
                    input_ortho_path = prev_ortho_path
                    logging.debug(f"    Using previous transform result: {prev_ortho_path.name}")
                else:
                    # Fall back to original (shouldn't happen in normal flow)
                    input_ortho_path = self._get_or_create_downsampled_orthomosaic(target_scale)
                    logging.warning(f"    Previous transform not found, using original: {input_ortho_path.name}")
            
            if input_ortho_path is None:
                continue
            
            # Determine output path - use current transform type in filename (in intermediate directory)
            output_path = self.intermediate_dir / f'orthomosaic_scale{target_scale:.3f}_{transform_type}.tif'
            
            # region agent log
            _dbg_log(
                hypothesis_id="H1",
                location="register_orthomosaic.py:_apply_cumulative_transformation_to_pyramid",
                message="apply_transform_choice",
                data={
                    "current_scale": current_scale,
                    "target_scale": target_scale,
                    "transform_type": transform_type,
                    "input_path": str(input_ortho_path),
                    "output_path": str(output_path),
                    "prev_scale": self.scales[i - 1] if i > 0 else None
                }
            )
            # endregion agent log

            # Skip if already exists
            if output_path.exists():
                logging.info(f"  Skipping scale {target_scale:.3f} (already transformed: {output_path.name})")
                cumulative_orthos[target_scale] = output_path
                continue
            
            logging.info(f"  Applying {transform_type} to orthomosaic at scale {target_scale:.3f}...")
            logging.info(f"    Input: {input_ortho_path.name}")
            
            # Apply transformation (scaled appropriately)
            transformed_path = self._apply_transform_to_orthomosaic(
                input_ortho_path, new_transform_result, current_scale, target_scale
            )
            
            if transformed_path:
                cumulative_orthos[target_scale] = transformed_path
                logging.info(f"    ✓ Applied to scale {target_scale:.3f} -> {transformed_path.name}")
            else:
                logging.error(f"    ✗ Failed to apply to scale {target_scale:.3f}")
        
        return cumulative_orthos
    
    def _get_or_create_downsampled_orthomosaic(self, scale: float) -> Optional[Path]:
        """Create or load downsampled orthomosaic at specified scale."""
        ortho_path = self.preprocessing_dir / f'orthomosaic_scale{scale:.3f}.tif'
        
        if ortho_path.exists():
            logging.debug(f"  Using existing downsampled orthomosaic: {ortho_path.name}")
            return ortho_path
        
        logging.info(f"  Creating downsampled orthomosaic at scale {scale:.3f}...")
        
        try:
            with rasterio.open(self.source_path) as src:
                orig_height = src.height
                orig_width = src.width
                orig_transform = src.transform
                orig_crs = src.crs
                orig_count = src.count
                
                # Calculate new dimensions
                new_height = max(int(orig_height * scale), 100)
                new_width = max(int(orig_width * scale), 100)
                
                # Calculate new transform (pixel size increases by 1/scale)
                new_transform = rasterio.Affine(
                    orig_transform.a / scale,  # x pixel size
                    orig_transform.b,          # row rotation
                    orig_transform.c,          # x origin
                    orig_transform.d,          # column rotation
                    orig_transform.e / scale,  # y pixel size (negative)
                    orig_transform.f           # y origin
                )
                
                # Read and downsample each band
                output_data = []
                for band_idx in range(1, min(orig_count + 1, 4)):  # Max 3 bands for RGB
                    band_data = src.read(
                        band_idx,
                        out_shape=(new_height, new_width),
                        resampling=Resampling.bilinear
                    )
                    output_data.append(band_data)
                
                # Ensure 3 bands for RGB
                if len(output_data) == 1:
                    output_data = [output_data[0], output_data[0], output_data[0]]
                elif len(output_data) == 2:
                    output_data.append(output_data[1])
                
                # Convert to uint8
                output_array = np.stack(output_data[:3], axis=0).astype(np.uint8)
                
                # Save
                ortho_path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(
                    ortho_path,
                    'w',
                    driver='GTiff',
                    height=new_height,
                    width=new_width,
                    count=3,
                    dtype=np.uint8,
                    crs=orig_crs,
                    transform=new_transform,
                    compress='JPEG',
                    jpeg_quality=90,
                    photometric='RGB',
                    tiled=True,
                    blockxsize=512,
                    blockysize=512
                ) as dst:
                    for band_idx in range(1, 4):
                        dst.write(output_array[band_idx - 1], band_idx)
                
                logging.info(f"    ✓ Created: {ortho_path.name}")
                return ortho_path
                
        except Exception as e:
            logging.error(f"    Failed to create downsampled orthomosaic: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _apply_transform_to_orthomosaic(self, input_path: Path, transform_result: Dict,
                                       source_scale: float, target_scale: float) -> Optional[Path]:
        """Apply transformation to orthomosaic, scaling appropriately."""
        transform_type = transform_result['type']
        
        # Determine output path in intermediate directory (transformed orthomosaics)
        output_path = self.intermediate_dir / f'orthomosaic_scale{target_scale:.3f}_{transform_type}.tif'
        
        if output_path.exists():
            logging.debug(f"  Output already exists: {output_path.name}")
            return output_path
        
        try:
            with rasterio.open(input_path) as src:
                height, width = src.height, src.width
                num_bands = src.count
                orig_transform = src.transform
                orig_crs = src.crs
                
                # Scale factor for transformation
                scale_factor = target_scale / source_scale

                # region agent log
                _dbg_log(
                    hypothesis_id="H2",
                    location="register_orthomosaic.py:_apply_transform_to_orthomosaic",
                    message="before_apply_transform",
                    data={
                        "transform_type": transform_type,
                        "source_scale": source_scale,
                        "target_scale": target_scale,
                        "width": width,
                        "height": height,
                        "num_bands": num_bands,
                        "scale_factor": scale_factor
                    }
                )
                # endregion agent log
                
                written_tiled = False
                
                if transform_type in ['shift', 'similarity', 'affine', 'homography']:
                    # Matrix-based transformation
                    # Use meters version if available, otherwise fall back to pixels
                    target_pixel_resolution = 0.02 / target_scale
                    
                    if 'matrix_meters' in transform_result and transform_result['matrix_meters'] is not None:
                        # Transformation is stored in meters - convert to pixels at target scale
                        M_meters = np.array(transform_result['matrix_meters'])
                        if M_meters.shape == (2, 3):
                            M_meters = np.vstack([M_meters, [0, 0, 1]])
                        
                        # Convert translation from meters to pixels at target scale
                        M_scaled = M_meters.copy()
                        M_scaled[0, 2] = M_meters[0, 2] / target_pixel_resolution
                        M_scaled[1, 2] = M_meters[1, 2] / target_pixel_resolution
                    else:
                        # Fallback: use pixels and scale (old behavior)
                        M = np.array(transform_result['matrix'])
                        if M.shape == (2, 3):
                            M = np.vstack([M, [0, 0, 1]])
                        
                        # Scale translation components
                        M_scaled = M.copy()
                        M_scaled[0, 2] *= scale_factor
                        M_scaled[1, 2] *= scale_factor
                    
                    # cv2.warpPerspective applies the transformation directly
                    # Our matrix M maps source -> target, so we pass M directly
                    # OpenCV will handle the backward mapping internally
                    logging.debug(f"    M_scaled translation: ({M_scaled[0, 2]:.2f}, {M_scaled[1, 2]:.2f}) pixels")
                    
                    max_pixels_direct = 50_000_000
                    use_tiled = (height * width) > max_pixels_direct or max(height, width) >= 32000
                    tile_size = 1024
                    
                    if not use_tiled:
                        data = src.read()  # Full read for manageable sizes
                        output_bands = []
                        for band_idx in range(num_bands):
                            band_image = data[band_idx]
                            transformed_band = cv2.warpPerspective(
                                band_image, M_scaled, (width, height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0
                            )
                            output_bands.append(transformed_band)
                        output_data = np.stack(output_bands, axis=0)
                    else:
                        # Tiled warp to avoid SHRT_MAX and huge buffers
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with rasterio.open(
                            output_path,
                            'w',
                            driver='GTiff',
                            height=height,
                            width=width,
                            count=num_bands,
                            dtype=np.uint8,
                            crs=orig_crs,
                            transform=orig_transform,
                            compress='JPEG',
                            jpeg_quality=90,
                            photometric='RGB' if num_bands == 3 else 'MINISBLACK',
                            tiled=True,
                            blockxsize=512,
                            blockysize=512
                        ) as dst:
                            # For cv2.remap, we need the inverse mapping: for each output pixel,
                            # where should we sample from in the source?
                            # M_scaled maps source -> target, so we need M^-1 to map target -> source
                            try:
                                M_inv = np.linalg.inv(M_scaled)
                            except np.linalg.LinAlgError as e:
                                logging.error(f"    Failed to invert transformation matrix for tiled remap: {e}")
                                return None
                            
                            for y0 in range(0, height, tile_size):
                                tile_h = min(tile_size, height - y0)
                                for x0 in range(0, width, tile_size):
                                    tile_w = min(tile_size, width - x0)
                                    
                                    y_coords, x_coords = np.mgrid[y0:y0 + tile_h, x0:x0 + tile_w]
                                    ones = np.ones_like(x_coords)
                                    coords = np.stack([x_coords, y_coords, ones], axis=0).reshape(3, -1)
                                    # Use inverse to map output coordinates to source coordinates
                                    mapped = M_inv @ coords
                                    new_x = mapped[0].reshape(tile_h, tile_w)
                                    new_y = mapped[1].reshape(tile_h, tile_w)
                                    
                                    min_x = int(np.floor(np.clip(new_x.min() - 1, 0, width - 1)))
                                    max_x = int(np.ceil(np.clip(new_x.max() + 1, 0, width - 1)))
                                    min_y = int(np.floor(np.clip(new_y.min() - 1, 0, height - 1)))
                                    max_y = int(np.ceil(np.clip(new_y.max() + 1, 0, height - 1)))
                                    
                                    win_w = max(1, max_x - min_x + 1)
                                    win_h = max(1, max_y - min_y + 1)
                                    read_window = Window(col_off=min_x, row_off=min_y, width=win_w, height=win_h)
                                    
                                    map_x_local = (new_x - min_x).astype(np.float32)
                                    map_y_local = (new_y - min_y).astype(np.float32)
                                    
                                    for band_idx in range(num_bands):
                                        band_image = src.read(band_idx + 1, window=read_window)
                                        transformed_band = cv2.remap(
                                            band_image, map_x_local, map_y_local, cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0
                                        )
                                        dst.write(
                                            transformed_band,
                                            band_idx + 1,
                                            window=Window(x0, y0, tile_w, tile_h)
                                        )
                        written_tiled = True
                    
                elif transform_type in ['polynomial_2', 'polynomial_3']:
                    # Polynomial transformation
                    coeffs_x = np.array(transform_result['coefficients_x'])
                    coeffs_y = np.array(transform_result['coefficients_y'])
                    degree = 2 if transform_type == 'polynomial_2' else 3
                    
                    # Decide whether to use tiled remap to reduce memory for large images
                    max_pixels_direct = 50_000_000  # heuristic threshold for full-size maps
                    use_tiled = (height * width) > max_pixels_direct
                    tile_size = 1024
                    
                    # region agent log
                    _dbg_log(
                        hypothesis_id="H2",
                        location="register_orthomosaic.py:_apply_transform_to_orthomosaic",
                        message="poly_tile_strategy",
                        data={
                            "width": width,
                            "height": height,
                            "num_bands": num_bands,
                            "max_pixels_direct": max_pixels_direct,
                            "use_tiled": use_tiled,
                            "tile_size": tile_size,
                            "scale_factor": scale_factor,
                            "transform_type": transform_type
                        }
                    )
                    # endregion agent log
                    
                    if not use_tiled:
                        data = src.read()  # Full read only when safe
                        # Build coordinate grid
                        y_coords, x_coords = np.mgrid[0:height, 0:width]
                        
                        # Scale coordinates to match transformation's coordinate space
                        x_coords_scaled = x_coords / scale_factor
                        y_coords_scaled = y_coords / scale_factor
                        
                        # Build polynomial terms
                        if degree == 2:
                            terms = np.array([
                                np.ones_like(x_coords_scaled),
                                x_coords_scaled,
                                y_coords_scaled,
                                x_coords_scaled**2,
                                x_coords_scaled * y_coords_scaled,
                                y_coords_scaled**2
                            ])
                        else:  # degree == 3
                            terms = np.array([
                                np.ones_like(x_coords_scaled),
                                x_coords_scaled,
                                y_coords_scaled,
                                x_coords_scaled**2,
                                x_coords_scaled * y_coords_scaled,
                                y_coords_scaled**2,
                                x_coords_scaled**3,
                                x_coords_scaled**2 * y_coords_scaled,
                                x_coords_scaled * y_coords_scaled**2,
                                y_coords_scaled**3
                            ])
                        
                        # Apply transformation
                        new_x = np.sum(coeffs_x[:, np.newaxis, np.newaxis] * terms, axis=0)
                        new_y = np.sum(coeffs_y[:, np.newaxis, np.newaxis] * terms, axis=0)
                        
                        # Scale back to target scale coordinate space
                        new_x = new_x * scale_factor
                        new_y = new_y * scale_factor
                        
                        map_x = new_x.astype(np.float32)
                        map_y = new_y.astype(np.float32)
                        
                        # Apply remap
                        output_bands = []
                        for band_idx in range(num_bands):
                            band_image = data[band_idx]
                            transformed_band = cv2.remap(
                                band_image, map_x, map_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0
                            )
                            output_bands.append(transformed_band)
                        output_data = np.stack(output_bands, axis=0)
                    else:
                        # Tiled remap to reduce peak memory (compute map per tile and read windows)
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with rasterio.open(
                            output_path,
                            'w',
                            driver='GTiff',
                            height=height,
                            width=width,
                            count=num_bands,
                            dtype=np.uint8,
                            crs=orig_crs,
                            transform=orig_transform,
                            compress='JPEG',
                            jpeg_quality=90,
                            photometric='RGB' if num_bands == 3 else 'MINISBLACK',
                            tiled=True,
                            blockxsize=512,
                            blockysize=512
                        ) as dst:
                            tiles_written = 0
                            for y0 in range(0, height, tile_size):
                                tile_h = min(tile_size, height - y0)
                                for x0 in range(0, width, tile_size):
                                    tile_w = min(tile_size, width - x0)
                                    
                                    # Coordinate grid for this tile
                                    y_coords, x_coords = np.mgrid[y0:y0 + tile_h, x0:x0 + tile_w]
                                    x_coords_scaled = x_coords / scale_factor
                                    y_coords_scaled = y_coords / scale_factor
                                    
                                    if degree == 2:
                                        terms = np.array([
                                            np.ones_like(x_coords_scaled),
                                            x_coords_scaled,
                                            y_coords_scaled,
                                            x_coords_scaled**2,
                                            x_coords_scaled * y_coords_scaled,
                                            y_coords_scaled**2
                                        ])
                                    else:
                                        terms = np.array([
                                            np.ones_like(x_coords_scaled),
                                            x_coords_scaled,
                                            y_coords_scaled,
                                            x_coords_scaled**2,
                                            x_coords_scaled * y_coords_scaled,
                                            y_coords_scaled**2,
                                            x_coords_scaled**3,
                                            x_coords_scaled**2 * y_coords_scaled,
                                            x_coords_scaled * y_coords_scaled**2,
                                            y_coords_scaled**3
                                        ])
                                    
                                    new_x = np.sum(coeffs_x[:, np.newaxis, np.newaxis] * terms, axis=0)
                                    new_y = np.sum(coeffs_y[:, np.newaxis, np.newaxis] * terms, axis=0)
                                    
                                    new_x = new_x * scale_factor
                                    new_y = new_y * scale_factor
                                    
                                    # Determine the source window needed for this tile
                                    min_x = int(np.floor(np.clip(new_x.min() - 1, 0, width - 1)))
                                    max_x = int(np.ceil(np.clip(new_x.max() + 1, 0, width - 1)))
                                    min_y = int(np.floor(np.clip(new_y.min() - 1, 0, height - 1)))
                                    max_y = int(np.ceil(np.clip(new_y.max() + 1, 0, height - 1)))
                                    
                                    win_w = max(1, max_x - min_x + 1)
                                    win_h = max(1, max_y - min_y + 1)
                                    read_window = Window(col_off=min_x, row_off=min_y, width=win_w, height=win_h)
                                    
                                    map_x_local = (new_x - min_x).astype(np.float32)
                                    map_y_local = (new_y - min_y).astype(np.float32)
                                    
                                    for band_idx in range(num_bands):
                                        band_image = src.read(band_idx + 1, window=read_window)
                                        transformed_band = cv2.remap(
                                            band_image, map_x_local, map_y_local, cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0
                                        )
                                        dst.write(
                                            transformed_band,
                                            band_idx + 1,
                                            window=Window(x0, y0, tile_w, tile_h)
                                        )
                                    
                                    tiles_written += 1
                            
                        written_tiled = True
                    
                else:
                    logging.warning(f"    Unsupported transform type for orthomosaic: {transform_type}")
                    return None
                
                # Save transformed orthomosaic
                if written_tiled:
                    # region agent log
                    _dbg_log(
                        hypothesis_id="H2",
                        location="register_orthomosaic.py:_apply_transform_to_orthomosaic",
                        message="after_apply_transform_tiled",
                        data={
                            "transform_type": transform_type,
                            "target_scale": target_scale,
                            "output_path": str(output_path),
                            "width": width,
                            "height": height
                        }
                    )
                    # endregion agent log
                    return output_path
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=num_bands,
                    dtype=np.uint8,
                    crs=orig_crs,
                    transform=orig_transform,
                    compress='JPEG',
                    jpeg_quality=90,
                    photometric='RGB' if num_bands == 3 else 'MINISBLACK',
                    tiled=True,
                    blockxsize=512,
                    blockysize=512
                ) as dst:
                    for band_idx in range(1, num_bands + 1):
                        dst.write(output_data[band_idx - 1], band_idx)
                
                # region agent log
                _dbg_log(
                    hypothesis_id="H2",
                    location="register_orthomosaic.py:_apply_transform_to_orthomosaic",
                    message="after_apply_transform",
                    data={
                        "transform_type": transform_type,
                        "target_scale": target_scale,
                        "output_path": str(output_path),
                        "width": width,
                        "height": height
                    }
                )
                # endregion agent log

                return output_path
                
        except Exception as e:
            logging.error(f"    Failed to apply transformation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_matches_json(self, matches_result: Dict, output_path: Path, scale: float,
                          source_image: Optional[str] = None, target_image: Optional[str] = None):
        """Save matches to JSON file."""
        method = matches_result.get('method', self.matcher)
        matches = matches_result.get('matches', [])
        scale_factor = matches_result.get('scale_factor', 1.0)
        pixel_resolution = 0.02 / scale
        
        # Collect all distances for summary statistics
        distances_pixels = []
        distances_meters = []
        
        # Initialize matches_data with summary_statistics placeholder at top
        matches_data = {
            'summary_statistics': None,  # Will be computed and filled in later
            'method': method,
            'scale': scale,
            'num_matches': len(matches),
            'scale_factor': scale_factor,
            'source_image': source_image,
            'target_image': target_image,
            'matches': []
        }
        
        # Extract match data based on matcher type
        method_upper = method.upper() if method else ''
        logging.debug(f"  Processing matches for method: {method} (normalized: {method_upper})")
        if method_upper in ['SIFT', 'ORB', 'LIGHTGLUE']:
            # Feature-based matching
            kp1 = matches_result.get('kp1', [])
            kp2 = matches_result.get('kp2', [])
            
            for match in matches:
                if hasattr(match, 'queryIdx') and hasattr(match, 'trainIdx'):
                    src_kp = kp1[match.queryIdx]
                    tgt_kp = kp2[match.trainIdx]
                    
                    src_x = float(src_kp.pt[0])
                    src_y = float(src_kp.pt[1])
                    tgt_x_upsampled = float(tgt_kp.pt[0])
                    tgt_y_upsampled = float(tgt_kp.pt[1])
                    tgt_x_original = tgt_x_upsampled / scale_factor if scale_factor != 1.0 else tgt_x_upsampled
                    tgt_y_original = tgt_y_upsampled / scale_factor if scale_factor != 1.0 else tgt_y_upsampled
                    
                    pixel_distance = np.sqrt((src_x - tgt_x_upsampled)**2 + (src_y - tgt_y_upsampled)**2)
                    distance_meters = pixel_distance * pixel_resolution
                    
                    distances_pixels.append(pixel_distance)
                    distances_meters.append(distance_meters)
                    
                    matches_data['matches'].append({
                        'source': {'x': src_x, 'y': src_y, 'response': float(src_kp.response)},
                        'target_upsampled': {'x': tgt_x_upsampled, 'y': tgt_y_upsampled},
                        'target_original': {'x': tgt_x_original, 'y': tgt_y_original},
                        'distance': {
                            'pixels': float(pixel_distance),
                            'meters': float(distance_meters)
                        }
                    })
        
        elif method == 'Patch_NCC':
            # Patch-based matching
            for match in matches:
                if isinstance(match, dict):
                    src_pt = match.get('source', (0, 0))
                    tgt_pt = match.get('target', (0, 0))
                    
                    if isinstance(src_pt, (list, tuple)) and isinstance(tgt_pt, (list, tuple)):
                        src_x, src_y = float(src_pt[0]), float(src_pt[1])
                        tgt_x, tgt_y = float(tgt_pt[0]), float(tgt_pt[1])
                        
                        pixel_distance = np.sqrt((src_x - tgt_x)**2 + (src_y - tgt_y)**2)
                        distance_meters = pixel_distance * pixel_resolution
                        
                        distances_pixels.append(pixel_distance)
                        distances_meters.append(distance_meters)
                        
                        matches_data['matches'].append({
                            'source': {'x': src_x, 'y': src_y},
                            'target_original': {'x': tgt_x, 'y': tgt_y},
                            'target_upsampled': {'x': tgt_x, 'y': tgt_y},  # Same for patch NCC
                            'distance': {
                                'pixels': float(pixel_distance),
                                'meters': float(distance_meters)
                            },
                            'confidence': match.get('confidence', 0.0)
                        })
        
        # Compute summary statistics
        logging.debug(f"  Computing summary statistics: {len(distances_pixels)} distances collected")
        if len(distances_pixels) > 0:
            distances_pixels_arr = np.array(distances_pixels, dtype=float)
            distances_meters_arr = np.array(distances_meters, dtype=float)
            summary_stats = {
                'distance_pixels': {
                    'mean': float(np.mean(distances_pixels_arr)),
                    'median': float(np.median(distances_pixels_arr)),
                    'stddev': float(np.std(distances_pixels_arr)),
                    'min': float(np.min(distances_pixels_arr)),
                    'max': float(np.max(distances_pixels_arr))
                },
                'distance_meters': {
                    'mean': float(np.mean(distances_meters_arr)),
                    'median': float(np.median(distances_meters_arr)),
                    'stddev': float(np.std(distances_meters_arr)),
                    'min': float(np.min(distances_meters_arr)),
                    'max': float(np.max(distances_meters_arr))
                }
            }
            logging.debug(f"  Summary stats computed: mean={summary_stats['distance_meters']['mean']:.2f}m, "
                        f"median={summary_stats['distance_meters']['median']:.2f}m")
        else:
            # No matches found, set empty statistics
            logging.warning(f"  No distances collected for summary statistics!")
            summary_stats = {
                'distance_pixels': {'mean': 0.0, 'median': 0.0, 'stddev': 0.0, 'min': 0.0, 'max': 0.0},
                'distance_meters': {'mean': 0.0, 'median': 0.0, 'stddev': 0.0, 'min': 0.0, 'max': 0.0},
                'note': 'No matches found'
            }
        
        # Rebuild matches_data with summary_statistics at the top
        final_matches_data = {
            'summary_statistics': summary_stats,
            'method': matches_data['method'],
            'scale': matches_data['scale'],
            'num_matches': matches_data['num_matches'],
            'scale_factor': matches_data['scale_factor'],
            'source_image': matches_data['source_image'],
            'target_image': matches_data['target_image'],
            'matches': matches_data['matches']
        }
        
        # Verify summary statistics are in the data
        if 'summary_statistics' not in final_matches_data or final_matches_data['summary_statistics'] is None:
            logging.error(f"  ERROR: Summary statistics not computed! This should not happen.")
        else:
            logging.debug(f"  Summary statistics verified in final_matches_data")
            logging.debug(f"  Summary stats keys: {list(final_matches_data['summary_statistics'].keys())}")
        
        # Write JSON file
        with open(output_path, 'w') as f:
            json.dump(final_matches_data, f, indent=2)
        
        # Verify file was written correctly by reading it back
        try:
            with open(output_path, 'r') as f:
                written_data = json.load(f)
            if 'summary_statistics' in written_data:
                logging.debug(f"  ✓ Verified: summary_statistics present in written file")
                if 'distance_meters' in written_data['summary_statistics']:
                    logging.debug(f"  ✓ Verified: distance_meters present in written file")
                else:
                    logging.error(f"  ✗ ERROR: distance_meters missing in written file!")
            else:
                logging.error(f"  ✗ ERROR: summary_statistics missing in written file!")
        except Exception as e:
            logging.warning(f"  Could not verify written file: {e}")
        
        logging.info(f"  Saved {len(final_matches_data['matches'])} matches to {output_path.name}")
        if summary_stats and 'distance_meters' in summary_stats:
            logging.info(f"    Summary: mean={summary_stats['distance_meters']['mean']:.2f}m, "
                        f"median={summary_stats['distance_meters']['median']:.2f}m, "
                        f"stddev={summary_stats['distance_meters']['stddev']:.2f}m")
        else:
            logging.warning(f"    WARNING: Summary statistics not available!")
    
    def _save_transform_json(self, transform_result: Dict, output_path: Path, scale: float):
        """Save transformation to JSON file."""
        # Add pixel and meter conversions
        pixel_resolution = 0.02 / scale
        
        transform_data = {
            'scale': scale,
            'transform_type': transform_result['type'],
            'pixel_resolution_meters': pixel_resolution,
            'median_error_pixels': transform_result.get('median_error', 0),
            'median_error_meters': transform_result.get('median_error', 0) * pixel_resolution,
            'robust_rmse_pixels': transform_result.get('robust_rmse', 0),
            'robust_rmse_meters': transform_result.get('robust_rmse', 0) * pixel_resolution,
            'matrix': transform_result.get('matrix', None),  # Pixels (for backward compatibility)
            'matrix_meters': transform_result.get('matrix_meters', None),  # Meters (primary)
            'coefficients_x': transform_result.get('coefficients_x', None),  # Pixels
            'coefficients_y': transform_result.get('coefficients_y', None),  # Pixels
            'coefficients_x_meters': transform_result.get('coefficients_x_meters', None),  # Meters
            'coefficients_y_meters': transform_result.get('coefficients_y_meters', None),  # Meters
            'control_points_src': transform_result.get('control_points_src', None),
            'control_points_dst': transform_result.get('control_points_dst', None),
            'pixel_resolution_at_computation': transform_result.get('pixel_resolution_at_computation', pixel_resolution)
        }
        
        # Convert numpy arrays to lists for JSON
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        transform_data = convert_to_serializable(transform_data)
        
        with open(output_path, 'w') as f:
            json.dump(transform_data, f, indent=2)
        
        logging.info(f"  Saved transformation to {output_path.name}")
    
    def _create_error_histogram(self, transform_result: Dict, scale: float,
                                src_pts: Optional[np.ndarray] = None,
                                dst_pts: Optional[np.ndarray] = None,
                                inlier_mask: Optional[np.ndarray] = None,
                                json_name: Optional[str] = None,
                                matches_path: Optional[Path] = None):
        """Create histogram visualization of distances (all matches vs inliers)."""
        import matplotlib.pyplot as plt
        
        if matches_path is None or not matches_path.exists():
            return
        with open(matches_path, 'r') as f:
            matches_data = json.load(f)
        matches_list = matches_data.get('matches', [])
        if len(matches_list) == 0:
            return
        
        pixel_resolution = 0.02 / scale
        
        # Distances for all matches from matches JSON (as-is, no filtering)
        distances_all_m = []
        for m in matches_list:
            dist = m.get('distance', {})
            if 'meters' in dist:
                distances_all_m.append(float(dist['meters']))
            elif 'pixels' in dist:
                distances_all_m.append(float(dist['pixels']) * pixel_resolution)
        if len(distances_all_m) == 0:
            return
        distances_all_m = np.array(distances_all_m, dtype=float)
        
        # Inlier mask for reprojection errors
        reproj_errors = transform_result.get('reproj_errors', [])
        reproj_errors = np.array(reproj_errors, dtype=float)
        if inlier_mask is None or len(inlier_mask) != len(reproj_errors):
            inlier_mask = np.ones_like(reproj_errors, dtype=bool)
        inlier_errors_m = reproj_errors[inlier_mask] * pixel_resolution
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].hist(distances_all_m, bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Distance (meters)', fontweight='bold')
        axes[0].set_ylabel('Count', fontweight='bold')
        matches_json_name = matches_path.name if matches_path else 'unknown'
        subtitle = f'All matches - Scale {scale:.3f}\n{matches_json_name}'
        axes[0].set_title(subtitle, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(np.median(distances_all_m), color='blue', linestyle='--', linewidth=2,
                        label=f'Median: {np.median(distances_all_m):.2f}m')
        axes[0].axvline(np.mean(distances_all_m), color='green', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(distances_all_m):.2f}m')
        axes[0].legend()
        
        axes[1].hist(inlier_errors_m, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1].set_xlabel('Distance (meters)', fontweight='bold')
        axes[1].set_ylabel('Count', fontweight='bold')
        matches_json_name = matches_path.name if matches_path else 'unknown'
        transform_json_name = json_name if json_name else 'unknown'
        subtitle_inliers = f'Inliers from {matches_json_name}\nwhen computing {transform_json_name}\n({transform_result["type"]})'
        axes[1].set_title(subtitle_inliers, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(np.median(inlier_errors_m), color='red', linestyle='--', linewidth=2,
                        label=f'Median: {np.median(inlier_errors_m):.2f}m')
        axes[1].axvline(np.mean(inlier_errors_m), color='green', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(inlier_errors_m):.2f}m')
        axes[1].legend()
        
        plt.tight_layout()
        # Only save histogram if matching_dir exists (high debug level)
        if self.matching_dir:
            output_path = self.matching_dir / f'error_histogram_scale{scale:.3f}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logging.info(f"  Saved distance histograms to {output_path.name}")
        plt.close()
        
        # region agent log
        _dbg_log(
            hypothesis_id="H3",
            location="register_orthomosaic.py:_create_error_histogram",
            message="histogram_stats",
            data={
                "scale": scale,
                "transform_type": transform_result.get("type"),
                "num_all": len(distances_all_m),
                "num_inliers": int(inlier_mask.sum()),
                "median_all_m": float(np.median(distances_all_m)),
                "median_inliers_m": float(np.median(inlier_errors_m))
            }
        )
        # endregion agent log
    
    def _extract_overlap_from_orthomosaic(self, ortho_path: Path, scale: float,
                                          preprocessing_dir: Path,
                                          output_name: Optional[str] = None) -> Optional[Path]:
        """Extract overlap region from orthomosaic with custom naming."""
        if output_name:
            overlap_path = preprocessing_dir / output_name
        else:
            overlap_path = preprocessing_dir / f'source_overlap_scale{scale:.3f}_pretransformed.png'
        
        if overlap_path.exists():
            return overlap_path
        
        try:
            # Get overlap info
            overlap_info = self.preprocessor.compute_overlap_region(scale)
            if overlap_info is None:
                return None
            
            # Crop to overlap region directly from GeoTIFF (more memory efficient for large images)
            src_crop = overlap_info['source']
            from rasterio.windows import Window
            
            # Read only the overlap region
            window = Window(
                col_off=src_crop['x1'],
                row_off=src_crop['y1'],
                width=src_crop['x2'] - src_crop['x1'],
                height=src_crop['y2'] - src_crop['y1']
            )
            
            with rasterio.open(ortho_path) as src:
                # Read only the overlap window
                if src.count >= 3:
                    data = src.read([1, 2, 3], window=window)
                    rgb = np.moveaxis(data, 0, -1)
                    # Normalize to 0-255
                    rgb_min, rgb_max = rgb.min(), rgb.max()
                    if rgb_max > rgb_min:
                        rgb_norm = ((rgb - rgb_min) / (rgb_max - rgb_min) * 255).astype(np.uint8)
                    else:
                        rgb_norm = np.zeros_like(rgb, dtype=np.uint8)
                    # Convert to grayscale using weighted average (avoid OpenCV for large images)
                    gray = (0.299 * rgb_norm[:, :, 0] + 0.587 * rgb_norm[:, :, 1] + 0.114 * rgb_norm[:, :, 2]).astype(np.uint8)
                else:
                    data = src.read(1, window=window)
                    single_min, single_max = data.min(), data.max()
                    if single_max > single_min:
                        gray = ((data - single_min) / (single_max - single_min) * 255).astype(np.uint8)
                    else:
                        gray = np.zeros_like(data, dtype=np.uint8)
            
            # Save using PIL to avoid OpenCV size limits
            from PIL import Image
            preprocessing_dir.mkdir(parents=True, exist_ok=True)
            img = Image.fromarray(gray, mode='L')
            img.save(str(overlap_path), 'PNG', compress_level=1)
            
            return overlap_path
            
        except Exception as e:
            logging.error(f"Failed to extract overlap: {e}")
            return None
    
    def _create_final_orthomosaic(self, cumulative_orthos: Dict) -> Path:
        """Create final registered orthomosaic at full resolution."""
        output_path = self.output_dir / 'orthomosaic_registered.tif'
        
        if not cumulative_orthos:
            logging.warning("No transformations computed, copying source as output")
            import shutil
            shutil.copy(self.source_path, output_path)
            return output_path
        
        # Get the final scale (1.0) transformation if available, otherwise use the last one
        final_scale = 1.0
        if final_scale not in cumulative_orthos:
            # Use the last scale that has a transformation
            final_scale = max(cumulative_orthos.keys())
        
        final_ortho_path = cumulative_orthos[final_scale]
        
        logging.info(f"Creating final registered orthomosaic from scale {final_scale:.3f}...")
        
        # If final scale is 1.0, we already have the full-resolution registered orthomosaic
        if final_scale == 1.0:
            # Copy the transformed orthomosaic to final output
            import shutil
            shutil.copy(final_ortho_path, output_path)
            logging.info(f"  ✓ Final orthomosaic: {output_path.name}")
            return output_path
        
        # Otherwise, we need to apply the transformation to the full-resolution source
        # This would require applying all cumulative transformations
        # For now, copy the highest resolution transformed orthomosaic
        logging.warning(f"Final scale is {final_scale}, not 1.0. Using transformed orthomosaic at this scale.")
        import shutil
        shutil.copy(final_ortho_path, output_path)
        
        return output_path


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path_obj, 'r') as f:
        config = json.load(f)
    
    return config


def save_config(config: dict, output_dir: Path):
    """Save configuration to output directory for reproducibility."""
    config_path = output_dir / 'run_config.json'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logging.info(f"Configuration saved to: {config_path}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description='Register orthomosaic to basemap')
    parser.add_argument('source', nargs='?', help='Path to source orthomosaic (overrides config)')
    parser.add_argument('target', nargs='?', help='Path to target basemap (overrides config)')
    parser.add_argument('output', nargs='?', help='Output directory (overrides config)')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--scales', nargs='+', type=float,
                       help='Resolution scales (overrides config)')
    parser.add_argument('--algorithms', nargs='+', type=str,
                       help='Transform algorithms for each scale (overrides config). Must match number of scales.')
    parser.add_argument('--matcher', choices=['lightglue', 'sift', 'orb', 'patch_ncc'],
                       help='Matching method (overrides config)')
    parser.add_argument('--debug', choices=['none', 'intermediate', 'high'],
                       default=DEFAULT_DEBUG_LEVEL, help='Debug level: none (log + final only), intermediate (+ intermediate files), high (+ all debug files)')
    parser.add_argument('--get-basemap', type=str, choices=['esri', 'esri_world_imagery', 'openstreetmap', 'google_satellite', 'google_hybrid'],
                       help='Download basemap from specified source (requires --basemap-area)')
    parser.add_argument('--basemap-area', type=str,
                       help='Area for basemap download: bounding box as "min_lat,min_lon,max_lat,max_lon" or path to file with H3 cells (one per line)')
    parser.add_argument('--basemap-zoom', type=int,
                       help='Zoom level for basemap download (auto-calculated if not specified)')
    parser.add_argument('--basemap-resolution', type=float,
                       help='Target resolution in meters per pixel for basemap download (alternative to --basemap-zoom)')
    parser.add_argument('--gcp-analysis', type=str,
                       help='Path to GCP file (CSV or KMZ) for GCP analysis. Extracts 300x300 pixel patches from registered orthomosaic centered at each GCP location.')
    parser.add_argument('--gcp-evaluation', type=str,
                       help='Path to GCP file (CSV or KMZ) for GCP evaluation. Evaluates registration quality at the second-to-last scale using evaluate_gcps.py.')
    
    args = parser.parse_args()
    
    # Load configuration from file if provided
    config_dict = {}
    if args.config:
        config_dict = load_config(args.config)
        logging.info(f"Loaded configuration from: {args.config}")
    
    # Map config fields to arguments (with CLI overrides taking precedence)
    source_path = args.source or config_dict.get('source_path')
    target_path = args.target or config_dict.get('target_path')
    output_dir = args.output or config_dict.get('output_dir', DEFAULT_OUTPUT_DIR)
    
    # Get debug level from CLI or config (CLI takes precedence)
    debug_level = args.debug if args.debug else config_dict.get('debug_level', DEFAULT_DEBUG_LEVEL)
    
    # Handle basemap download if requested
    downloaded_bbox = None
    if args.get_basemap:
        if not args.basemap_area:
            parser.error('--get-basemap requires --basemap-area to be specified')
        
        # Warn if target is also provided
        if target_path:
            logging.warning(f'Both --get-basemap and target basemap provided. Using downloaded basemap (ignoring target: {target_path})')
        
        from basemap_downloader import (
            download_basemap, h3_cells_to_bbox, load_h3_cells_from_file,
            parse_bbox_string
        )
        
        # Determine bounding box
        basemap_area_path = Path(args.basemap_area)
        if basemap_area_path.exists() and basemap_area_path.is_file():
            # Assume it's an H3 cells file
            try:
                h3_cells = load_h3_cells_from_file(basemap_area_path)
                logging.info(f"Loaded {len(h3_cells)} H3 cells from {basemap_area_path}")
                bbox = h3_cells_to_bbox(h3_cells)
                downloaded_bbox = bbox
            except ImportError:
                # h3 not installed, try parsing as bbox string
                try:
                    bbox = parse_bbox_string(basemap_area_path.read_text().strip())
                    downloaded_bbox = bbox
                except Exception as e2:
                    parser.error(f'Failed to parse --basemap-area. H3 package not installed. Install with: pip install h3. Or provide as bbox string: {e2}')
            except Exception as e:
                # If H3 loading fails, try parsing as bbox string
                try:
                    bbox = parse_bbox_string(basemap_area_path.read_text().strip())
                    downloaded_bbox = bbox
                except Exception as e2:
                    parser.error(f'Failed to parse --basemap-area as H3 cells file or bbox: {e}, {e2}')
        else:
            # Try parsing as bbox string
            try:
                bbox = parse_bbox_string(args.basemap_area)
                downloaded_bbox = bbox
            except Exception as e:
                parser.error(f'Failed to parse --basemap-area as bounding box: {e}. Provide as "min_lat,min_lon,max_lat,max_lon" or path to H3 cells file.')
        
        logging.info(f"Bounding box for basemap download: {bbox}")
        
        # Determine output path for downloaded basemap
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        basemap_source_name = args.get_basemap.replace('_', '_').lower()
        downloaded_basemap_path = output_dir_path / f'downloaded_basemap_{basemap_source_name}.tif'
        
        # Check if basemap already exists
        if downloaded_basemap_path.exists():
            logging.info(f"✓ Basemap already exists at: {downloaded_basemap_path}")
            target_path = str(downloaded_basemap_path)
        else:
            # Download basemap
            logging.info(f"Downloading basemap from {args.get_basemap}...")
            try:
                downloaded_basemap = download_basemap(
                    bbox=bbox,
                    output_path=str(downloaded_basemap_path),
                    source=args.get_basemap,
                    zoom=args.basemap_zoom,
                    target_resolution=args.basemap_resolution
                )
                logging.info(f"✓ Basemap downloaded to: {downloaded_basemap}")
                target_path = downloaded_basemap
            except Exception as e:
                parser.error(f'Failed to download basemap: {e}')
    
    # Validate required parameters
    if not source_path:
        parser.error('source_path is required. Provide it via --config file, as positional argument, or use --get-basemap with --basemap-area.')
    if not target_path:
        parser.error('target_path is required. Provide it via --config file, as positional argument, or use --get-basemap with --basemap-area.')
    
    # Get scales and algorithms from config or CLI
    if args.scales:
        scales = args.scales
    elif 'hierarchical_scales' in config_dict:
        scales = config_dict['hierarchical_scales']
    else:
        scales = DEFAULT_SCALES.copy()
    
    if args.algorithms:
        algorithms = args.algorithms
    elif 'algorithms' in config_dict:
        algorithms = config_dict['algorithms']
    else:
        # Use default algorithms, extending if needed
        if len(scales) == len(DEFAULT_SCALES) and scales == DEFAULT_SCALES:
            algorithms = DEFAULT_ALGORITHMS.copy()
        else:
            # Default: first two scales get 'shift', rest get 'homography'
            algorithms = ['shift'] * min(2, len(scales)) + ['homography'] * max(0, len(scales) - 2)
            # Ensure we have enough algorithms
            while len(algorithms) < len(scales):
                algorithms.append('homography')
    
    # Get matcher from config or CLI
    if args.matcher:
        matcher = args.matcher
    elif 'method' in config_dict:
        # Map old config 'method' to new 'matcher' choices
        method_map = {
            'lightglue': 'lightglue',
            'sift': 'sift',
            'orb': 'orb',
            'patch_ncc': 'patch_ncc'
        }
        config_method = config_dict['method'].lower()
        matcher = method_map.get(config_method, DEFAULT_MATCHER)
    else:
        matcher = DEFAULT_MATCHER
    
    # Validate that scales and algorithms have the same length
    if len(scales) != len(algorithms):
        parser.error(f'Number of scales ({len(scales)}) must match number of algorithms ({len(algorithms)})')
    
    # Create transform types dict from paired scales and algorithms
    transform_types = {scale: algo for scale, algo in zip(scales, algorithms)}
    
    # Create output directory early to save config
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save effective configuration for reproducibility
    effective_config = {
        'source_path': str(source_path),
        'target_path': str(target_path),
        'output_dir': str(output_dir),
        'hierarchical_scales': scales,
        'algorithms': algorithms,
        'method': matcher,
        'transform_types': {str(k): v for k, v in transform_types.items()},
        'debug_level': debug_level
    }
    
    # Add basemap download info if basemap was downloaded
    if args.get_basemap and downloaded_bbox:
        effective_config['basemap_download'] = {
            'source': args.get_basemap,
            'basemap_area': args.basemap_area,
            'bbox': list(downloaded_bbox),
            'zoom': args.basemap_zoom,
            'target_resolution': args.basemap_resolution,
            'downloaded_path': str(target_path)
        }
    
    # Add any additional config values that might be useful
    if 'ransac_threshold' in config_dict:
        effective_config['ransac_threshold'] = config_dict['ransac_threshold']
    if 'max_features' in config_dict:
        effective_config['max_features'] = config_dict['max_features']
    if 'patch_size' in config_dict:
        effective_config['patch_size'] = config_dict['patch_size']
    
    save_config(effective_config, output_path)
    
    # Run registration
    registration = OrthomosaicRegistration(
        source_path, target_path, output_dir,
        scales=scales,
        matcher=matcher,
        transform_types=transform_types,
        debug_level=debug_level,
        gcp_evaluation_path=args.gcp_evaluation if hasattr(args, 'gcp_evaluation') and args.gcp_evaluation else None
    )
    
    result = registration.register()
    
    if result:
        print(f"\n✓ Registration complete: {result}")
        
        # Run GCP analysis if requested
        if args.gcp_analysis:
            from gcp_analysis import analyze_gcps
            
            gcp_analysis_output_dir = output_path / 'gcp_analysis'
            print(f"\n{'='*80}")
            print("Running GCP Analysis")
            print(f"{'='*80}")
            
            try:
                analyze_gcps(
                    registered_orthomosaic_path=str(result),
                    gcp_file_path=args.gcp_analysis,
                    output_dir=str(gcp_analysis_output_dir),
                    patch_size=300
                )
                print(f"\n✓ GCP analysis complete. Results saved to: {gcp_analysis_output_dir}")
            except Exception as e:
                print(f"\n✗ GCP analysis failed: {e}")
                import traceback
                traceback.print_exc()
                # Don't fail the entire registration if GCP analysis fails
    else:
        print("\n✗ Registration failed")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

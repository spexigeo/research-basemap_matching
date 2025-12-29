#!/usr/bin/env python3
"""Debug script to regenerate histograms and keypoint visualizations."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import rasterio
import argparse
from matching import visualize_matches
from defaults import DEFAULT_SCALES
from gcp_analysis import load_gcps_from_file, lonlat_to_pixel


def overlay_gcps_on_transformed_image(
    transformed_tif_path: Path,
    gcp_file_path: Path,
    output_image_path: Path,
    dot_radius: int = 15
):
    """
    Overlay GCPs as red dots on a transformed orthomosaic image.
    
    Args:
        transformed_tif_path: Path to transformed TIF file (e.g., scale0.500_after_affine.tif)
        gcp_file_path: Path to GCP file (CSV or KMZ)
        output_image_path: Path to save the output image with GCPs overlaid
        dot_radius: Radius of red dots in pixels
    """
    print(f"\n{'='*80}")
    print("Overlaying GCPs on Transformed Image")
    print(f"{'='*80}")
    print(f"Transformed image: {transformed_tif_path}")
    print(f"GCP file: {gcp_file_path}")
    print(f"Output: {output_image_path}")
    
    if not transformed_tif_path.exists():
        print(f"✗ Error: Transformed image not found: {transformed_tif_path}")
        return False
    
    if not gcp_file_path.exists():
        print(f"✗ Error: GCP file not found: {gcp_file_path}")
        return False
    
    # Load GCPs
    try:
        gcps = load_gcps_from_file(str(gcp_file_path))
        print(f"✓ Loaded {len(gcps)} GCPs")
    except Exception as e:
        print(f"✗ Error loading GCPs: {e}")
        return False
    
    # Load transformed image
    with rasterio.open(transformed_tif_path) as src:
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
        
        # Read image data
        if src.count >= 3:
            data = src.read([1, 2, 3])
            # Convert from (bands, height, width) to (height, width, bands)
            rgb = np.moveaxis(data, 0, -1)
            # Normalize to 0-255
            rgb_min, rgb_max = rgb.min(), rgb.max()
            if rgb_max > rgb_min:
                rgb_norm = ((rgb - rgb_min) / (rgb_max - rgb_min) * 255).astype(np.uint8)
            else:
                rgb_norm = np.zeros_like(rgb, dtype=np.uint8)
        else:
            # Single band - convert to RGB
            data = src.read(1)
            dmin, dmax = data.min(), data.max()
            if dmax > dmin:
                gray = ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
            else:
                gray = np.zeros_like(data, dtype=np.uint8)
            rgb_norm = np.stack([gray, gray, gray], axis=-1)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2BGR)
        
        # Convert GCPs to pixel coordinates and draw dots
        gcps_within_bounds = 0
        for gcp in gcps:
            try:
                col, row = lonlat_to_pixel(gcp['lon'], gcp['lat'], transform, crs)
                
                # Check if GCP is within image bounds
                if 0 <= col < width and 0 <= row < height:
                    # Draw red circle (BGR format: red = (0, 0, 255))
                    cv2.circle(image_bgr, 
                              (int(col), int(row)), 
                              dot_radius, 
                              (0, 0, 255),  # Red in BGR
                              -1)  # Filled circle
                    
                    # Draw white border for visibility
                    cv2.circle(image_bgr,
                              (int(col), int(row)),
                              dot_radius + 2,
                              (255, 255, 255),  # White in BGR
                              2)  # Border width
                    
                    gcps_within_bounds += 1
                else:
                    print(f"  Warning: GCP {gcp.get('id', 'unknown')} outside image bounds (col={col:.1f}, row={row:.1f})")
            except Exception as e:
                print(f"  Warning: Failed to convert GCP {gcp.get('id', 'unknown')}: {e}")
        
        print(f"✓ Overlaid {gcps_within_bounds} GCPs within image bounds")
        
        # Convert back to RGB for saving
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Save using PIL for better compatibility
        from PIL import Image
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(image_rgb)
        img.save(output_image_path, 'PNG', compress_level=1)
        print(f"✓ Saved image with GCPs: {output_image_path}")
        
        return True


def regenerate_visualizations(output_dir: Path):
    """Regenerate histograms and keypoint visualizations."""
    matching_dir = output_dir / "matching_and_transformations"
    preprocessing_dir = output_dir / "preprocessing"
    scales = DEFAULT_SCALES.copy()
    
    # Regenerate keypoint visualizations
    print("Regenerating keypoint visualizations...")
    for scale in scales:
        matches_json = matching_dir / f'matches_scale{scale:.3f}.json'
        if not matches_json.exists():
            print(f"  Skipping scale {scale:.3f} - matches JSON not found")
            continue
        
        # Find corresponding source and target overlap images
        source_overlap = preprocessing_dir / f'source_overlap_scale{scale:.3f}.png'
        target_overlap = preprocessing_dir / f'target_overlap_scale{scale:.3f}.png'
        
        # Also check for pre-transformed sources
        if not source_overlap.exists():
            # Try to find pre-transformed versions
            pattern = f'orthomosaic_scale{scale:.3f}_*.png'
            candidates = list(preprocessing_dir.glob(pattern))
            if candidates:
                source_overlap = candidates[0]
                print(f"  Using pre-transformed source: {source_overlap.name}")
        
        if not source_overlap.exists() or not target_overlap.exists():
            print(f"  Skipping scale {scale:.3f} - overlap images not found")
            continue
        
        # Load images
        source_img = cv2.imread(str(source_overlap), cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(str(target_overlap), cv2.IMREAD_GRAYSCALE)
        
        if source_img is None or target_img is None:
            print(f"  Skipping scale {scale:.3f} - failed to load images")
            continue
        
        # Load matches JSON
        with open(matches_json, 'r') as f:
            matches_data = json.load(f)
        
        # Reconstruct matches_result structure for visualize_matches
        # This is a simplified version - we need the actual keypoints and matches
        # For now, skip if we can't reconstruct properly
        print(f"  Note: Keypoint visualization at scale {scale:.3f} requires full match result structure")
        print(f"        This will be handled by the main pipeline")

    # Regenerate histograms
    print("\nRegenerating histograms...")
    for scale in scales:
        matches_json = matching_dir / f'matches_scale{scale:.3f}.json'
        transform_json = matching_dir / f'transform_scale{scale:.3f}.json'
        
        if not matches_json.exists() or not transform_json.exists():
            print(f"  Skipping scale {scale:.3f} - JSON files not found")
            continue
        
        print(f"  Processing scale {scale:.3f}...")
        
        # Load matches JSON
        with open(matches_json, 'r') as f:
            matches_data = json.load(f)
        matches_list = matches_data.get('matches', [])
        
        # Load transform JSON
        with open(transform_json, 'r') as f:
            transform_data = json.load(f)
        
        pixel_resolution = 0.02 / scale
        
        # Extract distances for all matches
        distances_all_m = []
        for m in matches_list:
            dist = m.get('distance', {})
            if 'meters' in dist:
                distances_all_m.append(float(dist['meters']))
            elif 'pixels' in dist:
                distances_all_m.append(float(dist['pixels']) * pixel_resolution)
        
        if len(distances_all_m) == 0:
            print(f"    No distances found in matches JSON")
            continue
        
        distances_all_m = np.array(distances_all_m, dtype=float)
        
        # Get reprojection errors from transform
        reproj_errors = transform_data.get('reproj_errors', [])
        if not reproj_errors:
            print(f"    No reprojection errors in transform JSON")
            continue
        
        reproj_errors = np.array(reproj_errors, dtype=float)
        reproj_errors_m = reproj_errors * pixel_resolution
        
        # Get inlier mask if available
        inliers = transform_data.get('inliers', None)
        if inliers is not None:
            inlier_mask = np.array(inliers, dtype=bool)
            if len(inlier_mask) == len(reproj_errors):
                inlier_errors_m = reproj_errors_m[inlier_mask]
            else:
                inlier_errors_m = reproj_errors_m
        else:
            inlier_errors_m = reproj_errors_m
        
        # Create histogram
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left panel: All matches
        axes[0].hist(distances_all_m, bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Distance (meters)', fontweight='bold')
        axes[0].set_ylabel('Count', fontweight='bold')
        subtitle = f'All matches - Scale {scale:.3f}\n{matches_json.name}'
        axes[0].set_title(subtitle, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        median_all = np.median(distances_all_m)
        mean_all = np.mean(distances_all_m)
        axes[0].axvline(median_all, color='blue', linestyle='--', linewidth=2,
                        label=f'Median: {median_all:.2f}m')
        axes[0].axvline(mean_all, color='green', linestyle='--', linewidth=2,
                        label=f'Mean: {mean_all:.2f}m')
        axes[0].legend()
        
        # Right panel: Inliers only
        axes[1].hist(inlier_errors_m, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1].set_xlabel('Distance (meters)', fontweight='bold')
        axes[1].set_ylabel('Count', fontweight='bold')
        transform_type = transform_data.get('transform_type', 'unknown')
        subtitle_inliers = f'Inliers only ({transform_type})\n{transform_json.name}'
        axes[1].set_title(subtitle_inliers, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        median_inliers = np.median(inlier_errors_m)
        mean_inliers = np.mean(inlier_errors_m)
        axes[1].axvline(median_inliers, color='red', linestyle='--', linewidth=2,
                        label=f'Median: {median_inliers:.2f}m')
        axes[1].axvline(mean_inliers, color='green', linestyle='--', linewidth=2,
                        label=f'Mean: {mean_inliers:.2f}m')
        axes[1].legend()
        
        plt.tight_layout()
        output_path = matching_dir / f'error_histogram_scale{scale:.3f}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Regenerated: {output_path.name}")
        print(f"      All matches: {len(distances_all_m)} points, median={median_all:.2f}m, mean={mean_all:.2f}m")
        print(f"      Inliers: {len(inlier_errors_m)} points, median={median_inliers:.2f}m, mean={mean_inliers:.2f}m")


# Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debug visualizations and GCP overlay')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory (default: outputs)')
    parser.add_argument('--gcp-file', type=str, 
                       help='Path to GCP file (CSV or KMZ) for overlay visualization')
    parser.add_argument('--overlay-only', action='store_true',
                       help='Only create GCP overlay, skip other visualizations')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    intermediate_dir = output_dir / "intermediate"
    scales = DEFAULT_SCALES.copy()
    
    if not args.overlay_only:
        regenerate_visualizations(output_dir)
        print("\nDone!")
    
    if args.gcp_file:
        # Find the highest scale transformed image
        highest_scale = max(scales) if scales else 0.5
        
        # Look for transformed images at highest scale
        transformed_patterns = [
            f'scale{highest_scale:.3f}_after_affine.tif',
            f'scale{highest_scale:.3f}_after_homography.tif',
            f'scale{highest_scale:.3f}_after_shift.tif',
            f'scale{highest_scale:.3f}_after_polynomial_2.tif',
        ]
        
        transformed_tif = None
        for pattern in transformed_patterns:
            candidate = intermediate_dir / pattern
            if candidate.exists():
                transformed_tif = candidate
                print(f"\nFound transformed image: {candidate.name}")
                break
        
        if not transformed_tif:
            # Try to find any transformed image at highest scale
            candidates = list(intermediate_dir.glob(f'scale{highest_scale:.3f}_after_*.tif'))
            if candidates:
                transformed_tif = candidates[0]
                print(f"\nFound transformed image: {transformed_tif.name}")
        
        if transformed_tif:
            output_image_path = output_dir / f'transformed_with_gcps_scale{highest_scale:.3f}.png'
            overlay_gcps_on_transformed_image(
                transformed_tif,
                Path(args.gcp_file),
                output_image_path,
                dot_radius=75
            )
        else:
            print(f"\n✗ No transformed image found at scale {highest_scale:.3f}")
            print(f"  Looked in: {intermediate_dir}")
            print(f"  Patterns: {transformed_patterns}")

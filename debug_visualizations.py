#!/usr/bin/env python3
"""Debug script to regenerate histograms and keypoint visualizations."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from matching import visualize_matches
from constants import DEFAULT_SCALES

output_dir = Path("outputs/test_registration")
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

print("\nDone!")






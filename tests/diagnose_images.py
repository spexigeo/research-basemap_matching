#!/usr/bin/env python3
"""
Diagnostic tool to check if orthomosaic and basemap actually overlap
and are in compatible coordinate systems.
"""

import rasterio
from rasterio.warp import transform_bounds
import numpy as np
from shapely.geometry import box
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def diagnose_geospatial_alignment(source_path, target_path):
    """
    Comprehensive diagnostic of two geospatial images.
    """
    print("=" * 80)
    print("GEOSPATIAL ALIGNMENT DIAGNOSTIC")
    print("=" * 80)

    # Read metadata
    with rasterio.open(source_path) as src:
        src_crs = src.crs
        src_bounds = src.bounds
        src_shape = (src.height, src.width)
        src_transform = src.transform
        src_res = (src_transform.a, abs(src_transform.e))

    with rasterio.open(target_path) as tgt:
        tgt_crs = tgt.crs
        tgt_bounds = tgt.bounds
        tgt_shape = (tgt.height, tgt.width)
        tgt_transform = tgt.transform
        tgt_res = (tgt_transform.a, abs(tgt_transform.e))

    print("\n" + "=" * 80)
    print("SOURCE (Orthomosaic)")
    print("=" * 80)
    print(f"Path: {source_path}")
    print(f"CRS: {src_crs}")
    print(f"Shape: {src_shape[1]} x {src_shape[0]} pixels")
    print(f"Resolution: {src_res[0]:.6f} x {src_res[1]:.6f} m/pixel")
    print(f"Bounds (minx, miny, maxx, maxy):")
    print(f"  {src_bounds.left:.2f}, {src_bounds.bottom:.2f}, {src_bounds.right:.2f}, {src_bounds.top:.2f}")
    print(f"Width:  {src_bounds.right - src_bounds.left:.2f} m")
    print(f"Height: {src_bounds.top - src_bounds.bottom:.2f} m")

    print("\n" + "=" * 80)
    print("TARGET (Basemap)")
    print("=" * 80)
    print(f"Path: {target_path}")
    print(f"CRS: {tgt_crs}")
    print(f"Shape: {tgt_shape[1]} x {tgt_shape[0]} pixels")
    print(f"Resolution: {tgt_res[0]:.6f} x {tgt_res[1]:.6f} m/pixel")
    print(f"Bounds (minx, miny, maxx, maxy):")
    print(f"  {tgt_bounds.left:.2f}, {tgt_bounds.bottom:.2f}, {tgt_bounds.right:.2f}, {tgt_bounds.top:.2f}")
    print(f"Width:  {tgt_bounds.right - tgt_bounds.left:.2f} m")
    print(f"Height: {tgt_bounds.top - tgt_bounds.bottom:.2f} m")

    # Check CRS compatibility
    print("\n" + "=" * 80)
    print("CRS COMPATIBILITY")
    print("=" * 80)

    if src_crs != tgt_crs:
        print(f"❌ CRS MISMATCH!")
        print(f"   Source: {src_crs}")
        print(f"   Target: {tgt_crs}")
        print(f"\n   Transforming source bounds to target CRS...")

        # Transform source bounds to target CRS
        src_bounds_in_tgt_crs = transform_bounds(src_crs, tgt_crs, *src_bounds)
        print(f"   Source bounds in target CRS:")
        print(f"   {src_bounds_in_tgt_crs[0]:.2f}, {src_bounds_in_tgt_crs[1]:.2f}, "
              f"{src_bounds_in_tgt_crs[2]:.2f}, {src_bounds_in_tgt_crs[3]:.2f}")

        # Use transformed bounds for overlap check
        src_bounds_check = src_bounds_in_tgt_crs
    else:
        print(f"✓ CRS MATCH: {src_crs}")
        src_bounds_check = src_bounds

    # Check overlap
    print("\n" + "=" * 80)
    print("OVERLAP ANALYSIS")
    print("=" * 80)

    src_box = box(src_bounds_check[0], src_bounds_check[1],
                  src_bounds_check[2], src_bounds_check[3])
    tgt_box = box(tgt_bounds.left, tgt_bounds.bottom,
                  tgt_bounds.right, tgt_bounds.top)

    intersection = src_box.intersection(tgt_box)

    if intersection.is_empty:
        print("❌ NO OVERLAP - Images are in completely different locations!")
        print("\nDistance between centers:")
        src_center = ((src_bounds_check[0] + src_bounds_check[2]) / 2,
                      (src_bounds_check[1] + src_bounds_check[3]) / 2)
        tgt_center = ((tgt_bounds.left + tgt_bounds.right) / 2,
                      (tgt_bounds.bottom + tgt_bounds.top) / 2)
        dist = np.sqrt((src_center[0] - tgt_center[0]) ** 2 +
                       (src_center[1] - tgt_center[1]) ** 2)
        print(f"  {dist:.2f} meters apart")

    else:
        overlap_area = intersection.area
        src_area = src_box.area
        tgt_area = tgt_box.area

        overlap_pct_src = (overlap_area / src_area) * 100
        overlap_pct_tgt = (overlap_area / tgt_area) * 100

        print(f"✓ OVERLAP EXISTS")
        print(f"  Overlap area: {overlap_area:.2f} m²")
        print(f"  Source coverage: {overlap_pct_src:.1f}%")
        print(f"  Target coverage: {overlap_pct_tgt:.1f}%")

        if overlap_pct_src < 10 or overlap_pct_tgt < 10:
            print(f"\n⚠️  WARNING: Overlap is very small (<10%)")
            print(f"     Registration may fail or be inaccurate")

    # Resolution check
    print("\n" + "=" * 80)
    print("RESOLUTION ANALYSIS")
    print("=" * 80)

    res_ratio = src_res[0] / tgt_res[0]
    print(f"Resolution ratio (source/target): {res_ratio:.2f}x")

    if res_ratio > 10:
        print(f"⚠️  Source is MUCH higher resolution ({res_ratio:.1f}x)")
        print(f"   Consider downsampling source or using coarser matching scales")
    elif res_ratio < 0.1:
        print(f"⚠️  Target is MUCH higher resolution ({1 / res_ratio:.1f}x)")
        print(f"   Consider downsampling target or adjusting scales")
    else:
        print(f"✓ Resolution ratio is reasonable")

    # Recommended scales
    print("\n" + "=" * 80)
    print("RECOMMENDED SETTINGS")
    print("=" * 80)

    if not intersection.is_empty:
        # Calculate what scale would make the overlap ~1000x1000 pixels in target space
        overlap_width = intersection.bounds[2] - intersection.bounds[0]
        overlap_height = intersection.bounds[3] - intersection.bounds[1]

        target_pixels_width = overlap_width / tgt_res[0]
        target_pixels_height = overlap_height / tgt_res[1]

        print(f"Overlap dimensions in target pixels: {target_pixels_width:.0f} x {target_pixels_height:.0f}")

        # Suggest scales that give reasonable sizes
        scales = []
        for target_size in [500, 1000, 2000]:
            scale = target_size / max(target_pixels_width, target_pixels_height)
            if 0.01 <= scale <= 1.0:
                scales.append(scale)

        if scales:
            print(f"\nSuggested hierarchical_scales: {sorted(scales)}")
        else:
            print(f"\n⚠️  Images are very large or very small - manual scale adjustment needed")

    # Visualize
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot target bounds (blue)
    tgt_rect = patches.Rectangle(
        (tgt_bounds.left, tgt_bounds.bottom),
        tgt_bounds.right - tgt_bounds.left,
        tgt_bounds.top - tgt_bounds.bottom,
        linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3,
        label='Target (Basemap)'
    )
    ax.add_patch(tgt_rect)

    # Plot source bounds (red) - use transformed bounds if different CRS
    if src_crs != tgt_crs:
        src_rect = patches.Rectangle(
            (src_bounds_check[0], src_bounds_check[1]),
            src_bounds_check[2] - src_bounds_check[0],
            src_bounds_check[3] - src_bounds_check[1],
            linewidth=2, edgecolor='red', facecolor='red', alpha=0.3,
            label='Source (Orthomosaic)'
        )
    else:
        src_rect = patches.Rectangle(
            (src_bounds.left, src_bounds.bottom),
            src_bounds.right - src_bounds.left,
            src_bounds.top - src_bounds.bottom,
            linewidth=2, edgecolor='red', facecolor='red', alpha=0.3,
            label='Source (Orthomosaic)'
        )
    ax.add_patch(src_rect)

    # Plot overlap if exists (green)
    if not intersection.is_empty:
        int_bounds = intersection.bounds
        overlap_rect = patches.Rectangle(
            (int_bounds[0], int_bounds[1]),
            int_bounds[2] - int_bounds[0],
            int_bounds[3] - int_bounds[1],
            linewidth=3, edgecolor='green', facecolor='green', alpha=0.5,
            label='Overlap'
        )
        ax.add_patch(overlap_rect)

    # Set limits with some padding
    all_bounds = [
        tgt_bounds.left, tgt_bounds.bottom, tgt_bounds.right, tgt_bounds.top,
        src_bounds_check[0], src_bounds_check[1], src_bounds_check[2], src_bounds_check[3]
    ]

    x_min, x_max = min(all_bounds[0], all_bounds[4]), max(all_bounds[2], all_bounds[6])
    y_min, y_max = min(all_bounds[1], all_bounds[5]), max(all_bounds[3], all_bounds[7])

    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = 0.1

    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)

    ax.set_xlabel('Easting (m)' if 'UTM' in str(tgt_crs) else 'X', fontsize=12)
    ax.set_ylabel('Northing (m)' if 'UTM' in str(tgt_crs) else 'Y', fontsize=12)
    ax.set_title('Geospatial Extent Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('geospatial_diagnostic.png', dpi=150, bbox_inches='tight')
    print("Saved visualization: geospatial_diagnostic.png")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    issues = []
    if src_crs != tgt_crs:
        issues.append("CRS mismatch - images need reprojection")
    if intersection.is_empty:
        issues.append("NO OVERLAP - images are in different locations")
    elif overlap_pct_src < 10 or overlap_pct_tgt < 10:
        issues.append("Very small overlap - registration may be unreliable")
    if res_ratio > 10 or res_ratio < 0.1:
        issues.append(f"Large resolution difference ({res_ratio:.1f}x)")

    if issues:
        print("❌ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print("\nREGISTRATION WILL LIKELY FAIL until these are resolved.")
    else:
        print("✓ Images appear compatible for registration")
        print("  - Same CRS")
        print("  - Good overlap")
        print("  - Reasonable resolution ratio")

    print("=" * 80)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print("Usage: python diagnose_images.py <source_orthomosaic> <target_basemap>")
        print("\nExample:")
        print("  python diagnose_images.py orthomosaic_utm.tif basemap_utm.tif")
        sys.exit(1)

    source = sys.argv[1]
    target = sys.argv[2]

    diagnose_geospatial_alignment(source, target)
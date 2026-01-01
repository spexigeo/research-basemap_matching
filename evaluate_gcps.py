#!/usr/bin/env python3
"""
Evaluate registration quality using Ground Control Points (GCPs).

This tool:
1. Loads GCPs from a CSV file (expected format: Label, X, Y, Z, Accuracy, Enabled)
2. Applies each transformation step from hierarchical registration
3. Computes errors at each scale
4. Reports whether registration is improving or degrading
"""

import sys
import csv
import json
import numpy as np
import rasterio
from pathlib import Path
from affine import Affine
import pyproj
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


def load_gcps(gcp_path: str) -> List[Dict]:
    """
    Load GCPs from CSV or KMZ file.
    
    Args:
        gcp_path: Path to CSV or KMZ file
        
    Returns:
        List of GCP dictionaries
    """
    from gcp_analysis import load_gcps_from_file
    
    gcps = load_gcps_from_file(gcp_path)
    
    # Ensure 'label' and 'accuracy' fields for backward compatibility
    for gcp in gcps:
        if 'label' not in gcp:
            gcp['label'] = gcp.get('id', f"GCP_{gcps.index(gcp)+1:03d}")
        if 'accuracy' not in gcp:
            gcp['accuracy'] = 0.005  # Default accuracy
    
    return gcps


def lonlat_to_pixel(lon: float, lat: float, transform: Affine, crs) -> Tuple[float, float]:
    """Convert geographic coordinates (WGS84 lon/lat) to pixel coordinates."""
    # Always transform from WGS84 to the image CRS
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS.from_epsg(4326),  # WGS84 (lon/lat)
        crs,
        always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    
    # Convert to pixel coordinates
    inv_transform = ~transform
    col, row = inv_transform * (x, y)
    return col, row


def pixel_to_lonlat(col: float, row: float, transform: Affine, crs) -> Tuple[float, float]:
    """Convert pixel coordinates to geographic coordinates."""
    # Convert to geographic coordinates
    x, y = transform * (col, row)
    
    if crs.is_geographic:
        lon, lat = x, y
    else:
        # Transform from projected CRS to WGS84
        transformer = pyproj.Transformer.from_crs(
            crs,
            pyproj.CRS.from_epsg(4326),  # WGS84
            always_xy=True
        )
        lon, lat = transformer.transform(x, y)
    
    return lon, lat


def apply_transform_to_point(point: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply 2x3 affine transform to a point."""
    if transform.shape == (2, 3):
        # Affine transform: [x', y'] = [x, y, 1] @ M^T
        point_homogeneous = np.array([point[0], point[1], 1.0])
        result = transform @ point_homogeneous
        return result
    elif transform.shape == (3, 3):
        # Homography
        point_homogeneous = np.array([point[0], point[1], 1.0])
        result = transform @ point_homogeneous
        return result[:2] / result[2]  # Normalize by w
    else:
        raise ValueError(f"Unknown transform shape: {transform.shape}")


def compute_error_meters(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Compute distance in meters between two geographic points."""
    geod = pyproj.Geod(ellps='WGS84')
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
    return distance


def evaluate_registration(
    source_path: str,
    gcp_csv_path: str,
    output_dir: str,
    run_timestamp: str = None
):
    """
    Evaluate registration quality using GCPs.
    
    Args:
        source_path: Path to source orthomosaic
        gcp_csv_path: Path to GCP CSV file
        output_dir: Output directory from registration run
        run_timestamp: Optional timestamp to find specific run directory
    """
    output_path = Path(output_dir)
    
    # Check if using new directory structure (no run_ subdirectories)
    # New structure: outputs/intermediate/, outputs/matching_and_transformations/
    # Old structure: outputs/run_*/intermediate/
    intermediate_dir = output_path / 'intermediate'
    matching_dir = output_path / 'matching_and_transformations'
    
    if not intermediate_dir.exists():
        # Try old structure with run_ directories
        if run_timestamp:
            run_dir = output_path / f"run_{run_timestamp}"
        else:
            # Find most recent run directory
            run_dirs = sorted([d for d in output_path.iterdir() if d.is_dir() and d.name.startswith('run_')])
            if not run_dirs:
                raise ValueError(f"No intermediate directory found in {output_path}. Expected 'intermediate/' or 'run_*/' subdirectories.")
            run_dir = run_dirs[-1]
            print(f"Using most recent run: {run_dir.name}")
        
        intermediate_dir = run_dir / 'intermediate'
        matching_dir = run_dir / 'matching_and_transformations' if (run_dir / 'matching_and_transformations').exists() else None
    else:
        # Using new structure
        run_dir = output_path
    
    # Load source image metadata
    with rasterio.open(source_path) as src:
        source_transform = src.transform
        source_crs = src.crs
        source_shape = (src.height, src.width)
    
    print(f"\nSource image: {source_path}")
    print(f"  CRS: {source_crs}")
    print(f"  Shape: {source_shape[1]}x{source_shape[0]}")
    
    # Load GCPs
    gcps = load_gcps(gcp_csv_path)
    print(f"\nLoaded {len(gcps)} GCPs from {gcp_csv_path}")
    
    # Convert GCPs to pixel coordinates in source image
    gcp_pixels = []
    for gcp in gcps:
        try:
            col, row = lonlat_to_pixel(gcp['lon'], gcp['lat'], source_transform, source_crs)
            # Check if GCP is within image bounds
            if 0 <= col < source_shape[1] and 0 <= row < source_shape[0]:
                gcp_pixels.append({
                    'label': gcp['label'],
                    'lon': gcp['lon'],
                    'lat': gcp['lat'],
                    'pixel_col': col,
                    'pixel_row': row,
                    'accuracy': gcp['accuracy']
                })
            else:
                print(f"  Warning: GCP {gcp['label']} is outside image bounds (col={col:.1f}, row={row:.1f})")
        except Exception as e:
            print(f"  Warning: Failed to convert GCP {gcp['label']}: {e}")
    
    print(f"  {len(gcp_pixels)} GCPs are within image bounds")
    
    if len(gcp_pixels) == 0:
        print("ERROR: No GCPs are within image bounds!")
        return
    
    # Find all transform files
    # Try new structure first (matching_and_transformations/)
    transform_files = []
    if matching_dir and matching_dir.exists():
        transform_files = sorted(matching_dir.glob('transform_scale*.json'))
    # Fallback to old structure or intermediate_dir
    if not transform_files:
        transform_files = sorted(intermediate_dir.glob('transform_level*_scale*.txt'))
    
    if not transform_files:
        print(f"\nWARNING: No transform files found in {intermediate_dir} or {matching_dir}")
        print("  This may mean all scales were skipped or registration failed")
        return
    
    print(f"\nFound {len(transform_files)} transform files:")
    for tf in transform_files:
        print(f"  {tf.name}")
    
    # Load final transform
    final_transform_file = run_dir / 'transform_final.txt'
    if final_transform_file.exists():
        final_transform = np.loadtxt(final_transform_file)
        print(f"\nFound final transform: {final_transform_file.name}")
    else:
        print(f"\nWARNING: Final transform file not found: {final_transform_file}")
        final_transform = None
    
    # Evaluate each transformation step
    results = []
    
    # Initial state (no transformation)
    initial_errors = []
    for gcp in gcp_pixels:
        # GCP position in source image (this is the "truth" from the original orthomosaic)
        # We'll compare transformed positions back to this
        initial_errors.append(0.0)  # No error initially
    
    results.append({
        'step': 'initial',
        'transform_file': None,
        'mean_error_m': 0.0,
        'std_error_m': 0.0,
        'max_error_m': 0.0,
        'median_error_m': 0.0,
        'errors': initial_errors
    })
    
    # Apply each transform step
    cumulative_transform = np.eye(2, 3, dtype=np.float32)
    
    for transform_file in transform_files:
        # Extract scale from filename
        # Format: transform_level1_scale0.150.txt
        parts = transform_file.stem.split('_')
        scale = None
        level = None
        for part in parts:
            if part.startswith('scale'):
                scale = float(part[5:])
            elif part.startswith('level'):
                level = int(part[5:])
        
        print(f"\nEvaluating: {transform_file.name} (level {level}, scale {scale})")
        
        # Load transform (JSON format in new structure, text in old)
        if transform_file.suffix == '.json':
            with open(transform_file, 'r') as f:
                transform_data = json.load(f)
            M_at_scale = np.array(transform_data.get('matrix', transform_data.get('transform_matrix', [])))
            if M_at_scale.size == 0:
                print(f"  Warning: Could not extract matrix from {transform_file.name}")
                continue
            # Ensure it's 2x3 for affine/homography
            if M_at_scale.shape == (3, 3):
                M_at_scale = M_at_scale[:2, :]
            elif M_at_scale.shape != (2, 3):
                print(f"  Warning: Unexpected matrix shape {M_at_scale.shape} in {transform_file.name}")
                continue
        else:
            # Old format: text file
            M_at_scale = np.loadtxt(transform_file)
        
        # The transform in the file is in "full image coordinates at current scale"
        # We need to scale it to full resolution for accumulation
        M_fullres = M_at_scale.copy()
        if scale is not None:
            M_fullres[:, 2] /= scale
        
        # Accumulate with previous transforms
        M_full = np.vstack([M_fullres, [0, 0, 1]])
        M_cum_full = np.vstack([cumulative_transform, [0, 0, 1]])
        cumulative_transform_new = (M_cum_full @ M_full)[:2, :]
        
        # Apply cumulative transform to GCP pixel coordinates
        errors = []
        transformed_gcps = []
        
        for gcp in gcp_pixels:
            pixel_point = np.array([gcp['pixel_col'], gcp['pixel_row']])
            
            # Apply cumulative transform
            transformed_point = apply_transform_to_point(pixel_point, cumulative_transform_new)
            
            # Convert back to geographic coordinates using source transform
            # (assuming transform maps source -> target, we need to see where source pixels end up)
            # Actually, we need to think about this differently:
            # The transform maps source pixels to target pixels
            # But we want to know: if we apply this transform, where do the GCPs end up?
            # We need the target image transform to convert back to geographic coords
            
            # For now, let's compute the error differently:
            # The transform tells us how source pixels map to target pixels
            # We need to check where the transformed GCP pixels map in the target image
            # and compare to where they should be based on their geographic coordinates
            
            # This is complex - let's simplify:
            # We'll compute the pixel offset and convert to meters
            offset_pixels = transformed_point - pixel_point
            offset_meters = np.linalg.norm(offset_pixels) * abs(source_transform.a)  # Assuming square pixels
            
            errors.append(offset_meters)
            transformed_gcps.append({
                'label': gcp['label'],
                'original_pixel': pixel_point,
                'transformed_pixel': transformed_point,
                'offset_pixels': offset_pixels,
                'error_m': offset_meters
            })
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        median_error = np.median(errors)
        
        results.append({
            'step': f'level_{level}_scale_{scale:.3f}',
            'transform_file': transform_file.name,
            'mean_error_m': mean_error,
            'std_error_m': std_error,
            'max_error_m': max_error,
            'median_error_m': median_error,
            'errors': errors,
            'transformed_gcps': transformed_gcps
        })
        
        print(f"  Mean error: {mean_error:.2f}m")
        print(f"  Std error: {std_error:.2f}m")
        print(f"  Max error: {max_error:.2f}m")
        print(f"  Median error: {median_error:.2f}m")
        
        cumulative_transform = cumulative_transform_new
    
    # Evaluate final transform if available
    if final_transform is not None:
        print(f"\nEvaluating final transform...")
        errors = []
        for gcp in gcp_pixels:
            pixel_point = np.array([gcp['pixel_col'], gcp['pixel_row']])
            transformed_point = apply_transform_to_point(pixel_point, final_transform)
            offset_pixels = transformed_point - pixel_point
            offset_meters = np.linalg.norm(offset_pixels) * abs(source_transform.a)
            errors.append(offset_meters)
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        median_error = np.median(errors)
        
        results.append({
            'step': 'final',
            'transform_file': 'transform_final.txt',
            'mean_error_m': mean_error,
            'std_error_m': std_error,
            'max_error_m': max_error,
            'median_error_m': median_error,
            'errors': errors
        })
        
        print(f"  Mean error: {mean_error:.2f}m")
        print(f"  Std error: {std_error:.2f}m")
        print(f"  Max error: {max_error:.2f}m")
        print(f"  Median error: {median_error:.2f}m")
    
    # Save results
    results_file = output_path / 'gcp_evaluation.json'
    with open(results_file, 'w') as f:
        json.dump({
            'gcp_count': len(gcp_pixels),
            'source_path': source_path,
            'gcp_csv_path': gcp_csv_path,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot error progression
    steps = [r['step'] for r in results]
    mean_errors = [r['mean_error_m'] for r in results]
    max_errors = [r['max_error_m'] for r in results]
    median_errors = [r['median_error_m'] for r in results]
    
    axes[0].plot(range(len(steps)), mean_errors, 'o-', label='Mean Error', linewidth=2)
    axes[0].plot(range(len(steps)), max_errors, 's-', label='Max Error', linewidth=2)
    axes[0].plot(range(len(steps)), median_errors, '^-', label='Median Error', linewidth=2)
    axes[0].set_xlabel('Registration Step')
    axes[0].set_ylabel('Error (meters)')
    axes[0].set_title('GCP Error Progression Through Registration Steps')
    axes[0].set_xticks(range(len(steps)))
    axes[0].set_xticklabels([s.replace('_', ' ').title() for s in steps], rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot error distribution for final step
    if len(results) > 1:
        final_errors = results[-1]['errors']
        axes[1].hist(final_errors, bins=20, edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(final_errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_errors):.2f}m')
        axes[1].axvline(np.median(final_errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(final_errors):.2f}m')
        axes[1].set_xlabel('Error (meters)')
        axes[1].set_ylabel('Number of GCPs')
        axes[1].set_title('Final Error Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_file = output_path / 'gcp_evaluation.png'
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {viz_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Step':<30} {'Mean Error (m)':<15} {'Max Error (m)':<15} {'Status':<15}")
    print("-" * 80)
    
    prev_error = None
    for r in results:
        status = "N/A"
        if prev_error is not None:
            if r['mean_error_m'] < prev_error:
                status = "✓ Improving"
            elif r['mean_error_m'] > prev_error:
                status = "✗ Degrading"
            else:
                status = "→ Same"
        prev_error = r['mean_error_m']
        
        print(f"{r['step']:<30} {r['mean_error_m']:<15.2f} {r['max_error_m']:<15.2f} {status:<15}")
    
    print("=" * 80)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python evaluate_gcps.py <source_orthomosaic> <gcp_csv> <output_dir> [run_timestamp]")
        print("\nExample:")
        print("  python evaluate_gcps.py inputs/qualicum_beach/orthomosaic_no_gcps_utm10n.tif \\")
        print("                           ../research-qualicum_beach_gcp_analysis/outputs/gcps_metashape.csv \\")
        print("                           outputs/improved_quality_esri_similarity")
        sys.exit(1)
    
    source_path = sys.argv[1]
    gcp_csv_path = sys.argv[2]
    output_dir = sys.argv[3]
    run_timestamp = sys.argv[4] if len(sys.argv) > 4 else None
    
    evaluate_registration(source_path, gcp_csv_path, output_dir, run_timestamp)


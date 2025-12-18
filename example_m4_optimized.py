#!/usr/bin/env python3
"""
Improved example script with new registration features.

NEW FEATURES:
1. Pre-alignment at each hierarchical scale
2. Adaptive search regions
3. Transform type selection (similarity/affine/homography)
4. Iterative refinement at finest scale
5. Better outlier filtering
"""

import sys
from pathlib import Path
import json
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# IMPORTANT: Import and setup M4 optimizations FIRST
from performance_optimizations import (
    setup_m4_optimizations,
    get_preset_config,
    estimate_processing_time,
    format_time
)

# Setup M4 optimizations before importing other modules
print("Setting up M4 Pro optimizations...")
m4_config = setup_m4_optimizations()
print(f"✓ Configured for {m4_config['performance_cores']} performance cores\n")

# Now import the main registration module
from main import main as run_registration


def run_improved_registration(preset_name='quality', target_basemap='esri',
                              transform_type='similarity'):
    """
    Run improved registration with new features.

    Args:
        preset_name: 'fast', 'balanced', 'quality', or 'maximum'
        target_basemap: 'esri' or 'google'
        transform_type: 'similarity', 'affine', or 'homography'
    """
    print("\n" + "=" * 80)
    print(f"IMPROVED REGISTRATION - Preset: {preset_name.upper()}, Transform: {transform_type}")
    print("=" * 80)

    # Base configuration
    config = {
        "source_path": "inputs/orthomosaic_no_gcps_utm10n.tif",
        "target_path": "inputs/qualicum_beach_basemap_esri_utm10n.tif",
        "method": "patch_ncc",

        # Hierarchical scales - now with pre-alignment at each level
        "hierarchical_scales": [0.06, 0.12, 0.24],

        # Patch matching
        "patch_size": 384,
        "patch_grid_spacing": 192,
        "ncc_threshold": 0.35,  # Lower threshold to get more matches

        # NEW: Transform type
        "transform_type": transform_type,  # 'similarity', 'affine', or 'homography'

        # NEW: Iterative refinement
        "enable_refinement": True,
        "refinement_iterations": 2,

        # NEW: Adaptive search
        "adaptive_search": True,
        "min_search_margin": 50,
        "max_search_margin": 200,

        # RANSAC
        "ransac_threshold": 5.0,

        # Preprocessing
        "preprocess_method": "edges",

        # Output
        "output_dir": f"outputs/improved_{preset_name}_{target_basemap}_{transform_type}",
        "verbose": True,
        "visualization": {
            "create_match_visualizations": True,
            "create_difference_maps": True,
            "create_png_overviews": True
        },
        "output_format": {
            "compression": "JPEG",
            "jpeg_quality": 90,
            "tiled": True,
            "blocksize": 1024
        },
        "reproject_to_metric": True
    }

    # Apply preset optimizations
    preset = get_preset_config(preset_name)
    config.update(preset)

    # Save config
    config_path = f"config_improved_{preset_name}_{target_basemap}_{transform_type}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nConfiguration: {config_path}")
    print(f"Method: {config['method']}")
    print(f"Transform type: {config['transform_type']}")
    print(f"Scales: {config['hierarchical_scales']}")
    print(f"Adaptive search: {config['adaptive_search']}")
    print(f"Refinement: {config['enable_refinement']} ({config['refinement_iterations']} iterations)")
    print(f"Max features: {config['max_features']}")
    print(f"Patch size: {config['patch_size']}")

    # Estimate processing time
    source_shape = (46405, 54537)  # Your orthomosaic size
    estimated_time = estimate_processing_time(
        source_shape,
        method=config['method'],
        scales=config['hierarchical_scales']
    )
    print(f"\nEstimated processing time: ~{format_time(estimated_time)}")
    print("Starting registration...\n")

    # Run registration
    start_time = time.time()

    sys.argv = ['example_improved.py', '--config', config_path]
    try:
        run_registration()

        elapsed = time.time() - start_time
        print(f"\n✓ Actual processing time: {format_time(int(elapsed))}")

    except Exception as e:
        print(f"\n✗ Registration failed: {e}")
        elapsed = time.time() - start_time
        print(f"Time before failure: {format_time(int(elapsed))}")
        raise


def compare_transform_types(preset='balanced'):
    """
    Compare similarity vs affine vs homography transforms.
    """
    print("\n" + "=" * 80)
    print("TRANSFORM TYPE COMPARISON")
    print("=" * 80)
    print("Will test similarity, affine, and homography transforms")
    print(f"Using preset: {preset}")

    results = {}

    for transform_type in ['similarity', 'affine', 'homography']:
        print("\n" + "=" * 80)
        print(f"Testing: {transform_type.upper()}")
        print("=" * 80)

        try:
            start = time.time()
            run_improved_registration(
                preset_name=preset,
                target_basemap='esri',
                transform_type=transform_type
            )
            elapsed = time.time() - start

            results[transform_type] = {
                'time': elapsed,
                'status': 'success'
            }

        except Exception as e:
            results[transform_type] = {
                'time': None,
                'status': f'failed: {e}'
            }

    # Print summary
    print("\n" + "=" * 80)
    print("TRANSFORM COMPARISON RESULTS")
    print("=" * 80)

    for transform_type, result in results.items():
        if result['status'] == 'success':
            print(f"{transform_type:15s}: {format_time(int(result['time']))} ✓")
        else:
            print(f"{transform_type:15s}: {result['status']} ✗")

    print("\nCompare difference maps to choose best transform type for your data:")
    for transform_type in results.keys():
        if results[transform_type]['status'] == 'success':
            print(f"  - outputs/improved_{preset}_esri_{transform_type}/visualizations/")


def test_preprocessing_methods(preset='balanced', transform_type='similarity'):
    """
    Test different preprocessing methods.
    """
    print("\n" + "=" * 80)
    print("PREPROCESSING METHOD COMPARISON")
    print("=" * 80)

    methods = ['edges', 'gradient', 'clahe', 'none']
    results = {}

    for method in methods:
        print("\n" + "=" * 80)
        print(f"Testing preprocessing: {method.upper()}")
        print("=" * 80)

        # Modify config temporarily
        config_backup = {
            "source_path": "inputs/orthomosaic_no_gcps_utm10n.tif",
            "target_path": "inputs/qualicum_beach_basemap_esri_utm10n.tif",
            "method": "patch_ncc",
            "hierarchical_scales": [0.1, 0.2],  # Faster for testing
            "patch_size": 256,
            "patch_grid_spacing": 128,
            "ncc_threshold": 0.35,
            "transform_type": transform_type,
            "enable_refinement": False,  # Disable for faster comparison
            "adaptive_search": True,
            "preprocess_method": method,
            "output_dir": f"outputs/preprocess_{method}_{transform_type}",
            "verbose": True,
            "visualization": {
                "create_match_visualizations": True,
                "create_difference_maps": True,
                "create_png_overviews": True
            }
        }

        preset_config = get_preset_config(preset)
        config_backup.update(preset_config)

        config_path = f"config_preprocess_{method}.json"
        with open(config_path, 'w') as f:
            json.dump(config_backup, f, indent=2)

        try:
            start = time.time()
            sys.argv = ['example_improved.py', '--config', config_path]
            run_registration()
            elapsed = time.time() - start

            results[method] = {
                'time': elapsed,
                'status': 'success'
            }

        except Exception as e:
            results[method] = {
                'time': None,
                'status': f'failed: {e}'
            }

    # Print summary
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPARISON RESULTS")
    print("=" * 80)

    for method, result in results.items():
        if result['status'] == 'success':
            print(f"{method:15s}: {format_time(int(result['time']))} ✓")
        else:
            print(f"{method:15s}: {result['status']} ✗")


if __name__ == '__main__':
    """
    Choose which example to run.
    """

    # ========== RECOMMENDED: Single run with improved features ==========
    # Uses similarity transform (rotation + uniform scale + translation)
    run_improved_registration(
        preset_name='quality',
        target_basemap='esri',
        transform_type='similarity'
    )

    # ========== OPTION 2: Try affine transform (allows shear/non-uniform scale) ==========
    # run_improved_registration(
    #     preset_name='quality',
    #     target_basemap='esri',
    #     transform_type='affine'
    # )

    # ========== OPTION 3: Try homography (full perspective transform) ==========
    # run_improved_registration(
    #     preset_name='quality',
    #     target_basemap='esri',
    #     transform_type='homography'
    # )

    # ========== OPTION 4: Compare all transform types ==========
    # compare_transform_types(preset='balanced')

    # ========== OPTION 5: Test preprocessing methods ==========
    # test_preprocessing_methods(preset='balanced', transform_type='similarity')

    # ========== OPTION 6: Different presets ==========
    # run_improved_registration(preset_name='balanced', target_basemap='esri')
    # run_improved_registration(preset_name='fast', target_basemap='esri')
    # run_improved_registration(preset_name='maximum', target_basemap='esri')

    print("\n" + "=" * 80)
    print("IMPROVED REGISTRATION COMPLETE")
    print("=" * 80)
    print("\nKEY IMPROVEMENTS APPLIED:")
    print("  ✓ Pre-alignment at each hierarchical scale")
    print("  ✓ Adaptive search regions based on expected error")
    print("  ✓ Support for affine and homography transforms")
    print("  ✓ Iterative refinement at finest scale")
    print("  ✓ Better outlier filtering before RANSAC")
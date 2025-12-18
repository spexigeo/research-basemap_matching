#!/usr/bin/env python3
"""
Example script optimized for Apple M4 Pro with 48GB RAM.
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


def run_with_preset(preset_name='quality', target_basemap='esri'):
    """
    Run registration with a performance preset.

    Args:
        preset_name: 'fast', 'balanced', 'quality', or 'maximum'
        target_basemap: 'esri' or 'google'
    """
    print("\n" + "=" * 80)
    print(f"OPTIMIZED REGISTRATION - Preset: {preset_name.upper()}")
    print("=" * 80)

    # Base configuration
    config = {
        "source_path": "inputs/orthomosaic_no_gcps_utm10n.tif",
        "target_path": "inputs/qualicum_beach_basemap_esri_utm10n.tif",
        "method": "patch_ncc",
        "hierarchical_scales": [0.06, 0.12, 0.24],
        "patch_size": 384,
        "patch_grid_spacing": 192,
        "ransac_threshold": 5.0,
        "output_dir": f"outputs/m4_{preset_name}_{target_basemap}",
        "verbose": True,
        "preprocess_method": "edges",
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
        "ransac_threshold": 5.0,
        "ncc_threshold": 0.35,
        "reproject_to_metric": True
    }

    # Apply preset optimizations
    preset = get_preset_config(preset_name)
    config.update(preset)

    # Save config
    config_path = f"config_m4_{preset_name}_{target_basemap}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration: {config_path}")
    print(f"Method: {config['method']}")
    print(f"Scales: {config['hierarchical_scales']}")
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

    sys.argv = ['example_m4_optimized.py', '--config', config_path]
    try:
        run_registration()

        elapsed = time.time() - start_time
        print(f"\n✓ Actual processing time: {format_time(int(elapsed))}")

    except Exception as e:
        print(f"\n✗ Registration failed: {e}")
        elapsed = time.time() - start_time
        print(f"Time before failure: {format_time(int(elapsed))}")
        raise


def run_comparison_on_both_basemaps(preset='balanced'):
    """
    Run registration against both ESRI and Google basemaps.
    """
    print("\n" + "=" * 80)
    print("DUAL BASEMAP COMPARISON")
    print("=" * 80)
    print(f"Will register orthomosaic against both ESRI and Google basemaps")
    print(f"Using preset: {preset}")

    total_start = time.time()

    # ESRI
    print("\n" + "=" * 80)
    print("Phase 1: ESRI Basemap")
    print("=" * 80)
    try:
        run_with_preset(preset_name=preset, target_basemap='esri')
    except Exception as e:
        print(f"ESRI registration failed: {e}")

    # Google
    print("\n" + "=" * 80)
    print("Phase 2: Google Basemap")
    print("=" * 80)
    try:
        run_with_preset(preset_name=preset, target_basemap='google')
    except Exception as e:
        print(f"Google registration failed: {e}")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Total time: {format_time(int(total_elapsed))}")
    print("\nCompare results in:")
    print(f"  - outputs/m4_{preset}_esri/")
    print(f"  - outputs/m4_{preset}_google/")


def quick_test():
    """
    Quick test with minimal processing to verify setup.
    """
    print("\n" + "=" * 80)
    print("QUICK TEST - Verifying M4 Setup")
    print("=" * 80)

    config = {
        "source_path": "/Users/mauriciohessflores/Documents/Code/MyCode/research-qualicum_beach_gcp_analysis/outputs/orthomosaics/orthomosaic_no_gcps_utm.tif",
        "target_path": "/Users/mauriciohessflores/Documents/Code/MyCode/research-qualicum_beach_gcp_analysis/outputs/qualicum_beach_basemap_esri_utm.tif",
        "method": "phase_correlation",  # Fastest method
        "output_dir": "outputs/m4_quick_test",
        "verbose": False,
        "hierarchical_scales": [0.05],  # Very coarse, very fast
        "preprocess_method": "none",
        "visualization": {
            "create_match_visualizations": False,
            "create_difference_maps": False,
            "create_png_overviews": True
        },
        "output_format": {
            "compression": "JPEG",
            "jpeg_quality": 85,
            "tiled": True,
            "blocksize": 512
        },
        "reproject_to_metric": True
    }

    config_path = "config_m4_quick_test.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("Running quick test with phase correlation at 5% scale...")
    print("This should complete in < 30 seconds\n")

    start_time = time.time()
    sys.argv = ['example_m4_optimized.py', '--config', config_path]

    try:
        run_registration()
        elapsed = time.time() - start_time
        print(f"\n✓ Quick test completed in {elapsed:.1f}s")
        print("M4 optimization is working correctly!")
    except Exception as e:
        print(f"\n✗ Quick test failed: {e}")


def benchmark_all_presets():
    """
    Benchmark all presets to find the sweet spot for your images.
    """
    print("\n" + "=" * 80)
    print("PRESET BENCHMARK - Testing all performance levels")
    print("=" * 80)
    print("This will run registration 4 times with different quality/speed tradeoffs")
    print("Use results to choose your preferred preset\n")

    results = {}

    for preset in ['fast', 'balanced', 'quality', 'maximum']:
        print(f"\n{'=' * 80}")
        print(f"Testing preset: {preset.upper()}")
        print(f"{'=' * 80}")

        try:
            start = time.time()
            run_with_preset(preset_name=preset, target_basemap='esri')
            elapsed = time.time() - start

            results[preset] = {
                'time': elapsed,
                'status': 'success'
            }

        except Exception as e:
            results[preset] = {
                'time': None,
                'status': f'failed: {e}'
            }

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    for preset, result in results.items():
        if result['status'] == 'success':
            print(f"{preset:10s}: {format_time(int(result['time']))} ✓")
        else:
            print(f"{preset:10s}: {result['status']} ✗")

    print("\nRecommendation based on your M4 Pro with 48GB RAM:")
    print("  - For routine processing: use 'balanced' preset")
    print("  - For best quality: use 'quality' preset")
    print("  - For testing/iteration: use 'fast' preset")


if __name__ == '__main__':
    """
    Choose which example to run.
    """

    # ========== OPTION 1: Quick test (recommended first) ==========
    # quick_test()

    # ========== OPTION 2: Single run with quality preset (recommended) ==========
    run_with_preset(preset_name='quality', target_basemap='esri')

    # ========== OPTION 3: Try different preset ==========
    # run_with_preset(preset_name='balanced', target_basemap='esri')
    # run_with_preset(preset_name='fast', target_basemap='esri')
    # run_with_preset(preset_name='maximum', target_basemap='esri')  # Uses full 48GB

    # ========== OPTION 4: Compare both basemaps ==========
    # run_comparison_on_both_basemaps(preset='balanced')

    # ========== OPTION 5: Benchmark all presets ==========
    # benchmark_all_presets()

    print("\n" + "=" * 80)
    print("M4-OPTIMIZED REGISTRATION COMPLETE")
    print("=" * 80)
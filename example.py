#!/usr/bin/env python3
"""
Example script showing how to use the registration system.
Modify the paths and parameters below for your data.
"""

import sys
from pathlib import Path

# Add parent directory to path if running as script
sys.path.insert(0, str(Path(__file__).parent))

from main import main as run_registration
import json


def create_example_config():
    """
    Create an example configuration file.
    Modify these paths and parameters for your specific data.
    """
    config = {
        # ============ INPUT FILES ============
        "source_path": "/Users/mauriciohessflores/Documents/Code/MyCode/research-qualicum_beach_gcp_analysis/outputs/orthomosaics/orthomosaic_no_gcps_utm.tif",
        "target_path": "/Users/mauriciohessflores/Documents/Code/MyCode/research-qualicum_beach_gcp_analysis/outputs/qualicum_beach_basemap_esri_utm.tif",
        # "target_path": "/Users/mauriciohessflores/Documents/Code/MyCode/research-qualicum_beach_gcp_analysis/outputs/qualicum_beach_basemap_google_utm.tif",

        # ============ REGISTRATION METHOD ============
        # Options: 'phase_correlation', 'ecc', 'sift', 'orb', 'patch_ncc', 'arosics'
        # Recommended: 'patch_ncc' for seasonal changes, 'sift' for scale differences
        "method": "patch_ncc",

        # ============ OUTPUT SETTINGS ============
        "output_dir": "outputs",
        "verbose": True,  # Set to False for minimal output

        # ============ HIERARCHICAL PROCESSING ============
        # Coarse-to-fine scales (smaller = faster but coarser)
        # For large images, start with [0.05, 0.1]
        # For moderate images, use [0.1, 0.2, 0.4]
        "hierarchical_scales": [0.01, 0.05, 0.1, 0.2],

        # ============ PREPROCESSING ============
        # Options: 'histogram', 'edges', 'gradient', 'clahe', 'none'
        # 'gradient' - good for seasonal changes (structure-based)
        # 'histogram' - good for illumination differences
        # 'clahe' - good for contrast enhancement
        # 'edges' - robust to appearance changes
        "preprocess_method": "gradient",

        # ============ VISUALIZATION OPTIONS ============
        "visualization": {
            "create_match_visualizations": True,  # Show feature matches
            "create_difference_maps": True,  # Show before/after difference
            "create_png_overviews": True  # Create PNG summaries
        },

        # ============ OUTPUT FORMAT ============
        "output_format": {
            "compression": "JPEG",  # Options: 'JPEG', 'LZW', 'DEFLATE'
            "jpeg_quality": 90,  # 1-100, higher = better quality
            "tiled": True,  # Enable tiled structure
            "blocksize": 512  # Tile size in pixels
        },

        # ============ METHOD-SPECIFIC PARAMETERS ============
        # RANSAC outlier rejection threshold (pixels at target resolution)
        "ransac_threshold": 5.0,

        # Maximum number of SIFT/ORB features to detect
        "max_features": 5000,

        # Patch-based matching parameters
        "patch_size": 256,  # Size of patches to match
        "patch_grid_spacing": 128,  # Spacing between patch centers
        "ncc_threshold": 0.3  # Minimum correlation to accept match
    }

    return config


def run_example_with_esri():
    """Example: Register orthomosaic to ESRI basemap."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Registering to ESRI Basemap")
    print("=" * 80 + "\n")

    config = create_example_config()
    config[
        "target_path"] = "/Users/mauriciohessflores/Documents/Code/MyCode/research-qualicum_beach_gcp_analysis/outputs/qualicum_beach_basemap_esri.tif"
    config["output_dir"] = "outputs/esri_registration"

    # Save config
    config_path = "config_esri.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {config_path}")
    print("Running registration...\n")

    # Run registration
    sys.argv = ['example.py', '--config', config_path]
    run_registration()


def run_example_with_google():
    """Example: Register orthomosaic to Google basemap."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Registering to Google Basemap")
    print("=" * 80 + "\n")

    config = create_example_config()
    config[
        "target_path"] = "/Users/mauriciohessflores/Documents/Code/MyCode/research-qualicum_beach_gcp_analysis/outputs/qualicum_beach_basemap_google.tif"
    config["output_dir"] = "outputs/google_registration"

    # Save config
    config_path = "config_google.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {config_path}")
    print("Running registration...\n")

    # Run registration
    sys.argv = ['example.py', '--config', config_path]
    run_registration()


def run_comparison_experiment():
    """
    Example: Compare different methods on the same data.
    This will run multiple registration methods and save results separately.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Method Comparison Experiment")
    print("=" * 80 + "\n")

    base_config = create_example_config()
    base_config["verbose"] = True

    # Methods to compare
    methods = ['patch_ncc', 'sift', 'ecc', 'phase_correlation']

    for method in methods:
        print(f"\n{'=' * 60}")
        print(f"Testing method: {method}")
        print(f"{'=' * 60}\n")

        config = base_config.copy()
        config["method"] = method
        config["output_dir"] = f"outputs/comparison/{method}"

        # Adjust parameters based on method
        if method == 'phase_correlation':
            config["hierarchical_scales"] = [0.1]  # Single scale is sufficient

        # Save config
        config_path = f"config_{method}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Run registration
        sys.argv = ['example.py', '--config', config_path]
        try:
            run_registration()
        except Exception as e:
            print(f"Method {method} failed: {e}")
            continue

    print("\n" + "=" * 80)
    print("Comparison experiment complete!")
    print("Check outputs/comparison/ for results from each method")
    print("=" * 80)


def quick_test():
    """Quick test with minimal processing for debugging."""
    print("\n" + "=" * 80)
    print("QUICK TEST: Minimal processing")
    print("=" * 80 + "\n")

    config = create_example_config()
    config["hierarchical_scales"] = [0.05]  # Very coarse, very fast
    config["verbose"] = False
    config["visualization"]["create_match_visualizations"] = False
    config["output_dir"] = "outputs/quick_test"

    config_path = "config_quick_test.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {config_path}")
    print("Running quick test...\n")

    sys.argv = ['example.py', '--config', config_path]
    run_registration()


if __name__ == '__main__':
    """
    Choose which example to run by uncommenting the appropriate line below.
    """

    # ========== OPTION 1: Register to ESRI basemap ==========
    run_example_with_esri()

    # ========== OPTION 2: Register to Google basemap ==========
    # run_example_with_google()

    # ========== OPTION 3: Compare all methods ==========
    # run_comparison_experiment()

    # ========== OPTION 4: Quick test ==========
    # quick_test()

    print("\n" + "=" * 80)
    print("Example execution complete!")
    print("=" * 80)
#!/usr/bin/env python3
"""
Example script demonstrating how to use the orthomosaic registration pipeline.

This script shows various ways to register an orthomosaic to a basemap:
1. Using command-line interface directly
2. Using configuration files
3. Different debug levels
4. Different matchers and transform algorithms
5. Programmatic usage

For learning purposes - modify paths and parameters for your data.
"""

import subprocess
import sys
from pathlib import Path
import json


def example_1_basic_cli():
    """
    Example 1: Basic command-line usage
    
    This is the simplest way to register an orthomosaic.
    Uses default settings: scales [0.125, 0.25, 0.5, 1.0] with algorithms [shift, shift, homography, homography]
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic CLI Usage")
    print("=" * 80)
    print("\nThis example shows the simplest way to register an orthomosaic.")
    print("It uses default settings and saves only the final registered orthomosaic.\n")
    
    source = "inputs/orthomosaic_no_gcps.tif"
    target = "inputs/qualicum_beach_basemap_esri.tif"
    output = "outputs/example1_basic"
    
    cmd = [
        sys.executable,
        "register_orthomosaic.py",
        source,
        target,
        output,
        "--debug", "none"  # Only save log and final orthomosaic
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    print("Running registration...")
    subprocess.run(cmd, check=True)
    print(f"\n✓ Registration complete! Check {output}/ for results.")


def example_2_custom_scales_and_algorithms():
    """
    Example 2: Custom scales and algorithms
    
    This example shows how to specify custom resolution scales and 
    corresponding transform algorithms for each scale.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Custom Scales and Algorithms")
    print("=" * 80)
    print("\nThis example uses custom scales and algorithms.")
    print("You can specify different transform types for different scales.\n")
    
    source = "inputs/orthomosaic_no_gcps.tif"
    target = "inputs/qualicum_beach_basemap_esri.tif"
    output = "outputs/example2_custom"
    
    cmd = [
        sys.executable,
        "register_orthomosaic.py",
        source,
        target,
        output,
        "--scales", "0.125", "0.25", "0.5", "1.0",
        "--algorithms", "shift", "shift", "affine", "homography",
        "--debug", "intermediate"  # Save intermediate files too
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    print("Running registration...")
    subprocess.run(cmd, check=True)
    print(f"\n✓ Registration complete! Check {output}/ for results.")


def example_3_using_config_file():
    """
    Example 3: Using a configuration file
    
    Configuration files are useful for:
    - Reproducible experiments
    - Sharing settings with others
    - Complex parameter sets
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Using Configuration File")
    print("=" * 80)
    print("\nThis example shows how to use a JSON configuration file.")
    print("See sample_configs/ for example configuration files.\n")
    
    # Create a sample config file
    config = {
        "source_path": "inputs/orthomosaic_no_gcps.tif",
        "target_path": "inputs/qualicum_beach_basemap_esri.tif",
        "output_dir": "outputs/example3_config",
        "hierarchical_scales": [0.125, 0.25, 0.5, 1.0],
        "algorithms": ["shift", "shift", "homography", "homography"],
        "method": "lightglue",
        "debug_level": "high"  # Save all debug files
    }
    
    config_path = "example_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created config file: {config_path}")
    print(f"Config contents:\n{json.dumps(config, indent=2)}\n")
    
    cmd = [
        sys.executable,
        "register_orthomosaic.py",
        "--config", config_path
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    print("Running registration...")
    subprocess.run(cmd, check=True)
    print(f"\n✓ Registration complete! Check {config['output_dir']}/ for results.")


def example_4_different_matchers():
    """
    Example 4: Comparing different matchers
    
    This example shows how to use different feature matching methods:
    - lightglue: Deep learning-based (recommended, requires installation)
    - sift: Traditional feature matching
    - orb: Fast feature matching
    - patch_ncc: Patch-based normalized cross-correlation
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Different Matchers")
    print("=" * 80)
    print("\nThis example compares different matching methods.")
    print("Each matcher has different strengths:\n")
    print("  - lightglue: Best accuracy, handles large appearance changes")
    print("  - sift: Good for scale/rotation differences")
    print("  - orb: Fast, good for similar images")
    print("  - patch_ncc: Robust to seasonal/illumination changes\n")
    
    source = "inputs/orthomosaic_no_gcps.tif"
    target = "inputs/qualicum_beach_basemap_esri.tif"
    
    matchers = ['lightglue', 'sift', 'orb', 'patch_ncc']
    
    for matcher in matchers:
        print(f"\n{'=' * 60}")
        print(f"Testing matcher: {matcher.upper()}")
        print(f"{'=' * 60}")
        
        output = f"outputs/example4_{matcher}"
        
        cmd = [
            sys.executable,
            "register_orthomosaic.py",
            source,
            target,
            output,
            "--matcher", matcher,
            "--debug", "none"  # Minimal output for comparison
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ {matcher} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ {matcher} failed: {e}")
    
    print("\n" + "=" * 80)
    print("Comparison complete! Check outputs/example4_*/ for results from each matcher.")
    print("=" * 80)


def example_5_debug_levels():
    """
    Example 5: Understanding debug levels
    
    Debug levels control what files are saved:
    - none: Only registration.log and final orthomosaic
    - intermediate: + intermediate/ directory (transformed orthomosaics at each scale)
    - high: + matching_and_transformations/ + preprocessing/ + registration_verbose.log
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Debug Levels")
    print("=" * 80)
    print("\nThis example demonstrates the different debug levels.\n")
    
    source = "inputs/orthomosaic_no_gcps.tif"
    target = "inputs/qualicum_beach_basemap_esri.tif"
    
    debug_levels = [
        ("none", "Only log and final orthomosaic"),
        ("intermediate", "+ intermediate files (transformed orthomosaics)"),
        ("high", "+ all debug files (matches, transforms, visualizations)")
    ]
    
    for debug_level, description in debug_levels:
        print(f"\n{'=' * 60}")
        print(f"Debug level: {debug_level.upper()}")
        print(f"Description: {description}")
        print(f"{'=' * 60}")
        
        output = f"outputs/example5_{debug_level}"
        
        cmd = [
            sys.executable,
            "register_orthomosaic.py",
            source,
            target,
            output,
            "--debug", debug_level,
            "--scales", "0.125", "0.25"  # Use fewer scales for faster demo
        ]
        
        print(f"Running with debug level '{debug_level}'...")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ Completed - check {output}/ to see what files were created")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {e}")
    
    print("\n" + "=" * 80)
    print("Debug level comparison complete!")
    print("Compare the output directories to see the difference in saved files.")
    print("=" * 80)


def example_6_programmatic_usage():
    """
    Example 6: Programmatic usage
    
    This example shows how to use the registration pipeline programmatically
    instead of through the command line.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Programmatic Usage")
    print("=" * 80)
    print("\nThis example shows how to use the registration class directly.\n")
    
    from register_orthomosaic import OrthomosaicRegistration
    
    source = "inputs/orthomosaic_no_gcps.tif"
    target = "inputs/qualicum_beach_basemap_esri.tif"
    output = "outputs/example6_programmatic"
    
    # Create registration instance
    registration = OrthomosaicRegistration(
        source_path=source,
        target_path=target,
        output_dir=output,
        scales=[0.125, 0.25, 0.5, 1.0],
        matcher='lightglue',
        transform_types={
            0.125: 'shift',
            0.25: 'shift',
            0.5: 'homography',
            1.0: 'homography'
        },
        debug_level='intermediate'
    )
    
    print("Running registration programmatically...")
    result = registration.register()
    
    if result:
        print(f"\n✓ Registration complete! Result: {result}")
    else:
        print("\n✗ Registration failed")


def example_7_quick_test():
    """
    Example 7: Quick test
    
    A minimal example for quick testing and verification.
    Uses coarse scales and minimal output for speed.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Quick Test")
    print("=" * 80)
    print("\nThis is a minimal example for quick testing.\n")
    
    source = "inputs/orthomosaic_no_gcps.tif"
    target = "inputs/qualicum_beach_basemap_esri.tif"
    output = "outputs/example7_quick_test"
    
    cmd = [
        sys.executable,
        "register_orthomosaic.py",
        source,
        target,
        output,
        "--scales", "0.125", "0.25",  # Only 2 scales for speed
        "--algorithms", "shift", "shift",  # Simple shifts only
        "--matcher", "sift",  # Fast matcher
        "--debug", "none"  # Minimal output
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    print("Running quick test...")
    subprocess.run(cmd, check=True)
    print(f"\n✓ Quick test complete! Check {output}/ for results.")


def main():
    """
    Main function - choose which example to run.
    
    Uncomment the example you want to run, or modify to run multiple examples.
    """
    print("\n" + "=" * 80)
    print("ORTHOMOSAIC REGISTRATION - EXAMPLES")
    print("=" * 80)
    print("\nThis script demonstrates various ways to use the registration pipeline.")
    print("Modify the file paths in each example to match your data.\n")
    
    # ========== CHOOSE AN EXAMPLE TO RUN ==========
    
    # Basic usage examples
    # example_1_basic_cli()
    # example_2_custom_scales_and_algorithms()
    # example_3_using_config_file()
    
    # Advanced examples
    # example_4_different_matchers()
    # example_5_debug_levels()
    # example_6_programmatic_usage()
    
    # Quick test
    example_7_quick_test()
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nFor more information, see:")
    print("  - README.md for documentation")
    print("  - sample_configs/ for example configuration files")
    print("  - register_orthomosaic.py --help for CLI options")


if __name__ == '__main__':
    main()

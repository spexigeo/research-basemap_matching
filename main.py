#!/usr/bin/env python3
"""
Main entry point for orthomosaic registration.
Supports command-line execution and configuration file input.
"""

import argparse
import json
import sys
from pathlib import Path
import logging
from datetime import datetime

from registration_lib import OrthoRegistration, RegistrationConfig
from utils import setup_logging, create_output_directory


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: dict, output_dir: Path):
    """Save configuration to output directory for reproducibility."""
    config_path = output_dir / 'run_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logging.info(f"Configuration saved to: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Register orthomosaic to basemap using various algorithms'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file'
    )

    parser.add_argument(
        '--source',
        type=str,
        help='Path to source orthomosaic (overrides config)'
    )

    parser.add_argument(
        '--target',
        type=str,
        help='Path to target basemap (overrides config)'
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=['phase_correlation', 'ecc', 'sift', 'orb', 'patch_ncc', 'arosics'],
        help='Registration method (overrides config)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (overrides config)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output and intermediate visualizations'
    )

    parser.add_argument(
        '--scales',
        type=float,
        nargs='+',
        help='Hierarchical scales, e.g., 0.1 0.25 0.5 (overrides config)'
    )

    parser.add_argument(
        '--preprocess',
        type=str,
        choices=['histogram', 'edges', 'gradient', 'clahe', 'none'],
        help='Preprocessing method (overrides config)'
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config_dict = load_config(args.config)
    else:
        # Use default config
        config_dict = {
            'source_path': args.source or '',
            'target_path': args.target or '',
            'method': args.method or 'patch_ncc',
            'output_dir': args.output_dir or 'outputs',
            'verbose': args.verbose or False,
            'hierarchical_scales': args.scales or [0.1, 0.2],
            'preprocess_method': args.preprocess or 'gradient',
            'visualization': {
                'create_match_visualizations': True,
                'create_difference_maps': True,
                'create_png_overviews': True
            },
            'output_format': {
                'compression': 'JPEG',
                'jpeg_quality': 90,
                'tiled': True,
                'blocksize': 512
            }
        }

    # Override config with command-line arguments
    if args.source:
        config_dict['source_path'] = args.source
    if args.target:
        config_dict['target_path'] = args.target
    if args.method:
        config_dict['method'] = args.method
    if args.output_dir:
        config_dict['output_dir'] = args.output_dir
    if args.verbose:
        config_dict['verbose'] = True
    if args.scales:
        config_dict['hierarchical_scales'] = args.scales
    if args.preprocess:
        config_dict['preprocess_method'] = args.preprocess

    # Validate required parameters
    if not config_dict.get('source_path') or not config_dict.get('target_path'):
        print("Error: source_path and target_path are required")
        print("Provide them via --config file or --source/--target arguments")
        sys.exit(1)

    # Create configuration object
    config = RegistrationConfig.from_dict(config_dict)

    # Setup output directory and logging
    output_dir = create_output_directory(config.output_dir)
    setup_logging(output_dir, verbose=config.verbose)

    logging.info("=" * 80)
    logging.info("ORTHOMOSAIC REGISTRATION")
    logging.info("=" * 80)
    logging.info(f"Timestamp: {datetime.now().isoformat()}")
    logging.info(f"Source: {config.source_path}")
    logging.info(f"Target: {config.target_path}")
    logging.info(f"Method: {config.method}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Verbose mode: {config.verbose}")
    logging.info("=" * 80)

    # Save configuration for reproducibility
    save_config(config_dict, output_dir)

    # Initialize registration
    registrator = OrthoRegistration(config, output_dir)

    try:
        # Run hierarchical registration
        logging.info("\nStarting hierarchical registration...")
        transform_matrix = registrator.hierarchical_registration()

        if transform_matrix is None:
            logging.error("Registration failed - no valid transform computed")
            sys.exit(1)

        # STOP HERE - Only run matching, not registration
        logging.info("\n" + "=" * 80)
        logging.info("MATCHING COMPLETED - Stopping before registration")
        logging.info("=" * 80)
        logging.info("Transform matrix computed but not applied")
        logging.info("Check match visualizations in visualizations/ directory")
        logging.info(f"All outputs saved to: {output_dir}")
        logging.info("=" * 80)
        
        # Skip registration steps
        return
        
        # Apply transformation to full resolution
        logging.info("\nApplying transformation to full-resolution orthomosaic...")
        output_path = registrator.apply_transform()

        # Create visualizations
        if config.visualization['create_png_overviews']:
            logging.info("\nCreating PNG overviews...")
            registrator.create_png_overviews()

        if config.visualization['create_difference_maps']:
            logging.info("\nCreating difference maps...")
            registrator.create_difference_map()

        # Generate final report
        logging.info("\nGenerating registration report...")
        registrator.generate_report()

        logging.info("\n" + "=" * 80)
        logging.info("REGISTRATION COMPLETED SUCCESSFULLY")
        logging.info(f"Registered orthomosaic: {output_path}")
        logging.info(f"All outputs saved to: {output_dir}")
        logging.info("=" * 80)

    except Exception as e:
        logging.error(f"Registration failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
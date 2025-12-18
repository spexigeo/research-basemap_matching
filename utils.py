"""
Utility functions for logging, directory management, and file I/O.
"""

import logging
from pathlib import Path
from datetime import datetime
import sys


def setup_logging(output_dir: Path, verbose: bool = False):
    """
    Setup logging configuration with proper file placement.

    Args:
        output_dir: Directory to save log files
        verbose: If True, set DEBUG level; otherwise INFO
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(message)s'  # Very simple for console
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler (simple format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # File handler in logs subdirectory (detailed format)
    logs_dir = output_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'registration_{timestamp}.log'

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)  # Always capture all details in file
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)

    logging.info(f"Log file: {log_file}")
    logging.info(f"Verbose mode: {verbose}")


def create_output_directory(base_dir: str) -> Path:
    """
    Create output directory structure.

    Args:
        base_dir: Base output directory path

    Returns:
        Path object for the created directory
    """
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(base_dir) / f'run_{timestamp}'

    # Create directory and subdirectories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    (output_dir / 'intermediate').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)

    return output_dir


def format_bytes(num_bytes: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def get_file_info(filepath: Path) -> dict:
    """Get file information."""
    if not filepath.exists():
        return {'exists': False}

    stat = filepath.stat()
    return {
        'exists': True,
        'size': stat.st_size,
        'size_formatted': format_bytes(stat.st_size),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
    }
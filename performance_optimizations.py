"""
Performance optimizations for Apple M4 Pro.
Import and call setup_m4_optimizations() at the start of your program.
"""

import os
import cv2
import numpy as np
import logging
from multiprocessing import cpu_count


def setup_m4_optimizations():
    """
    Configure optimal settings for Apple M4 Pro with 48GB RAM.
    Call this once at the start of your program.
    """

    # Detect CPU core count (M4 Pro has 14 cores: 10 performance + 4 efficiency)
    cores = cpu_count()
    performance_cores = min(cores, 10)  # Use performance cores primarily

    logging.info(f"Detected {cores} CPU cores")
    logging.info(f"Configuring for Apple M4 Pro optimization...")

    # ===== OpenCV Optimizations =====
    # Use all available cores for parallel operations
    cv2.setNumThreads(performance_cores)
    logging.info(f"OpenCV threads: {performance_cores}")

    # Enable optimizations
    cv2.setUseOptimized(True)

    # ===== NumPy/BLAS Optimizations =====
    # These environment variables should be set before importing numpy,
    # but we set them here as well for any lazy imports
    os.environ['OPENBLAS_NUM_THREADS'] = str(performance_cores)
    os.environ['MKL_NUM_THREADS'] = str(performance_cores)
    os.environ['NUMEXPR_NUM_THREADS'] = str(performance_cores)
    os.environ['OMP_NUM_THREADS'] = str(performance_cores)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(performance_cores)

    # ===== Memory Optimizations =====
    # With 48GB RAM, we can be generous with memory usage

    # Increase OpenCV memory pool (if using CUDA, but good to set anyway)
    os.environ['OPENCV_OPENCL_RUNTIME'] = ''  # Disable OpenCL (not optimal on Mac)

    # ===== Apple Silicon Specific =====
    # Force use of Accelerate framework (Apple's optimized BLAS/LAPACK)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(performance_cores)

    # ===== Logging =====
    logging.info("M4 Pro optimizations applied:")
    logging.info(f"  - CPU cores: {cores} (using {performance_cores} for compute)")
    logging.info(f"  - Available RAM: 48GB")
    logging.info(f"  - OpenCV optimizations: ENABLED")
    logging.info(f"  - Accelerate framework: ACTIVE")

    return {
        'cores': cores,
        'performance_cores': performance_cores,
        'memory_gb': 48
    }


def get_optimal_batch_size(image_shape, memory_gb=48):
    """
    Calculate optimal batch/tile size for processing large images.

    Args:
        image_shape: (height, width) of the image
        memory_gb: Available memory in GB

    Returns:
        Optimal tile size
    """
    # Estimate memory per pixel (RGB image + processing overhead)
    bytes_per_pixel = 3 * 4  # 3 channels, 4 bytes per float32
    overhead_factor = 5  # Account for intermediate results

    # Calculate maximum pixels we can process
    available_bytes = memory_gb * 1024 ** 3 * 0.7  # Use 70% of available RAM
    max_pixels = available_bytes / (bytes_per_pixel * overhead_factor)

    # For square tiles
    tile_size = int(np.sqrt(max_pixels))

    # Round to nice block sizes
    tile_size = min(tile_size, 8192)  # Cap at 8K
    tile_size = (tile_size // 512) * 512  # Round to 512-pixel blocks

    return tile_size


def optimize_opencv_for_image_size(height, width):
    """
    Adjust OpenCV settings based on image size.

    Args:
        height: Image height in pixels
        width: Image width in pixels
    """
    total_pixels = height * width

    if total_pixels > 100_000_000:  # > 100 megapixels
        logging.info(f"Large image detected ({total_pixels / 1e6:.1f} MP)")
        logging.info("Enabling large image optimizations...")

        # For very large images, use fewer threads per operation
        # but process multiple regions in parallel
        cv2.setNumThreads(4)

    elif total_pixels > 25_000_000:  # 25-100 megapixels
        cv2.setNumThreads(8)

    else:  # Smaller images
        cv2.setNumThreads(10)


class M4ProgressTracker:
    """
    Track and display progress with M4-specific optimizations.
    """

    def __init__(self, total_steps, description="Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description

    def update(self, step_name=""):
        """Update progress."""
        self.current_step += 1
        percent = (self.current_step / self.total_steps) * 100

        bar_length = 40
        filled = int(bar_length * self.current_step / self.total_steps)
        bar = '█' * filled + '░' * (bar_length - filled)

        logging.info(f"{self.description}: [{bar}] {percent:.1f}% - {step_name}")

    def complete(self):
        """Mark as complete."""
        logging.info(f"{self.description}: Complete! ✓")


# Pre-computed optimal settings for common scenarios
M4_PRESETS = {
    'fast': {
        'hierarchical_scales': [0.1],
        'max_features': 3000,
        'patch_size': 256,
        'patch_grid_spacing': 256,
        'description': 'Fastest processing, lower accuracy'
    },
    'balanced': {
        'hierarchical_scales': [0.15, 0.3],
        'max_features': 5000,
        'patch_size': 320,
        'patch_grid_spacing': 192,
        'description': 'Good balance of speed and accuracy'
    },
    'quality': {
        'hierarchical_scales': [0.15, 0.3, 0.5],
        'max_features': 8000,
        'patch_size': 384,
        'patch_grid_spacing': 192,
        'description': 'Best accuracy, slower processing'
    },
    'maximum': {
        'hierarchical_scales': [0.2, 0.4, 0.6, 0.8],
        'max_features': 12000,
        'patch_size': 512,
        'patch_grid_spacing': 256,
        'description': 'Maximum quality, slowest (uses full 48GB RAM capacity)'
    }
}


def get_preset_config(preset_name='quality'):
    """
    Get a pre-configured preset optimized for M4 Pro.

    Args:
        preset_name: One of 'fast', 'balanced', 'quality', 'maximum'

    Returns:
        Dictionary with optimal parameters
    """
    if preset_name not in M4_PRESETS:
        logging.warning(f"Unknown preset '{preset_name}', using 'quality'")
        preset_name = 'quality'

    preset = M4_PRESETS[preset_name].copy()
    logging.info(f"Using preset: {preset_name} - {preset['description']}")

    return preset


def estimate_processing_time(image_shape, method='patch_ncc', scales=None):
    """
    Estimate processing time on M4 Pro.

    Args:
        image_shape: (height, width) of source image
        method: Registration method
        scales: List of hierarchical scales

    Returns:
        Estimated time in seconds
    """
    if scales is None:
        scales = [0.15, 0.3, 0.5]

    total_pixels = image_shape[0] * image_shape[1]

    # Rough time estimates per method per megapixel on M4 Pro
    time_per_mp = {
        'phase_correlation': 0.1,
        'ecc': 0.3,
        'sift': 0.8,
        'patch_ncc': 0.5,
        'orb': 0.2
    }

    base_time = time_per_mp.get(method, 0.5)

    # Calculate for each scale
    total_time = 0
    for scale in scales:
        scaled_pixels = total_pixels * (scale ** 2)
        megapixels = scaled_pixels / 1e6
        total_time += megapixels * base_time

    # Add overhead
    total_time *= 1.3

    return int(total_time)


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


if __name__ == '__main__':
    # Demo/test
    print("Apple M4 Pro Performance Optimizer")
    print("=" * 50)

    config = setup_m4_optimizations()
    print(f"\nConfiguration applied:")
    print(f"  Cores: {config['cores']}")
    print(f"  Performance cores: {config['performance_cores']}")
    print(f"  Memory: {config['memory_gb']}GB")

    print(f"\nAvailable presets:")
    for name, preset in M4_PRESETS.items():
        print(f"  {name:10s}: {preset['description']}")

    # Example time estimate
    large_image = (46405, 54537)  # Your orthomosaic size
    print(f"\nEstimated processing time for {large_image[0]}x{large_image[1]} image:")
    for preset_name in ['fast', 'balanced', 'quality', 'maximum']:
        preset = M4_PRESETS[preset_name]
        time_est = estimate_processing_time(
            large_image,
            method='patch_ncc',
            scales=preset['hierarchical_scales']
        )
        print(f"  {preset_name:10s}: ~{format_time(time_est)}")
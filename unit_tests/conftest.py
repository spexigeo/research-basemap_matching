"""
Shared fixtures for unit tests.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import rasterio
from rasterio.transform import from_bounds


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image_100x100():
    """Create a sample 100x100 grayscale image."""
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


@pytest.fixture
def sample_image_200x200():
    """Create a sample 200x200 grayscale image."""
    return np.random.randint(0, 255, (200, 200), dtype=np.uint8)


@pytest.fixture
def sample_geotiff_source(temp_dir):
    """Create a sample GeoTIFF file for source image."""
    output_path = temp_dir / "source.tif"
    
    # Create a simple 100x100 image with georeferencing
    height, width = 100, 100
    transform = from_bounds(
        -123.5, 49.0, -123.4, 49.1,  # Bounds in WGS84
        width, height
    )
    
    data = np.random.randint(0, 255, (1, height, width), dtype=np.uint8)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
        compress='jpeg',
        jpeg_quality=90
    ) as dst:
        dst.write(data)
    
    return output_path


@pytest.fixture
def sample_geotiff_target(temp_dir):
    """Create a sample GeoTIFF file for target image."""
    output_path = temp_dir / "target.tif"
    
    # Create a simple 200x200 image with georeferencing (overlapping with source)
    height, width = 200, 200
    transform = from_bounds(
        -123.45, 48.95, -123.35, 49.05,  # Overlapping bounds
        width, height
    )
    
    data = np.random.randint(0, 255, (1, height, width), dtype=np.uint8)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
        compress='jpeg',
        jpeg_quality=90
    ) as dst:
        dst.write(data)
    
    return output_path


@pytest.fixture
def sample_matches():
    """Create sample match data."""
    n_matches = 10
    src_points = np.random.rand(n_matches, 2) * 100
    tgt_points = src_points + np.random.rand(n_matches, 2) * 5  # Small offset
    return src_points, tgt_points




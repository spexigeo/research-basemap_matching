"""
Unit tests for preprocessing module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import ImagePreprocessor
from conftest import sample_geotiff_source, sample_geotiff_target, temp_dir


class TestImagePreprocessor:
    """Test ImagePreprocessor class."""
    
    def test_initialization(self, sample_geotiff_source, sample_geotiff_target, temp_dir):
        """Test that ImagePreprocessor initializes correctly."""
        preprocessor = ImagePreprocessor(
            source_path=str(sample_geotiff_source),
            target_path=str(sample_geotiff_target),
            output_dir=temp_dir
        )
        
        assert preprocessor.source_path == Path(sample_geotiff_source)
        assert preprocessor.target_path == Path(sample_geotiff_target)
        assert preprocessor.output_dir == temp_dir
        assert temp_dir.exists()
    
    def test_lazy_metadata_loading(self, sample_geotiff_source, sample_geotiff_target, temp_dir):
        """Test that metadata is loaded lazily."""
        preprocessor = ImagePreprocessor(
            source_path=str(sample_geotiff_source),
            target_path=str(sample_geotiff_target),
            output_dir=temp_dir
        )
        
        # Metadata should not be loaded yet
        assert not preprocessor._source_metadata_loaded
        assert not preprocessor._target_metadata_loaded
        
        # Accessing properties should trigger loading
        source_crs = preprocessor.source_crs
        assert preprocessor._source_metadata_loaded
        assert source_crs is not None
        
        target_crs = preprocessor.target_crs
        assert preprocessor._target_metadata_loaded
        assert target_crs is not None
    
    def test_source_properties(self, sample_geotiff_source, sample_geotiff_target, temp_dir):
        """Test source image properties."""
        preprocessor = ImagePreprocessor(
            source_path=str(sample_geotiff_source),
            target_path=str(sample_geotiff_target),
            output_dir=temp_dir
        )
        
        # Test all source properties
        assert preprocessor.source_crs is not None
        assert preprocessor.source_transform is not None
        assert preprocessor.source_shape == (100, 100)  # From fixture
        assert preprocessor.source_bounds is not None
        assert preprocessor.source_count == 1
    
    def test_target_properties(self, sample_geotiff_source, sample_geotiff_target, temp_dir):
        """Test target image properties."""
        preprocessor = ImagePreprocessor(
            source_path=str(sample_geotiff_source),
            target_path=str(sample_geotiff_target),
            output_dir=temp_dir
        )
        
        # Test all target properties
        assert preprocessor.target_crs is not None
        assert preprocessor.target_transform is not None
        assert preprocessor.target_shape == (200, 200)  # From fixture
        assert preprocessor.target_bounds is not None
    
    def test_resolution_calculation(self, sample_geotiff_source, sample_geotiff_target, temp_dir):
        """Test resolution calculation."""
        preprocessor = ImagePreprocessor(
            source_path=str(sample_geotiff_source),
            target_path=str(sample_geotiff_target),
            output_dir=temp_dir
        )
        
        source_res = preprocessor.source_res
        target_res = preprocessor.target_res
        
        # Resolutions should be positive numbers
        assert source_res > 0
        assert target_res > 0
        assert isinstance(source_res, float)
        assert isinstance(target_res, float)
    
    def test_load_downsampled(self, sample_geotiff_source, sample_geotiff_target, temp_dir):
        """Test loading downsampled images."""
        preprocessor = ImagePreprocessor(
            source_path=str(sample_geotiff_source),
            target_path=str(sample_geotiff_target),
            output_dir=temp_dir
        )
        
        scale = 0.5
        source_img, target_img = preprocessor.load_downsampled(scale)
        
        # Images should be numpy arrays
        assert isinstance(source_img, np.ndarray)
        assert isinstance(target_img, np.ndarray)
        
        # Downsampled images should be smaller
        assert source_img.shape[0] <= 100
        assert source_img.shape[1] <= 100
        assert target_img.shape[0] <= 200
        assert target_img.shape[1] <= 200
    
    def test_compute_overlap_region(self, sample_geotiff_source, sample_geotiff_target, temp_dir):
        """Test overlap region computation."""
        preprocessor = ImagePreprocessor(
            source_path=str(sample_geotiff_source),
            target_path=str(sample_geotiff_target),
            output_dir=temp_dir
        )
        
        scale = 0.5
        overlap_info = preprocessor.compute_overlap_region(scale)
        
        # Should return overlap information or None
        if overlap_info is not None:
            assert 'source' in overlap_info
            assert 'target' in overlap_info
            assert 'bounds' in overlap_info


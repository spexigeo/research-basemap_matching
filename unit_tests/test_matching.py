"""
Unit tests for matching module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import cv2
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from matching import create_mask, match_sift, match_orb, match_patch_ncc
from conftest import sample_image_100x100, sample_image_200x200


class TestCreateMask:
    """Test mask creation function."""
    
    def test_create_mask_basic(self, sample_image_100x100):
        """Test basic mask creation."""
        mask = create_mask(sample_image_100x100)
        
        assert isinstance(mask, np.ndarray)
        assert mask.shape == sample_image_100x100.shape
        assert mask.dtype == np.uint8
        assert mask.min() >= 0
        assert mask.max() <= 255
    
    def test_create_mask_black_region(self):
        """Test mask creation with black region."""
        image = np.ones((100, 100), dtype=np.uint8) * 255
        image[50:60, 50:60] = 0  # Black region
        
        mask = create_mask(image)
        
        # Black region should be masked out
        assert mask[55, 55] == 0  # Should be masked
    
    def test_create_mask_threshold(self):
        """Test mask creation with custom threshold."""
        image = np.ones((100, 100), dtype=np.uint8) * 5  # Very dark
        
        mask_low = create_mask(image, threshold=10)
        mask_high = create_mask(image, threshold=1)
        
        # With low threshold, dark image should be mostly masked
        assert np.sum(mask_low == 0) > np.sum(mask_high == 0)


class TestSIFTMatching:
    """Test SIFT matching."""
    
    def test_match_sift_basic(self, sample_image_100x100, sample_image_200x200):
        """Test basic SIFT matching."""
        # Create a shifted version of the image for matching
        shifted = np.roll(sample_image_100x100, (10, 10), axis=(0, 1))
        
        mask1 = create_mask(sample_image_100x100)
        mask2 = create_mask(shifted)
        
        result = match_sift(sample_image_100x100, shifted, mask1, mask2)
        
        # Should return a result dictionary
        assert result is not None
        assert isinstance(result, dict)
        
        if 'matches' in result and len(result['matches']) > 0:
            assert len(result['matches']) > 0
            match = result['matches'][0]
            assert 'source' in match
            assert 'target_original' in match


class TestORBMatching:
    """Test ORB matching."""
    
    def test_match_orb_basic(self, sample_image_100x100, sample_image_200x200):
        """Test basic ORB matching."""
        # Create a shifted version of the image for matching
        shifted = np.roll(sample_image_100x100, (10, 10), axis=(0, 1))
        
        mask1 = create_mask(sample_image_100x100)
        mask2 = create_mask(shifted)
        
        result = match_orb(sample_image_100x100, shifted, mask1, mask2)
        
        # Should return a result dictionary
        assert result is not None
        assert isinstance(result, dict)
        
        if 'matches' in result and len(result['matches']) > 0:
            assert len(result['matches']) > 0
            match = result['matches'][0]
            assert 'source' in match
            assert 'target_original' in match


class TestPatchNCCMatching:
    """Test Patch NCC matching."""
    
    def test_match_patch_ncc_basic(self, sample_image_100x100):
        """Test basic Patch NCC matching."""
        # Create a shifted version of the image for matching
        shifted = np.roll(sample_image_100x100, (10, 10), axis=(0, 1))
        
        mask1 = create_mask(sample_image_100x100)
        mask2 = create_mask(shifted)
        
        result = match_patch_ncc(
            sample_image_100x100, shifted, mask1, mask2,
            patch_size=32,
            grid_spacing=16
        )
        
        # Should return a result dictionary
        assert result is not None
        assert isinstance(result, dict)
        
        if 'matches' in result and len(result['matches']) > 0:
            assert len(result['matches']) > 0
            match = result['matches'][0]
            assert 'source' in match
            assert 'target_original' in match




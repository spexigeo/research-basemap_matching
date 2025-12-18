"""
Unit tests for transformations module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import json
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformations import (
    compute_2d_shift,
    compute_similarity_transform,
    compute_affine_transform,
    compute_homography,
    remove_gross_outliers
)
from conftest import sample_matches


class Test2DShift:
    """Test 2D shift transformation computation."""
    
    def test_compute_2d_shift_basic(self, sample_matches):
        """Test basic 2D shift computation."""
        src_points, tgt_points = sample_matches
        
        result = compute_2d_shift(src_points, tgt_points)
        
        assert result is not None
        assert 'matrix' in result
        assert 'matrix_meters' in result
        assert 'error' in result
        assert 'inliers' in result
        
        # Matrix should be 2x3
        assert result['matrix'].shape == (2, 3)
        
        # Error should be non-negative
        assert result['error'] >= 0
    
    def test_compute_2d_shift_identity(self):
        """Test 2D shift with no actual shift."""
        src_points = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
        tgt_points = src_points.copy()  # No shift
        
        result = compute_2d_shift(src_points, tgt_points)
        
        # Translation should be close to zero
        assert abs(result['matrix'][0, 2]) < 1.0
        assert abs(result['matrix'][1, 2]) < 1.0


class TestSimilarityTransform:
    """Test similarity transformation computation."""
    
    def test_compute_similarity_basic(self, sample_matches):
        """Test basic similarity transform computation."""
        src_points, tgt_points = sample_matches
        
        result = compute_similarity_transform(src_points, tgt_points)
        
        assert result is not None
        assert 'matrix' in result
        assert 'matrix_meters' in result
        assert 'error' in result
        assert 'inliers' in result
        
        # Matrix should be 2x3
        assert result['matrix'].shape == (2, 3)
        
        # Error should be non-negative
        assert result['error'] >= 0


class TestAffineTransform:
    """Test affine transformation computation."""
    
    def test_compute_affine_basic(self, sample_matches):
        """Test basic affine transform computation."""
        src_points, tgt_points = sample_matches
        
        result = compute_affine_transform(src_points, tgt_points)
        
        assert result is not None
        assert 'matrix' in result
        assert 'matrix_meters' in result
        assert 'error' in result
        assert 'inliers' in result
        
        # Matrix should be 2x3
        assert result['matrix'].shape == (2, 3)
        
        # Error should be non-negative
        assert result['error'] >= 0


class TestHomography:
    """Test homography transformation computation."""
    
    def test_compute_homography_basic(self, sample_matches):
        """Test basic homography computation."""
        src_points, tgt_points = sample_matches
        
        result = compute_homography(src_points, tgt_points)
        
        assert result is not None
        assert 'matrix' in result
        assert 'matrix_meters' in result
        assert 'error' in result
        assert 'inliers' in result
        
        # Matrix should be 3x3
        assert result['matrix'].shape == (3, 3)
        
        # Error should be non-negative
        assert result['error'] >= 0


class TestOutlierRemoval:
    """Test outlier removal."""
    
    def test_remove_gross_outliers_basic(self, sample_matches):
        """Test basic outlier removal."""
        src_points, tgt_points = sample_matches
        
        # Add some outliers
        src_with_outliers = np.vstack([src_points, [[1000, 1000], [2000, 2000]]])
        tgt_with_outliers = np.vstack([tgt_points, [[10, 10], [20, 20]]])
        
        filtered_src, filtered_tgt = remove_gross_outliers(
            src_with_outliers, tgt_with_outliers, threshold=100.0
        )
        
        # Should have fewer points after filtering
        assert len(filtered_src) <= len(src_with_outliers)
        assert len(filtered_tgt) <= len(tgt_with_outliers)
        
        # Should have at least some points remaining
        assert len(filtered_src) > 0
        assert len(filtered_tgt) > 0
    
    def test_remove_gross_outliers_no_outliers(self, sample_matches):
        """Test outlier removal with no outliers."""
        src_points, tgt_points = sample_matches
        
        filtered_src, filtered_tgt = remove_gross_outliers(
            src_points, tgt_points, threshold=1000.0
        )
        
        # Should keep all points if threshold is high
        assert len(filtered_src) == len(src_points)
        assert len(filtered_tgt) == len(tgt_points)


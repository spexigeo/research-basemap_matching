"""
Unit tests for defaults module.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from defaults import (
    DEFAULT_SCALES,
    DEFAULT_ALGORITHMS,
    DEFAULT_MATCHER,
    DEFAULT_DEBUG_LEVEL,
    DEFAULT_OUTPUT_DIR
)


class TestDefaults:
    """Test default configuration values."""
    
    def test_default_scales(self):
        """Test that default scales are defined correctly."""
        assert DEFAULT_SCALES is not None
        assert isinstance(DEFAULT_SCALES, list)
        assert len(DEFAULT_SCALES) > 0
        
        # All scales should be between 0 and 1
        for scale in DEFAULT_SCALES:
            assert 0 < scale <= 1
            assert isinstance(scale, float)
        
        # Scales should be in ascending order
        assert DEFAULT_SCALES == sorted(DEFAULT_SCALES)
    
    def test_default_algorithms(self):
        """Test that default algorithms are defined correctly."""
        assert DEFAULT_ALGORITHMS is not None
        assert isinstance(DEFAULT_ALGORITHMS, list)
        assert len(DEFAULT_ALGORITHMS) == len(DEFAULT_SCALES)
        
        # Valid algorithm names
        valid_algorithms = ['shift', 'similarity', 'affine', 'homography', 
                          'polynomial_2', 'polynomial_3', 'spline']
        
        for algo in DEFAULT_ALGORITHMS:
            assert algo in valid_algorithms
    
    def test_default_matcher(self):
        """Test that default matcher is defined correctly."""
        assert DEFAULT_MATCHER is not None
        assert isinstance(DEFAULT_MATCHER, str)
        
        # Valid matcher names
        valid_matchers = ['lightglue', 'sift', 'orb', 'patch_ncc']
        assert DEFAULT_MATCHER in valid_matchers
    
    def test_default_debug_level(self):
        """Test that default debug level is defined correctly."""
        assert DEFAULT_DEBUG_LEVEL is not None
        assert isinstance(DEFAULT_DEBUG_LEVEL, str)
        
        # Valid debug levels
        valid_levels = ['none', 'intermediate', 'high']
        assert DEFAULT_DEBUG_LEVEL in valid_levels
    
    def test_default_output_dir(self):
        """Test that default output directory is defined correctly."""
        assert DEFAULT_OUTPUT_DIR is not None
        assert isinstance(DEFAULT_OUTPUT_DIR, str)
        assert len(DEFAULT_OUTPUT_DIR) > 0





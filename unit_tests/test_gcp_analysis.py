"""
Unit tests for GCP analysis module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import csv
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gcp_analysis import load_gcps_from_csv, load_gcps_from_file


class TestGCPLoading:
    """Test GCP loading functions."""
    
    def test_load_gcps_from_csv_basic(self, temp_dir):
        """Test loading GCPs from CSV file."""
        csv_path = temp_dir / "test_gcps.csv"
        
        # Create a sample CSV file
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Label', 'X', 'Y', 'Z', 'Accuracy', 'Enabled'])
            writer.writerow(['GCP_001', '-123.5', '49.0', '10.5', '0.005', '1'])
            writer.writerow(['GCP_002', '-123.4', '49.1', '11.0', '0.005', '1'])
            writer.writerow(['GCP_003', '-123.3', '49.2', '12.0', '0.005', '0'])  # Disabled
        
        gcps = load_gcps_from_csv(str(csv_path))
        
        # Should load 2 enabled GCPs
        assert len(gcps) == 2
        assert gcps[0]['id'] == 'GCP_001'
        assert gcps[0]['lon'] == -123.5
        assert gcps[0]['lat'] == 49.0
        assert gcps[0]['z'] == 10.5
    
    def test_load_gcps_from_csv_tab_separated(self, temp_dir):
        """Test loading GCPs from tab-separated CSV file."""
        csv_path = temp_dir / "test_gcps.tsv"
        
        # Create a sample TSV file
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['Label', 'X', 'Y', 'Z'])
            writer.writerow(['GCP_001', '-123.5', '49.0', '10.5'])
        
        gcps = load_gcps_from_csv(str(csv_path))
        
        # Should load 1 GCP
        assert len(gcps) == 1
        assert gcps[0]['id'] == 'GCP_001'
    
    def test_load_gcps_from_csv_case_insensitive(self, temp_dir):
        """Test that CSV loading is case-insensitive for column names."""
        csv_path = temp_dir / "test_gcps.csv"
        
        # Create CSV with lowercase column names
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['label', 'x', 'y', 'z'])
            writer.writerow(['GCP_001', '-123.5', '49.0', '10.5'])
        
        gcps = load_gcps_from_csv(str(csv_path))
        
        # Should still load correctly
        assert len(gcps) == 1
        assert gcps[0]['id'] == 'GCP_001'
    
    def test_load_gcps_from_file_csv(self, temp_dir):
        """Test load_gcps_from_file with CSV."""
        csv_path = temp_dir / "test_gcps.csv"
        
        # Create a sample CSV file
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Label', 'X', 'Y', 'Z'])
            writer.writerow(['GCP_001', '-123.5', '49.0', '10.5'])
        
        gcps = load_gcps_from_file(str(csv_path))
        
        # Should load correctly
        assert len(gcps) == 1
        assert gcps[0]['id'] == 'GCP_001'
    
    def test_load_gcps_from_file_missing_file(self, temp_dir):
        """Test load_gcps_from_file with missing file."""
        missing_path = temp_dir / "nonexistent.csv"
        
        with pytest.raises(FileNotFoundError):
            load_gcps_from_file(str(missing_path))




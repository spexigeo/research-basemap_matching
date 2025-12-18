"""
GCP Analysis Module

Extracts patches from registered orthomosaic centered at GCP locations
and visualizes them with red dots marking the GCP position.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import rasterio
from rasterio.windows import Window
import cv2
from PIL import Image
import pyproj
from affine import Affine

# Import KMZ parser from the other repo
# Try to import the functions we need
KMZ_PARSER_AVAILABLE = False
parse_kmz_file = None
load_gcps_from_kmz = None

try:
    # Try direct import if the module is in the same parent directory
    import importlib.util
    kmz_parser_path = Path(__file__).parent.parent / 'research-qualicum_beach_gcp_analysis' / 'qualicum_beach_gcp_analysis' / 'kmz_parser.py'
    if kmz_parser_path.exists():
        # Read the file and extract just the functions we need
        # This avoids import errors from dependencies
        import ast
        import types
        
        with open(kmz_parser_path, 'r') as f:
            code = f.read()
        
        # Compile and execute in a new namespace
        namespace = {}
        # Add minimal imports that the code needs
        namespace['zipfile'] = __import__('zipfile')
        namespace['ET'] = __import__('xml.etree.ElementTree')
        namespace['re'] = __import__('re')
        namespace['os'] = __import__('os')
        namespace['Path'] = Path
        namespace['List'] = List
        namespace['Dict'] = Dict
        namespace['Optional'] = Optional
        namespace['Tuple'] = Tuple
        
        exec(compile(code, str(kmz_parser_path), 'exec'), namespace)
        
        parse_kmz_file = namespace.get('parse_kmz_file')
        load_gcps_from_kmz = namespace.get('load_gcps_from_kmz')
        
        if parse_kmz_file and load_gcps_from_kmz:
            KMZ_PARSER_AVAILABLE = True
except Exception as e:
    # If import fails, we'll handle it gracefully
    pass


def load_gcps_from_file(gcp_path: str) -> List[Dict]:
    """
    Load GCPs from either CSV or KMZ file.
    
    Args:
        gcp_path: Path to CSV or KMZ file
        
    Returns:
        List of GCP dictionaries with keys: 'id', 'lat', 'lon', 'z' (optional)
    """
    gcp_path = Path(gcp_path)
    
    if not gcp_path.exists():
        raise FileNotFoundError(f"GCP file not found: {gcp_path}")
    
    if gcp_path.suffix.lower() == '.kmz':
        if not KMZ_PARSER_AVAILABLE:
            raise ImportError("KMZ parser not available. Cannot read KMZ files.")
        print(f"Loading GCPs from KMZ file: {gcp_path}")
        gcps = load_gcps_from_kmz(str(gcp_path))
        # Ensure consistent format
        for gcp in gcps:
            if 'id' not in gcp:
                gcp['id'] = gcp.get('label', f"GCP_{gcps.index(gcp)+1:03d}")
            if 'z' not in gcp:
                gcp['z'] = gcp.get('elevation', gcp.get('altitude', 0.0))
        return gcps
    elif gcp_path.suffix.lower() == '.csv':
        return load_gcps_from_csv(str(gcp_path))
    else:
        raise ValueError(f"Unsupported GCP file format: {gcp_path.suffix}. Use .csv or .kmz")


def load_gcps_from_csv(csv_path: str) -> List[Dict]:
    """
    Load GCPs from CSV file (supports both comma and tab separated).
    
    Expected format: Label, X (lon), Y (lat), Z (optional), Accuracy (optional), Enabled (optional)
    """
    import csv
    
    gcps = []
    with open(csv_path, 'r') as f:
        # Try to detect delimiter
        first_line = f.readline()
        f.seek(0)
        
        if '\t' in first_line:
            delimiter = '\t'
        else:
            delimiter = ','
        
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            # Handle case-insensitive column names
            label_key = next((k for k in row.keys() if k.lower() == 'label'), 'Label')
            x_key = next((k for k in row.keys() if k.lower() == 'x'), 'X')
            y_key = next((k for k in row.keys() if k.lower() == 'y'), 'Y')
            z_key = next((k for k in row.keys() if k.lower() == 'z'), 'Z')
            enabled_key = next((k for k in row.keys() if k.lower() == 'enabled'), 'Enabled')
            
            # Check if enabled (default to enabled if not specified)
            enabled = row.get(enabled_key, '1').strip() == '1'
            
            if enabled:
                gcps.append({
                    'id': row[label_key].strip() if label_key in row else f"GCP_{len(gcps)+1:03d}",
                    'lon': float(row[x_key]),
                    'lat': float(row[y_key]),
                    'z': float(row.get(z_key, '0.0')) if z_key in row and row[z_key].strip() else 0.0
                })
    
    return gcps


def lonlat_to_pixel(lon: float, lat: float, transform: Affine, crs) -> Tuple[float, float]:
    """Convert geographic coordinates (WGS84 lon/lat) to pixel coordinates."""
    # Always transform from WGS84 to the image CRS
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS.from_epsg(4326),  # WGS84 (lon/lat)
        crs,
        always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    
    # Convert to pixel coordinates
    inv_transform = ~transform
    col, row = inv_transform * (x, y)
    return col, row


def extract_gcp_patch(orthomosaic_path: str, gcp: Dict, patch_size: int = 300) -> Optional[np.ndarray]:
    """
    Extract a patch from the orthomosaic centered at the GCP location.
    
    Args:
        orthomosaic_path: Path to registered orthomosaic
        gcp: GCP dictionary with 'lat', 'lon', 'id'
        patch_size: Size of patch in pixels (default: 300x300)
        
    Returns:
        Image patch as numpy array, or None if GCP is outside image bounds
    """
    with rasterio.open(orthomosaic_path) as src:
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
        
        # Convert GCP lat/lon to pixel coordinates
        col, row = lonlat_to_pixel(gcp['lon'], gcp['lat'], transform, crs)
        
        # Check if GCP is within image bounds
        if col < 0 or col >= width or row < 0 or row >= height:
            print(f"  Warning: GCP {gcp.get('id', 'unknown')} is outside image bounds (col={col:.1f}, row={row:.1f})")
            return None
        
        # Calculate patch bounds
        half_size = patch_size // 2
        col_start = max(0, int(col) - half_size)
        col_end = min(width, int(col) + half_size)
        row_start = max(0, int(row) - half_size)
        row_end = min(height, int(row) + half_size)
        
        # Adjust if patch would be smaller than requested (at image edges)
        actual_col_size = col_end - col_start
        actual_row_size = row_end - row_start
        
        if actual_col_size < patch_size:
            if col_start == 0:
                col_end = min(width, patch_size)
            else:
                col_start = max(0, col_end - patch_size)
        
        if actual_row_size < patch_size:
            if row_start == 0:
                row_end = min(height, patch_size)
            else:
                row_start = max(0, row_end - patch_size)
        
        # Read patch
        window = Window.from_slices(
            (row_start, row_end),
            (col_start, col_end)
        )
        
        # Read all bands
        patch = src.read(window=window)
        
        # Handle multi-band images - convert to RGB if needed
        if len(patch.shape) == 3:
            if patch.shape[0] == 1:
                # Single band - convert to grayscale RGB
                patch = np.stack([patch[0], patch[0], patch[0]], axis=0)
            elif patch.shape[0] == 3:
                # RGB
                pass
            elif patch.shape[0] == 4:
                # RGBA - take first 3 bands
                patch = patch[:3]
            else:
                # Take first 3 bands
                patch = patch[:3]
        
        # Transpose from (bands, height, width) to (height, width, bands)
        patch = np.transpose(patch, (1, 2, 0))
        
        # Normalize to 0-255 if needed
        if patch.dtype != np.uint8:
            if patch.max() > 255:
                patch = (patch / patch.max() * 255).astype(np.uint8)
            else:
                patch = patch.astype(np.uint8)
        
        # Calculate GCP position relative to patch
        gcp_col_in_patch = col - col_start
        gcp_row_in_patch = row - row_start
        
        # Draw red dot at GCP location
        patch_with_marker = patch.copy()
        
        # Draw a large red circle (radius ~10 pixels)
        radius = 10
        cv2.circle(patch_with_marker, 
                  (int(gcp_col_in_patch), int(gcp_row_in_patch)), 
                  radius, 
                  (0, 0, 255),  # Red in BGR
                  -1)  # Filled circle
        
        # Draw a white border around the red dot for visibility
        cv2.circle(patch_with_marker,
                  (int(gcp_col_in_patch), int(gcp_row_in_patch)),
                  radius + 2,
                  (255, 255, 255),  # White in BGR
                  2)  # Border width
        
        return patch_with_marker


def analyze_gcps(registered_orthomosaic_path: str, gcp_file_path: str, output_dir: str, patch_size: int = 300):
    """
    Analyze GCPs by extracting patches from registered orthomosaic.
    
    Args:
        registered_orthomosaic_path: Path to final registered orthomosaic
        gcp_file_path: Path to CSV or KMZ file containing GCPs
        output_dir: Output directory for GCP analysis results
        patch_size: Size of patches to extract (default: 300x300 pixels)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("GCP Analysis")
    print(f"{'='*80}")
    print(f"Registered orthomosaic: {registered_orthomosaic_path}")
    print(f"GCP file: {gcp_file_path}")
    print(f"Output directory: {output_dir}")
    print(f"Patch size: {patch_size}x{patch_size} pixels")
    
    # Load GCPs
    try:
        gcps = load_gcps_from_file(gcp_file_path)
        print(f"\nLoaded {len(gcps)} GCPs from {gcp_file_path}")
    except Exception as e:
        print(f"\n✗ Error loading GCPs: {e}")
        return
    
    if len(gcps) == 0:
        print("\n✗ No GCPs found in file")
        return
    
    # Extract patches for each GCP
    successful_extractions = 0
    failed_extractions = 0
    
    for gcp in gcps:
        gcp_id = gcp.get('id', f"GCP_{gcps.index(gcp)+1:03d}")
        print(f"\nProcessing GCP: {gcp_id} (lat={gcp['lat']:.6f}, lon={gcp['lon']:.6f})")
        
        try:
            patch = extract_gcp_patch(registered_orthomosaic_path, gcp, patch_size)
            
            if patch is not None:
                # Save patch
                # Sanitize GCP ID for filename
                safe_id = "".join(c for c in gcp_id if c.isalnum() or c in ('_', '-')).strip()
                output_path = output_dir / f"{safe_id}.png"
                
                # Save using PIL to handle large images
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for PIL
                patch_image = Image.fromarray(patch_rgb)
                patch_image.save(output_path)
                
                print(f"  ✓ Saved patch to: {output_path.name}")
                successful_extractions += 1
            else:
                print(f"  ✗ Failed to extract patch (GCP outside image bounds)")
                failed_extractions += 1
                
        except Exception as e:
            print(f"  ✗ Error extracting patch: {e}")
            failed_extractions += 1
    
    print(f"\n{'='*80}")
    print("GCP Analysis Summary")
    print(f"{'='*80}")
    print(f"Total GCPs: {len(gcps)}")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions: {failed_extractions}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")


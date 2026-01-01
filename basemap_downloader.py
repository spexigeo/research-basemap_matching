"""Basemap downloader utilities for downloading map tiles."""

import math
import time
import requests
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import io
from pathlib import Path
import warnings


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to tile coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """Convert tile coordinates to lat/lon of top-left corner."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def get_tile_url(xtile: int, ytile: int, zoom: int, source: str = "openstreetmap") -> str:
    """Get URL for a tile."""
    if source == "openstreetmap":
        return f"https://tile.openstreetmap.org/{zoom}/{xtile}/{ytile}.png"
    elif source == "esri_world_imagery" or source == "esri":
        return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
    elif source == "google_satellite":
        return f"https://mt1.google.com/vt/lyrs=s&x={xtile}&y={ytile}&z={zoom}"
    elif source == "google_hybrid":
        return f"https://mt1.google.com/vt/lyrs=y&x={xtile}&y={ytile}&z={zoom}"
    else:
        raise ValueError(f"Unknown tile source: {source}. Supported: 'openstreetmap', 'esri_world_imagery'/'esri', 'google_satellite', 'google_hybrid'")


def download_tile(xtile: int, ytile: int, zoom: int, source: str = "openstreetmap", 
                  verbose: bool = False, retries: int = 2) -> Optional[Image.Image]:
    """Download a single tile with retry logic."""
    url = get_tile_url(xtile, ytile, zoom, source)
    
    headers = {
        'User-Agent': 'research-basemap-matching/1.0.0'
    }
    
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            return img
        except requests.exceptions.HTTPError as e:
            if attempt < retries:
                time.sleep(0.1 * (attempt + 1))
                continue
            if verbose:
                status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'unknown'
                print(f"Warning: HTTP error downloading tile {zoom}/{xtile}/{ytile}: {e} (Status: {status_code})")
            return None
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(0.1 * (attempt + 1))
                continue
            if verbose:
                print(f"Warning: Request error downloading tile {zoom}/{xtile}/{ytile}: {e}")
            return None
        except Exception as e:
            if attempt < retries:
                time.sleep(0.1 * (attempt + 1))
                continue
            if verbose:
                print(f"Warning: Failed to download tile {zoom}/{xtile}/{ytile}: {e}")
            return None
    
    return None


def calculate_zoom_level(bbox: Tuple[float, float, float, float], 
                         max_tiles: int = 64, 
                         target_resolution: Optional[float] = None) -> int:
    """
    Calculate appropriate zoom level based on bounding box size or target resolution.
    
    Args:
        bbox: Bounding box as (min_lat, min_lon, max_lat, max_lon)
        max_tiles: Maximum number of tiles to download (default 64)
        target_resolution: Target resolution in meters per pixel (optional)
        
    Returns:
        Zoom level
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    
    if target_resolution:
        # Calculate zoom based on target resolution using the same formula as the notebook
        center_lat = (min_lat + max_lat) / 2
        
        # Earth's circumference at equator in meters
        earth_circumference = 40075017.0  # meters
        
        # Find the zoom level that gives resolution closest to (but not too far above) target_resolution
        # Iterate from high zoom (19) down to low zoom (12) to find best match
        best_zoom = None
        best_diff = float('inf')
        
        for zoom in range(19, 11, -1):
            # Resolution at equator for given zoom level
            resolution_equator = earth_circumference / (256 * (2 ** zoom))
            # Adjust for latitude (pixels get smaller as you move away from equator)
            resolution = resolution_equator * math.cos(math.radians(center_lat))
            
            # Prefer resolutions <= target, but if none found, use closest
            if resolution <= target_resolution:
                # This is ideal - use it
                best_zoom = zoom
                break
            else:
                # Resolution is above target, but might be closest
                diff = resolution - target_resolution
                if diff < best_diff:
                    best_diff = diff
                    best_zoom = zoom
        
        if best_zoom is None:
            # Fallback to zoom 18 if no suitable zoom found
            best_zoom = 18
        
        return best_zoom
    
    # Calculate based on bounding box size
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    area_deg2 = lat_range * lon_range
    
    if area_deg2 < 0.0001:
        base_zoom = 16
    elif area_deg2 < 0.001:
        base_zoom = 15
    elif area_deg2 < 0.01:
        base_zoom = 13
    elif area_deg2 < 0.1:
        base_zoom = 11
    else:
        base_zoom = 9
    
    # Check tile count and adjust if needed
    for zoom in range(base_zoom, base_zoom - 5, -1):
        if zoom < 1:
            break
        xtile_min, ytile_min = deg2num(min_lat, min_lon, zoom)
        xtile_max, ytile_max = deg2num(max_lat, max_lon, zoom)
        
        num_tiles = (xtile_max - xtile_min + 1) * (ytile_max - ytile_min + 1)
        if num_tiles <= max_tiles:
            return zoom
    
    return max(1, base_zoom)


def h3_cells_to_bbox(h3_cells: List[str]) -> Tuple[float, float, float, float]:
    """
    Convert a list of H3 cell identifiers to a latitude/longitude bounding box.
    
    Args:
        h3_cells: List of H3 cell identifiers (hex strings)
        
    Returns:
        Tuple of (min_lat, min_lon, max_lat, max_lon)
    """
    try:
        import h3
    except ImportError:
        raise ImportError("h3 package is required for H3 cell support. Install with: pip install h3")
    
    if not h3_cells:
        raise ValueError("H3 cells list cannot be empty")
    
    all_lats = []
    all_lons = []
    
    for cell in h3_cells:
        # Validate H3 cell
        if not h3.is_valid_cell(cell):
            raise ValueError(f"Invalid H3 cell: {cell}")
        
        # Get cell boundary (tuple of (lat, lng) tuples)
        boundary = h3.cell_to_boundary(cell)
        
        # Extract latitudes and longitudes
        for lat, lon in boundary:
            all_lats.append(lat)
            all_lons.append(lon)
    
    min_lat = min(all_lats)
    max_lat = max(all_lats)
    min_lon = min(all_lons)
    max_lon = max(all_lons)
    
    return (min_lat, min_lon, max_lat, max_lon)


def load_h3_cells_from_file(file_path: Path) -> List[str]:
    """
    Load H3 cells from a text file (one per line) or XML file.
    
    Args:
        file_path: Path to file containing H3 cells (text or XML)
        
    Returns:
        List of H3 cell identifiers
    """
    h3_cells = []
    file_path = Path(file_path)
    
    # Check if it's an XML file
    if file_path.suffix.lower() == '.xml':
        # Parse XML file (Word XML format)
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Find all text elements containing H3 cells
            # H3 cells are typically 15-character hex strings
            for elem in root.iter():
                if elem.text:
                    text = elem.text.strip()
                    # H3 cells are typically 15-character hex strings
                    if len(text) == 15 and all(c in '0123456789abcdef' for c in text.lower()):
                        h3_cells.append(text)
            
            # Remove duplicates and return
            return list(set(h3_cells))
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse XML file {file_path}: {e}")
    else:
        # Plain text file (one H3 cell per line)
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    h3_cells.append(line)
        return h3_cells


def parse_bbox_string(bbox_str: str) -> Tuple[float, float, float, float]:
    """
    Parse bounding box from string format.
    
    Formats supported:
    - "min_lat,min_lon,max_lat,max_lon"
    - "min_lat min_lon max_lat max_lon"
    
    Args:
        bbox_str: Bounding box string
        
    Returns:
        Tuple of (min_lat, min_lon, max_lat, max_lon)
    """
    # Try comma-separated first
    if ',' in bbox_str:
        parts = [float(x.strip()) for x in bbox_str.split(',')]
    else:
        parts = [float(x.strip()) for x in bbox_str.split()]
    
    if len(parts) != 4:
        raise ValueError(f"Bounding box must have 4 values, got {len(parts)}")
    
    min_lat, min_lon, max_lat, max_lon = parts
    
    if min_lat >= max_lat:
        raise ValueError(f"min_lat ({min_lat}) must be less than max_lat ({max_lat})")
    if min_lon >= max_lon:
        raise ValueError(f"min_lon ({min_lon}) must be less than max_lon ({max_lon})")
    
    return (min_lat, min_lon, max_lat, max_lon)


def download_basemap(
    bbox: Tuple[float, float, float, float],
    output_path: str,
    source: str = "openstreetmap",
    zoom: Optional[int] = None,
    target_resolution: Optional[float] = None
) -> str:
    """
    Download basemap tiles and create a GeoTIFF.
    
    Args:
        bbox: Bounding box as (min_lat, min_lon, max_lat, max_lon)
        output_path: Path to save GeoTIFF
        source: Tile source ('openstreetmap', 'esri_world_imagery'/'esri', 'google_satellite', 'google_hybrid')
        zoom: Zoom level (auto-calculated if None)
        target_resolution: Target resolution in meters per pixel
        
    Returns:
        Path to saved GeoTIFF
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    
    # Normalize source name
    if source.lower() == "esri":
        source = "esri_world_imagery"
    
    # Warn about Google tile usage
    if source in ("google_satellite", "google_hybrid"):
        warnings.warn(
            "Google Satellite tiles: Direct tile access may be subject to Google's Terms of Service. "
            "For production/commercial use, consider Google Earth Engine API with proper authentication.",
            UserWarning
        )
    
    # Validate bounding box
    if min_lat >= max_lat:
        raise ValueError(f"Invalid bounding box: min_lat ({min_lat}) must be less than max_lat ({max_lat})")
    if min_lon >= max_lon:
        raise ValueError(f"Invalid bounding box: min_lon ({min_lon}) must be less than max_lon ({max_lon})")
    
    if zoom is None:
        zoom = calculate_zoom_level(bbox, target_resolution=target_resolution)
    
    print(f"Downloading basemap at zoom level {zoom}...")
    
    # Calculate tile range
    xtile_min, ytile_max = deg2num(min_lat, min_lon, zoom)
    xtile_max, ytile_min = deg2num(max_lat, max_lon, zoom)
    
    # Ensure correct ordering
    if xtile_min > xtile_max:
        xtile_min, xtile_max = xtile_max, xtile_min
    if ytile_min > ytile_max:
        ytile_min, ytile_max = ytile_max, ytile_min
    
    print(f"Tile range: X [{xtile_min}, {xtile_max}], Y [{ytile_min}, {ytile_max}]")
    
    # Download tiles
    tiles = []
    total_tiles = (xtile_max - xtile_min + 1) * (ytile_max - ytile_min + 1)
    downloaded = 0
    
    for y in range(ytile_min, ytile_max + 1):
        row = []
        for x in range(xtile_min, xtile_max + 1):
            tile = download_tile(x, y, zoom, source, verbose=True)
            if tile is None:
                # Create blank tile
                tile = Image.new('RGB', (256, 256), color=(128, 128, 128))
            row.append(tile)
            downloaded += 1
            if downloaded % 10 == 0:
                print(f"  Downloaded {downloaded}/{total_tiles} tiles...")
        tiles.append(row)
    
    print(f"  Downloaded {downloaded}/{total_tiles} tiles")
    
    # Stitch tiles together
    tile_height = tiles[0][0].height
    tile_width = tiles[0][0].width
    
    stitched = Image.new('RGB', 
                        ((xtile_max - xtile_min + 1) * tile_width,
                         (ytile_max - ytile_min + 1) * tile_height))
    
    for y_idx, row in enumerate(tiles):
        for x_idx, tile in enumerate(row):
            x_pos = (x_idx) * tile_width
            y_pos = (y_idx) * tile_height
            stitched.paste(tile, (x_pos, y_pos))
    
    # Get bounds of stitched image
    top_left_lat, top_left_lon = num2deg(xtile_min, ytile_min, zoom)
    bottom_right_lat, bottom_right_lon = num2deg(xtile_max + 1, ytile_max + 1, zoom)
    
    # Crop to requested bounds
    pixels_per_degree_lon = stitched.width / (bottom_right_lon - top_left_lon)
    pixels_per_degree_lat = stitched.height / (top_left_lat - bottom_right_lat)
    
    left_pixel = int((min_lon - top_left_lon) * pixels_per_degree_lon)
    top_pixel = int((top_left_lat - max_lat) * pixels_per_degree_lat)
    right_pixel = int((max_lon - top_left_lon) * pixels_per_degree_lon)
    bottom_pixel = int((top_left_lat - min_lat) * pixels_per_degree_lat)
    
    left_pixel = max(0, left_pixel)
    top_pixel = max(0, top_pixel)
    right_pixel = min(stitched.width, right_pixel)
    bottom_pixel = min(stitched.height, bottom_pixel)
    
    # Ensure valid crop rectangle
    if right_pixel <= left_pixel:
        right_pixel = left_pixel + 1
        if right_pixel > stitched.width:
            left_pixel = stitched.width - 1
            right_pixel = stitched.width
    if bottom_pixel <= top_pixel:
        bottom_pixel = top_pixel + 1
        if bottom_pixel > stitched.height:
            top_pixel = stitched.height - 1
            bottom_pixel = stitched.height
    
    cropped = stitched.crop((left_pixel, top_pixel, right_pixel, bottom_pixel))
    
    # Save as GeoTIFF
    width, height = cropped.size
    
    # Validate dimensions
    if width == 0 or height == 0:
        raise ValueError(f"Invalid cropped image dimensions: {width}x{height}. "
                        f"This may indicate an invalid bounding box or tile calculation issue.")
    
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
    
    array = np.array(cropped)
    
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=array.dtype,
        crs=CRS.from_epsg(4326),  # WGS84
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(array.transpose(2, 0, 1))
    
    print(f"Basemap saved to {output_path}")
    return output_path


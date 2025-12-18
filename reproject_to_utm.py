#!/usr/bin/env python3
"""
Properly reproject orthomosaic and basemap to the same UTM zone.
Handles CRS detection and ensures both images end up in compatible coordinates.
"""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import numpy as np
from pathlib import Path
import pyproj


def detect_utm_zone(lon, lat):
    """
    Detect appropriate UTM zone for a given lon/lat.
    """
    zone_number = int((lon + 180) / 6) + 1

    # Handle special cases
    if lat >= 56.0 and lat < 64.0 and lon >= 3.0 and lon < 12.0:
        zone_number = 32
    elif lat >= 72.0 and lat < 84.0:
        if lon >= 0.0 and lon < 9.0:
            zone_number = 31
        elif lon >= 9.0 and lon < 21.0:
            zone_number = 33
        elif lon >= 21.0 and lon < 33.0:
            zone_number = 35
        elif lon >= 33.0 and lon < 42.0:
            zone_number = 37

    # Northern or Southern hemisphere
    hemisphere = 'north' if lat >= 0 else 'south'

    # EPSG code
    if hemisphere == 'north':
        epsg = 32600 + zone_number
    else:
        epsg = 32700 + zone_number

    return zone_number, hemisphere, epsg


def get_image_center(src_path):
    """Get center coordinates of an image."""
    with rasterio.open(src_path) as src:
        bounds = src.bounds
        crs = src.crs

        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2

        # If in geographic coords, return as-is
        if crs and crs.is_geographic:
            return center_x, center_y, crs

        # If in projected coords, convert to geographic
        if crs:
            transformer = pyproj.Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
            lon, lat = transformer.transform(center_x, center_y)
            return lon, lat, crs

        return None, None, None


def reproject_image(input_path, output_path, target_crs, resampling=Resampling.bilinear):
    """
    Reproject an image to target CRS.
    """
    print(f"\nReprojecting: {Path(input_path).name}")
    print(f"  Output: {Path(output_path).name}")

    with rasterio.open(input_path) as src:
        # Check if already in target CRS
        if src.crs == target_crs:
            print(f"  ✓ Already in {target_crs}, copying with JPEG compression...")
            # Copy with JPEG compression
            profile = src.profile.copy()
            profile.update({
                'compress': 'JPEG',
                'jpeg_quality': 90,
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512
            })

            with rasterio.open(output_path, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    dst.write(src.read(i), i)
            return

        print(f"  Source CRS: {src.crs}")
        print(f"  Target CRS: {target_crs}")
        print(f"  Source shape: {src.width} x {src.height}")
        print(f"  Source resolution: {src.res[0]:.6f} x {src.res[1]:.6f}")

        # Calculate transform and dimensions for target CRS
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        print(f"  Target shape: {width} x {height}")
        print(f"  Target resolution: {transform.a:.6f} x {abs(transform.e):.6f}")

        # Update profile with JPEG compression
        profile = src.profile.copy()
        profile.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'compress': 'JPEG',
            'jpeg_quality': 90,
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512
        })

        # Reproject each band
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                print(f"  Processing band {i}/{src.count}...", end='', flush=True)

                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling
                )
                print(" done")

        # Verify output
        with rasterio.open(output_path) as dst:
            file_size_mb = Path(output_path).stat().st_size / (1024 ** 2)
            print(f"  ✓ Created: {dst.width} x {dst.height} @ {dst.res[0]:.6f} m/px")
            print(f"  ✓ CRS: {dst.crs}")
            print(
                f"  ✓ Compression: {dst.profile.get('compress', 'none')} (quality={dst.profile.get('jpeg_quality', 'N/A')})")
            print(f"  ✓ File size: {file_size_mb:.1f} MB")


def main():
    """
    Main reprojection workflow.
    """
    print("=" * 80)
    print("ORTHOMOSAIC AND BASEMAP REPROJECTION TO UTM")
    print("=" * 80)

    # Paths - UPDATE THESE
    orthomosaic_path = "/Users/mauriciohessflores/Documents/Code/MyCode/research-qualicum_beach_gcp_analysis/outputs/orthomosaics/orthomosaic_no_gcps.tif"
    basemap_esri_path = "/Users/mauriciohessflores/Documents/Code/MyCode/research-qualicum_beach_gcp_analysis/outputs/qualicum_beach_basemap_esri.tif"
    basemap_google_path = "/Users/mauriciohessflores/Documents/Code/MyCode/research-qualicum_beach_gcp_analysis/outputs/qualicum_beach_basemap_google.tif"

    output_dir = Path("/Users/mauriciohessflores/Documents/Code/MyCode/research-qualicum_beach_gcp_analysis/outputs")

    # Check files exist
    for path in [orthomosaic_path, basemap_esri_path]:
        if not Path(path).exists():
            print(f"ERROR: File not found: {path}")
            return

    # Detect appropriate UTM zone from orthomosaic center
    print("\nDetecting appropriate UTM zone...")
    lon, lat, src_crs = get_image_center(orthomosaic_path)

    if lon is None or lat is None:
        print("ERROR: Could not determine image center coordinates")
        return

    zone_num, hemisphere, epsg = detect_utm_zone(lon, lat)
    target_crs = CRS.from_epsg(epsg)

    print(f"  Center coordinates: {lon:.6f}°, {lat:.6f}°")
    print(f"  Detected UTM Zone: {zone_num}{hemisphere[0].upper()}")
    print(f"  Target CRS: EPSG:{epsg} ({target_crs})")

    # Reproject orthomosaic
    ortho_output = output_dir / "orthomosaic_no_gcps_utm10n.tif"
    reproject_image(orthomosaic_path, ortho_output, target_crs, Resampling.bilinear)

    # Reproject ESRI basemap
    esri_output = output_dir / "qualicum_beach_basemap_esri_utm10n.tif"
    reproject_image(basemap_esri_path, esri_output, target_crs, Resampling.bilinear)

    # Reproject Google basemap if it exists
    if Path(basemap_google_path).exists():
        google_output = output_dir / "qualicum_beach_basemap_google_utm10n.tif"
        reproject_image(basemap_google_path, google_output, target_crs, Resampling.bilinear)

    print("\n" + "=" * 80)
    print("REPROJECTION COMPLETE")
    print("=" * 80)
    print("\nOutput files:")
    print(f"  1. {ortho_output}")
    print(f"  2. {esri_output}")
    if Path(basemap_google_path).exists():
        print(f"  3. {google_output}")

    print("\nNext steps:")
    print("  1. Verify reprojection with diagnostic:")
    print(f"     python diagnose_images.py {ortho_output} {esri_output}")
    print("\n  2. Update your config to use these files:")
    print(f'     "source_path": "{ortho_output}"')
    print(f'     "target_path": "{esri_output}"')
    print("\n  3. Run registration:")
    print("     python example_m4_optimized.py")

    print("=" * 80)


if __name__ == '__main__':
    main()
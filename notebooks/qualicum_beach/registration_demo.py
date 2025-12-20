# ---
# marimo-version: 0.4.0
# ---
"""
# Orthomosaic Registration Pipeline Demo (Marimo)

This Marimo notebook demonstrates the complete hierarchical orthomosaic registration pipeline using the iterative upsample workflow.

The algorithm:
1. Creates a downsampled orthomosaic at the lowest scale (0.125)
2. Matches features, computes transformation, and applies it
3. Upsamples the transformed version to the next scale (0.25)
4. Repeats matching, transformation, and upsampling for each scale
5. Applies cumulative transformations to the full-resolution input
"""

import marimo

__generated_with = "0.4.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import sys
    from pathlib import Path
    import json
    import xml.etree.ElementTree as ET
    import numpy as np
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.plot import show
    import warnings
    warnings.filterwarnings('ignore')
    return ET, Path, json, np, plt, rasterio, show, sys, warnings


@app.cell
def __(Path, sys):
    # Add parent directory to path to import modules
    notebook_dir = Path.cwd()
    repo_root = notebook_dir.parent
    sys.path.insert(0, str(repo_root))
    
    print(f"Working directory: {notebook_dir}")
    print(f"Repository root: {repo_root}")
    return notebook_dir, repo_root


@app.cell
def __(repo_root):
    from defaults import DEFAULT_SCALES, DEFAULT_ALGORITHMS, DEFAULT_MATCHER
    from basemap_downloader import (
        download_basemap, h3_cells_to_bbox, load_h3_cells_from_file,
        parse_bbox_string
    )
    from preprocessing import ImagePreprocessor
    from matching import match_lightglue, visualize_matches, create_mask, LIGHTGLUE_AVAILABLE
    from transformations import (
        load_matches, remove_gross_outliers, compute_2d_shift, compute_homography
    )
    from register_orthomosaic import OrthomosaicRegistration
    
    print("✓ Modules imported")
    print(f"Default scales: {DEFAULT_SCALES}")
    print(f"Default matcher: {DEFAULT_MATCHER}")
    return (
        DEFAULT_ALGORITHMS,
        DEFAULT_MATCHER,
        DEFAULT_SCALES,
        ImagePreprocessor,
        LIGHTGLUE_AVAILABLE,
        OrthomosaicRegistration,
        compute_2d_shift,
        compute_homography,
        create_mask,
        download_basemap,
        h3_cells_to_bbox,
        load_h3_cells_from_file,
        load_matches,
        match_lightglue,
        parse_bbox_string,
        remove_gross_outliers,
        visualize_matches,
    )


@app.cell
def __(ET, Path, repo_root):
    # Parse H3 cells from XML file
    def parse_h3_cells_from_xml(xml_path: Path) -> list:
        """Extract H3 cell IDs from Word XML format."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        h3_cells = []
        for elem in root.iter():
            if elem.text:
                text = elem.text.strip()
                if len(text) == 15 and all(c in '0123456789abcdef' for c in text.lower()):
                    h3_cells.append(text)
        
        return list(set(h3_cells))
    
    # Load H3 cells
    h3_xml_path = repo_root / "inputs" / "qualicum_beach" / "h3_cells.xml"
    h3_cells = parse_h3_cells_from_xml(h3_xml_path)
    print(f"Found {len(h3_cells)} unique H3 cells:")
    for i, cell in enumerate(h3_cells[:5]):
        print(f"  {i+1}. {cell}")
    if len(h3_cells) > 5:
        print(f"  ... and {len(h3_cells) - 5} more")
    return h3_cells, h3_xml_path, parse_h3_cells_from_xml


@app.cell
def __(h3_cells, h3_cells_to_bbox):
    # Convert H3 cells to bounding box
    bbox = h3_cells_to_bbox(h3_cells)
    min_lat, min_lon, max_lat, max_lon = bbox
    print(f"Bounding box: ({min_lat:.6f}, {min_lon:.6f}, {max_lat:.6f}, {max_lon:.6f})")
    print(f"  Latitude range: {max_lat - min_lat:.6f} degrees")
    print(f"  Longitude range: {max_lon - min_lon:.6f} degrees")
    return bbox, max_lat, max_lon, min_lat, min_lon


@app.cell
def __(Path, bbox, download_basemap, repo_root):
    # Download basemap
    output_dir = repo_root / "outputs" / "notebook_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    basemap_path = output_dir / "downloaded_basemap_esri.tif"
    
    if not basemap_path.exists():
        print("Downloading basemap from ESRI World Imagery...")
        downloaded_path = download_basemap(
            bbox=bbox,
            output_path=str(basemap_path),
            source="esri",
            target_resolution=0.5
        )
        print(f"✓ Basemap downloaded to: {downloaded_path}")
    else:
        print(f"✓ Basemap already exists: {basemap_path}")
        downloaded_path = str(basemap_path)
    return basemap_path, downloaded_path, output_dir


@app.cell
def __(Path, downloaded_path, plt, rasterio, repo_root, show):
    # Visualize downloaded basemap
    with rasterio.open(downloaded_path) as src:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        show(src, ax=ax, title="Downloaded Basemap (ESRI World Imagery)")
        plt.tight_layout()
        plt.show()
        
        print(f"Basemap info:")
        print(f"  Size: {src.width} x {src.height} pixels")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")
    return ax, fig, src


@app.cell
def __(ImagePreprocessor, Path, downloaded_path, output_dir, repo_root):
    # Load source orthomosaic
    source_path = repo_root / "inputs" / "qualicum_beach" / "orthomosaic_no_gcps.tif"
    
    print(f"Source orthomosaic: {source_path}")
    print(f"  Exists: {source_path.exists()}")
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(
        source_path=str(source_path),
        target_path=downloaded_path,
        output_dir=output_dir
    )
    
    # Log metadata
    preprocessor.log_metadata()
    return preprocessor, source_path


@app.cell
def __(Path, plt, rasterio, repo_root, show, source_path):
    # Visualize source orthomosaic
    with rasterio.open(source_path) as src:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        show(src, ax=ax, title="Source Orthomosaic")
        plt.tight_layout()
        plt.show()
        
        print(f"Source orthomosaic info:")
        print(f"  Size: {src.width} x {src.height} pixels")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")
    return ax, fig, src


@app.cell
def __(DEFAULT_SCALES):
    # Define scales for hierarchical registration (using defaults from defaults module)
    scales = DEFAULT_SCALES.copy()
    print(f"Processing scales: {scales}")
    return scales,


@app.cell
def __(preprocessor, scales):
    # Create resolution pyramid
    for scale in scales:
        print(f"\n--- Scale {scale:.3f} ---")
        
        source_img, target_img = preprocessor.load_downsampled(scale)
        print(f"  Source shape: {source_img.shape}")
        print(f"  Target shape: {target_img.shape}")
        
        overlap_info = preprocessor.compute_overlap_region(scale)
        if overlap_info:
            print(f"  Overlap region: {overlap_info['source']}")
            
            source_overlap, target_overlap = preprocessor.crop_to_overlap(
                source_img, target_img, overlap_info
            )
            print(f"  Source overlap shape: {source_overlap.shape}")
            print(f"  Target overlap shape: {target_overlap.shape}")
        else:
            print(f"  ⚠ No overlap found at scale {scale}")
    return overlap_info, scale, source_img, source_overlap, target_img, target_overlap


@app.cell
def __(LIGHTGLUE_AVAILABLE):
    # Check if LightGlue is available
    if not LIGHTGLUE_AVAILABLE:
        print("⚠ LightGlue not available. Install with: pip install lightglue")
        print("Falling back to SIFT matching...")
    else:
        print("✓ LightGlue is available")
    return


@app.cell
def __(Path, create_mask, match_lightglue, output_dir, preprocessor, scales, visualize_matches):
    # Match features at each scale
    matches_by_scale = {}
    match_visualizations = {}
    
    for scale in scales:
        print(f"\n{'='*60}")
        print(f"Matching at scale {scale:.3f}")
        print(f"{'='*60}")
        
        source_img, target_img = preprocessor.load_downsampled(scale)
        overlap_info = preprocessor.compute_overlap_region(scale)
        
        if not overlap_info:
            print(f"  ⚠ No overlap at scale {scale}, skipping...")
            continue
        
        source_overlap, target_overlap = preprocessor.crop_to_overlap(
            source_img, target_img, overlap_info
        )
        
        source_mask = create_mask(source_overlap)
        target_mask = create_mask(target_overlap)
        
        pixel_resolution = 0.02 / scale
        print(f"  Pixel resolution: {pixel_resolution:.4f} m/pixel")
        
        if LIGHTGLUE_AVAILABLE:
            matches_result = match_lightglue(
                source_overlap, target_overlap, source_mask, target_mask,
                use_tiles=True,
                pixel_resolution_meters=pixel_resolution
            )
        else:
            from matching import match_sift
            matches_result = match_sift(source_overlap, target_overlap, source_mask, target_mask)
        
        if matches_result and 'matches' in matches_result and len(matches_result['matches']) > 0:
            num_matches = len(matches_result['matches'])
            print(f"  ✓ Found {num_matches} matches")
            matches_by_scale[scale] = matches_result
            
            viz_path = output_dir / f"matches_scale{scale:.3f}.png"
            visualize_matches(
                source_overlap, target_overlap, matches_result, viz_path,
                source_name=f"source_scale{scale:.3f}",
                target_name=f"target_scale{scale:.3f}",
                skip_json=True
            )
            match_visualizations[scale] = viz_path
            print(f"  ✓ Saved visualization: {viz_path.name}")
        else:
            print(f"  ✗ No matches found at scale {scale}")
    return (
        match_visualizations,
        matches_by_scale,
        matches_result,
        num_matches,
        pixel_resolution,
        scale,
        source_mask,
        source_overlap,
        target_mask,
        target_overlap,
        viz_path,
    )


@app.cell
def __(OrthomosaicRegistration, downloaded_path, output_dir, scales, source_path):
    # Run the complete registration pipeline
    print("Running full hierarchical registration pipeline...")
    print(f"Source: {source_path}")
    print(f"Target: {downloaded_path}")
    print(f"Output: {output_dir}")
    print(f"Scales: {scales}")
    
    registration = OrthomosaicRegistration(
        source_path=str(source_path),
        target_path=downloaded_path,
        output_dir=str(output_dir),
        scales=scales,  # Uses DEFAULT_SCALES
        matcher=DEFAULT_MATCHER,
        transform_types={scale: algo for scale, algo in zip(DEFAULT_SCALES, DEFAULT_ALGORITHMS)},
        debug_level='high'
    )
    
    result = registration.register()
    
    if result:
        print(f"\n✓ Registration complete!")
        print(f"Final output: {result}")
    else:
        print("\n✗ Registration failed")
    return registration, result


@app.cell
def __(Path, plt, rasterio, repo_root, show, source_path):
    # Load and visualize final registered orthomosaic
    if 'result' in locals() and result and result.exists():
        with rasterio.open(result) as src:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            with rasterio.open(source_path) as orig:
                show(orig, ax=axes[0], title="Original Source Orthomosaic")
            
            show(src, ax=axes[1], title="Registered Orthomosaic")
            
            plt.tight_layout()
            plt.show()
            
            print(f"Registered orthomosaic info:")
            print(f"  Size: {src.width} x {src.height} pixels")
            print(f"  CRS: {src.crs}")
            print(f"  Bounds: {src.bounds}")
    else:
        print("Final registered orthomosaic not found")
    return axes, fig, orig, src


@app.cell
def __(Path, output_dir, plt):
    # Display error histograms if available
    matching_dir = output_dir / "matching_and_transformations"
    if matching_dir.exists():
        import glob
        histograms = sorted(glob.glob(str(matching_dir / "error_histogram_scale*.png")))
        
        if histograms:
            fig, axes = plt.subplots(1, len(histograms), figsize=(20, 5))
            if len(histograms) == 1:
                axes = [axes]
            
            for idx, hist_path in enumerate(histograms):
                img = plt.imread(hist_path)
                axes[idx].imshow(img)
                axes[idx].set_title(Path(hist_path).stem)
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.show()
    return axes, fig, hist_path, histograms, idx, img, matching_dir


if __name__ == "__main__":
    app.run()


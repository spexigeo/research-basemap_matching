# Orthomosaic Registration System

A comprehensive hierarchical system for registering high-resolution orthomosaics to basemaps using multiple computer vision algorithms. Designed for drone imagery with resolution differences and seasonal variations.

## Features

- **Hierarchical Registration**: Coarse-to-fine alignment using resolution pyramids (default: 0.125, 0.25, 0.5)
- **Multiple Matching Methods**: LightGlue (default), SIFT, ORB, Patch NCC
- **Multiple Transform Types**: 2D shift, similarity, affine, homography, polynomial (2nd/3rd order), spline, rubber sheeting
- **Cumulative Transformations**: Each scale builds upon previous transformations for improved accuracy
- **Basemap Download**: Automatic download from ESRI, Google Satellite, OpenStreetMap
- **GCP Analysis**: Extract and visualize patches from registered orthomosaic at GCP locations
- **Comprehensive Logging**: Detailed logs and intermediate results for debugging
- **Rich Visualizations**: Match flow diagrams, error histograms, and comparison views
- **Flexible Configuration**: JSON-based configuration or command-line arguments
- **GeoTIFF Output**: Properly georeferenced outputs with JPEG compression

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install LightGlue for best matching performance
pip install lightglue

# Optional: Install h3 for H3 cell support
pip install h3
```

## Quick Start

### Option 1: Command Line (Recommended)

```bash
python register_orthomosaic.py \
    inputs/qualicum_beach/orthomosaic_no_gcps.tif \
    inputs/qualicum_beach/qualicum_beach_basemap_esri.tif \
    outputs/ \
    --scales 0.125 0.25 0.5 \
    --algorithms shift shift shift \
    --matcher lightglue \
    --debug intermediate
```

### Option 2: Using Configuration File

```bash
# 1. Copy and edit a sample configuration
cp sample_configs/config_template.json my_config.json
# Edit my_config.json with your paths and parameters

# 2. Run registration
python register_orthomosaic.py --config my_config.json
```

### Option 3: Using the Example Script

```bash
# Edit example_qualicum_beach.py to set your file paths
python example_qualicum_beach.py
```

### Option 4: Download Basemap Automatically

```bash
python register_orthomosaic.py \
    inputs/qualicum_beach/orthomosaic_no_gcps.tif \
    outputs/ \
    --get-basemap esri \
    --basemap-area inputs/qualicum_beach/h3_cells.xml \
    --basemap-resolution 0.5
```

### Option 5: With GCP Analysis

```bash
python register_orthomosaic.py \
    inputs/qualicum_beach/orthomosaic_no_gcps.tif \
    inputs/qualicum_beach/qualicum_beach_basemap_esri.tif \
    outputs/ \
    --gcp-analysis inputs/qualicum_beach/QualicumBeach_AOI.kmz
```

## Registration Methods

### Matching Algorithms

1. **lightglue** (Default)
   - Best for: High accuracy, robust to appearance changes
   - Speed: Moderate to Fast (with GPU)
   - Accuracy: Very High
   - Use when: You want the best possible matches

2. **sift** (Scale-Invariant Feature Transform)
   - Best for: Scale differences, rotation
   - Speed: Moderate to Slow
   - Accuracy: High
   - Use when: LightGlue is not available

3. **orb** (Oriented FAST and Rotated BRIEF)
   - Best for: Speed-critical applications
   - Speed: Very Fast
   - Accuracy: Moderate

4. **patch_ncc** (Patch-based Normalized Cross-Correlation)
   - Best for: Seasonal changes, appearance variations
   - Speed: Moderate
   - Accuracy: High
   - Use when: Images have different vegetation states or lighting

### Transform Types

- **shift**: 2D translation (2 parameters) - Fast, good for coarse alignment
- **similarity**: Translation + rotation + uniform scale (4 parameters)
- **affine**: Translation + rotation + scale + shear (6 parameters)
- **homography**: Full projective transform (8 parameters) - Best for perspective changes
- **polynomial_2**: 2nd order polynomial (12 parameters) - Handles local distortions
- **polynomial_3**: 3rd order polynomial (20 parameters) - More flexible, slower
- **spline**: Thin-plate spline - Handles complex local distortions
- **rubber_sheeting**: Piecewise affine - Good for local corrections

**Default Strategy**: Use `shift` for all scales (0.125, 0.25, 0.5)

## Configuration

### Command-Line Arguments

```bash
python register_orthomosaic.py [source] [target] [output] [OPTIONS]

Positional arguments:
  source              Path to source orthomosaic (optional if --config used)
  target              Path to target basemap (optional if --config used)
  output              Output directory (optional if --config used)

Options:
  --config PATH       Path to JSON configuration file
  --scales FLOAT ...  Resolution scales (default: 0.125 0.25 0.5)
  --algorithms STR ... Transform algorithms for each scale
  --matcher STR       Matching method: lightglue, sift, orb, patch_ncc
  --debug LEVEL       Debug level: none, intermediate, high
  --get-basemap STR   Download basemap: esri, google_satellite, openstreetmap
  --basemap-area STR  Bounding box or H3 cells file for basemap download
  --basemap-zoom INT  Zoom level for basemap download
  --basemap-resolution FLOAT  Target resolution in meters per pixel
  --gcp-analysis PATH Path to GCP file (CSV or KMZ) for analysis
```

### Configuration File Format

```json
{
  "source_path": "inputs/qualicum_beach/orthomosaic_no_gcps.tif",
  "target_path": "inputs/qualicum_beach/qualicum_beach_basemap_esri.tif",
  "output_dir": "outputs/",
  "hierarchical_scales": [0.125, 0.25, 0.5],
  "algorithms": ["shift", "shift", "shift"],
  "method": "lightglue",
  "debug_level": "intermediate"
}
```

### Debug Levels

- **none**: Only log file and final registered orthomosaic
- **intermediate**: `none` + intermediate/ directory with transformed orthomosaics
- **high**: `intermediate` + matching_and_transformations/ directory with all debug files

## Output Structure

The system generates outputs in the specified output directory:

```
outputs/
├── orthomosaic_registered.tif          # Final registered GeoTIFF
├── registration.log                    # Log file (or registration_verbose.log for high debug)
├── config.json                         # Effective configuration used
├── preprocessing/                      # (if debug_level='high')
│   ├── source_overlap_scale*.png
│   ├── target_overlap_scale*.png
│   └── orthomosaic_scale*.tif
├── matching_and_transformations/       # (if debug_level='high')
│   ├── matches_scale*.json            # Match data with statistics
│   ├── matches_scale*.png             # Match visualizations
│   ├── lightglue_keypoints_scale*.png  # Feature keypoints
│   ├── transform_scale*.json          # Transformation matrices
│   └── error_histogram_scale*.png      # Error distributions
├── intermediate/                       # (if debug_level='intermediate' or 'high')
│   ├── orthomosaic_scale*_*.tif        # Transformed orthomosaics
│   └── orthomosaic_scale*_*.png       # PNG previews
└── gcp_analysis/                       # (if --gcp-analysis used)
    └── GCP_*.png                       # 300x300 pixel patches at GCP locations
```

## Project Structure

```
research-basemap_matching/
├── register_orthomosaic.py    # Main entry point and registration pipeline
├── preprocessing.py           # Image preprocessing and overlap computation
├── matching.py                # Feature matching algorithms
├── transformations.py         # Geometric transformation computation and application
├── basemap_downloader.py      # Basemap tile downloading utilities
├── gcp_analysis.py            # GCP patch extraction and visualization
├── evaluate_gcps.py           # GCP-based registration quality evaluation
├── defaults.py                # Default configuration constants
├── example_qualicum_beach.py  # Usage examples for Qualicum Beach dataset
├── sample_configs/            # Configuration file templates
│   ├── config_template.json
│   ├── config_esri.json
│   └── ...
├── inputs/                    # Input data directory
│   └── qualicum_beach/        # Qualicum Beach dataset
│       ├── orthomosaic_no_gcps.tif
│       ├── qualicum_beach_basemap_esri.tif
│       ├── h3_cells.xml
│       └── QualicumBeach_AOI.kmz
├── notebooks/                 # Interactive notebooks
│   └── qualicum_beach/        # Qualicum Beach demo notebooks
│       ├── registration_demo.ipynb
│       ├── registration_demo_colab.ipynb
│       └── registration_demo.py
├── debug_tests/               # Testing and debugging utilities
│   ├── test_matching.py
│   └── diagnose_images.py
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Key Features

### Hierarchical Registration

The system uses a resolution pyramid approach:
1. Start with coarse scale (0.125) to find approximate alignment
2. Apply transformation and move to next scale (0.25)
3. Refine alignment at each scale
4. Final scale (1.0) produces full-resolution registered orthomosaic

Transformations are **cumulative** - each scale's transformation is applied to the result from the previous scale.

### Basemap Download

Automatically download basemaps from:
- **ESRI World Imagery**: High-quality satellite imagery
- **Google Satellite**: Google's satellite imagery
- **OpenStreetMap**: OpenStreetMap tiles

Provide either:
- Bounding box: `"min_lat,min_lon,max_lat,max_lon"`
- H3 cells file: Path to file containing H3 cell IDs (one per line or XML format)

### GCP Analysis

After registration, extract 300x300 pixel patches centered at each Ground Control Point:
- Loads GCPs from CSV or KMZ files
- Extracts patches from registered orthomosaic
- Draws red dots at GCP locations
- Saves patches to `outputs/gcp_analysis/`

### Match Visualizations

For each scale, generates:
- **Match flow diagrams**: Show feature correspondences with connecting lines
- **Keypoint visualizations**: Highlight detected features
- **Error histograms**: Distribution of match distances (all matches and inliers only)

## Tips for Best Results

### For Seasonal Changes
- Use `lightglue` or `patch_ncc` matcher
- Use hierarchical scales: `[0.125, 0.25, 0.5]`
- Start with `shift` transforms, refine with `homography`

### For Large Images
- Start with coarse scales: `[0.125, 0.25]`
- Use `debug_level='none'` for faster processing
- Enable JPEG compression (default)

### For High Accuracy
- Use multiple hierarchical scales
- Use `homography` or `polynomial_2` for fine scales
- Set `debug_level='high'` to inspect intermediate results

### For Speed
- Use fewer scales: `[0.25, 0.5]`
- Use `shift` transforms only
- Set `debug_level='none'`

## Troubleshooting

### Issue: Not enough matches found
**Solution**: 
- Try different matcher (`lightglue` recommended)
- Use coarser initial scale
- Check that images have sufficient overlap

### Issue: Poor alignment with seasonal changes
**Solution**:
- Use `lightglue` or `patch_ncc` matcher
- Use hierarchical scales for progressive refinement
- Check error histograms to see if matches improve at each scale

### Issue: Out of memory
**Solution**:
- Use coarser scales: `[0.125, 0.25]`
- Set `debug_level='none'` to avoid saving intermediate files
- Process smaller regions

### Issue: Registration taking too long
**Solution**:
- Use fewer scales
- Use `shift` transforms for coarse scales
- Set `debug_level='none'`

## Interactive Notebooks

See `notebooks/qualicum_beach/README.md` for detailed information about:
- Jupyter notebook (`registration_demo.ipynb`)
- Google Colab notebook (`registration_demo_colab.ipynb`)
- Marimo notebook (`registration_demo.py`)

## Testing and Debugging

See `debug_tests/README.md` for information about:
- `test_matching.py`: Unit tests for matching algorithms
- `diagnose_images.py`: Image diagnostics and analysis

## Citation

If you use this system in your research, please cite the relevant algorithms:

- LightGlue: Lindenberger et al., 2023. LightGlue: Local Feature Matching at Light Speed
- SIFT: Lowe, D.G., 2004. Distinctive image features from scale-invariant keypoints
- Homography: Hartley & Zisserman, 2003. Multiple View Geometry in Computer Vision

## License

MIT License - feel free to use and modify as needed.

## Contact

For questions and issues, please open a GitHub issue or contact the maintainer.

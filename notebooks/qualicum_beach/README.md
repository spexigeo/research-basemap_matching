# Notebooks Directory

This directory contains interactive notebooks demonstrating the complete orthomosaic registration pipeline.

## Available Notebooks

### 1. `registration_demo.ipynb` - Jupyter Notebook
Standard Jupyter notebook for local use or JupyterLab.

**Usage:**
```bash
cd notebooks/qualicum_beach
jupyter notebook registration_demo.ipynb
```

**Requirements:**
- Jupyter installed: `pip install jupyter`
- All dependencies from `../../requirements.txt` (in repository root)
- Input files in `../../inputs/qualicum_beach/`:
  - `orthomosaic_no_gcps.tif`
  - `h3_cells.xml`

### 2. `registration_demo_colab.ipynb` - Google Colab Notebook
Optimized for Google Colab with automatic dependency installation.

**Usage:**
1. Upload to Google Colab
2. Upload required files:
   - `inputs/qualicum_beach/orthomosaic_no_gcps.tif` → `/content/inputs/qualicum_beach/`
   - `inputs/qualicum_beach/h3_cells.xml` → `/content/inputs/qualicum_beach/`
   - All Python modules to `/content/` (or clone from GitHub)
3. Run all cells in order

**Features:**
- Automatic dependency installation
- Google Drive mounting support
- Optimized for Colab's environment

### 3. `registration_demo.py` - Marimo Notebook
Marimo notebook for reactive Python notebooks.

**Usage:**
```bash
# Install Marimo
pip install marimo

# Run notebook
cd notebooks/qualicum_beach
marimo edit registration_demo.py
```

**Features:**
- Reactive cells (auto-updates when dependencies change)
- Modern UI
- Great for interactive exploration

## What the Notebooks Demonstrate

All notebooks walk through the complete registration pipeline:

1. **H3 Cell Parsing**: Extract H3 cell IDs from XML and convert to bounding box
2. **Basemap Download**: Download basemap tiles from ESRI World Imagery
3. **Preprocessing**: 
   - Load source orthomosaic
   - Create resolution pyramid at multiple scales (0.125, 0.25, 0.5, 1.0)
   - Compute overlap regions
4. **Feature Matching**: 
   - Match features using LightGlue at each scale
   - Visualize matches
5. **Transformation Computation**: 
   - Compute transformations (shift for coarse scales, homography for fine scales)
   - Remove outliers using RANSAC
6. **Full Pipeline**: Run the complete hierarchical registration
7. **Visualization**: Display results, error histograms, and final registered orthomosaic

## Output Structure

When run, the notebooks create:

```
outputs/notebook_demo/
├── downloaded_basemap_esri.tif          # Downloaded basemap
├── orthomosaic_registered.tif            # Final registered orthomosaic
├── preprocessing/                         # (if debug_level='high')
│   ├── source_overlap_scale*.png
│   ├── target_overlap_scale*.png
│   └── orthomosaic_scale*.tif
├── matching_and_transformations/         # (if debug_level='high')
│   ├── matches_scale*.json
│   ├── matches_scale*.png
│   ├── lightglue_keypoints_scale*.png
│   ├── transform_scale*.json
│   └── error_histogram_scale*.png
└── intermediate/                         # (if debug_level='intermediate' or 'high')
    └── orthomosaic_scale*_*.tif
```

## Notes

- The notebooks use `debug_level='high'` to generate all intermediate files for visualization
- For faster runs, you can set `debug_level='none'` to only generate the final output
- The H3 cells XML parser handles Word XML format (extracts 15-character hex strings)
- If LightGlue is not available, the notebooks fall back to SIFT matching


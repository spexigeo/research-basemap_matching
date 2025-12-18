# Test and Debugging Tools

This directory contains tools for testing and debugging the registration pipeline.

## diagnose_images.py

**Purpose**: Diagnostic tool to check if orthomosaic and basemap are compatible for registration.

**Usage**:
```bash
python tests/diagnose_images.py <source_orthomosaic> <target_basemap>
```

**What it checks**:
- CRS compatibility
- Geographic overlap
- Resolution differences
- Recommended scale settings

**Output**:
- Detailed diagnostic report
- Visualization: `geospatial_diagnostic.png`

**When to use**:
- Before running registration to verify images are compatible
- When registration fails - check for overlap/CRS issues
- To get recommended scale settings for your images

## test_matching.py

**Purpose**: Test patch matching functionality to debug matching issues.

**Usage**:
```bash
python tests/test_matching.py <source> <target> [scale] [max_patches]
```

**What it does**:
- Tests patch matching at a given scale
- Tries different preprocessing methods (none, gradient, clahe, edges)
- Reports which preprocessing works best
- Helps diagnose why matching might be failing

**When to use**:
- When registration produces poor matches
- To determine best preprocessing method for your images
- To test if images are matchable at a given scale

**Example**:
```bash
python tests/test_matching.py inputs/orthomosaic.tif inputs/basemap.tif 0.15 100
```


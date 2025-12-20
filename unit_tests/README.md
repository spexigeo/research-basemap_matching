# Unit Tests

This directory contains unit tests for the orthomosaic registration pipeline. These tests are designed to catch regressions and ensure core functionality works correctly.

## Running Tests

### Install Dependencies

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
# From the repository root
pytest unit_tests/

# With verbose output
pytest unit_tests/ -v

# With coverage report
pytest unit_tests/ --cov=. --cov-report=html
```

### Run Specific Test Files

```bash
# Test preprocessing
pytest unit_tests/test_preprocessing.py -v

# Test matching
pytest unit_tests/test_matching.py -v

# Test transformations
pytest unit_tests/test_transformations.py -v

# Test GCP analysis
pytest unit_tests/test_gcp_analysis.py -v

# Test defaults
pytest unit_tests/test_defaults.py -v
```

### Run Specific Test Classes or Functions

```bash
# Run a specific test class
pytest unit_tests/test_preprocessing.py::TestImagePreprocessor -v

# Run a specific test function
pytest unit_tests/test_preprocessing.py::TestImagePreprocessor::test_initialization -v
```

## Test Structure

### `conftest.py`
Shared fixtures for all tests:
- `temp_dir`: Temporary directory for test outputs
- `sample_image_100x100`: Sample 100x100 grayscale image
- `sample_image_200x200`: Sample 200x200 grayscale image
- `sample_geotiff_source`: Sample GeoTIFF source image
- `sample_geotiff_target`: Sample GeoTIFF target image
- `sample_matches`: Sample match data

### `test_preprocessing.py`
Tests for `preprocessing.py`:
- ImagePreprocessor initialization
- Lazy metadata loading
- Source and target properties
- Resolution calculation
- Downsampled image loading
- Overlap region computation

### `test_matching.py`
Tests for `matching.py`:
- Mask creation
- SIFT matching
- ORB matching
- Patch NCC matching

### `test_transformations.py`
Tests for `transformations.py`:
- 2D shift computation
- Similarity transform computation
- Affine transform computation
- Homography computation
- Outlier removal

### `test_gcp_analysis.py`
Tests for `gcp_analysis.py`:
- CSV GCP loading
- Tab-separated CSV loading
- Case-insensitive column names
- File loading wrapper

### `test_defaults.py`
Tests for `defaults.py`:
- Default scales validation
- Default algorithms validation
- Default matcher validation
- Default debug level validation
- Default output directory validation

## Writing New Tests

When adding new functionality, add corresponding unit tests:

1. Create a new test file or add to existing one
2. Use fixtures from `conftest.py` when possible
3. Test both success and failure cases
4. Test edge cases (empty inputs, invalid inputs, etc.)
5. Use descriptive test names that explain what is being tested

Example:
```python
def test_new_functionality_basic():
    """Test basic functionality."""
    result = new_function(input_data)
    assert result is not None
    assert result['expected_key'] == expected_value
```

## Continuous Integration

These tests are designed to run quickly and catch regressions. They should:
- Run in < 1 minute total
- Not require large input files
- Use synthetic data when possible
- Be deterministic (same inputs = same outputs)

## Notes

- Tests use temporary directories that are automatically cleaned up
- Tests create synthetic GeoTIFF files for testing
- Some tests may be skipped if optional dependencies are not installed (e.g., LightGlue)
- Tests are designed to be independent and can run in any order




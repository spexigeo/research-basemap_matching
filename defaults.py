"""
Default values for orthomosaic registration pipeline.

These defaults define the default values used throughout the registration pipeline,
which uses scale [1.000] (full resolution) with affine transformation and saves to outputs/.
They can be overridden by user input via command-line arguments, config files, or programmatic API.
"""

# Default scales for hierarchical registration
DEFAULT_SCALES = [1.000]

# Default transform algorithms for each scale (must match length of DEFAULT_SCALES)
DEFAULT_ALGORITHMS = ['affine']

# Default matcher
DEFAULT_MATCHER = 'lightglue'

# Default debug level
DEFAULT_DEBUG_LEVEL = 'none'

# Default output directory
DEFAULT_OUTPUT_DIR = 'outputs'




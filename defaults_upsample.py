"""
Default values for orthomosaic registration pipeline (upsample variant).

These defaults define the default values used throughout the registration pipeline
for the upsample variant, which uses scales [0.125, 0.25, 0.5] and saves to outputs_upsample/.
They can be overridden by user input via command-line arguments, config files, or programmatic API.
"""

# Default scales for hierarchical registration (upsample variant)
DEFAULT_SCALES = [0.125, 0.25, 0.5]

# Default transform algorithms for each scale (must match length of DEFAULT_SCALES)
DEFAULT_ALGORITHMS = ['shift', 'shift', 'affine']

# Default matcher
DEFAULT_MATCHER = 'lightglue'

# Default debug level
DEFAULT_DEBUG_LEVEL = 'none'

# Default output directory (upsample variant)
DEFAULT_OUTPUT_DIR = 'outputs_upsample'




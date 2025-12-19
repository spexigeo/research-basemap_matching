"""
Default values for orthomosaic registration pipeline.

These defaults define the default values used throughout the registration pipeline.
They can be overridden by user input via command-line arguments, config files, or programmatic API.
"""

# Default scales for hierarchical registration
DEFAULT_SCALES = [0.125, 0.25, 1.0]

# Default transform algorithms for each scale (must match length of DEFAULT_SCALES)
DEFAULT_ALGORITHMS = ['shift', 'shift', 'shift']

# Default matcher
DEFAULT_MATCHER = 'lightglue'

# Default debug level
DEFAULT_DEBUG_LEVEL = 'none'

# Default output directory
DEFAULT_OUTPUT_DIR = 'outputs'


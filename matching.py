"""
Matching module for orthomosaic registration.
Provides feature matching algorithms: LightGlue (default), SIFT, ORB, and Patch NCC.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
from typing import Tuple, List, Dict, Optional

# Try to import arosics
try:
    from arosics import COREG_LOCAL, COREG_GLOBAL
    AROSICS_AVAILABLE = True
except ImportError:
    AROSICS_AVAILABLE = False
    print("Warning: arosics not available. Install with: pip install arosics")

# Try to import SuperGlue/LightGlue
try:
    import torch
    from lightglue import LightGlue, SuperPoint, DISK
    from lightglue.utils import load_image, rbd
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    print("Warning: lightglue not available. Install with: pip install lightglue")


def load_images(source_path: Path, target_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load and convert images to grayscale."""
    source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
    
    if source is None:
        raise ValueError(f"Could not load source image: {source_path}")
    if target is None:
        raise ValueError(f"Could not load target image: {target_path}")
    
    print(f"Source shape: {source.shape}, dtype: {source.dtype}, range: [{source.min()}, {source.max()}]")
    print(f"Target shape: {target.shape}, dtype: {target.dtype}, range: [{target.min()}, {target.max()}]")
    
    return source, target


def create_mask(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Create mask to ignore black/blank regions."""
    # Black pixels (blank space) should be masked out
    mask = image > threshold
    return mask.astype(np.uint8) * 255


def match_sift(source: np.ndarray, target: np.ndarray, 
               source_mask: Optional[np.ndarray] = None,
               target_mask: Optional[np.ndarray] = None,
               match_resolution: bool = True) -> Dict:
    """SIFT feature matching."""
    print("\n=== SIFT Matching ===")
    
    # Optionally match resolutions
    source_work = source.copy()
    target_work = target.copy()
    source_mask_work = source_mask.copy() if source_mask is not None else None
    target_mask_work = target_mask.copy() if target_mask is not None else None
    scale_factor = 1.0
    
    if match_resolution:
        # Upsample target to match source resolution
        if target.shape != source.shape:
            print(f"Upsampling target from {target.shape} to {source.shape}")
            target_work = cv2.resize(target, (source.shape[1], source.shape[0]), 
                                    interpolation=cv2.INTER_CUBIC)
            if target_mask is not None:
                target_mask_work = cv2.resize(target_mask, (source.shape[1], source.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
            scale_factor = source.shape[0] / target.shape[0]
    
    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=5000)
    
    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(source_work, source_mask_work)
    kp2, des2 = sift.detectAndCompute(target_work, target_mask_work)
    
    print(f"Source keypoints: {len(kp1)}")
    print(f"Target keypoints: {len(kp2)}")
    
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return {'matches': [], 'kp1': kp1, 'kp2': kp2, 'method': 'SIFT', 'scale_factor': scale_factor}
    
    # Match using FLANN or BFMatcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    print(f"Good matches (ratio test): {len(good_matches)}")
    
    return {
        'matches': good_matches,
        'kp1': kp1,
        'kp2': kp2,
        'method': 'SIFT',
        'scale_factor': scale_factor,
        'source_shape': source_work.shape,
        'target_shape': target_work.shape
    }


def match_orb(source: np.ndarray, target: np.ndarray,
              source_mask: Optional[np.ndarray] = None,
              target_mask: Optional[np.ndarray] = None,
              match_resolution: bool = True) -> Dict:
    """ORB feature matching."""
    print("\n=== ORB Matching ===")
    
    # Optionally match resolutions
    source_work = source.copy()
    target_work = target.copy()
    source_mask_work = source_mask.copy() if source_mask is not None else None
    target_mask_work = target_mask.copy() if target_mask is not None else None
    scale_factor = 1.0
    
    if match_resolution:
        # Upsample target to match source resolution
        if target.shape != source.shape:
            print(f"Upsampling target from {target.shape} to {source.shape}")
            target_work = cv2.resize(target, (source.shape[1], source.shape[0]), 
                                    interpolation=cv2.INTER_CUBIC)
            if target_mask is not None:
                target_mask_work = cv2.resize(target_mask, (source.shape[1], source.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
            scale_factor = source.shape[0] / target.shape[0]
    
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=5000)
    
    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(source_work, source_mask_work)
    kp2, des2 = orb.detectAndCompute(target_work, target_mask_work)
    
    print(f"Source keypoints: {len(kp1)}")
    print(f"Target keypoints: {len(kp2)}")
    
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return {'matches': [], 'kp1': kp1, 'kp2': kp2, 'method': 'ORB', 'scale_factor': scale_factor}
    
    # Match using Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    print(f"Good matches (ratio test): {len(good_matches)}")
    
    return {
        'matches': good_matches,
        'kp1': kp1,
        'kp2': kp2,
        'method': 'ORB',
        'scale_factor': scale_factor,
        'source_shape': source_work.shape,
        'target_shape': target_work.shape
    }


def match_patch_ncc(source: np.ndarray, target: np.ndarray,
                    source_mask: Optional[np.ndarray] = None,
                    target_mask: Optional[np.ndarray] = None,
                    patch_size: int = 64, grid_spacing: int = 32,
                    ncc_threshold: float = 0.5, scale: float = 0.15) -> Dict:
    """Patch-based NCC matching."""
    print("\n=== Patch NCC Matching ===")
    
    # Resize source to match target if needed
    scale_x = target.shape[1] / source.shape[1]
    scale_y = target.shape[0] / source.shape[0]
    
    if source.shape != target.shape:
        print(f"Resizing source from {source.shape} to {target.shape} (scale: {scale_x:.3f}, {scale_y:.3f})")
        source_resized = cv2.resize(source, (target.shape[1], target.shape[0]), 
                                   interpolation=cv2.INTER_AREA)
        if source_mask is not None:
            source_mask_resized = cv2.resize(source_mask, (target.shape[1], target.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
        else:
            source_mask_resized = None
    else:
        source_resized = source
        source_mask_resized = source_mask
        scale_x = scale_y = 1.0
    
    h, w = target.shape
    matches = []
    all_attempts = []
    
    # Scale-adaptive parameters
    # At coarser scales, we need larger patches and search margins
    # Expected error is ~2-3m, which at scale 0.15 = ~13.3cm/pixel = ~15-22 pixels
    # At scale 0.30 = ~6.7cm/pixel = ~30-45 pixels
    # At scale 0.50 = ~4cm/pixel = ~50-75 pixels
    expected_error_meters = 3.0
    pixel_resolution_meters = 0.02 / scale  # Source GSD at scaled resolution
    expected_error_pixels = int(expected_error_meters / pixel_resolution_meters)
    
    # Scale-adaptive patch size and search margin
    # Larger patches at coarser scales for better feature detection
    adaptive_patch_size = max(patch_size, int(patch_size * (1.0 / scale)))
    adaptive_grid_spacing = max(grid_spacing, int(grid_spacing * (1.0 / scale)))
    # Search margin should be ~3x expected error to account for uncertainty
    search_margin = max(150, int(expected_error_pixels * 3))
    
    print(f"  Scale-adaptive parameters:")
    print(f"    Scale: {scale:.3f}, Pixel resolution: {pixel_resolution_meters*100:.2f} cm/pixel")
    print(f"    Expected error: {expected_error_meters}m = {expected_error_pixels} pixels")
    print(f"    Patch size: {adaptive_patch_size} (base: {patch_size})")
    print(f"    Grid spacing: {adaptive_grid_spacing} (base: {grid_spacing})")
    print(f"    Search margin: {search_margin} pixels")
    
    # Grid-based patch matching
    # For each target patch, search for corresponding patch in source
    # Ensure we cover the ENTIRE image, not just upper portion
    # Use a more systematic grid that covers all valid regions
    y_start = adaptive_patch_size // 2
    y_end = h - adaptive_patch_size // 2
    x_start = adaptive_patch_size // 2
    x_end = w - adaptive_patch_size // 2
    
    # Log coverage
    print(f"  Grid coverage: y=[{y_start}:{y_end}], x=[{x_start}:{x_end}], spacing={adaptive_grid_spacing}")
    print(f"  Expected grid points: {(y_end-y_start)//adaptive_grid_spacing * (x_end-x_start)//adaptive_grid_spacing}")
    
    # Track where we're actually testing and finding matches
    tested_y_coords = []
    matched_y_coords = []
    
    for y in range(y_start, y_end, adaptive_grid_spacing):
        for x in range(x_start, x_end, adaptive_grid_spacing):
            # Skip if in masked region (target)
            if target_mask is not None and target_mask[y, x] == 0:
                continue
            
            # Extract target patch
            ty1, ty2 = y - adaptive_patch_size // 2, y + adaptive_patch_size // 2
            tx1, tx2 = x - adaptive_patch_size // 2, x + adaptive_patch_size // 2
            target_patch = target[ty1:ty2, tx1:tx2]
            
            # Search in resized source around corresponding location
            # Since both are now same size, search around (x, y) with margin
            # Use scale-adaptive search margin
            
            # Define search region with scale-adaptive margin
            sx1 = max(0, x - search_margin)
            sx2 = min(w, x + search_margin + adaptive_patch_size)
            sy1 = max(0, y - search_margin)
            sy2 = min(h, y + search_margin + adaptive_patch_size)
            
            # Check if search region has enough valid pixels
            # Be less restrictive - only require 20% valid pixels instead of 30%
            if source_mask_resized is not None:
                search_region_mask = source_mask_resized[sy1:sy2, sx1:sx2]
                search_area = (sy2 - sy1) * (sx2 - sx1)
                valid_pixels = np.sum(search_region_mask > 0)
                if valid_pixels < (adaptive_patch_size * adaptive_patch_size * 0.2) and valid_pixels < (search_area * 0.1):
                    # Not enough valid pixels in search region
                    continue
            
            search_region = source_resized[sy1:sy2, sx1:sx2]
            
            if search_region.shape[0] >= adaptive_patch_size and search_region.shape[1] >= adaptive_patch_size:
                try:
                    result = cv2.matchTemplate(search_region, target_patch, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    
                    # Match location in resized source coordinates
                    src_x_resized = sx1 + max_loc[0] + adaptive_patch_size // 2
                    src_y_resized = sy1 + max_loc[1] + adaptive_patch_size // 2
                    
                    # Convert back to original source coordinates for visualization
                    src_x_original = int(src_x_resized / scale_x)
                    src_y_original = int(src_y_resized / scale_y)
                    
                    # Verify match is in valid region of original source
                    # Be less strict - check a small region around the match point
                    if source_mask is not None:
                        # Check if match point is in bounds
                        if src_x_original < 0 or src_y_original < 0 or \
                           src_y_original >= source_mask.shape[0] or src_x_original >= source_mask.shape[1]:
                            continue
                        # Check if match point or nearby region is valid (allow small tolerance)
                        check_radius = 5
                        y_check_min = max(0, src_y_original - check_radius)
                        y_check_max = min(source_mask.shape[0], src_y_original + check_radius + 1)
                        x_check_min = max(0, src_x_original - check_radius)
                        x_check_max = min(source_mask.shape[1], src_x_original + check_radius + 1)
                        check_region = source_mask[y_check_min:y_check_max, x_check_min:x_check_max]
                        if np.sum(check_region > 0) == 0:
                            # No valid pixels in check region, skip
                            continue
                    
                    all_attempts.append({
                        'target': (x, y),
                        'source_resized': (src_x_resized, src_y_resized),
                        'source_original': (src_x_original, src_y_original),
                        'confidence': float(max_val)
                    })
                    
                    if max_val > ncc_threshold:
                        matched_y_coords.append(y)
                        matches.append({
                            'target': (x, y),
                            'source': (src_x_original, src_y_original),  # Use original coordinates
                            'source_resized': (src_x_resized, src_y_resized),
                            'confidence': float(max_val)
                        })
                except cv2.error:
                    continue
    
    print(f"Tested {len(all_attempts)} patches, found {len(matches)} matches (threshold={ncc_threshold})")
    if tested_y_coords:
        print(f"  Y coordinate range tested: {min(tested_y_coords)} to {max(tested_y_coords)} (target height: {h})")
        print(f"  Tested Y distribution: top 25%={sum(1 for y in tested_y_coords if y < h//4)}, "
              f"mid 50%={sum(1 for y in tested_y_coords if h//4 <= y < 3*h//4)}, "
              f"bottom 25%={sum(1 for y in tested_y_coords if y >= 3*h//4)}")
    if matched_y_coords:
        print(f"  Y coordinate range matched: {min(matched_y_coords)} to {max(matched_y_coords)}")
        print(f"  Matched Y distribution: top 25%={sum(1 for y in matched_y_coords if y < h//4)}, "
              f"mid 50%={sum(1 for y in matched_y_coords if h//4 <= y < 3*h//4)}, "
              f"bottom 25%={sum(1 for y in matched_y_coords if y >= 3*h//4)}")
        if max(matched_y_coords) < h * 0.5:
            print(f"  WARNING: All matches are in upper half! This suggests:")
            print(f"    - Geographic overlap may only be in upper portion, OR")
            print(f"    - Source resized image may only have valid pixels in upper portion")
    
    return {
        'matches': matches,
        'all_attempts': all_attempts,
        'method': 'Patch_NCC',
        'scale_x': scale_x,
        'scale_y': scale_y
    }


# Default tile settings (can be overridden via command line)
DEFAULT_TILE_SIZE = 2048
DEFAULT_OVERLAP = 256
DEFAULT_EXPECTED_ERROR = 3.0

def match_lightglue(source: np.ndarray, target: np.ndarray,
                   source_mask: Optional[np.ndarray] = None,
                   target_mask: Optional[np.ndarray] = None,
                   use_tiles: bool = True,
                   tile_size: int = None,
                   overlap: int = None,
                   expected_error_meters: float = None,
                   pixel_resolution_meters: float = 0.02) -> Dict:
    """
    LightGlue/SuperGlue matching (if available).
    
    Args:
        source: Source image
        target: Target image
        source_mask: Optional mask for source
        target_mask: Optional mask for target
        use_tiles: Whether to use tiled processing for large images
        tile_size: Size of tiles for processing
        overlap: Overlap between tiles (pixels)
        expected_error_meters: Expected matching error in meters (for region-based search)
        pixel_resolution_meters: Pixel resolution in meters at current scale
    """
    if not LIGHTGLUE_AVAILABLE:
        return {'matches': [], 'method': 'LightGlue', 'error': 'lightglue not available'}
    
    # Use defaults if not provided
    if tile_size is None:
        tile_size = DEFAULT_TILE_SIZE
    if overlap is None:
        overlap = DEFAULT_OVERLAP
    if expected_error_meters is None:
        expected_error_meters = DEFAULT_EXPECTED_ERROR
    
    print("\n=== LightGlue Matching ===")
    
    try:
        # Upsample target to match source resolution
        if target.shape != source.shape:
            print(f"Upsampling target from {target.shape} to {source.shape}")
            target_work = cv2.resize(target, (source.shape[1], source.shape[0]), 
                                    interpolation=cv2.INTER_CUBIC)
            if target_mask is not None:
                target_mask_work = cv2.resize(target_mask, (source.shape[1], source.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
            else:
                target_mask_work = None
        else:
            target_work = target
            target_mask_work = target_mask
        
        # Determine if we should use tiled processing
        # Use tiles if image is large (more than 2x tile_size in any dimension)
        h, w = source.shape
        should_tile = use_tiles and (h > 2 * tile_size or w > 2 * tile_size)
        
        if should_tile:
            print(f"Using tiled processing (image size: {h}x{w}, tile size: {tile_size})")
            return match_lightglue_tiled(source, target_work, source_mask, target_mask_work,
                                       tile_size, overlap, expected_error_meters, pixel_resolution_meters)
        
        # Convert to RGB (LightGlue expects RGB)
        source_rgb = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        target_rgb = cv2.cvtColor(target_work, cv2.COLOR_GRAY2RGB)
        
        # Apply masks if available
        if source_mask is not None:
            source_rgb[source_mask == 0] = 0
        if target_mask_work is not None:
            target_rgb[target_mask_work == 0] = 0
        
        # Load models for LightGlue - use best available device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')  # Apple Silicon GPU
        else:
            device = torch.device('cpu')
        print(f"Using device: {device}")
        extractor = SuperPoint(max_num_keypoints=5000).eval().to(device)
        matcher = LightGlue(features='superpoint').eval().to(device)
        
        # Convert numpy arrays to tensors (LightGlue expects tensors)
        # Images should be in [C, H, W] format, normalized to [0, 1]
        from torchvision.transforms import ToTensor
        
        # Convert to tensor and normalize
        to_tensor = ToTensor()
        source_tensor = to_tensor(source_rgb).unsqueeze(0).to(device)  # Add batch dimension [1, C, H, W]
        target_tensor = to_tensor(target_rgb).unsqueeze(0).to(device)
        
        # Extract features - LightGlue extractor expects tensors
        feats0 = extractor.extract(source_tensor)
        feats1 = extractor.extract(target_tensor)
        
        # Get keypoint counts
        kp0_count = feats0['keypoints'].shape[1] if len(feats0['keypoints'].shape) > 1 else len(feats0['keypoints'])
        kp1_count = feats1['keypoints'].shape[1] if len(feats1['keypoints'].shape) > 1 else len(feats1['keypoints'])
        print(f"Source keypoints: {kp0_count}")
        print(f"Target keypoints: {kp1_count}")
        
        # Match - features are already on device from extractor
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        
        
        # Convert to numpy - handle both tensor and list formats
        # CRITICAL: Always move to CPU first for MPS/CUDA tensors
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                # Move to CPU before converting to numpy (required for MPS and CUDA)
                if x.device.type != 'cpu':
                    x = x.detach().cpu()
                else:
                    x = x.detach()
                return x.numpy()
            elif isinstance(x, list):
                return np.array(x)
            return x
        
        feats0_np = {k: to_numpy(v) for k, v in feats0.items()}
        feats1_np = {k: to_numpy(v) for k, v in feats1.items()}
        
        # Extract matches - matches01['matches'] can be a tensor or a list containing tensors
        matches_tensor = matches01['matches']
        # Handle different formats: tensor, list of tensors, or list of arrays
        if isinstance(matches_tensor, torch.Tensor):
            # Direct tensor - move to CPU first
            if matches_tensor.device.type != 'cpu':
                matches_np = matches_tensor.detach().cpu().numpy()
            else:
                matches_np = matches_tensor.detach().numpy()
        elif isinstance(matches_tensor, list):
            # List - check if it contains tensors
            if len(matches_tensor) > 0 and isinstance(matches_tensor[0], torch.Tensor):
                # List of tensors - convert each to numpy
                matches_np = np.array([item.detach().cpu().numpy() if item.device.type != 'cpu' 
                                     else item.detach().numpy() for item in matches_tensor])
            else:
                # List of arrays or numbers - convert directly
                matches_np = np.array(matches_tensor)
        elif hasattr(matches_tensor, 'device') and hasattr(matches_tensor, 'detach'):
            # Tensor-like object with device
            if matches_tensor.device.type != 'cpu':
                matches_np = matches_tensor.detach().cpu().numpy()
            else:
                matches_np = matches_tensor.detach().numpy()
        else:
            # Try direct conversion
            matches_np = np.array(matches_tensor)
        
        # Handle different match tensor shapes
        # Remove batch dimension if present (shape [1, N, 2] -> [N, 2])
        if len(matches_np.shape) == 3 and matches_np.shape[0] == 1:
            matches_np = matches_np[0]  # Remove batch dimension
        
        if len(matches_np.shape) == 2 and matches_np.shape[1] == 2:
            # Shape [N, 2] - correct format
            # Filter valid matches (indices >= 0)
            valid = matches_np[:, 0] >= 0
            matches_valid = matches_np[valid]
        else:
            # Unexpected shape
            print(f"Warning: Unexpected matches shape: {matches_np.shape}")
            matches_valid = np.array([]).reshape(0, 2)
        
        num_matches = len(matches_valid)
        print(f"LightGlue matches: {num_matches}")
        
        # Get keypoints - handle different tensor shapes
        if len(feats0_np['keypoints'].shape) == 2:
            # Shape [N, 2] - already correct
            kp0_coords = feats0_np['keypoints']
            kp1_coords = feats1_np['keypoints']
        else:
            # Shape [1, N, 2] - remove batch dimension
            kp0_coords = feats0_np['keypoints'][0] if feats0_np['keypoints'].shape[0] == 1 else feats0_np['keypoints']
            kp1_coords = feats1_np['keypoints'][0] if feats1_np['keypoints'].shape[0] == 1 else feats1_np['keypoints']
        
        # Get match scores/confidence if available
        match_scores = None
        # LightGlue returns 'scores' key with match confidence
        # 'scores' is a list containing a tensor: [tensor([...])]
        if 'scores' in matches01:
            match_scores_data = matches01['scores']
            try:
                # Handle list containing tensor
                if isinstance(match_scores_data, list) and len(match_scores_data) > 0:
                    # Extract tensor from list
                    scores_tensor = match_scores_data[0]
                    if isinstance(scores_tensor, torch.Tensor):
                        # Move to CPU before converting to numpy (required for MPS/CUDA)
                        if scores_tensor.device.type != 'cpu':
                            match_scores_np = scores_tensor.detach().cpu().numpy()
                        else:
                            match_scores_np = scores_tensor.detach().numpy()
                    else:
                        match_scores_np = np.array(scores_tensor)
                elif isinstance(match_scores_data, torch.Tensor):
                    # Move to CPU before converting to numpy
                    if match_scores_data.device.type != 'cpu':
                        match_scores_np = match_scores_data.detach().cpu().numpy()
                    else:
                        match_scores_np = match_scores_data.detach().numpy()
                elif hasattr(match_scores_data, 'detach'):  # Handle other tensor-like objects
                    if hasattr(match_scores_data, 'device') and match_scores_data.device.type != 'cpu':
                        match_scores_np = match_scores_data.detach().cpu().numpy()
                    else:
                        match_scores_np = match_scores_data.detach().numpy()
                else:
                    match_scores_np = np.array(match_scores_data)
                
                # Remove batch dimension if present
                if match_scores_np is not None:
                    if len(match_scores_np.shape) == 2 and match_scores_np.shape[0] == 1:
                        match_scores_np = match_scores_np[0]
                    elif len(match_scores_np.shape) == 3 and match_scores_np.shape[0] == 1:
                        match_scores_np = match_scores_np[0]
                    # Apply same valid filter as matches
                    match_scores = match_scores_np[valid] if valid is not None else match_scores_np
                    if match_scores is not None and len(match_scores) > 0:
                        print(f"Extracted {len(match_scores)} match scores, range: [{match_scores.min():.3f}, {match_scores.max():.3f}]")
            except Exception as e:
                print(f"Warning: Could not extract match scores from LightGlue: {e}")
                import traceback
                traceback.print_exc()
                match_scores = None
        
        # Get keypoint scores/responses if available
        kp0_scores = feats0_np.get('keypoint_scores', None)
        kp1_scores = feats1_np.get('keypoint_scores', None)
        if kp0_scores is not None and len(kp0_scores.shape) > 1 and kp0_scores.shape[0] == 1:
            kp0_scores = kp0_scores[0]
        if kp1_scores is not None and len(kp1_scores.shape) > 1 and kp1_scores.shape[0] == 1:
            kp1_scores = kp1_scores[0]
        
        # Convert to OpenCV format for visualization
        kp0 = []
        for i, kp in enumerate(kp0_coords):
            response = float(kp0_scores[i]) if kp0_scores is not None and i < len(kp0_scores) else 0.0
            kp0.append(cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=10, response=response))
        
        kp1 = []
        for i, kp in enumerate(kp1_coords):
            response = float(kp1_scores[i]) if kp1_scores is not None and i < len(kp1_scores) else 0.0
            kp1.append(cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=10, response=response))
        
        cv_matches = []
        for i, (idx0, idx1) in enumerate(matches_valid):
            # Get match confidence score
            if match_scores is not None and i < len(match_scores):
                confidence = float(match_scores[i])
                # Convert confidence to distance (lower is better for OpenCV)
                # LightGlue scores are typically in [0, 1] range, higher is better
                distance = 1.0 - confidence
            else:
                # No score available, use a default
                distance = 0.5  # Default distance
                confidence = 0.5
            cv_matches.append(cv2.DMatch(_queryIdx=int(idx0), _trainIdx=int(idx1), _distance=distance))
        
        return {
            'matches': cv_matches,
            'kp1': kp0,
            'kp2': kp1,
            'method': 'LightGlue',
            'scale_factor': source.shape[0] / target.shape[0] if target.shape != source.shape else 1.0,
            'source_shape': source.shape,
            'target_shape': target_work.shape,
            'match_scores': match_scores.tolist() if match_scores is not None and hasattr(match_scores, 'tolist') else (list(match_scores) if match_scores is not None else None)
        }
    except Exception as e:
        print(f"LightGlue failed: {e}")
        import traceback
        traceback.print_exc()
        return {'matches': [], 'method': 'LightGlue', 'error': str(e)}


def visualize_from_json(json_path: Path, source: np.ndarray, target: np.ndarray, output_path: Path):
    """Create visualization from JSON match file."""
    import json
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    method = data['method']
    matches_data = data['matches']
    scale_factor = data.get('scale_factor', 1.0)
    
    # Convert JSON matches to OpenCV format
    kp1 = []
    kp2 = []
    cv_matches = []
    
    # Collect unique keypoints
    src_kp_dict = {}
    tgt_kp_dict = {}
    
    for i, match in enumerate(matches_data):
        src_pt = (match['source']['x'], match['source']['y'])
        tgt_upsampled_pt = (match['target_upsampled']['x'], match['target_upsampled']['y'])
        
        # Add source keypoint if not already added
        if src_pt not in src_kp_dict:
            src_idx = len(kp1)
            src_kp_dict[src_pt] = src_idx
            kp1.append(cv2.KeyPoint(x=src_pt[0], y=src_pt[1], size=10, 
                                   response=match['source'].get('response', 0.0)))
        else:
            src_idx = src_kp_dict[src_pt]
        
        # Add target keypoint if not already added
        if tgt_upsampled_pt not in tgt_kp_dict:
            tgt_idx = len(kp2)
            tgt_kp_dict[tgt_upsampled_pt] = tgt_idx
            kp2.append(cv2.KeyPoint(x=tgt_upsampled_pt[0], y=tgt_upsampled_pt[1], size=10, 
                                   response=match['source'].get('response', 0.0)))
        else:
            tgt_idx = tgt_kp_dict[tgt_upsampled_pt]
        
        # Create match
        distance = match.get('distance', 0.0)
        cv_matches.append(cv2.DMatch(_queryIdx=src_idx, _trainIdx=tgt_idx, _distance=distance))
    
    # Create result dict in same format as match functions
    result = {
        'matches': cv_matches,
        'kp1': kp1,
        'kp2': kp2,
        'method': method,
        'scale_factor': scale_factor,
        'source_shape': source.shape,
        'target_shape': target.shape
    }
    
    # Use existing visualization function
    visualize_matches(source, target, result, output_path)


def match_lightglue_tiled(source: np.ndarray, target: np.ndarray,
                         source_mask: Optional[np.ndarray] = None,
                         target_mask: Optional[np.ndarray] = None,
                         tile_size: int = 2048,
                         overlap: int = 256,
                         expected_error_meters: float = 3.0,
                         pixel_resolution_meters: float = 0.02) -> Dict:
    """
    LightGlue matching with tiled processing for large images.
    
    Args:
        source: Source image (already upsampled to match target if needed)
        target: Target image (already upsampled to match source)
        source_mask: Optional mask for source
        target_mask: Optional mask for target
        tile_size: Size of tiles for processing
        overlap: Overlap between tiles (pixels)
        expected_error_meters: Expected matching error in meters
        pixel_resolution_meters: Pixel resolution in meters at current scale
    """
    print(f"  Tiled processing: tile_size={tile_size}, overlap={overlap}")
    
    # Get device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Initialize models
    extractor = SuperPoint(max_num_keypoints=2000).eval().to(device)  # Fewer keypoints per tile
    matcher = LightGlue(features='superpoint').eval().to(device)
    from torchvision.transforms import ToTensor
    to_tensor = ToTensor()
    
    # Calculate search radius in pixels based on expected error
    search_radius_pixels = int(expected_error_meters / pixel_resolution_meters)
    print(f"  Expected error: {expected_error_meters}m = {search_radius_pixels} pixels")
    
    # Process in tiles
    h, w = source.shape
    all_matches = []
    all_kp0 = []
    all_kp1 = []
    kp0_idx_offset = 0
    kp1_idx_offset = 0
    
    # Calculate tile grid
    num_tiles_y = max(1, (h + tile_size - overlap - 1) // (tile_size - overlap))
    num_tiles_x = max(1, (w + tile_size - overlap - 1) // (tile_size - overlap))
    print(f"  Processing {num_tiles_y} x {num_tiles_x} = {num_tiles_y * num_tiles_x} tiles")
    
    tile_idx = 0
    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            tile_idx += 1
            if tile_idx % 10 == 0 or tile_idx == num_tiles_y * num_tiles_x:
                print(f"  Processing tile {tile_idx}/{num_tiles_y * num_tiles_x} ({ty+1},{tx+1})...", end=' ')
            
            # Calculate tile bounds
            y_start = ty * (tile_size - overlap)
            y_end = min(y_start + tile_size, h)
            x_start = tx * (tile_size - overlap)
            x_end = min(x_start + tile_size, w)
            
            # Extract tiles
            source_tile = source[y_start:y_end, x_start:x_end]
            target_tile = target[y_start:y_end, x_start:x_end]
            
            # Skip if tile is too small
            if source_tile.shape[0] < 64 or source_tile.shape[1] < 64:
                if tile_idx % 10 == 0 or tile_idx == num_tiles_y * num_tiles_x:
                    print("skipped (too small)")
                continue
            
            # Check if tile has enough valid pixels
            if source_mask is not None:
                mask_tile = source_mask[y_start:y_end, x_start:x_end]
                if np.sum(mask_tile > 0) < (mask_tile.size * 0.1):  # Less than 10% valid
                    if tile_idx % 10 == 0 or tile_idx == num_tiles_y * num_tiles_x:
                        print("skipped (mostly masked)")
                    continue
            
            try:
                # Convert to RGB
                if len(source_tile.shape) == 2:
                    source_tile_rgb = cv2.cvtColor(source_tile, cv2.COLOR_GRAY2RGB)
                    target_tile_rgb = cv2.cvtColor(target_tile, cv2.COLOR_GRAY2RGB)
                else:
                    source_tile_rgb = source_tile
                    target_tile_rgb = target_tile
                
                # Apply masks
                if source_mask is not None:
                    mask_tile = source_mask[y_start:y_end, x_start:x_end]
                    source_tile_rgb[mask_tile == 0] = 0
                if target_mask is not None:
                    mask_tile = target_mask[y_start:y_end, x_start:x_end]
                    target_tile_rgb[mask_tile == 0] = 0
                
                # Convert to tensors
                source_tensor = to_tensor(source_tile_rgb).unsqueeze(0).to(device)
                target_tensor = to_tensor(target_tile_rgb).unsqueeze(0).to(device)
                
                # Extract features
                feats0 = extractor.extract(source_tensor)
                feats1 = extractor.extract(target_tensor)
                
                # Match
                matches01 = matcher({'image0': feats0, 'image1': feats1})
                
                # Process matches
                matches_tensor = matches01['matches']
                if isinstance(matches_tensor, torch.Tensor):
                    if matches_tensor.device.type != 'cpu':
                        matches_np = matches_tensor.detach().cpu().numpy()
                    else:
                        matches_np = matches_tensor.detach().numpy()
                elif isinstance(matches_tensor, list):
                    if len(matches_tensor) > 0 and isinstance(matches_tensor[0], torch.Tensor):
                        matches_np = np.array([item.detach().cpu().numpy() if item.device.type != 'cpu' 
                                             else item.detach().numpy() for item in matches_tensor])
                    else:
                        matches_np = np.array(matches_tensor)
                else:
                    matches_np = np.array(matches_tensor)
                
                # Remove batch dimension if present
                if len(matches_np.shape) == 3 and matches_np.shape[0] == 1:
                    matches_np = matches_np[0]
                
                # Filter valid matches
                if len(matches_np.shape) == 2 and matches_np.shape[1] == 2:
                    valid = matches_np[:, 0] >= 0
                    matches_valid = matches_np[valid]
                else:
                    matches_valid = np.array([]).reshape(0, 2)
                
                if len(matches_valid) > 0:
                    # Convert features to numpy
                    def to_numpy(x):
                        if isinstance(x, torch.Tensor):
                            if x.device.type != 'cpu':
                                return x.detach().cpu().numpy()
                            return x.detach().numpy()
                        elif isinstance(x, list):
                            if len(x) > 0 and isinstance(x[0], torch.Tensor):
                                return np.array([item.detach().cpu().numpy() if item.device.type != 'cpu' 
                                               else item.detach().numpy() for item in x])
                            return np.array(x)
                        return x
                    
                    feats0_np = {k: to_numpy(v) for k, v in feats0.items()}
                    feats1_np = {k: to_numpy(v) for k, v in feats1.items()}
                    
                    # Get keypoints
                    kp0_coords = feats0_np['keypoints']
                    kp1_coords = feats1_np['keypoints']
                    if len(kp0_coords.shape) == 3 and kp0_coords.shape[0] == 1:
                        kp0_coords = kp0_coords[0]
                    if len(kp1_coords.shape) == 3 and kp1_coords.shape[0] == 1:
                        kp1_coords = kp1_coords[0]
                    
                    # Get keypoint scores
                    kp0_scores = feats0_np.get('keypoint_scores', None)
                    kp1_scores = feats1_np.get('keypoint_scores', None)
                    if kp0_scores is not None and len(kp0_scores.shape) > 1 and kp0_scores.shape[0] == 1:
                        kp0_scores = kp0_scores[0]
                    if kp1_scores is not None and len(kp1_scores.shape) > 1 and kp1_scores.shape[0] == 1:
                        kp1_scores = kp1_scores[0]
                    
                    # Get match scores
                    match_scores = None
                    if 'scores' in matches01:
                        scores_data = matches01['scores']
                        if isinstance(scores_data, list) and len(scores_data) > 0:
                            scores_tensor = scores_data[0]
                            if isinstance(scores_tensor, torch.Tensor):
                                if scores_tensor.device.type != 'cpu':
                                    match_scores = scores_tensor.detach().cpu().numpy()[valid]
                                else:
                                    match_scores = scores_tensor.detach().numpy()[valid]
                        elif isinstance(scores_data, torch.Tensor):
                            if scores_data.device.type != 'cpu':
                                match_scores = scores_data.detach().cpu().numpy()[valid]
                            else:
                                match_scores = scores_data.detach().numpy()[valid]
                    
                    # Convert to OpenCV format and adjust coordinates to full image space
                    for i, (idx0, idx1) in enumerate(matches_valid):
                        kp0_pt = kp0_coords[int(idx0)]
                        kp1_pt = kp1_coords[int(idx1)]
                        
                        # Adjust coordinates to full image space
                        kp0_full = (float(kp0_pt[0]) + x_start, float(kp0_pt[1]) + y_start)
                        kp1_full = (float(kp1_pt[0]) + x_start, float(kp1_pt[1]) + y_start)
                        
                        # Create keypoints
                        response0 = float(kp0_scores[int(idx0)]) if kp0_scores is not None and int(idx0) < len(kp0_scores) else 0.0
                        response1 = float(kp1_scores[int(idx1)]) if kp1_scores is not None and int(idx1) < len(kp1_scores) else 0.0
                        
                        kp0_cv = cv2.KeyPoint(x=kp0_full[0], y=kp0_full[1], size=10, response=response0)
                        kp1_cv = cv2.KeyPoint(x=kp1_full[0], y=kp1_full[1], size=10, response=response1)
                        
                        all_kp0.append(kp0_cv)
                        all_kp1.append(kp1_cv)
                        
                        # Create match
                        distance = 0.5
                        if match_scores is not None and i < len(match_scores):
                            confidence = float(match_scores[i])
                            distance = 1.0 - confidence
                        
                        match_cv = cv2.DMatch(_queryIdx=len(all_kp0)-1, _trainIdx=len(all_kp1)-1, _distance=distance)
                        all_matches.append(match_cv)
                    
                    if tile_idx % 10 == 0 or tile_idx == num_tiles_y * num_tiles_x:
                        print(f"found {len(matches_valid)} matches")
                else:
                    if tile_idx % 10 == 0 or tile_idx == num_tiles_y * num_tiles_x:
                        print("no matches")
            except Exception as e:
                if tile_idx % 10 == 0 or tile_idx == num_tiles_y * num_tiles_x:
                    print(f"error: {e}")
                continue
    
    print(f"\nTotal matches across all tiles: {len(all_matches)}")
    
    return {
        'matches': all_matches,
        'kp1': all_kp0,
        'kp2': all_kp1,
        'method': 'LightGlue',
        'scale_factor': source.shape[0] / target.shape[0] if target.shape != source.shape else 1.0,
        'source_shape': source.shape,
        'target_shape': target.shape,
        'match_scores': None
    }

def match_arosics(source_path: Path, target_path: Path) -> Dict:
    """AROSICS matching (if available)."""
    if not AROSICS_AVAILABLE:
        return {'matches': [], 'method': 'AROSICS', 'error': 'arosics not available'}
    
    print("\n=== AROSICS Matching ===")
    
    try:
        # AROSICS expects file paths
        CRL = COREG_LOCAL(
            source_path,
            target_path,
            path_out=str(source_path.parent / 'arosics_result.tif'),
            window_size=(256, 256),
            max_shift=50,
            max_iter=5
        )
        
        # Get tie points
        tie_points = CRL.tie_points
        print(f"AROSICS found {len(tie_points)} tie points")
        
        return {
            'matches': tie_points,
            'method': 'AROSICS',
            'transform': CRL.coreg_info
        }
    except Exception as e:
        print(f"AROSICS failed: {e}")
        return {'matches': [], 'method': 'AROSICS', 'error': str(e)}


def visualize_matches(source: np.ndarray, target: np.ndarray, 
                     result: Dict, output_path: Path,
                     source_name: Optional[str] = None,
                     target_name: Optional[str] = None,
                     skip_json: bool = False):
    """Create visualization of matches.
    
    Args:
        skip_json: If True, skip writing JSON file (useful when JSON is written separately)
    """
    method = result['method']
    
    if method in ['SIFT', 'ORB', 'LightGlue']:
        # Feature-based matching visualization
        kp1 = result['kp1']
        kp2 = result['kp2']
        matches = result['matches']
        scale_factor = result.get('scale_factor', 1.0)
        
        # Upsample target to match source size for better visualization
        if scale_factor != 1.0 or target.shape != source.shape:
            target_vis = cv2.resize(target, (source.shape[1], source.shape[0]), 
                                   interpolation=cv2.INTER_CUBIC)
            # CRITICAL: Do NOT scale keypoints - they are already in upsampled coordinates!
            # kp2 was detected on target_work which is already upsampled to source.shape
            # So kp2.pt coordinates are already in the upsampled space (same as source)
            # No scaling needed!
        else:
            target_vis = target
        
        # Create side-by-side visualization with flow lines
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        
        # Source with matches
        axes[0].imshow(source, cmap='gray')
        # Draw keypoints
        for kp in kp1:
            axes[0].plot(kp.pt[0], kp.pt[1], 'r.', markersize=1, alpha=0.3)
        # Draw matched keypoints
        src_matched = [kp1[m.queryIdx] for m in matches]
        axes[0].plot([kp.pt[0] for kp in src_matched], [kp.pt[1] for kp in src_matched], 
                    'go', markersize=4, alpha=0.7, label=f'Matched ({len(matches)})')
        src_title = source_name if source_name else 'Source'
        axes[0].set_title(f'{src_title} - {len(kp1)} keypoints, {len(matches)} matches', 
                         fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].axis('off')
        
        # Target with matches (upsampled to match source)
        axes[1].imshow(target_vis, cmap='gray')
        # Draw keypoints
        for kp in kp2:
            axes[1].plot(kp.pt[0], kp.pt[1], 'b.', markersize=1, alpha=0.3)
        # Draw matched keypoints
        tgt_matched = [kp2[m.trainIdx] for m in matches]
        axes[1].plot([kp.pt[0] for kp in tgt_matched], [kp.pt[1] for kp in tgt_matched], 
                    'go', markersize=4, alpha=0.7, label=f'Matched ({len(matches)})')
        tgt_title = target_name if target_name else 'Target (upsampled)'
        axes[1].set_title(f'{tgt_title} - {len(kp2)} keypoints, {len(matches)} matches', 
                         fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].axis('off')
        
        plt.tight_layout()
        # Extract scale from output_path if present, otherwise use method name
        output_stem = output_path.stem
        scale_suffix = ''
        # Try to extract scale from patterns like "matches_scale0.125" or "matches_0.150"
        import re
        # Look for "scale" followed by a number pattern
        scale_match = re.search(r'scale([0-9]+\.[0-9]+)', output_stem)
        if scale_match:
            scale_suffix = '_scale' + scale_match.group(1)
        else:
            # Fallback: look for number at the end after underscore
            parts = output_stem.split('_')
            if len(parts) > 1:
                last_part = parts[-1]
                # Check if it's a number (with optional decimal)
                if re.match(r'^[0-9]+\.[0-9]+$', last_part):
                    scale_suffix = '_' + last_part
        plt.savefig(output_path.parent / f'{method.lower()}_keypoints{scale_suffix}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create flow visualization with thick lines
        fig2, axes2 = plt.subplots(1, 2, figsize=(24, 12))
        
        axes2[0].imshow(target_vis, cmap='gray')
        axes2[1].imshow(source, cmap='gray')
        
        # Draw flow lines (thicker)
        # Show evenly distributed matches throughout the image
        num_to_show = min(200, len(matches))
        if len(matches) > num_to_show:
            # Sample evenly by Y coordinate to ensure coverage
            matches_sorted = sorted(matches, key=lambda m: kp1[m.queryIdx].pt[1])  # Sort by source Y
            step = len(matches_sorted) // num_to_show
            matches_to_show = matches_sorted[::step][:num_to_show]
        else:
            matches_to_show = matches
        
        for match in matches_to_show:
            src_kp = kp1[match.queryIdx]
            tgt_kp = kp2[match.trainIdx]
            
            src_pt = (src_kp.pt[0], src_kp.pt[1])
            tgt_pt = (tgt_kp.pt[0], tgt_kp.pt[1])
            
            axes2[0].plot(tgt_pt[0], tgt_pt[1], 'go', markersize=3, alpha=0.6)
            axes2[1].plot(src_pt[0], src_pt[1], 'go', markersize=3, alpha=0.6)
            
            # Draw connection line (thicker)
            con = mpatches.ConnectionPatch(
                xyA=src_pt, xyB=tgt_pt,
                coordsA=axes2[1].transData, coordsB=axes2[0].transData,
                color='green', linewidth=1.5, alpha=0.5, arrowstyle='->',
                mutation_scale=10
            )
            fig2.add_artist(con)
        
        tgt_flow_title = target_name if target_name else 'Target with Match Flow'
        src_flow_title = source_name if source_name else 'Source with Match Flow'
        axes2[0].set_title(tgt_flow_title, fontsize=14, fontweight='bold')
        axes2[1].set_title(src_flow_title, fontsize=14, fontweight='bold')
        axes2[0].axis('off')
        axes2[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        
        # Save match data with coordinates (unless skipped)
        if not skip_json:
            json_path = output_path.with_suffix('.json')
            match_data = {
                'method': method,
                'num_matches': len(matches),
                'num_source_keypoints': len(kp1),
                'num_target_keypoints': len(kp2),
                'scale_factor': scale_factor,
                'matches': []
            }
            
            for match in matches:
                src_kp = kp1[match.queryIdx]
                tgt_kp = kp2[match.trainIdx]
                # tgt_kp coordinates are in upsampled target space (same as source.shape)
                # If we want original target coordinates, divide by scale_factor
                tgt_x_upsampled = float(tgt_kp.pt[0])
                tgt_y_upsampled = float(tgt_kp.pt[1])
                tgt_x_original = tgt_x_upsampled / scale_factor if scale_factor != 1.0 else tgt_x_upsampled
                tgt_y_original = tgt_y_upsampled / scale_factor if scale_factor != 1.0 else tgt_y_upsampled
                
                # Calculate pixel distance between matched points
                src_x = float(src_kp.pt[0])
                src_y = float(src_kp.pt[1])
                pixel_distance = np.sqrt((src_x - tgt_x_upsampled)**2 + (src_y - tgt_y_upsampled)**2)
                
                # Pixel resolution in meters - extract scale from result
                # For SIFT/ORB, we upsampled target to match source, so use source resolution
                scale_used = result.get('scale', 0.15)  # Get scale from result, default to 0.15
                source_resolution_m_per_pixel = 0.02 / scale_used  # 2cm/pixel at full res, scaled
                distance_meters = pixel_distance * source_resolution_m_per_pixel
                
                match_data['matches'].append({
                    'source': {'x': src_x, 'y': src_y, 
                              'response': float(src_kp.response)},
                    'target_upsampled': {'x': tgt_x_upsampled, 'y': tgt_y_upsampled,
                                        'note': 'Coordinates in upsampled target space (matches source size)'},
                    'target_original': {'x': tgt_x_original, 'y': tgt_y_original,
                                       'note': 'Coordinates in original target space'},
                    'distance': {
                        'match_confidence': float(match.distance),
                        'pixels': float(pixel_distance),
                        'meters': float(distance_meters),
                        'note': 'Distance between matched points in image space'
                    }
                })
            
            with open(json_path, 'w') as f:
                json.dump(match_data, f, indent=2)
            
            print(f"Saved visualization to {output_path}")
            print(f"Saved match data to {json_path}")
        else:
            print(f"Saved visualization to {output_path} (JSON skipped)")
        
    elif method == 'Patch_NCC':
        # Patch-based matching visualization
        matches = result['matches']
        all_attempts = result.get('all_attempts', [])
        scale_x = result.get('scale_x', 1.0)
        scale_y = result.get('scale_y', 1.0)
        
        # Upsample target to match source size for visualization
        target_upsampled = cv2.resize(target, (source.shape[1], source.shape[0]), 
                                     interpolation=cv2.INTER_CUBIC)
        scale_factor_x = source.shape[1] / target.shape[1]
        scale_factor_y = source.shape[0] / target.shape[0]
        
        # Create side-by-side visualization
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        
        # Source with match points
        axes[0].imshow(source, cmap='gray')
        if matches:
            src_points = np.array([m['source'] for m in matches])
            axes[0].plot(src_points[:, 0], src_points[:, 1], 'go', markersize=4, alpha=0.7, label=f'Matches ({len(matches)})')
        axes[0].set_title('Source with Matches', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].axis('off')
        
        # Target (upsampled) with match points
        axes[1].imshow(target_upsampled, cmap='gray')
        if matches:
            tgt_points = np.array([m['target'] for m in matches])
            # Scale target points to upsampled target size
            tgt_points_scaled = tgt_points * np.array([scale_factor_x, scale_factor_y])
            axes[1].plot(tgt_points_scaled[:, 0], tgt_points_scaled[:, 1], 'go', markersize=4, alpha=0.7, label=f'Matches ({len(matches)})')
        axes[1].set_title('Target (upsampled) with Matches', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also create flow visualization with thick lines
        if matches:
            fig2, axes2 = plt.subplots(1, 2, figsize=(24, 12))
            
            axes2[0].imshow(target_upsampled, cmap='gray')
            axes2[1].imshow(source, cmap='gray')
            
            # Show evenly distributed matches throughout the image, not just first 200
            # Sample matches to ensure coverage across the entire image
            num_to_show = min(200, len(matches))
            if len(matches) > num_to_show:
                # Sample evenly by Y coordinate to ensure coverage
                matches_sorted = sorted(matches, key=lambda m: m['source'][1])  # Sort by Y
                step = len(matches_sorted) // num_to_show
                matches_to_show = matches_sorted[::step][:num_to_show]
            else:
                matches_to_show = matches
            
            for match in matches_to_show:
                tgt_pt = np.array(match['target']) * np.array([scale_factor_x, scale_factor_y])
                src_pt = np.array(match['source'])
                conf = match['confidence']
                
                axes2[0].plot(tgt_pt[0], tgt_pt[1], 'go', markersize=3, alpha=0.6)
                axes2[1].plot(src_pt[0], src_pt[1], 'go', markersize=3, alpha=0.6)
                
                # Draw connection line (thicker)
                con = mpatches.ConnectionPatch(
                    xyA=tuple(src_pt), xyB=tuple(tgt_pt),
                    coordsA=axes2[1].transData, coordsB=axes2[0].transData,
                    color='green', linewidth=1.5, alpha=0.5, arrowstyle='->',
                    mutation_scale=10
                )
                fig2.add_artist(con)
            
            axes2[0].set_title('Target with Match Flow', fontsize=14, fontweight='bold')
            axes2[1].set_title('Source with Match Flow', fontsize=14, fontweight='bold')
            axes2[0].axis('off')
            axes2[1].axis('off')
            
            plt.tight_layout()
            # Extract scale from output_path if present
            output_stem = output_path.stem
            if '_' in output_stem:
                parts = output_stem.split('_')
                scale_suffix = '_' + parts[-1] if parts[-1].replace('.', '').isdigit() else ''
            else:
                scale_suffix = ''
            plt.savefig(output_path.parent / f'{method.lower()}_flow{scale_suffix}.png', dpi=150, bbox_inches='tight')
            plt.close(fig2)
        
        # Save match data as JSON with coordinates (unless skipped)
        if not skip_json:
            json_path = output_path.with_suffix('.json')
            
            # Add pixel and meter distances to matches
        # CRITICAL: For Patch NCC, source is resized to match target size for matching
        # - source: original source coordinates (7111x8308) at 0.02 m/pixel
        # - target: original target coordinates (1071x1250) at 0.19 m/pixel
        # - source_resized: resized source coordinates (1071x1250) - source downsampled by ~0.15
        # 
        # When source is downsampled to match target size, its GSD becomes:
        #   source_GSD_resized = 0.02 / scale_x ≈ 0.133 m/pixel
        # The target remains at 0.19 m/pixel
        #
        # For distance calculation in the resized coordinate space (where matching occurred),
        # we use the source's GSD at the resized scale, since that's the resolution of the
        # source in that coordinate space. This gives distances that match expected 2-3m range.
            scale_x = result.get('scale_x', 0.15)
            scale_y = result.get('scale_y', 0.15)
            source_resolution_m_per_pixel = 0.02 / scale_x  # Source GSD at resized scale
            
            matches_with_distances = []
            for match in matches[:200]:  # Process first 200
                # Use resized coordinates for distance calculation (both at same pixel scale)
                src_resized_pt = match['source_resized']  # Coordinates in resized space (1071x1250)
                tgt_pt = match['target']  # Coordinates in target space (1071x1250)
                
                # Both are now at the same pixel scale, compute distance in pixels
                pixel_distance = np.sqrt((src_resized_pt[0] - tgt_pt[0])**2 + (src_resized_pt[1] - tgt_pt[1])**2)
                
                # Convert to meters using source GSD at resized scale
                # This gives us the ground distance between matched points
                distance_meters = pixel_distance * source_resolution_m_per_pixel
                
                match_with_dist = match.copy()
                match_with_dist['distance'] = {
                    'confidence': match.get('confidence', 0.0),
                    'pixels': float(pixel_distance),
                    'meters': float(distance_meters),
                    'note': 'Distance between matched points in resized coordinate space (both at target scale)'
                }
                matches_with_distances.append(match_with_dist)
            
            match_data = {
                'method': method,
                'num_matches': len(matches),
                'scale_x': scale_x,
                'scale_y': scale_y,
                'matches': matches_with_distances
            }
            with open(json_path, 'w') as f:
                json.dump(match_data, f, indent=2, default=str)
            
            print(f"Saved visualization to {output_path}")
            print(f"Saved match data to {json_path}")
        else:
            print(f"Saved visualization to {output_path} (JSON skipped)")


def main(scale: float = 0.150, method: str = 'lightglue'):
    """Main test function.
    
    Args:
        scale: Scale factor (e.g., 0.150, 0.300, 0.500)
        method: Matching method ('lightglue', 'sift', 'orb', 'patch_ncc', 'all')
    """
    test_dir = Path(__file__).parent
    # Format scale as 0.150, 0.300, 0.500 (always 3 decimal places)
    scale_str = f"{scale:.3f}"
    source_path = test_dir / f'source_overlap_scale{scale_str}.png'
    target_path = test_dir / f'target_overlap_scale{scale_str}.png'
    
    print("=" * 80)
    print(f"MATCHING TEST - Scale {scale_str}")
    print("=" * 80)
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    
    # Load images
    source, target = load_images(source_path, target_path)
    
    # Create masks for black regions
    source_mask = create_mask(source, threshold=10)
    target_mask = create_mask(target, threshold=10)
    
    print(f"\nSource mask: {np.sum(source_mask > 0)} / {source_mask.size} pixels valid ({100 * np.sum(source_mask > 0) / source_mask.size:.1f}%)")
    print(f"Target mask: {np.sum(target_mask > 0)} / {target_mask.size} pixels valid ({100 * np.sum(target_mask > 0) / target_mask.size:.1f}%)")
    
    # Save masks for inspection
    cv2.imwrite(str(test_dir / f'source_mask_{scale_str}.png'), source_mask)
    cv2.imwrite(str(test_dir / f'target_mask_{scale_str}.png'), target_mask)
    print(f"Saved masks to {test_dir}")
    
    # Test different matching methods
    results = []
    
    # Determine which methods to run
    run_all = (method == 'all')
    run_sift = (run_all or method == 'sift')
    run_orb = (run_all or method == 'orb')
    run_patch_ncc = (run_all or method == 'patch_ncc')
    run_lightglue = (run_all or method == 'lightglue')
    
    # Default to LightGlue if method not specified and LightGlue is available
    if method == 'lightglue' and not LIGHTGLUE_AVAILABLE:
        print("Warning: LightGlue not available, falling back to all methods")
        run_all = True
        run_lightglue = False
    
    # 1. SIFT (with resolution matching)
    if run_sift:
        try:
            result = match_sift(source, target, source_mask, target_mask, match_resolution=True)
            result['scale'] = scale  # Store scale for distance calculations
            results.append(result)
            visualize_matches(source, target, result, test_dir / f'sift_matches_{scale_str}.png')
        except Exception as e:
            print(f"SIFT failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 2. ORB (with resolution matching)
    if run_orb:
        try:
            result = match_orb(source, target, source_mask, target_mask, match_resolution=True)
            result['scale'] = scale  # Store scale for distance calculations
            results.append(result)
            visualize_matches(source, target, result, test_dir / f'orb_matches_{scale_str}.png')
        except Exception as e:
            print(f"ORB failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 3. Patch NCC
    if run_patch_ncc:
        try:
            result = match_patch_ncc(source, target, source_mask, target_mask,
                                    patch_size=64, grid_spacing=32, ncc_threshold=0.4, scale=scale)
            result['scale'] = scale  # Store scale for distance calculations
            results.append(result)
            visualize_matches(source, target, result, test_dir / f'patch_ncc_matches_{scale_str}.png')
        except Exception as e:
            print(f"Patch NCC failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. LightGlue/SuperGlue (if available) - DEFAULT METHOD
    if run_lightglue and LIGHTGLUE_AVAILABLE:
        try:
            # Calculate pixel resolution at this scale
            pixel_resolution_meters = 0.02 / scale  # Source GSD at scaled resolution
            # Use tiled processing for large images (automatic based on image size)
            result = match_lightglue(source, target, source_mask, target_mask,
                                   use_tiles=True,
                                   tile_size=DEFAULT_TILE_SIZE,
                                   overlap=DEFAULT_OVERLAP,
                                   expected_error_meters=DEFAULT_EXPECTED_ERROR,
                                   pixel_resolution_meters=pixel_resolution_meters)
            if result.get('matches'):
                results.append(result)
                print(f"Creating LightGlue visualization...")
                visualize_matches(source, target, result, test_dir / f'lightglue_matches_{scale_str}.png')
                print(f"Saved LightGlue visualization")
                
                # Also save JSON file with matches
                json_path = test_dir / f'lightglue_matches_{scale_str}.json'
                match_data = {
                    'method': 'LightGlue',
                    'num_matches': len(result['matches']),
                    'num_source_keypoints': len(result['kp1']),
                    'num_target_keypoints': len(result['kp2']),
                    'scale_factor': result.get('scale_factor', 1.0),
                    'source_shape': result.get('source_shape', source.shape),
                    'target_shape': result.get('target_shape', target.shape),
                    'matches': []
                }
                
                # Get match scores if available
                match_scores_list = result.get('match_scores', None)
                
                # Pixel resolution in meters (at current scale)
                # Using average for visualization - source is 2cm/pixel at full res, target is 19cm/pixel
                scale_used = scale  # Use the scale parameter passed to main()
                source_resolution_m_per_pixel = 0.02 / scale_used  # 2cm/pixel at full res, scaled
                target_resolution_m_per_pixel = 0.19 / scale_used  # 19cm/pixel at full res, scaled
                
                for i, match in enumerate(result['matches']):
                    src_kp = result['kp1'][match.queryIdx]
                    tgt_kp = result['kp2'][match.trainIdx]
                    scale_factor = result.get('scale_factor', 1.0)
                    tgt_x_upsampled = float(tgt_kp.pt[0])
                    tgt_y_upsampled = float(tgt_kp.pt[1])
                    tgt_x_original = tgt_x_upsampled / scale_factor if scale_factor != 1.0 else tgt_x_upsampled
                    tgt_y_original = tgt_y_upsampled / scale_factor if scale_factor != 1.0 else tgt_y_upsampled
                    
                    # Get actual response and distance values
                    src_response = float(src_kp.response) if hasattr(src_kp, 'response') and src_kp.response > 0 else 0.0
                    tgt_response = float(tgt_kp.response) if hasattr(tgt_kp, 'response') and tgt_kp.response > 0 else 0.0
                    match_distance = float(match.distance) if hasattr(match, 'distance') else 0.0
                    
                    # If we have match scores stored, use them
                    if match_scores_list is not None and i < len(match_scores_list):
                        confidence = float(match_scores_list[i])
                        match_distance = 1.0 - confidence  # Convert to distance
                    
                    # Calculate pixel distance between matched points
                    src_x = float(src_kp.pt[0])
                    src_y = float(src_kp.pt[1])
                    pixel_distance = np.sqrt((src_x - tgt_x_upsampled)**2 + (src_y - tgt_y_upsampled)**2)
                    
                    # Convert to meters (using source resolution as reference)
                    distance_meters = pixel_distance * source_resolution_m_per_pixel
                    
                    match_data['matches'].append({
                        'source': {'x': src_x, 'y': src_y, 
                                  'response': src_response},
                        'target_upsampled': {'x': tgt_x_upsampled, 'y': tgt_y_upsampled,
                                            'note': 'Coordinates in upsampled target space (matches source size)'},
                        'target_original': {'x': tgt_x_original, 'y': tgt_y_original,
                                           'note': 'Coordinates in original target space'},
                        'distance': {
                            'match_confidence': 1.0 - match_distance if match_distance < 1.0 else 0.0,
                            'pixels': float(pixel_distance),
                            'meters': float(distance_meters),
                            'note': 'Distance between matched points in image space'
                        }
                    })
                
                with open(json_path, 'w') as f:
                    json.dump(match_data, f, indent=2)
                print(f"Saved LightGlue match data to {json_path}")
            else:
                print(f"LightGlue returned no matches")
        except Exception as e:
            print(f"LightGlue failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. AROSICS
    if AROSICS_AVAILABLE:
        try:
            result = match_arosics(source_path, target_path)
            results.append(result)
            if result.get('matches'):
                # AROSICS visualization would need special handling
                print(f"AROSICS found {len(result['matches'])} tie points")
        except Exception as e:
            print(f"AROSICS failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Create visualization from JSON if LightGlue visualization failed
    lightglue_json = test_dir / f'lightglue_matches_{scale_str}.json'
    lightglue_png = test_dir / f'lightglue_matches_{scale_str}.png'
    if lightglue_json.exists() and not lightglue_png.exists():
        print(f"\nCreating LightGlue visualization from JSON file...")
        try:
            visualize_from_json(lightglue_json, source, target, lightglue_png)
            print(f"Saved LightGlue visualization from JSON")
        except Exception as e:
            print(f"Failed to create visualization from JSON: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for result in results:
        method = result['method']
        num_matches = len(result.get('matches', []))
        print(f"{method:15s}: {num_matches:4d} matches")
    
    # Create comparison visualization
    try:
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        axes = axes.flatten()
        
        # Show source and target
        axes[0].imshow(source, cmap='gray')
        axes[0].set_title('Source Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(target, cmap='gray')
        axes[1].set_title('Target Image', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Show masks
        axes[2].imshow(source_mask, cmap='gray')
        axes[2].set_title(f'Source Mask ({100 * np.sum(source_mask > 0) / source_mask.size:.1f}% valid)', 
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        axes[3].imshow(target_mask, cmap='gray')
        axes[3].set_title(f'Target Mask ({100 * np.sum(target_mask > 0) / target_mask.size:.1f}% valid)', 
                         fontsize=14, fontweight='bold')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(test_dir / f'comparison_overview_{scale_str}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison overview to {test_dir / f'comparison_overview_{scale_str}.png'}")
    except Exception as e:
        print(f"Could not create comparison visualization: {e}")
    
    print(f"\nAll results saved to: {test_dir}")
    print("\nGenerated files:")
    print(f"  - sift_matches_{scale_str}.png, sift_keypoints_{scale_str}.png, sift_matches_{scale_str}.json")
    print(f"  - orb_matches_{scale_str}.png, orb_keypoints_{scale_str}.png, orb_matches_{scale_str}.json")
    print(f"  - patch_ncc_matches_{scale_str}.png, patch_ncc_flow_{scale_str}.png, patch_ncc_matches_{scale_str}.json")
    print(f"  - lightglue_matches_{scale_str}.png, lightglue_keypoints_{scale_str}.png, lightglue_matches_{scale_str}.json")
    print(f"  - source_mask_{scale_str}.png, target_mask_{scale_str}.png")
    print(f"  - comparison_overview_{scale_str}.png")


if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Test image matching at different scales')
    parser.add_argument('scale', type=float, nargs='?', default=0.150,
                       help='Scale factor (e.g., 0.150, 0.300, 0.500). Default: 0.150')
    parser.add_argument('--method', type=str, default='lightglue',
                       choices=['lightglue', 'sift', 'orb', 'patch_ncc', 'all'],
                       help='Matching method. Default: lightglue')
    parser.add_argument('--tile-size', type=int, default=2048,
                       help='Tile size for tiled processing (default: 2048)')
    parser.add_argument('--overlap', type=int, default=256,
                       help='Overlap between tiles in pixels (default: 256)')
    parser.add_argument('--expected-error', type=float, default=3.0,
                       help='Expected matching error in meters (default: 3.0)')
    
    args = parser.parse_args()
    
    # Update global tile settings if provided
    if args.tile_size:
        import test_matching
        # Store in a way that match_lightglue can access
        test_matching.DEFAULT_TILE_SIZE = args.tile_size
        test_matching.DEFAULT_OVERLAP = args.overlap
        test_matching.DEFAULT_EXPECTED_ERROR = args.expected_error
    
    main(scale=args.scale, method=args.method)


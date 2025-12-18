"""
Transformations module for orthomosaic registration.
Provides transformation computation and application: shift, similarity, affine, homography,
polynomial (2nd and 3rd order), spline, and rubber sheeting.
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Try to import scipy for spline transformations
try:
    from scipy.interpolate import RBFInterpolator, griddata
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Spline transformations will be disabled.")

# Try to import rasterio for GeoTIFF output
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.enums import Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not available. GeoTIFF output will be disabled.")

# Note: Matching functions are in matching.py, not imported here to avoid circular dependencies


def load_matches(json_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load matches from JSON file and return source and target points."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    src_points = []
    tgt_points = []
    
    for match in data.get('matches', []):
        src = match.get('source', {})
        # Use target_upsampled if available (same scale as source), otherwise target_original
        tgt = match.get('target_upsampled', match.get('target_original', {}))
        
        if isinstance(src, dict) and isinstance(tgt, dict):
            if 'x' in src and 'y' in src and 'x' in tgt and 'y' in tgt:
                src_points.append([float(src['x']), float(src['y'])])
                tgt_points.append([float(tgt['x']), float(tgt['y'])])
        elif isinstance(src, (list, tuple)) and isinstance(tgt, (list, tuple)):
            # Handle list/tuple format
            if len(src) >= 2 and len(tgt) >= 2:
                src_points.append([float(src[0]), float(src[1])])
                tgt_points.append([float(tgt[0]), float(tgt[1])])
    
    return np.array(src_points, dtype=np.float32), np.array(tgt_points, dtype=np.float32)


def remove_gross_outliers(src_pts: np.ndarray, dst_pts: np.ndarray, 
                          max_offset_pixels: float = 1000.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove gross outliers based on offset distance.
    
    Args:
        src_pts: Source points (N, 2)
        dst_pts: Destination points (N, 2)
        max_offset_pixels: Maximum allowed offset in pixels
        
    Returns:
        (filtered_src, filtered_dst, inlier_mask)
    """
    offsets = dst_pts - src_pts
    distances = np.linalg.norm(offsets, axis=1)
    
    # Use robust statistics: median + 3*MAD (Median Absolute Deviation)
    median_dist = np.median(distances)
    mad = np.median(np.abs(distances - median_dist))
    threshold = median_dist + 3 * mad
    
    # Also apply absolute threshold
    threshold = min(threshold, max_offset_pixels)
    
    inlier_mask = distances <= threshold
    
    print(f"  Gross outlier removal: {np.sum(inlier_mask)}/{len(src_pts)} points kept "
          f"(threshold: {threshold:.1f} pixels, median: {median_dist:.1f} pixels)")
    
    return src_pts[inlier_mask], dst_pts[inlier_mask], inlier_mask


def compute_2d_shift(src_pts: np.ndarray, dst_pts: np.ndarray,
                    ransac_threshold: float = 5.0) -> Dict:
    """Compute 2D shift (translation only) transformation."""
    if len(src_pts) < 1:
        return {'type': 'shift', 'matrix': None, 'error': 'insufficient_points'}
    
    # Simple mean shift (no RANSAC needed for translation)
    offsets = dst_pts - src_pts
    shift = np.mean(offsets, axis=0)
    
    # Compute errors
    errors = offsets - shift
    reproj_errors = np.linalg.norm(errors, axis=1)
    
    # Robust statistics
    median_error = np.median(reproj_errors)
    mad = np.median(np.abs(reproj_errors - median_error))
    robust_rmse = median_error + 1.4826 * mad  # Convert MAD to std estimate
    
    # Create transformation matrix
    M = np.array([
        [1.0, 0.0, shift[0]],
        [0.0, 1.0, shift[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    return {
        'type': 'shift',
        'matrix': M,
        'shift_x': float(shift[0]),
        'shift_y': float(shift[1]),
        'num_points': len(src_pts),
        'median_error': float(median_error),
        'robust_rmse': float(robust_rmse),
        'mean_error': float(np.mean(reproj_errors)),
        'std_error': float(np.std(reproj_errors)),
        'max_error': float(np.max(reproj_errors)),
        'reproj_errors': [float(x) for x in reproj_errors.tolist()]  # Ensure all are Python floats
    }


def compute_affine_transform(src_pts: np.ndarray, dst_pts: np.ndarray,
                             ransac_threshold: float = 5.0) -> Dict:
    """Compute affine transformation using RANSAC."""
    if len(src_pts) < 3:
        return {'type': 'affine', 'matrix': None, 'error': 'insufficient_points'}
    
    # Use OpenCV's RANSAC
    M, inliers = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=2000,
        confidence=0.99
    )
    
    if M is None:
        return {'type': 'affine', 'matrix': None, 'error': 'ransac_failed'}
    
    # Convert to 3x3 matrix
    M_3x3 = np.vstack([M, [0, 0, 1]])
    
    # Compute reprojection errors for inliers
    src_pts_homogeneous = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    try:
        dst_pts_transformed = (M_3x3 @ src_pts_homogeneous.T).T[:, :2]
        reproj_errors = np.linalg.norm(dst_pts - dst_pts_transformed, axis=1)
        # Handle any invalid values
        reproj_errors = np.nan_to_num(reproj_errors, nan=1e6, posinf=1e6, neginf=1e6)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"    Warning: Error computing reprojection errors: {e}")
        reproj_errors = np.full(len(src_pts), 1e6)
    
    inlier_errors = reproj_errors[inliers.ravel() > 0] if inliers is not None else reproj_errors
    
    # Robust statistics
    median_error = np.median(inlier_errors) if len(inlier_errors) > 0 else np.median(reproj_errors)
    mad = np.median(np.abs(inlier_errors - median_error)) if len(inlier_errors) > 0 else np.median(np.abs(reproj_errors - median_error))
    robust_rmse = median_error + 1.4826 * mad
    
    inlier_count = np.sum(inliers) if inliers is not None else len(src_pts)
    
    return {
        'type': 'affine',
        'matrix': M_3x3,
        'num_points': len(src_pts),
        'inlier_count': int(inlier_count),
        'inlier_ratio': float(inlier_count / len(src_pts)),
        'median_error': float(median_error),
        'robust_rmse': float(robust_rmse),
        'mean_error': float(np.mean(inlier_errors)) if len(inlier_errors) > 0 else float(np.mean(reproj_errors)),
        'std_error': float(np.std(inlier_errors)) if len(inlier_errors) > 0 else float(np.std(reproj_errors)),
        'max_error': float(np.max(inlier_errors)) if len(inlier_errors) > 0 else float(np.max(reproj_errors)),
        'reproj_errors': reproj_errors.tolist(),
        'inliers': inliers.ravel().tolist() if inliers is not None else None
    }


def compute_homography(src_pts: np.ndarray, dst_pts: np.ndarray,
                      ransac_threshold: float = 5.0) -> Dict:
    """Compute homography transformation using RANSAC."""
    if len(src_pts) < 4:
        return {'type': 'homography', 'matrix': None, 'error': 'insufficient_points'}
    
    # Use OpenCV's RANSAC
    M, inliers = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=2000,
        confidence=0.99
    )
    
    if M is None:
        return {'type': 'homography', 'matrix': None, 'error': 'ransac_failed'}
    
    # Compute reprojection errors for inliers
    src_pts_homogeneous = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    try:
        dst_pts_transformed_homogeneous = (M @ src_pts_homogeneous.T).T
        # Handle division by zero for w coordinate
        w_coords = dst_pts_transformed_homogeneous[:, 2:3]
        w_coords = np.where(np.abs(w_coords) < 1e-6, 1.0, w_coords)  # Avoid division by zero
        dst_pts_transformed = dst_pts_transformed_homogeneous[:, :2] / w_coords
        reproj_errors = np.linalg.norm(dst_pts - dst_pts_transformed, axis=1)
        # Handle any invalid values
        reproj_errors = np.nan_to_num(reproj_errors, nan=1e6, posinf=1e6, neginf=1e6)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"    Warning: Error computing reprojection errors: {e}")
        reproj_errors = np.full(len(src_pts), 1e6)
    
    inlier_errors = reproj_errors[inliers.ravel() > 0] if inliers is not None else reproj_errors
    
    # Robust statistics
    median_error = np.median(inlier_errors) if len(inlier_errors) > 0 else np.median(reproj_errors)
    mad = np.median(np.abs(inlier_errors - median_error)) if len(inlier_errors) > 0 else np.median(np.abs(reproj_errors - median_error))
    robust_rmse = median_error + 1.4826 * mad
    
    inlier_count = np.sum(inliers) if inliers is not None else len(src_pts)
    
    return {
        'type': 'homography',
        'matrix': M,
        'num_points': len(src_pts),
        'inlier_count': int(inlier_count),
        'inlier_ratio': float(inlier_count / len(src_pts)),
        'median_error': float(median_error),
        'robust_rmse': float(robust_rmse),
        'mean_error': float(np.mean(inlier_errors)) if len(inlier_errors) > 0 else float(np.mean(reproj_errors)),
        'std_error': float(np.std(inlier_errors)) if len(inlier_errors) > 0 else float(np.std(reproj_errors)),
        'max_error': float(np.max(inlier_errors)) if len(inlier_errors) > 0 else float(np.max(reproj_errors)),
        'reproj_errors': [float(x) for x in reproj_errors.tolist()],
        'inliers': inliers.ravel().astype(int).tolist() if inliers is not None else None
    }


def compute_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray,
                                 ransac_threshold: float = 5.0) -> Dict:
    """Compute similarity transformation (rotation, uniform scale, translation) using RANSAC."""
    if len(src_pts) < 2:
        return {'type': 'similarity', 'matrix': None, 'error': 'insufficient_points'}
    
    # Use OpenCV's estimateAffinePartial2D (similarity = rigid + uniform scale)
    M, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=2000,
        confidence=0.99
    )
    
    if M is None:
        return {'type': 'similarity', 'matrix': None, 'error': 'ransac_failed'}
    
    # Convert to 3x3 matrix
    M_3x3 = np.vstack([M, [0, 0, 1]])
    
    # Compute reprojection errors
    src_pts_homogeneous = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    try:
        dst_pts_transformed = (M_3x3 @ src_pts_homogeneous.T).T[:, :2]
        reproj_errors = np.linalg.norm(dst_pts - dst_pts_transformed, axis=1)
        reproj_errors = np.nan_to_num(reproj_errors, nan=1e6, posinf=1e6, neginf=1e6)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"    Warning: Error computing reprojection errors: {e}")
        reproj_errors = np.full(len(src_pts), 1e6)
    
    inlier_mask = inliers.ravel() > 0 if inliers is not None else np.ones(len(src_pts), dtype=bool)
    inlier_errors = reproj_errors[inlier_mask] if np.any(inlier_mask) else reproj_errors
    
    # Robust statistics
    median_error = np.median(inlier_errors) if len(inlier_errors) > 0 else np.median(reproj_errors)
    mad = np.median(np.abs(inlier_errors - median_error)) if len(inlier_errors) > 0 else np.median(np.abs(reproj_errors - median_error))
    robust_rmse = median_error + 1.4826 * mad
    
    inlier_count = int(np.sum(inlier_mask))
    
    return {
        'type': 'similarity',
        'matrix': M_3x3,
        'num_points': len(src_pts),
        'inlier_count': inlier_count,
        'inlier_ratio': float(inlier_count / len(src_pts)) if len(src_pts) > 0 else 0.0,
        'median_error': float(median_error),
        'robust_rmse': float(robust_rmse),
        'mean_error': float(np.mean(inlier_errors)) if len(inlier_errors) > 0 else float(np.mean(reproj_errors)),
        'std_error': float(np.std(inlier_errors)) if len(inlier_errors) > 0 else float(np.std(reproj_errors)),
        'max_error': float(np.max(inlier_errors)) if len(inlier_errors) > 0 else float(np.max(reproj_errors)),
        'reproj_errors': [float(x) for x in reproj_errors.tolist()],
        'inliers': inliers.ravel().astype(int).tolist() if inliers is not None else None
    }


def compute_polynomial_transform(src_pts: np.ndarray, dst_pts: np.ndarray,
                                 degree: int = 2, ransac_threshold: float = 5.0) -> Dict:
    """Compute polynomial transformation (2nd or 3rd order)."""
    min_points = (degree + 1) * (degree + 2) // 2
    if len(src_pts) < min_points:
        return {'type': f'polynomial_{degree}', 'matrix': None, 'error': 'insufficient_points'}
    
    # Use RANSAC to find inliers first
    M_affine, inliers_affine = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold * 2,
        maxIters=2000,
        confidence=0.99
    )
    
    if M_affine is None:
        return {'type': f'polynomial_{degree}', 'matrix': None, 'error': 'ransac_failed'}
    
    inlier_mask = inliers_affine.ravel() > 0 if inliers_affine is not None else np.ones(len(src_pts), dtype=bool)
    src_inliers = src_pts[inlier_mask]
    dst_inliers = dst_pts[inlier_mask]
    
    if len(src_inliers) < min_points:
        return {'type': f'polynomial_{degree}', 'matrix': None, 'error': 'insufficient_inliers'}
    
    # Build polynomial basis functions
    def build_poly_basis(pts, deg):
        n = len(pts)
        if deg == 2:
            return np.column_stack([
                np.ones(n), pts[:, 0], pts[:, 1],
                pts[:, 0]**2, pts[:, 0]*pts[:, 1], pts[:, 1]**2
            ])
        elif deg == 3:
            return np.column_stack([
                np.ones(n), pts[:, 0], pts[:, 1],
                pts[:, 0]**2, pts[:, 0]*pts[:, 1], pts[:, 1]**2,
                pts[:, 0]**3, pts[:, 0]**2*pts[:, 1], pts[:, 0]*pts[:, 1]**2, pts[:, 1]**3
            ])
        else:
            raise ValueError(f"Unsupported polynomial degree: {deg}")
    
    basis = build_poly_basis(src_inliers, degree)
    
    # Solve for polynomial coefficients
    try:
        coeffs_x, residuals_x, rank_x, s_x = np.linalg.lstsq(basis, dst_inliers[:, 0], rcond=None)
        coeffs_y, residuals_y, rank_y, s_y = np.linalg.lstsq(basis, dst_inliers[:, 1], rcond=None)
        
        # Check if solution is valid (rank should equal number of basis functions)
        # Allow slight rank deficiency (e.g., rank >= expected_rank - 1) for robustness
        expected_rank = basis.shape[1]
        min_rank = min(rank_x, rank_y)
        if min_rank < expected_rank - 1:
            return {'type': f'polynomial_{degree}', 'matrix': None, 'error': f'rank_deficient (rank={min_rank}/{expected_rank})'}
        # If slightly rank-deficient, proceed but note it
        if min_rank < expected_rank:
            print(f"     Warning: Polynomial {degree} has rank {min_rank}/{expected_rank}, proceeding with regularization")
        
        # Check for extreme coefficients that would cause numerical issues
        if np.any(np.abs(coeffs_x) > 1e10) or np.any(np.abs(coeffs_y) > 1e10):
            return {'type': f'polynomial_{degree}', 'matrix': None, 'error': 'extreme_coefficients'}
    except np.linalg.LinAlgError as e:
        return {'type': f'polynomial_{degree}', 'matrix': None, 'error': f'solve_failed: {str(e)}'}
    except Exception as e:
        return {'type': f'polynomial_{degree}', 'matrix': None, 'error': f'unexpected_error: {str(e)}'}
    
    # Apply to all points to compute errors
    try:
        basis_all = build_poly_basis(src_pts, degree)
        dst_predicted = np.column_stack([
            basis_all @ coeffs_x,
            basis_all @ coeffs_y
        ])
        # Check for invalid predictions
        valid_pred = ~(np.isnan(dst_predicted).any(axis=1) | np.isinf(dst_predicted).any(axis=1))
        if np.sum(valid_pred) == 0:
            return {'type': f'polynomial_{degree}', 'matrix': None, 'error': 'all_predictions_invalid'}
        
        reproj_errors = np.linalg.norm(dst_pts - dst_predicted, axis=1)
        # Clip extreme errors
        if np.sum(valid_pred) > 0:
            max_reasonable = np.percentile(reproj_errors[valid_pred], 99) * 10
            reproj_errors[~valid_pred] = max_reasonable
    except Exception as e:
        return {'type': f'polynomial_{degree}', 'matrix': None, 'error': f'prediction_failed: {str(e)}'}
    
    inlier_errors = reproj_errors[inlier_mask] if np.any(inlier_mask) else reproj_errors
    
    # Robust statistics
    median_error = np.median(inlier_errors) if len(inlier_errors) > 0 else np.median(reproj_errors)
    mad = np.median(np.abs(inlier_errors - median_error)) if len(inlier_errors) > 0 else np.median(np.abs(reproj_errors - median_error))
    robust_rmse = median_error + 1.4826 * mad
    
    inlier_count = int(np.sum(inlier_mask))
    
    return {
        'type': f'polynomial_{degree}',
        'matrix': None,
        'coefficients_x': [float(x) for x in coeffs_x.tolist()],
        'coefficients_y': [float(x) for x in coeffs_y.tolist()],
        'degree': degree,
        'num_points': len(src_pts),
        'inlier_count': inlier_count,
        'inlier_ratio': float(inlier_count / len(src_pts)) if len(src_pts) > 0 else 0.0,
        'median_error': float(median_error),
        'robust_rmse': float(robust_rmse),
        'mean_error': float(np.mean(inlier_errors)) if len(inlier_errors) > 0 else float(np.mean(reproj_errors)),
        'std_error': float(np.std(inlier_errors)) if len(inlier_errors) > 0 else float(np.std(reproj_errors)),
        'max_error': float(np.max(inlier_errors)) if len(inlier_errors) > 0 else float(np.max(reproj_errors)),
        'reproj_errors': [float(x) for x in reproj_errors.tolist()],
        'inliers': [int(x) for x in inlier_mask.astype(int).tolist()]
    }


def compute_spline_transform(src_pts: np.ndarray, dst_pts: np.ndarray,
                             ransac_threshold: float = 5.0) -> Dict:
    """Compute thin-plate spline (TPS) transformation."""
    if not SCIPY_AVAILABLE:
        return {'type': 'spline', 'matrix': None, 'error': 'scipy_not_available'}
    
    if len(src_pts) < 3:
        return {'type': 'spline', 'matrix': None, 'error': 'insufficient_points'}
    
    # Use RANSAC to find inliers first
    M_affine, inliers_affine = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold * 2,
        maxIters=2000,
        confidence=0.99
    )
    
    if M_affine is None:
        return {'type': 'spline', 'matrix': None, 'error': 'ransac_failed'}
    
    inlier_mask = inliers_affine.ravel() > 0 if inliers_affine is not None else np.ones(len(src_pts), dtype=bool)
    src_inliers = src_pts[inlier_mask]
    dst_inliers = dst_pts[inlier_mask]
    
    if len(src_inliers) < 3:
        return {'type': 'spline', 'matrix': None, 'error': 'insufficient_inliers'}
    
    # Use RBFInterpolator for thin-plate spline
    # Check for collinear points that would cause singular matrix
    # Compute area of triangles formed by points
    # Don't fail immediately - try with smoothing first
    is_collinear = False
    if len(src_inliers) >= 3:
        # Check if points are collinear by computing triangle areas
        areas = []
        for i in range(min(100, len(src_inliers) - 2)):  # Sample to avoid too much computation
            p1, p2, p3 = src_inliers[i], src_inliers[i+1], src_inliers[i+2]
            area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
            areas.append(area)
        min_area = min(areas) if areas else 0
        if min_area < 1e-6:
            is_collinear = True
            print(f"     Warning: Points appear collinear (min_area={min_area:.2e}), using smoothing")
    
    # Try with increasing smoothing to avoid singular matrix
    # Start with higher smoothing if points are collinear
    if is_collinear:
        smoothing_values = [1.0, 10.0, 100.0, 1000.0]
    else:
        smoothing_values = [0.0, 0.1, 1.0, 10.0]
    
    rbf_x = None
    rbf_y = None
    used_smoothing = None
    
    for smoothing in smoothing_values:
        try:
            rbf_x = RBFInterpolator(src_inliers, dst_inliers[:, 0], kernel='thin_plate_spline', smoothing=smoothing)
            rbf_y = RBFInterpolator(src_inliers, dst_inliers[:, 1], kernel='thin_plate_spline', smoothing=smoothing)
            # Test if it works by predicting a few points
            test_pred = np.column_stack([rbf_x(src_inliers[:min(10, len(src_inliers))]), 
                                        rbf_y(src_inliers[:min(10, len(src_inliers))])])
            if not np.any(np.isnan(test_pred)) and not np.any(np.isinf(test_pred)):
                used_smoothing = smoothing
                break
        except Exception as e:
            if smoothing == smoothing_values[-1]:  # Last attempt failed
                return {'type': 'spline', 'matrix': None, 'error': f'rbf_failed: {str(e)}'}
            continue
    
    if rbf_x is None or rbf_y is None:
        return {'type': 'spline', 'matrix': None, 'error': 'rbf_initialization_failed'}
    
    # Apply to all points to compute errors
    try:
        dst_predicted = np.column_stack([
            rbf_x(src_pts),
            rbf_y(src_pts)
        ])
        # Check for invalid predictions
        valid_pred = ~(np.isnan(dst_predicted).any(axis=1) | np.isinf(dst_predicted).any(axis=1))
        if np.sum(valid_pred) == 0:
            return {'type': 'spline', 'matrix': None, 'error': 'all_predictions_invalid'}
        
        reproj_errors = np.linalg.norm(dst_pts - dst_predicted, axis=1)
        # Set large error for invalid predictions
        reproj_errors[~valid_pred] = np.percentile(reproj_errors[valid_pred], 99) * 10 if np.sum(valid_pred) > 0 else 1e6
    except Exception as e:
        return {'type': 'spline', 'matrix': None, 'error': f'prediction_failed: {str(e)}'}
    
    inlier_errors = reproj_errors[inlier_mask] if np.any(inlier_mask) else reproj_errors
    
    # Robust statistics
    median_error = np.median(inlier_errors) if len(inlier_errors) > 0 else np.median(reproj_errors)
    mad = np.median(np.abs(inlier_errors - median_error)) if len(inlier_errors) > 0 else np.median(np.abs(reproj_errors - median_error))
    robust_rmse = median_error + 1.4826 * mad
    
    inlier_count = int(np.sum(inlier_mask))
    
    return {
        'type': 'spline',
        'matrix': None,
        'control_points_src': [[float(x), float(y)] for x, y in src_inliers.tolist()],
        'control_points_dst': [[float(x), float(y)] for x, y in dst_inliers.tolist()],
        'num_points': len(src_pts),
        'inlier_count': inlier_count,
        'inlier_ratio': float(inlier_count / len(src_pts)) if len(src_pts) > 0 else 0.0,
        'median_error': float(median_error),
        'robust_rmse': float(robust_rmse),
        'mean_error': float(np.mean(inlier_errors)) if len(inlier_errors) > 0 else float(np.mean(reproj_errors)),
        'std_error': float(np.std(inlier_errors)) if len(inlier_errors) > 0 else float(np.std(reproj_errors)),
        'max_error': float(np.max(inlier_errors)) if len(inlier_errors) > 0 else float(np.max(reproj_errors)),
        'reproj_errors': [float(x) for x in reproj_errors.tolist()],
        'inliers': [int(x) for x in inlier_mask.astype(int).tolist()]
    }


def compute_rubber_sheeting_transform(src_pts: np.ndarray, dst_pts: np.ndarray,
                                      ransac_threshold: float = 5.0) -> Dict:
    """Compute rubber sheeting transformation using Delaunay triangulation."""
    if not SCIPY_AVAILABLE:
        return {'type': 'rubber_sheeting', 'matrix': None, 'error': 'scipy_not_available'}
    
    if len(src_pts) < 3:
        return {'type': 'rubber_sheeting', 'matrix': None, 'error': 'insufficient_points'}
    
    # Use RANSAC to find inliers first
    M_affine, inliers_affine = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold * 2,
        maxIters=2000,
        confidence=0.99
    )
    
    if M_affine is None:
        return {'type': 'rubber_sheeting', 'matrix': None, 'error': 'ransac_failed'}
    
    inlier_mask = inliers_affine.ravel() > 0 if inliers_affine is not None else np.ones(len(src_pts), dtype=bool)
    src_inliers = src_pts[inlier_mask]
    dst_inliers = dst_pts[inlier_mask]
    
    if len(src_inliers) < 3:
        return {'type': 'rubber_sheeting', 'matrix': None, 'error': 'insufficient_inliers'}
    
    # Rubber sheeting uses piecewise affine transformations on Delaunay triangles
    try:
        from scipy.spatial import Delaunay
        from scipy.interpolate import griddata
        
        # Build Delaunay triangulation on source points
        tri = Delaunay(src_inliers)
        
        # For evaluation, use griddata for interpolation
        # Use 'nearest' as fallback for points outside convex hull to avoid NaN
        dst_predicted_x = griddata(src_inliers, dst_inliers[:, 0], src_pts, method='linear', fill_value=np.nan)
        dst_predicted_y = griddata(src_inliers, dst_inliers[:, 1], src_pts, method='linear', fill_value=np.nan)
        
        # Handle NaN values (points outside convex hull) - use nearest neighbor for these
        nan_mask = np.isnan(dst_predicted_x) | np.isnan(dst_predicted_y)
        if np.any(nan_mask):
            # For points outside convex hull, use nearest neighbor interpolation
            dst_predicted_x_nn = griddata(src_inliers, dst_inliers[:, 0], src_pts[nan_mask], method='nearest')
            dst_predicted_y_nn = griddata(src_inliers, dst_inliers[:, 1], src_pts[nan_mask], method='nearest')
            dst_predicted_x[nan_mask] = dst_predicted_x_nn
            dst_predicted_y[nan_mask] = dst_predicted_y_nn
        
        valid = ~(np.isnan(dst_predicted_x) | np.isnan(dst_predicted_y) | 
                 np.isinf(dst_predicted_x) | np.isinf(dst_predicted_y))
        if np.sum(valid) == 0:
            return {'type': 'rubber_sheeting', 'matrix': None, 'error': 'no_valid_predictions'}
        
        dst_predicted = np.column_stack([dst_predicted_x, dst_predicted_y])
        reproj_errors = np.linalg.norm(dst_pts - dst_predicted, axis=1)
        
        # Clip extreme errors to reasonable maximum
        # Use percentile-based clipping to handle outliers (points outside convex hull)
        if np.sum(valid) > 0:
            p99 = np.percentile(reproj_errors[valid], 99)
            max_reasonable_error = max(p99 * 5, 100.0)  # At least 100 pixels, or 5x the 99th percentile
            reproj_errors = np.clip(reproj_errors, 0, max_reasonable_error)
        
    except Exception as e:
        return {'type': 'rubber_sheeting', 'matrix': None, 'error': f'computation_failed: {str(e)}'}
    
    inlier_errors = reproj_errors[inlier_mask & valid] if np.any(inlier_mask & valid) else reproj_errors[valid]
    
    # Robust statistics
    median_error = np.median(inlier_errors) if len(inlier_errors) > 0 else np.median(reproj_errors[valid])
    mad = np.median(np.abs(inlier_errors - median_error)) if len(inlier_errors) > 0 else np.median(np.abs(reproj_errors[valid] - median_error))
    robust_rmse = median_error + 1.4826 * mad
    
    inlier_count = int(np.sum(inlier_mask))
    
    return {
        'type': 'rubber_sheeting',
        'matrix': None,
        'control_points_src': [[float(x), float(y)] for x, y in src_inliers.tolist()],
        'control_points_dst': [[float(x), float(y)] for x, y in dst_inliers.tolist()],
        'num_points': len(src_pts),
        'inlier_count': inlier_count,
        'inlier_ratio': float(inlier_count / len(src_pts)) if len(src_pts) > 0 else 0.0,
        'median_error': float(median_error),
        'robust_rmse': float(robust_rmse),
        'mean_error': float(np.mean(inlier_errors)) if len(inlier_errors) > 0 else float(np.mean(reproj_errors[valid])),
        'std_error': float(np.std(inlier_errors)) if len(inlier_errors) > 0 else float(np.std(reproj_errors[valid])),
        'max_error': float(np.max(inlier_errors)) if len(inlier_errors) > 0 else float(np.max(reproj_errors[valid])),
        'reproj_errors': [float(x) for x in reproj_errors.tolist()],
        'inliers': [int(x) for x in inlier_mask.astype(int).tolist()]
    }


def choose_best_transform(transforms: List[Dict]) -> Dict:
    """Choose the best transformation based on robust statistics."""
    valid_transforms = [t for t in transforms if t.get('matrix') is not None]
    
    if not valid_transforms:
        return None
    
    # Sort by robust_rmse (lower is better)
    valid_transforms.sort(key=lambda x: x.get('robust_rmse', float('inf')))
    
    best = valid_transforms[0]
    
    print(f"\n  Best transformation: {best['type']}")
    print(f"    Robust RMSE: {best['robust_rmse']:.3f} pixels")
    print(f"    Median error: {best['median_error']:.3f} pixels")
    print(f"    Inlier ratio: {best.get('inlier_ratio', 1.0):.1%}")
    print(f"    Points: {best['num_points']}")
    
    return best


def apply_transform_to_image(image: np.ndarray, M: np.ndarray, 
                             target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Apply transformation matrix to image.
    
    For affine (2x3), convert to 3x3 homography format for warpPerspective.
    """
    h, w = image.shape[:2]
    
    # Ensure M is 3x3
    if M.shape == (2, 3):
        # Affine matrix - convert to 3x3
        M_3x3 = np.vstack([M, [0, 0, 1]])
    elif M.shape == (3, 3):
        M_3x3 = M.copy()
    else:
        raise ValueError(f"Unsupported matrix shape: {M.shape}")
    
    if target_shape is None:
        # Determine output size from transformed corners
        corners = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1]
        ], dtype=np.float32).T
        
        transformed_corners = (M_3x3 @ corners).T
        # Handle homography (divide by w coordinate)
        if np.any(np.abs(transformed_corners[:, 2]) > 1e-6):
            transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:3]
        else:
            transformed_corners = transformed_corners[:, :2]
        
        min_x, min_y = transformed_corners.min(axis=0)
        max_x, max_y = transformed_corners.max(axis=0)
        
        out_w = int(np.ceil(max_x - min_x))
        out_h = int(np.ceil(max_y - min_y))
        
        # Adjust transformation for translation
        M_adjusted = M_3x3.copy()
        M_adjusted[0, 2] -= min_x
        M_adjusted[1, 2] -= min_y
    else:
        out_w, out_h = target_shape[1], target_shape[0]
        M_adjusted = M_3x3.copy()
    
    # Apply transformation
    if len(image.shape) == 2:
        # Grayscale
        transformed = cv2.warpPerspective(image, M_adjusted, (out_w, out_h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
    else:
        # Color
        transformed = cv2.warpPerspective(image, M_adjusted, (out_w, out_h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
    
    return transformed


def apply_polynomial_transform_to_image(image: np.ndarray, transform_dict: Dict,
                                        target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Apply polynomial transformation to image."""
    h, w = image.shape[:2]
    degree = transform_dict.get('degree', 2)
    coeffs_x = np.array(transform_dict['coefficients_x'])
    coeffs_y = np.array(transform_dict['coefficients_y'])
    
    # Build polynomial basis for all pixels
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
    
    def build_poly_basis(pts, deg):
        n = len(pts)
        if deg == 2:
            return np.column_stack([
                np.ones(n), pts[:, 0], pts[:, 1],
                pts[:, 0]**2, pts[:, 0]*pts[:, 1], pts[:, 1]**2
            ])
        elif deg == 3:
            return np.column_stack([
                np.ones(n), pts[:, 0], pts[:, 1],
                pts[:, 0]**2, pts[:, 0]*pts[:, 1], pts[:, 1]**2,
                pts[:, 0]**3, pts[:, 0]**2*pts[:, 1], pts[:, 0]*pts[:, 1]**2, pts[:, 1]**3
            ])
        else:
            raise ValueError(f"Unsupported polynomial degree: {deg}")
    
    basis = build_poly_basis(coords, degree)
    new_x = (basis @ coeffs_x).reshape(h, w)
    new_y = (basis @ coeffs_y).reshape(h, w)
    
    # Use remap to apply transformation
    map_x = new_x.astype(np.float32)
    map_y = new_y.astype(np.float32)
    
    if target_shape is not None:
        # Resize maps to target shape
        map_x = cv2.resize(map_x, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(map_y, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        out_h, out_w = target_shape
    else:
        # Determine output size from transformed bounds
        min_x, max_x = map_x.min(), map_x.max()
        min_y, max_y = map_y.min(), map_y.max()
        out_w = int(np.ceil(max_x - min_x))
        out_h = int(np.ceil(max_y - min_y))
        map_x -= min_x
        map_y -= min_y
    
    transformed = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return transformed


def apply_spline_transform_to_image(image: np.ndarray, transform_dict: Dict,
                                   target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Apply spline transformation to image."""
    if not SCIPY_AVAILABLE:
        raise ValueError("scipy not available for spline transformation")
    
    h, w = image.shape[:2]
    src_control = np.array(transform_dict['control_points_src'])
    dst_control = np.array(transform_dict['control_points_dst'])
    
    # Rebuild RBF interpolators
    rbf_x = RBFInterpolator(src_control, dst_control[:, 0], kernel='thin_plate_spline', smoothing=0.0)
    rbf_y = RBFInterpolator(src_control, dst_control[:, 1], kernel='thin_plate_spline', smoothing=0.0)
    
    # Build coordinate grid
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
    
    # Apply transformation
    new_x = rbf_x(coords).reshape(h, w)
    new_y = rbf_y(coords).reshape(h, w)
    
    map_x = new_x.astype(np.float32)
    map_y = new_y.astype(np.float32)
    
    if target_shape is not None:
        map_x = cv2.resize(map_x, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(map_y, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        out_h, out_w = target_shape
    else:
        min_x, max_x = map_x.min(), map_x.max()
        min_y, max_y = map_y.min(), map_y.max()
        out_w = int(np.ceil(max_x - min_x))
        out_h = int(np.ceil(max_y - min_y))
        map_x -= min_x
        map_y -= min_y
    
    transformed = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return transformed


def apply_rubber_sheeting_transform_to_image(image: np.ndarray, transform_dict: Dict,
                                            target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Apply rubber sheeting transformation to image."""
    if not SCIPY_AVAILABLE:
        raise ValueError("scipy not available for rubber sheeting")
    
    from scipy.interpolate import griddata
    
    h, w = image.shape[:2]
    src_control = np.array(transform_dict['control_points_src'])
    dst_control = np.array(transform_dict['control_points_dst'])
    
    # Build coordinate grid
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
    
    # Interpolate transformation
    new_x = griddata(src_control, dst_control[:, 0], coords, method='linear', fill_value=np.nan)
    new_y = griddata(src_control, dst_control[:, 1], coords, method='linear', fill_value=np.nan)
    
    # Handle NaN values
    valid = ~(np.isnan(new_x) | np.isnan(new_y))
    new_x[~valid] = coords[~valid, 0]  # Keep original x for invalid
    new_y[~valid] = coords[~valid, 1]  # Keep original y for invalid
    
    map_x = new_x.reshape(h, w).astype(np.float32)
    map_y = new_y.reshape(h, w).astype(np.float32)
    
    if target_shape is not None:
        map_x = cv2.resize(map_x, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(map_y, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        out_h, out_w = target_shape
    else:
        min_x, max_x = map_x.min(), map_x.max()
        min_y, max_y = map_y.min(), map_y.max()
        out_w = int(np.ceil(max_x - min_x))
        out_h = int(np.ceil(max_y - min_y))
        map_x -= min_x
        map_y -= min_y
    
    transformed = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return transformed


def save_geotiff(image: np.ndarray, output_path: Path, test_dir: Path, 
                 scale: float = 0.500, pixel_size_meters: float = 0.02):
    """Save image as GeoTIFF with JPEG compression (3-band RGB for QGIS compatibility).
    
    Uses georeferencing from the reference orthomosaic at the given scale.
    """
    if not RASTERIO_AVAILABLE:
        print(f"   Warning: rasterio not available, skipping GeoTIFF: {output_path.name}")
        return
    
    # Handle both grayscale and RGB images
    if len(image.shape) == 2:
        # Grayscale: convert to RGB by duplicating the channel
        h, w = image.shape
        # Ensure image is uint8 for JPEG compression
        if image.dtype != np.uint8:
            # Normalize to 0-255
            image_norm = ((image - image.min()) / (image.max() - image.min() + 1e-6) * 255).astype(np.uint8)
        else:
            image_norm = image
        
        # Convert grayscale to RGB (3 bands)
        image_rgb = np.stack([image_norm, image_norm, image_norm], axis=0)  # Shape: (3, H, W)
        num_bands = 3
    elif len(image.shape) == 3:
        # Already RGB or multi-band
        h, w = image.shape[:2]
        if image.shape[2] == 3:
            # RGB image
            if image.dtype != np.uint8:
                image_norm = ((image - image.min()) / (image.max() - image.min() + 1e-6) * 255).astype(np.uint8)
            else:
                image_norm = image
            # Convert from (H, W, 3) to (3, H, W) for rasterio
            image_rgb = np.transpose(image_norm, (2, 0, 1))
            num_bands = 3
        else:
            # Multi-band, use as-is
            if image.dtype != np.uint8:
                image_norm = ((image - image.min()) / (image.max() - image.min() + 1e-6) * 255).astype(np.uint8)
            else:
                image_norm = image
            image_rgb = np.transpose(image_norm, (2, 0, 1))
            num_bands = image.shape[2]
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    # Get georeferencing from reference orthomosaic
    # For overlap images, we need to extract the georeferencing for the overlap region
    # The overlap region is typically the top-left portion of the full orthomosaic
    scale_str = f"{scale:.3f}"
    if scale == 0.500:
        reference_ortho = test_dir / 'orthomosaic_no_gcps_utm10n_scale0.500_poly2.tif'
        if not reference_ortho.exists():
            reference_ortho = test_dir / f'orthomosaic_no_gcps_utm10n_scale{scale_str}.tif'
    else:
        reference_ortho = test_dir / f'orthomosaic_no_gcps_utm10n_scale{scale_str}.tif'
    
    if reference_ortho.exists():
        # Load georeferencing from reference orthomosaic
        with rasterio.open(reference_ortho) as ref:
            ref_transform = ref.transform
            ref_crs = ref.crs
            ref_width = ref.width
            ref_height = ref.height
            
            # Check if dimensions match (overlap region should match full orthomosaic at this scale)
            if w == ref_width and h == ref_height:
                # Use the exact transform from reference (overlap is the full image)
                transform = ref_transform
                crs = ref_crs
                print(f"   Using georeferencing from {reference_ortho.name} (full image)")
            else:
                # Overlap region is a subset - extract the transform for the overlap region
                # The overlap region is typically the top-left portion
                # Calculate the bounds of the overlap region
                # Get top-left corner coordinates from the transform
                left, top = ref_transform * (0, 0)  # Top-left corner
                right, bottom = ref_transform * (w, h)  # Bottom-right corner of overlap
                
                # Create transform for the overlap region
                # Pixel size remains the same, but origin is at (left, top)
                transform = rasterio.Affine(
                    ref_transform.a,  # x pixel size
                    ref_transform.b,  # row rotation
                    left,             # x origin (left edge)
                    ref_transform.d,  # column rotation
                    ref_transform.e,  # y pixel size (negative)
                    top               # y origin (top edge)
                )
                crs = ref_crs
                print(f"   Using georeferencing from {reference_ortho.name} (overlap region: {w}x{h})")
                print(f"     Bounds: ({left:.1f}, {bottom:.1f}) to ({right:.1f}, {top:.1f})")
    else:
        # Fallback: create simple georeferencing (shouldn't happen in normal workflow)
        print(f"   Warning: Reference orthomosaic not found, using simple georeferencing")
        transform = from_bounds(0, 0, w * pixel_size_meters, h * pixel_size_meters, w, h)
        crs = 'EPSG:32610'  # UTM Zone 10N
    
    # Save as RGB GeoTIFF with JPEG compression
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=h,
        width=w,
        count=num_bands,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        compress='JPEG',
        jpeg_quality=90,
        photometric='RGB' if num_bands == 3 else 'MINISBLACK',
        tiled=True,  # Tiled for better performance
        blockxsize=512,
        blockysize=512
    ) as dst:
        # Write each band
        for band_idx in range(1, num_bands + 1):
            dst.write(image_rgb[band_idx - 1], band_idx)
    
    print(f"   Saved GeoTIFF: {output_path.name} ({num_bands} bands, RGB format for QGIS)")


def process_scale_level(test_dir: Path, current_scale: float, next_scale: float,
                       matches_json: Path, ransac_threshold: float = 5.0) -> Dict:
    """
    Process one scale level: compute transformations, apply best, match at next scale.
    
    Returns:
        Dictionary with transformation results and next scale match data
    """
    current_scale_str = f"{current_scale:.3f}"
    next_scale_str = f"{next_scale:.3f}"
    
    print("\n" + "=" * 80)
    print(f"PROCESSING SCALE {current_scale_str} -> {next_scale_str}")
    print("=" * 80)
    
    # 1. Load matches
    print(f"\n1. Loading matches from {matches_json.name}...")
    src_pts, dst_pts = load_matches(matches_json)
    print(f"   Loaded {len(src_pts)} matches")
    
    # 2. Remove gross outliers
    print(f"\n2. Removing gross outliers...")
    src_pts_clean, dst_pts_clean, _ = remove_gross_outliers(src_pts, dst_pts)
    
    # 3. Compute transformations
    print(f"\n3. Computing transformations...")
    
    # Scale RANSAC threshold based on image resolution
    # At scale 0.15, 5 pixels = ~0.67m (5 * 0.133m/pixel)
    # Scale threshold proportionally
    pixel_resolution = 0.02 / current_scale  # meters per pixel
    scaled_ransac_threshold = ransac_threshold * (current_scale / 0.15)  # Scale with resolution
    
    transforms = []
    
    # Compute all transformation types
    # 2D Shift
    print("   Computing 2D shift...")
    shift_result = compute_2d_shift(src_pts_clean, dst_pts_clean, scaled_ransac_threshold)
    transforms.append(shift_result)
    
    # Similarity
    print("   Computing similarity transformation...")
    similarity_result = compute_similarity_transform(src_pts_clean, dst_pts_clean, scaled_ransac_threshold)
    transforms.append(similarity_result)
    
    # Affine
    print("   Computing affine transformation...")
    affine_result = compute_affine_transform(src_pts_clean, dst_pts_clean, scaled_ransac_threshold)
    transforms.append(affine_result)
    
    # Homography
    print("   Computing homography...")
    homography_result = compute_homography(src_pts_clean, dst_pts_clean, scaled_ransac_threshold)
    transforms.append(homography_result)
    
    # Polynomial (2nd order)
    print("   Computing polynomial (2nd order) transformation...")
    poly2_result = compute_polynomial_transform(src_pts_clean, dst_pts_clean, degree=2, ransac_threshold=scaled_ransac_threshold)
    transforms.append(poly2_result)
    
    # Polynomial (3rd order)
    print("   Computing polynomial (3rd order) transformation...")
    poly3_result = compute_polynomial_transform(src_pts_clean, dst_pts_clean, degree=3, ransac_threshold=scaled_ransac_threshold)
    transforms.append(poly3_result)
    
    # Spline and rubber sheeting: Skip at 0.5 scale due to computational issues
    if next_scale < 0.5:
        # Spline (thin-plate)
        print("   Computing spline (thin-plate) transformation...")
        spline_result = compute_spline_transform(src_pts_clean, dst_pts_clean, scaled_ransac_threshold)
        transforms.append(spline_result)
        
        # Rubber sheeting
        print("   Computing rubber sheeting transformation...")
        rubber_result = compute_rubber_sheeting_transform(src_pts_clean, dst_pts_clean, scaled_ransac_threshold)
        transforms.append(rubber_result)
    else:
        print("   Skipping spline and rubber_sheeting at 0.5 scale (computational issues)")
    
    # 4. Add meter conversions to all transforms and choose best
    print(f"\n4. Adding meter conversions and choosing best transformation...")
    
    # Add meter conversions to each transform (both pixels and meters)
    for t in transforms:
        if t.get('error') is None:
            # Add meter conversions for all error metrics (keep pixels too)
            if 'median_error' in t:
                t['median_error_pixels'] = float(t.get('median_error', 0))
                t['median_error_meters'] = float(t.get('median_error', 0) * pixel_resolution)
            if 'robust_rmse' in t:
                t['robust_rmse_pixels'] = float(t.get('robust_rmse', 0))
                t['robust_rmse_meters'] = float(t.get('robust_rmse', 0) * pixel_resolution)
            if 'mean_error' in t:
                t['mean_error_pixels'] = float(t.get('mean_error', 0))
                t['mean_error_meters'] = float(t.get('mean_error', 0) * pixel_resolution)
            if 'std_error' in t:
                t['std_error_pixels'] = float(t.get('std_error', 0))
                t['std_error_meters'] = float(t.get('std_error', 0) * pixel_resolution)
            if 'max_error' in t:
                t['max_error_pixels'] = float(t.get('max_error', 0))
                t['max_error_meters'] = float(t.get('max_error', 0) * pixel_resolution)
            # Convert reproj_errors to meters (keep pixels too)
            if 'reproj_errors' in t and t['reproj_errors']:
                t['reproj_errors_pixels'] = [float(e) for e in t['reproj_errors']]
                t['reproj_errors_meters'] = [float(e * pixel_resolution) for e in t['reproj_errors']]
    
    best_transform = choose_best_transform(transforms)
    
    if best_transform is None:
        print("ERROR: No valid transformation found!")
        return {'error': 'no_valid_transform'}
    
    # Save transformation results (convert all numpy arrays to lists)
    def convert_to_serializable(obj):
        """Recursively convert numpy arrays and other non-serializable objects to Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        else:
            return obj
    
    transform_results = {
        'current_scale': float(current_scale),
        'next_scale': float(next_scale),
        'pixel_resolution_meters': float(pixel_resolution),
        'transforms': [convert_to_serializable(t) for t in transforms],
        'best_transform': convert_to_serializable({
            'type': best_transform['type'],
            'matrix': best_transform.get('matrix', None),
            'robust_rmse_pixels': best_transform.get('robust_rmse_pixels', best_transform.get('robust_rmse', 0)),
            'robust_rmse_meters': best_transform.get('robust_rmse_meters', 0),
            'median_error_pixels': best_transform.get('median_error_pixels', best_transform.get('median_error', 0)),
            'median_error_meters': best_transform.get('median_error_meters', 0),
            'mean_error_pixels': best_transform.get('mean_error_pixels', best_transform.get('mean_error', 0)),
            'mean_error_meters': best_transform.get('mean_error_meters', 0),
            'std_error_pixels': best_transform.get('std_error_pixels', best_transform.get('std_error', 0)),
            'std_error_meters': best_transform.get('std_error_meters', 0),
            'max_error_pixels': best_transform.get('max_error_pixels', best_transform.get('max_error', 0)),
            'max_error_meters': best_transform.get('max_error_meters', 0)
        })
    }
    
    transform_json = test_dir / f'transform_{current_scale_str}_to_{next_scale_str}.json'
    with open(transform_json, 'w') as f:
        json.dump(transform_results, f, indent=2)
    print(f"   Saved transformation results to {transform_json.name}")
    
    # 5. Apply ALL transformations to source at next scale
    print(f"\n5. Applying all transformations to source at scale {next_scale_str}...")
    
    # For 0.500 scale, use the pre-transformed source (with polynomial_2 from 0.150->0.300 applied)
    if next_scale == 0.500:
        source_path_poly2 = test_dir / f'source_overlap_scale{next_scale_str}_poly2.png'
        if source_path_poly2.exists():
            print(f"   Using pre-transformed source: {source_path_poly2.name}")
            source_path = source_path_poly2
        else:
            # Fallback to regular source
            source_path = test_dir / f'source_overlap_scale{next_scale_str}.png'
            print(f"   Warning: Pre-transformed source not found, using: {source_path.name}")
    else:
        source_path = test_dir / f'source_overlap_scale{next_scale_str}.png'
    
    target_path = test_dir / f'target_overlap_scale{next_scale_str}.png'
    
    if not source_path.exists():
        print(f"ERROR: Source image not found: {source_path}")
        return {'error': 'source_not_found'}
    
    source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
    
    if source is None or target is None:
        print(f"ERROR: Could not load images")
        return {'error': 'image_load_failed'}
    
    # Load source at current_scale to understand coordinate system
    source_current_path = test_dir / f'source_overlap_scale{current_scale_str}.png'
    source_current = None
    if source_current_path.exists():
        source_current = cv2.imread(str(source_current_path), cv2.IMREAD_GRAYSCALE)
    
    # Calculate scale factors for transformation scaling
    scale_x = scale_y = 1.0
    if source_current is not None:
        source_current_h, source_current_w = source_current.shape
        source_next_h, source_next_w = source.shape
        scale_x = source_next_w / source_current_w
        scale_y = source_next_h / source_current_h
        print(f"   Coordinate system scaling: {scale_x:.3f} x {scale_y:.3f}")
    else:
        scale_ratio = next_scale / current_scale
        scale_x = scale_y = scale_ratio
        print(f"   Using scale ratio: {scale_ratio:.3f}")
    
    # Apply all valid transformations
    transformed_images = {}
    pixel_resolution_next = 0.02 / next_scale
    
    for t in transforms:
        if t.get('error') is not None:
            continue  # Skip transformations with errors
        
        transform_type = t['type']
        
        # Check if output files already exist
        png_path = test_dir / f'source_overlap_scale{next_scale_str}_{transform_type}.png'
        tif_path = test_dir / f'source_overlap_scale{next_scale_str}_{transform_type}.tif'
        
        if png_path.exists() and tif_path.exists():
            print(f"   Skipping {transform_type} (files already exist)")
            # Load existing image to add to transformed_images dict
            transformed = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
            if transformed is not None:
                transformed_images[transform_type] = transformed
            continue
        
        print(f"   Applying {transform_type}...")
        
        try:
            if t.get('matrix') is not None:
                # Matrix-based transformation (shift, similarity, affine, homography)
                M = np.array(t['matrix'])
                if M.shape == (2, 3):
                    M = np.vstack([M, [0, 0, 1]])
                elif M.shape != (3, 3):
                    print(f"     Warning: Unexpected matrix shape, skipping")
                    continue
                
                # Scale transformation
                M_scaled = M.copy()
                M_scaled[0, 2] *= scale_x
                M_scaled[1, 2] *= scale_y
                
                transformed = apply_transform_to_image(source, M_scaled, target_shape=None)
                
            elif 'coefficients_x' in t:
                # Polynomial transformation - need to scale coefficients appropriately
                # For polynomial, scaling is more complex - we'll apply as-is for now
                # (polynomial coefficients are in pixel space at current scale)
                transformed = apply_polynomial_transform_to_image(source, t, target_shape=None)
                
            elif 'control_points_src' in t:
                # Spline or rubber sheeting - scale control points
                t_scaled = t.copy()
                src_control = np.array(t['control_points_src'])
                dst_control = np.array(t['control_points_dst'])
                src_control_scaled = src_control * np.array([scale_x, scale_y])
                dst_control_scaled = dst_control * np.array([scale_x, scale_y])
                t_scaled['control_points_src'] = src_control_scaled.tolist()
                t_scaled['control_points_dst'] = dst_control_scaled.tolist()
                
                if transform_type == 'spline':
                    transformed = apply_spline_transform_to_image(source, t_scaled, target_shape=None)
                elif transform_type == 'rubber_sheeting':
                    transformed = apply_rubber_sheeting_transform_to_image(source, t_scaled, target_shape=None)
                else:
                    print(f"     Warning: Unknown type, skipping")
                    continue
            else:
                print(f"     Warning: Cannot apply {transform_type}, skipping")
                continue
            
            # Validate transformed image
            if transformed is None or transformed.size == 0:
                print(f"     Warning: Produced invalid image, skipping")
                continue
            
            valid_pixels = np.sum(transformed > 0)
            if valid_pixels < transformed.size * 0.1:
                print(f"     Warning: Only {100*valid_pixels/transformed.size:.1f}% valid pixels")
            
            # Save PNG
            png_path = test_dir / f'source_overlap_scale{next_scale_str}_{transform_type}.png'
            cv2.imwrite(str(png_path), transformed)
            print(f"     Saved PNG: {png_path.name}")
            
            # Save GeoTIFF
            tif_path = test_dir / f'source_overlap_scale{next_scale_str}_{transform_type}.tif'
            save_geotiff(transformed, tif_path, test_dir, scale=next_scale, pixel_size_meters=pixel_resolution_next)
            
            transformed_images[transform_type] = transformed
            
        except Exception as e:
            print(f"     Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 6. Prepare data for histograms (will create after matching)
    # Collect inlier errors for each transformation
    transform_inlier_data = {}
    transform_order = []  # Maintain order
    
    for t in transforms:
        if t.get('error') is not None or 'reproj_errors_meters' not in t:
            continue
        
        transform_type = t['type']
        transform_order.append(transform_type)
        
        # Get inliers
        inliers = t.get('inliers', None)
        reproj_errors_meters = np.array(t['reproj_errors_meters'])
        
        if inliers is None:
            # All points are inliers (e.g., for shift)
            inlier_errors = reproj_errors_meters.tolist()
        else:
            # For spline/rubber_sheeting: RANSAC inliers have near-zero error (fit perfectly)
            # So we use all points with meaningful reprojection errors as "inliers" for the histogram
            # For other transforms: use the actual inlier mask
            if transform_type in ['spline', 'rubber_sheeting']:
                # Use all points with non-zero errors (these are the "inliers" for evaluation)
                # The RANSAC inliers have zero error because the spline fits them perfectly
                inlier_errors = reproj_errors_meters[reproj_errors_meters > 1e-6].tolist()
            else:
                # For other transforms, use the actual inlier mask
                inlier_mask = np.array(inliers) > 0
                inlier_errors = reproj_errors_meters[inlier_mask].tolist()
        
        if len(inlier_errors) > 0:
            transform_inlier_data[transform_type] = inlier_errors
    
    histogram_path = None  # Will be set after matching
    
    # 7. Use best transformation for matching (maintain workflow)
    best_type = best_transform['type']
    
    # Load best transformation image if not already loaded
    if best_type not in transformed_images:
        best_png_path = test_dir / f'source_overlap_scale{next_scale_str}_{best_type}.png'
        if best_png_path.exists():
            transformed_source = cv2.imread(str(best_png_path), cv2.IMREAD_GRAYSCALE)
            if transformed_source is not None:
                transformed_images[best_type] = transformed_source
                print(f"   Loaded existing {best_type} transformation from {best_png_path.name}")
            else:
                print(f"ERROR: Best transformation {best_type} file exists but could not be loaded!")
                return {'error': 'best_transform_load_failed', 'transform': transform_results, 'histogram_path': histogram_path}
        else:
            print(f"ERROR: Best transformation {best_type} was not successfully applied and file doesn't exist!")
            return {'error': 'best_transform_not_applied', 'transform': transform_results, 'histogram_path': histogram_path}
    else:
        transformed_source = transformed_images[best_type]
    
    print(f"\n7. Using best transformation ({best_type}) for matching...")
    
    # 8. Match transformed source with target (skip if matches file exists)
    matches_json_next = test_dir / f'lightglue_matches_{next_scale_str.replace(".", "")}.json'
    
    if matches_json_next.exists():
        print(f"\n8. Matches file already exists, loading from {matches_json_next.name}...")
        with open(matches_json_next, 'r') as f:
            match_data = json.load(f)
        print(f"   Loaded {match_data.get('num_matches', 0)} matches from existing file")
    else:
        print(f"\n8. Matching transformed source with target at scale {next_scale_str}...")
        
        if not LIGHTGLUE_AVAILABLE:
            print("ERROR: LightGlue not available")
            return {'error': 'lightglue_not_available'}
        
        # Create masks
        source_mask = create_mask(transformed_source, threshold=10)
        target_mask = create_mask(target, threshold=10)
        
        # Calculate pixel resolution for matching
        pixel_resolution_meters = 0.02 / next_scale
        
        # Match - force tiled processing for transformed images to avoid MPS issues
        # For large images, always use tiled processing
        h, w = transformed_source.shape
        use_tiles_here = True  # Always use tiles for transformed images to avoid MPS issues
        
        # Try matching with error handling
        try:
            match_result = match_lightglue(
                transformed_source, target,
                source_mask, target_mask,
                use_tiles=use_tiles_here,
                tile_size=2048,
                overlap=256,
                expected_error_meters=3.0,
                pixel_resolution_meters=pixel_resolution_meters
            )
        except (RuntimeError, Exception) as e:
            error_msg = str(e)
            print(f"   Matching error: {error_msg}")
            if 'MPS' in error_msg or 'mps' in error_msg.lower() or 'Placeholder tensor' in error_msg:
                print(f"   MPS error detected. This is a known PyTorch MPS issue.")
                print(f"   The transformed image may need to be processed differently.")
                print(f"   Trying with smaller tile size...")
                try:
                    match_result = match_lightglue(
                        transformed_source, target,
                        source_mask, target_mask,
                        use_tiles=True,
                        tile_size=1024,  # Smaller tiles
                        overlap=128,
                        expected_error_meters=3.0,
                        pixel_resolution_meters=pixel_resolution_meters
                    )
                except Exception as e2:
                    print(f"   Still failed: {e2}")
                    return {'error': 'matching_failed', 'error_msg': str(e2), 'transform': transform_results}
            else:
                return {'error': 'matching_failed', 'error_msg': error_msg, 'transform': transform_results}
        
        if not match_result.get('matches'):
            print("ERROR: No matches found")
            return {'error': 'no_matches', 'transform': transform_results}
        
        print(f"   Found {len(match_result['matches'])} matches")
        
        # Convert to JSON format (similar to test_matching.py)
        match_data = {
            'method': 'LightGlue',
            'num_matches': len(match_result['matches']),
            'num_source_keypoints': len(match_result['kp1']),
            'num_target_keypoints': len(match_result['kp2']),
            'scale_factor': match_result.get('scale_factor', 1.0),
            'source_shape': match_result.get('source_shape', transformed_source.shape),
            'target_shape': match_result.get('target_shape', target.shape),
            'scale': next_scale,
            'matches': []
        }
        
        # Get match scores if available
        match_scores_list = match_result.get('match_scores', None)
        pixel_resolution_meters = 0.02 / next_scale
        
        for i, match in enumerate(match_result['matches']):
            src_kp = match_result['kp1'][match.queryIdx]
            tgt_kp = match_result['kp2'][match.trainIdx]
            scale_factor = match_result.get('scale_factor', 1.0)
            
            tgt_x_upsampled = float(tgt_kp.pt[0])
            tgt_y_upsampled = float(tgt_kp.pt[1])
            tgt_x_original = tgt_x_upsampled / scale_factor if scale_factor != 1.0 else tgt_x_upsampled
            tgt_y_original = tgt_y_upsampled / scale_factor if scale_factor != 1.0 else tgt_y_upsampled
            
            src_response = float(src_kp.response) if hasattr(src_kp, 'response') and src_kp.response > 0 else 0.0
            tgt_response = float(tgt_kp.response) if hasattr(tgt_kp, 'response') and tgt_kp.response > 0 else 0.0
            match_distance = float(match.distance) if hasattr(match, 'distance') else 0.0
            
            if match_scores_list is not None and i < len(match_scores_list):
                confidence = float(match_scores_list[i])
                match_distance = 1.0 - confidence
            
            src_x = float(src_kp.pt[0])
            src_y = float(src_kp.pt[1])
            pixel_distance = np.sqrt((src_x - tgt_x_upsampled)**2 + (src_y - tgt_y_upsampled)**2)
            distance_meters = pixel_distance * pixel_resolution_meters
            
            match_data['matches'].append({
                'source': {'x': src_x, 'y': src_y, 'response': src_response},
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
        
        with open(matches_json_next, 'w') as f:
            json.dump(match_data, f, indent=2)
        print(f"   Saved matches to {matches_json_next.name}")
    
    # 10. Create 3x3 grid histogram: all matches + 8 transformation inlier histograms
    print(f"\n10. Creating 3x3 grid histogram (all matches + 8 transformation methods)...")
    
    # Get all match distances from the original matches at current scale (no outlier filtering)
    # Try multiple possible filenames
    possible_matches_files = [
        test_dir / f'lightglue_matches_polynomial2_{current_scale_str}.json',
        test_dir / f'lightglue_matches_{current_scale_str}.json',
        test_dir / f'lightglue_matches_{current_scale_str.replace(".", "")}.json'
    ]
    
    matches_json_original = None
    for matches_file in possible_matches_files:
        if matches_file.exists():
            matches_json_original = matches_file
            break
    
    if matches_json_original is not None:
        print(f"   Loading original matches from {matches_json_original.name} for 'all matches' histogram...")
        with open(matches_json_original, 'r') as f:
            original_match_data = json.load(f)
        all_match_distances = [m['distance']['meters'] for m in original_match_data['matches']]
        print(f"   Using {len(all_match_distances)} matches from original scale {current_scale_str} (median={np.median(all_match_distances):.2f}m, mean={np.mean(all_match_distances):.2f}m)")
    else:
        print(f"   Warning: Original matches file not found, using current matches instead")
        all_match_distances = [m['distance']['meters'] for m in match_data['matches']]
    
    # Create 3x3 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    # Plot 0: All match distances (upper left)
    ax = axes[0]
    n, bins, patches = ax.hist(all_match_distances, bins=50, edgecolor='black', alpha=0.7)
    
    # Color bars by distance
    for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
        if bin_val < 5:
            patch.set_facecolor('green')
        elif bin_val < 10:
            patch.set_facecolor('yellow')
        elif bin_val < 20:
            patch.set_facecolor('orange')
        else:
            patch.set_facecolor('red')
    
    median = np.median(all_match_distances)
    mean = np.mean(all_match_distances)
    ax.axvline(median, color='blue', linestyle='--', linewidth=2, label=f'Median: {median:.2f}m')
    ax.axvline(mean, color='purple', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}m')
    
    ax.set_xlabel('Distance (meters)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Matches', fontsize=10, fontweight='bold')
    ax.set_title('All Matches (Including Outliers)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    stats_text = f'Count: {len(all_match_distances)}\n'
    stats_text += f'Median: {median:.2f} m\n'
    stats_text += f'Mean: {mean:.2f} m\n'
    stats_text += f'< 5m: {sum(1 for d in all_match_distances if d < 5)} ({100*sum(1 for d in all_match_distances if d < 5)/len(all_match_distances):.1f}%)'
    
    ax.text(0.98, 0.75, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=8, family='monospace')
    
    # Plots 1-8: Individual transformation inlier histograms
    plot_idx = 1
    for transform_type in transform_order:
        if transform_type not in transform_inlier_data:
            continue
        
        if plot_idx >= 9:  # Only 8 more plots (1-8)
            break
        
        ax = axes[plot_idx]
        inlier_errors = transform_inlier_data[transform_type]
        
        # Determine appropriate bins
        max_error = max(inlier_errors) if len(inlier_errors) > 0 else 10
        bins = np.linspace(0, max(max_error, 10), 30)
        
        n, bins_hist, patches = ax.hist(inlier_errors, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        
        # Color bars
        for i, (patch, bin_val) in enumerate(zip(patches, bins_hist[:-1])):
            if bin_val < 2:
                patch.set_facecolor('green')
            elif bin_val < 5:
                patch.set_facecolor('yellow')
            elif bin_val < 10:
                patch.set_facecolor('orange')
            else:
                patch.set_facecolor('red')
        
        median_err = np.median(inlier_errors)
        mean_err = np.mean(inlier_errors)
        ax.axvline(median_err, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_err:.2f}m')
        ax.axvline(mean_err, color='purple', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.2f}m')
        
        ax.set_xlabel('Reprojection Error (meters)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Number of Inliers', fontsize=10, fontweight='bold')
        ax.set_title(f'{transform_type} (Inliers Only)', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        stats_text = f'Inliers: {len(inlier_errors)}\n'
        stats_text += f'Median: {median_err:.3f} m\n'
        stats_text += f'Mean: {mean_err:.3f} m\n'
        stats_text += f'Max: {max(inlier_errors):.3f} m'
        
        ax.text(0.98, 0.75, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=8, family='monospace')
        
        plot_idx += 1
    
    # Hide unused subplots if we have fewer than 8 transformations
    for i in range(plot_idx, 9):
        axes[i].axis('off')
    
    plt.suptitle(f'Distance Histograms - Scale {next_scale_str}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    histogram_path = test_dir / f'distance_histogram_{next_scale_str.replace(".", "")}.png'
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved 3x3 grid histogram to {histogram_path.name}")
    
    return {
        'transform': transform_results,
        'matches_json': matches_json_next,
        'histogram_path': histogram_path,
        'transformed_images': list(transformed_images.keys())
    }


def main():
    """Main hierarchical transformation refinement."""
    test_dir = Path(__file__).parent
    
    # Define scale pyramid - Process 0.300 -> 0.500
    scales = [0.300, 0.500]
    
    print("=" * 80)
    print("HIERARCHICAL TRANSFORMATION REFINEMENT")
    print("=" * 80)
    print(f"Scales: {scales} (processing 0.300 -> 0.500)")
    print(f"Working directory: {test_dir}")
    
    results = []
    
    # Process 0.300 -> 0.500
    i = 0
    current_scale = scales[i]
    next_scale = scales[i + 1]
    
    # Get matches JSON path - use polynomial2 matches from 0.300
    matches_json = test_dir / 'lightglue_matches_polynomial2_0.300.json'
    if not matches_json.exists():
        # Fallback to regular matches
        matches_json = test_dir / f'lightglue_matches_{current_scale:.3f}.json'
    
    if not matches_json.exists():
        print(f"\nERROR: Matches file not found: {matches_json}")
        return
    
    # Process this level
    result = process_scale_level(test_dir, current_scale, next_scale, matches_json)
    
    if 'error' in result:
        print(f"\nERROR at scale {current_scale}: {result['error']}")
        return
    
    results.append(result)
    
    print(f"\n✓ Completed scale {current_scale} -> {next_scale}")
    if 'transform' in result and 'best_transform' in result['transform']:
        best = result['transform']['best_transform']
        print(f"  Best transform: {best['type']}")
        print(f"  Median error: {best.get('median_error_meters', 0):.3f} m")
        print(f"  Robust RMSE: {best.get('robust_rmse_meters', 0):.3f} m")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for i, result in enumerate(results):
        scale = scales[i + 1]
        print(f"Scale {scale:.3f}:")
        if 'transform' in result and 'best_transform' in result['transform']:
            best = result['transform']['best_transform']
            print(f"  Best transform: {best['type']}")
            print(f"  Median error: {best.get('median_error_meters', 0):.3f} m ({best.get('median_error_pixels', 0):.3f} px)")
            print(f"  Robust RMSE: {best.get('robust_rmse_meters', 0):.3f} m ({best.get('robust_rmse_pixels', 0):.3f} px)")
        if 'transformed_images' in result:
            print(f"  Applied transformations: {', '.join(result['transformed_images'])}")
    
    print("\n" + "=" * 80)
    print("HIERARCHICAL REFINEMENT COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()


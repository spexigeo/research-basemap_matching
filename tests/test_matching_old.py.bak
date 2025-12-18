#!/usr/bin/env python3
"""
Quick test to see if patch matching is working at all.
This will load the images at a coarse scale and try to find matches.
"""

import cv2
import numpy as np
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt


def test_patch_matching(source_path, target_path, scale=0.15):
    """Test if patch matching can find any matches."""

    print("=" * 80)
    print(f"PATCH MATCHING TEST - Scale: {scale}")
    print("=" * 80)

    # Load and downsample
    def load_image(path, scale):
        with rasterio.open(path) as src:
            out_h = int(src.height * scale)
            out_w = int(src.width * scale)

            data = src.read(
                out_shape=(src.count, out_h, out_w),
                resampling=rasterio.enums.Resampling.bilinear
            )

            if data.shape[0] >= 3:
                rgb = np.moveaxis(data[:3], 0, -1)

                # Normalize based on actual data range
                rgb_min = rgb.min()
                rgb_max = rgb.max()
                print(f"  Data range: {rgb_min} - {rgb_max}")

                if rgb_max > rgb_min:
                    rgb = ((rgb - rgb_min) / (rgb_max - rgb_min) * 255).astype(np.uint8)
                else:
                    rgb = np.zeros_like(rgb, dtype=np.uint8)

                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            else:
                single = data[0]

                # Normalize based on actual data range
                single_min = single.min()
                single_max = single.max()
                print(f"  Data range: {single_min} - {single_max}")

                if single_max > single_min:
                    gray = ((single - single_min) / (single_max - single_min) * 255).astype(np.uint8)
                else:
                    gray = np.zeros((out_h, out_w), dtype=np.uint8)

            print(f"  Normalized to: {gray.min()} - {gray.max()}")
            return gray

    print("\nLoading images...")
    source = load_image(source_path, scale)
    target = load_image(target_path, scale)

    print(f"Source: {source.shape}")
    print(f"Target: {target.shape}")

    # Save original images
    cv2.imwrite('test_source_original.png', source)
    cv2.imwrite('test_target_original.png', target)
    print("\nSaved: test_source_original.png, test_target_original.png")

    # Test different preprocessing methods
    preprocess_methods = {
        'none': (source, target),
        'gradient': (compute_gradient(source), compute_gradient(target)),
        'clahe': (apply_clahe(source), apply_clahe(target)),
        'edges': (cv2.Canny(source, 50, 150), cv2.Canny(target, 50, 150))
    }

    results = {}

    for method_name, (src_prep, tgt_prep) in preprocess_methods.items():
        print(f"\n{'=' * 60}")
        print(f"Testing with preprocessing: {method_name}")
        print(f"{'=' * 60}")

        # Save preprocessed
        cv2.imwrite(f'test_source_{method_name}.png', src_prep)
        cv2.imwrite(f'test_target_{method_name}.png', tgt_prep)

        # Try patch matching
        patch_size = 128
        grid_spacing = 64
        threshold = 0.4

        h, w = tgt_prep.shape
        matches = []

        test_count = 0
        for y in range(patch_size // 2, h - patch_size // 2, grid_spacing):
            for x in range(patch_size // 2, w - patch_size // 2, grid_spacing):
                if test_count >= 100:  # Test first 100 patches only
                    break

                test_count += 1
                ty1, ty2 = y - patch_size // 2, y + patch_size // 2
                tx1, tx2 = x - patch_size // 2, x + patch_size // 2
                target_patch = tgt_prep[ty1:ty2, tx1:tx2]

                if ty2 <= src_prep.shape[0] and tx2 <= src_prep.shape[1]:
                    search_margin = 100
                    sy1 = max(0, ty1 - search_margin)
                    sy2 = min(src_prep.shape[0], ty2 + search_margin)
                    sx1 = max(0, tx1 - search_margin)
                    sx2 = min(src_prep.shape[1], tx2 + search_margin)
                    search_region = src_prep[sy1:sy2, sx1:sx2]

                    if search_region.shape[0] > patch_size and search_region.shape[1] > patch_size:
                        try:
                            result = cv2.matchTemplate(search_region, target_patch, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, max_loc = cv2.minMaxLoc(result)

                            if max_val > threshold:
                                src_x = sx1 + max_loc[0] + patch_size // 2
                                src_y = sy1 + max_loc[1] + patch_size // 2
                                matches.append({
                                    'confidence': max_val,
                                    'target': (x, y),
                                    'source': (src_x, src_y)
                                })
                        except:
                            pass

            if test_count >= 100:
                break

        print(f"  Tested {test_count} patches")
        print(f"  Found {len(matches)} matches (threshold={threshold})")

        if matches:
            confidences = [m['confidence'] for m in matches]
            print(f"  Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"  Mean confidence: {np.mean(confidences):.3f}")

        results[method_name] = {
            'matches': len(matches),
            'tested': test_count,
            'match_details': matches[:5]  # First 5 for inspection
        }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    best_method = max(results.items(), key=lambda x: x[1]['matches'])

    for method, result in results.items():
        marker = "✓" if result['matches'] > 0 else "✗"
        print(f"{marker} {method:12s}: {result['matches']:3d} matches from {result['tested']:3d} patches")

    print(f"\nBest preprocessing: {best_method[0]} ({best_method[1]['matches']} matches)")

    if best_method[1]['matches'] == 0:
        print("\n❌ NO MATCHES FOUND WITH ANY METHOD!")
        print("\nPossible causes:")
        print("  1. Images are too different (different seasons, angles, etc.)")
        print("  2. Scale is wrong (try 0.05 or 0.3 instead of 0.15)")
        print("  3. Threshold too high (try 0.3 instead of 0.4)")
        print("  4. Images don't actually overlap despite what diagnostic says")
        print("\nTroubleshooting:")
        print("  - Check test_source_original.png and test_target_original.png")
        print("  - Do they show the same geographic area?")
        print("  - Are they recognizably similar?")
    else:
        print(f"\n✓ Matching WORKS with {best_method[0]} preprocessing!")
        print(f"  Update your config to use: 'preprocess_method': '{best_method[0]}'")
        print(f"  And consider lowering ncc_threshold to 0.4 or 0.35")


def compute_gradient(img):
    """Compute gradient magnitude."""
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def apply_clahe(img):
    """Apply CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print("Usage: python test_matching.py <source> <target>")
        print("\nExample:")
        print(
            "  python test_matching.py inputs/orthomosaic_no_gcps_utm10n.tif inputs/qualicum_beach_basemap_esri_utm10n.tif")
        sys.exit(1)

    source_path = sys.argv[1]
    target_path = sys.argv[2]

    test_patch_matching(source_path, target_path, scale=0.15)
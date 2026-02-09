# -*- coding: utf-8 -*-
"""
Text Overflow Detector (Gradient-based)
=======================================
Detect whether text content overflows from the designated textbox boundary.

Algorithm Pipeline:
1. Extract yellow highlight mask using HSV color space
2. Divide image into grid blocks and calculate gradient complexity
3. Find activated blocks (high gradient + yellow coverage)
4. Perform flood-fill to find connected regions
5. Check if any region extends outside the safe zone

Interface:
    from detector import detect_text_overflow
    
    result = detect_text_overflow(
        image_path="poster_slide.png",
        layer_coords=(4.0, 2.0, 5.0, 3.0),  # (x, y, w, h) in inches
        slide_size=(33.1, 46.8)              # slide size in inches
    )
    print(result)  # 0 (safe) or 1 (overflow)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


def detect_text_overflow(
    image_path: str,
    layer_coords: Tuple[float, float, float, float],
    slide_size: Tuple[float, float],
    grid_size: int = 256,
    complexity_threshold_multiplier: float = 1.5,
    overflow_grid_threshold: int = 1,
    yellow_coverage_threshold: float = 0.01,
    boundary_padding_inches: float = 0
) -> int:
    """
    Detect if text content overflows from the textbox boundary.
    
    The input image should have only ONE layer with content, 
    filled with yellow background for detection.
    
    Args:
        image_path: Path to the poster image (single layer highlighted with yellow)
        layer_coords: Textbox coordinates (x, y, width, height) in inches
        slide_size: Slide/poster dimensions (width, height) in inches
        grid_size: Grid division count (default 256)
        complexity_threshold_multiplier: Threshold multiplier for gradient complexity
        overflow_grid_threshold: Number of overflow grids to trigger overflow
        yellow_coverage_threshold: Yellow pixel coverage threshold (0-1)
        boundary_padding_inches: Boundary tolerance in inches
    
    Returns:
        int: 1 = overflow, 0 = safe
    """
    # Apply boundary padding
    layer_x_in, layer_y_in, layer_w_in, layer_h_in = layer_coords
    layer_x_in += boundary_padding_inches
    layer_y_in += boundary_padding_inches
    layer_w_in -= 2 * boundary_padding_inches
    layer_h_in -= 2 * boundary_padding_inches
    layer_w_in = max(0.1, layer_w_in)
    layer_h_in = max(0.1, layer_h_in)
    layer_coords = (layer_x_in, layer_y_in, layer_w_in, layer_h_in)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    img_h, img_w = img.shape[:2]
    slide_w_in, slide_h_in = slide_size
    
    # Convert inches to pixels
    scale_x = img_w / slide_w_in
    scale_y = img_h / slide_h_in
    
    safe_x = int(layer_x_in * scale_x)
    safe_y = int(layer_y_in * scale_y)
    safe_w = int(layer_w_in * scale_x)
    safe_h = int(layer_h_in * scale_y)
    
    # Compute gradient mask using grid-based analysis
    gradient_mask, _, _ = _compute_gradient_mask_grid(
        img, 
        grid_size=grid_size,
        complexity_threshold_multiplier=complexity_threshold_multiplier,
        yellow_coverage_threshold=yellow_coverage_threshold
    )
    
    # Calculate safe zone grid boundaries
    block_height = img_h / grid_size
    block_width = img_w / grid_size
    
    padding_grids = 1
    safe_grid_x_start = max(0, int(safe_x / block_width) - padding_grids)
    safe_grid_x_end = min(grid_size, int((safe_x + safe_w) / block_width) + padding_grids + 1)
    safe_grid_y_start = max(0, int(safe_y / block_height) - padding_grids)
    safe_grid_y_end = min(grid_size, int((safe_y + safe_h) / block_height) + padding_grids + 1)
    
    # Remove safe zone from mask (keep only overflow regions)
    for i in range(safe_grid_y_start, safe_grid_y_end):
        for j in range(safe_grid_x_start, safe_grid_x_end):
            y_start = int(i * block_height)
            y_end = int((i + 1) * block_height) if i < grid_size - 1 else img_h
            x_start = int(j * block_width)
            x_end = int((j + 1) * block_width) if j < grid_size - 1 else img_w
            gradient_mask[y_start:y_end, x_start:x_end] = 0
    
    # Count overflow grids
    overflow_grids = 0
    for i in range(grid_size):
        for j in range(grid_size):
            y_start = int(i * block_height)
            y_end = int((i + 1) * block_height) if i < grid_size - 1 else img_h
            x_start = int(j * block_width)
            x_end = int((j + 1) * block_width) if j < grid_size - 1 else img_w
            block_mask = gradient_mask[y_start:y_end, x_start:x_end]
            if cv2.countNonZero(block_mask) > 0:
                overflow_grids += 1
    
    return 1 if overflow_grids > overflow_grid_threshold else 0


def _extract_yellow_mask(img: np.ndarray) -> np.ndarray:
    """Extract yellow highlight region as binary mask."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Yellow HSV range
    lower_yellow = np.array([15, 80, 150])
    upper_yellow = np.array([45, 255, 255])
    
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Morphological processing
    kernel = np.ones((3, 3), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    
    return yellow_mask


def _calculate_block_complexity(block: np.ndarray) -> float:
    """Calculate gradient complexity of an image block."""
    gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY) if len(block.shape) == 3 else block
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return np.mean(gradient_magnitude)


def _calculate_block_yellow_coverage(block_mask: np.ndarray) -> float:
    """Calculate yellow pixel coverage ratio in a block."""
    if block_mask.size == 0:
        return 0.0
    return cv2.countNonZero(block_mask) / block_mask.size


def _compute_gradient_mask_grid(
    img: np.ndarray, 
    grid_size: int = 256,
    complexity_threshold_multiplier: float = 1.5,
    yellow_coverage_threshold: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, List[List[Tuple[int, int]]]]:
    """
    Compute gradient mask using grid-based analysis.
    
    Returns:
        Tuple[mask, complexity_matrix, regions]
    """
    height, width = img.shape[:2]
    block_height = height / grid_size
    block_width = width / grid_size
    
    # Extract yellow mask
    yellow_mask = _extract_yellow_mask(img)
    
    # Calculate complexity for each grid
    complexities = []
    for i in range(grid_size):
        for j in range(grid_size):
            y_start = int(i * block_height)
            y_end = int((i + 1) * block_height) if i < grid_size - 1 else height
            x_start = int(j * block_width)
            x_end = int((j + 1) * block_width) if j < grid_size - 1 else width
            
            block_mask = yellow_mask[y_start:y_end, x_start:x_end]
            yellow_coverage = _calculate_block_yellow_coverage(block_mask)
            
            if yellow_coverage > yellow_coverage_threshold:
                block = img[y_start:y_end, x_start:x_end]
                complexity = _calculate_block_complexity(block)
                complexities.append((complexity, (i, j), yellow_coverage))
            else:
                complexities.append((0, (i, j), 0))
    
    # Dynamic threshold
    complexity_values = [c for c, _, cov in complexities if cov > yellow_coverage_threshold]
    threshold = np.median(complexity_values) * complexity_threshold_multiplier if complexity_values else 0
    
    # Build complexity matrix
    complexity_matrix = np.zeros((grid_size, grid_size), dtype=bool)
    for complexity, (i, j), yellow_coverage in complexities:
        if yellow_coverage > yellow_coverage_threshold and complexity > threshold:
            complexity_matrix[i, j] = True
    
    # Find connected regions
    regions = _find_connected_regions(complexity_matrix)
    
    # Convert to pixel-level mask
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        if not region:
            continue
        for i, j in region:
            y_start = int(i * block_height)
            y_end = int((i + 1) * block_height) if i < grid_size - 1 else height
            x_start = int(j * block_width)
            x_end = int((j + 1) * block_width) if j < grid_size - 1 else width
            mask[y_start:y_end, x_start:x_end] = np.maximum(
                mask[y_start:y_end, x_start:x_end],
                yellow_mask[y_start:y_end, x_start:x_end]
            )
    
    return mask, complexity_matrix, regions


def _find_connected_regions(matrix: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Find all connected regions using BFS."""
    grid_size = matrix.shape[0]
    visited = np.zeros_like(matrix, dtype=bool)
    regions = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            if matrix[i, j] and not visited[i, j]:
                region = []
                queue = [(i, j)]
                visited[i, j] = True
                
                while queue:
                    current_i, current_j = queue.pop(0)
                    region.append((current_i, current_j))
                    
                    for di, dj in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                        ni, nj = current_i + di, current_j + dj
                        if (0 <= ni < grid_size and 0 <= nj < grid_size and 
                            matrix[ni, nj] and not visited[ni, nj]):
                            visited[ni, nj] = True
                            queue.append((ni, nj))
                
                if region:
                    regions.append(region)
    
    return regions


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    import os
    
    # Test case
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_image = os.path.join(test_dir, "test_images", "sample.png")
    
    if os.path.exists(test_image):
        result = detect_text_overflow(
            image_path=test_image,
            layer_coords=(4.0, 2.0, 5.0, 3.0),
            slide_size=(13.333, 7.5)
        )
        print(f"Detection result: {result} ({'overflow' if result else 'safe'})")
    else:
        print(f"Test image not found: {test_image}")
        print("Please provide a test image to run the detection.")


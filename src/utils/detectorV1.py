import time
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Any


# ======================
# 基础工具
# ======================
def stripe_bounds_by_count(length: int, n: int):
    n = max(1, min(int(n), length))
    bounds = np.linspace(0, length, n + 1)
    bounds = np.unique(bounds.astype(int))
    if len(bounds) < 2:
        return np.array([0, length], dtype=int)
    return bounds


def grad_score_along_x(rgb: np.ndarray):
    if rgb.shape[1] < 2:
        return 0.0
    diff = np.abs(rgb[:, 1:, :] - rgb[:, :-1, :])
    return float(diff.mean())


def grad_score_along_y(rgb: np.ndarray):
    if rgb.shape[0] < 2:
        return 0.0
    diff = np.abs(rgb[1:, :, :] - rgb[:-1, :, :])
    return float(diff.mean())


def robust_baseline(scores):
    return np.median(scores)


def compute_activated_rectangles(
    y_scores, x_scores,
    y_bounds, x_bounds,
    y_median, x_median
) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
    """计算 Y轴激活bins × X轴激活bins 的笛卡尔乘积矩形"""
    y_activated_indices = np.where(y_scores > y_median)[0]
    x_activated_indices = np.where(x_scores > x_median)[0]

    rectangles = []
    areas = []

    for yi in y_activated_indices:
        y0 = int(y_bounds[yi])
        y1 = int(y_bounds[yi + 1])

        for xi in x_activated_indices:
            x0 = int(x_bounds[xi])
            x1 = int(x_bounds[xi + 1])

            rect = (x0, y0, x1, y1)
            area = (x1 - x0) * (y1 - y0)

            rectangles.append(rect)
            areas.append(area)

    return rectangles, areas


def compute_union_coverage(rectangles, img_width, img_height) -> int:
    """计算矩形并集的覆盖面积"""
    if not rectangles:
        return 0

    mask = np.zeros((img_height, img_width), dtype=bool)

    for (x0, y0, x1, y1) in rectangles:
        mask[y0:y1, x0:x1] = True

    return int(np.sum(mask))


def compute_mbr(rectangles) -> Tuple[Tuple[int, int, int, int], int]:
    """计算最小外接矩形 (MBR)"""
    if not rectangles:
        return None, 0

    x0_min = min(r[0] for r in rectangles)
    y0_min = min(r[1] for r in rectangles)
    x1_max = max(r[2] for r in rectangles)
    y1_max = max(r[3] for r in rectangles)

    mbr = (x0_min, y0_min, x1_max, y1_max)
    mbr_area = (x1_max - x0_min) * (y1_max - y0_min)

    return mbr, mbr_area


# ======================
# 主接口
# ======================
def detect_panel_status(
    image_path: str,
    panel_bbox: Tuple[int, int, int, int],  # (x, y, w, h)
    bins: int = 32,
    sparse_threshold: float = 0.1
) -> Dict[str, Any]:
    """
    检测panel状态

    参数:
        image_path: 图片路径
        panel_bbox: panel边界框 (x, y, w, h)
        bins: 分割的bin数量
        sparse_threshold: sparse判断阈值 (激活面积/bbox面积 < 阈值 则为sparse)

    返回:
        {
            "result": "overflow" | "sparse" | "valid",
            "metadata": {
                "runtime_ms": float,
                "activated_regions": [(x, y, w, h), ...],
                "activated_total_area": int,
                "mbr_xywh": (x, y, w, h) or None,
                "mbr_area": int,
                "coverage_ratio": float
            }
        }
    """
    start_time = time.time()

    # 解析 panel_bbox
    panel_x, panel_y, panel_w, panel_h = panel_bbox
    panel_area = panel_w * panel_h

    # 加载图片
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    rgb = np.array(img, dtype=float)

    # Y轴分析
    y_bounds = stripe_bounds_by_count(h, bins)
    y_scores = []
    for i in range(len(y_bounds) - 1):
        y0, y1 = int(y_bounds[i]), int(y_bounds[i + 1])
        if y1 > y0:
            y_scores.append(grad_score_along_x(rgb[y0:y1]))
    y_scores = np.asarray(y_scores)
    y_median = robust_baseline(y_scores)

    # X轴分析
    x_bounds = stripe_bounds_by_count(w, bins)
    x_scores = []
    for i in range(len(x_bounds) - 1):
        x0, x1 = int(x_bounds[i]), int(x_bounds[i + 1])
        if x1 > x0:
            x_scores.append(grad_score_along_y(rgb[:, x0:x1]))
    x_scores = np.asarray(x_scores)
    x_median = robust_baseline(x_scores)

    # 计算激活矩形
    rectangles, areas = compute_activated_rectangles(
        y_scores, x_scores,
        y_bounds, x_bounds,
        y_median, x_median
    )

    # 计算并集面积
    union_area = compute_union_coverage(rectangles, w, h)

    # 计算 MBR
    mbr, mbr_area = compute_mbr(rectangles)

    # 转换为 xywh 格式
    activated_regions_xywh = [
        (x0, y0, x1 - x0, y1 - y0) for (x0, y0, x1, y1) in rectangles
    ]

    mbr_xywh = None
    if mbr:
        mbr_xywh = (mbr[0], mbr[1], mbr[2] - mbr[0], mbr[3] - mbr[1])

    # ======================
    # 判断结果
    # ======================
    result = "valid"

    # 1. 判断 overflow: MBR 是否完全在 panel_bbox 内部
    if mbr:
        mbr_x0, mbr_y0, mbr_x1, mbr_y1 = mbr
        panel_x1 = panel_x + panel_w
        panel_y1 = panel_y + panel_h

        # MBR 超出 bbox 则 overflow
        if not (mbr_x0 >= panel_x and mbr_y0 >= panel_y and
                mbr_x1 <= panel_x1 and mbr_y1 <= panel_y1):
            result = "overflow"

    # 2. 判断 sparse: 激活区域面积 / panel面积 < 阈值
    if result == "valid":
        coverage_ratio = union_area / panel_area if panel_area > 0 else 0
        if coverage_ratio < sparse_threshold:
            result = "sparse"
    else:
        coverage_ratio = union_area / panel_area if panel_area > 0 else 0

    runtime_ms = (time.time() - start_time) * 1000

    return {
        "result": result,
        "metadata": {
            "runtime_ms": runtime_ms,
            "activated_regions": activated_regions_xywh,
            "activated_total_area": union_area,
            "mbr_xywh": mbr_xywh,
            "mbr_area": mbr_area,
            "coverage_ratio": coverage_ratio
        }
    }


# ======================
# 调用示例
# ======================
if __name__ == "__main__":

    result = detect_panel_status(
        image_path="./camera/ceg-1.png",
        panel_bbox=(100, 100, 3000, 2500),  # (x, y, w, h)
        bins=32,
        sparse_threshold=0.1
    )

    print("=" * 50)
    print(f"  Result:              {result['result']}")
    print("=" * 50)
    print(f"  Runtime:             {result['metadata']['runtime_ms']:.2f} ms")
    print(f"  MBR (xywh):          {result['metadata']['mbr_xywh']}")
    print(f"  MBR Area:            {result['metadata']['mbr_area']}")
    print(f"  Activated Regions:   {len(result['metadata']['activated_regions'])} regions")
    print(f"  Activated Area:      {result['metadata']['activated_total_area']}")
    print(f"  Coverage Ratio:      {result['metadata']['coverage_ratio']:.2%}")
    print("=" * 50)
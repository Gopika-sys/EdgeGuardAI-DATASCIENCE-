"""
EdgeGuard AI — AI Engine 1: Computer Vision Intelligence
Module 05 of ml work.txt specification.

PURPOSE
-------
Convert mining-truck dump videos into operational intelligence:
  - Carryback Percentage      (% of payload stuck in bed after dump)
  - Payload Efficiency        (% of capacity loaded)
  - Material Occupancy        (% of bed volume filled)
  - Loading Quality           (0-1 score)
  - Hydraulic Stress Indicator (0-1 score from ram extension profile)
  - Material Distribution     (uniform / left-heavy / right-heavy)

TECH STACK
----------
  Python 3.10+  OpenCV 4  NumPy  Ultralytics YOLOv8
  Training images: synthetic (since we have no real mining videos for the
  hackathon). Production deployments would replace the synthetic dataset
  with frames annotated in CVAT/LabelImg from real CCTV footage.

DESIGN DECISION: SYNTHETIC FRAMES
---------------------------------
Real mining CCTV is hard to source for a hackathon. We synthesise
realistic frames programmatically:
  - Render a side view of a tipper truck at varying dump angles (0-55 deg)
  - Fill the bed with ore (texture) at a configurable fill ratio
  - Sprinkle carryback blobs below the truck bed
  - Render the hydraulic ram (piston) at varying extension lengths
The labels (bounding boxes) are derived from the geometric ground truth,
so the model trains on perfectly accurate labels without manual work.

INTEGRATION WITH OTHER ENGINES
------------------------------
After inference, the CV engine emits a JSON dict:
  {
    "carryback_pct":          float,  # 0-20
    "payload_efficiency":     float,  # 0-1
    "material_occupancy":     float,  # 0-1
    "loading_quality":        float,  # 0-1
    "hydraulic_stress_indicator": float,  # 0-1
    "material_distribution":  str,    # "uniform" | "left_heavy" | "right_heavy"
    "confidence_score":       float,  # 0-1, mean of detection confidences
  }
This JSON is consumed by the Multi-Modal Decision Engine
(engine3_multimodal/decision_engine.py).

USAGE
-----
    python engine1_vision/synth_frames.py        # generate training frames
    python engine1_vision/train.py               # train YOLOv8
    python engine1_vision/infer.py <image.jpg>   # run inference
"""

import math
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Geometry constants (in pixels, at the synthetic camera scale)
# ---------------------------------------------------------------------------
IMG_W, IMG_H = 640, 480
TRUCK_X = 220
TRUCK_Y_TOP = 220   # body top y
BED_W = 280
BED_H = 100
WHEEL_R = 28
RANDOM = random.Random(7)
NP_RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# Synthetic frame renderer
# ---------------------------------------------------------------------------
@dataclass
class FrameLabels:
    """Ground-truth labels for one synthetic frame."""
    dump_angle_deg: float
    bed_fill_pct:   float            # 0-1
    carryback_blobs: list            # list of (x,y,radius) carryback blobs
    ram_extension_pct: float         # 0-1
    material_left_pct: float         # 0-1, fraction on left half
    bboxes: list = field(default_factory=list)   # list of [x1,y1,x2,y2,class_id]


CLASS_MAP = {0: "truck_bed", 1: "payload", 2: "carryback", 3: "hydraulic_ram"}


def _draw_truck_body(img, angle_deg: float):
    """Draw the truck cab + body. Rotates the bed by angle_deg around its hinge."""
    # Cab (left side, fixed)
    cab = np.array([[TRUCK_X - 80, TRUCK_Y_TOP - 30],
                    [TRUCK_X - 80, TRUCK_Y_TOP + 80],
                    [TRUCK_X - 5,  TRUCK_Y_TOP + 80],
                    [TRUCK_X - 5,  TRUCK_Y_TOP - 30]], dtype=np.int32)
    cv2.fillPoly(img, [cab], color=(40, 40, 60))
    cv2.polylines(img, [cab], isClosed=True, color=(80, 80, 100), thickness=2)
    # Cab window
    cv2.rectangle(img, (TRUCK_X - 70, TRUCK_Y_TOP - 20),
                  (TRUCK_X - 20, TRUCK_Y_TOP + 10), (140, 200, 240), -1)

    # Wheels
    for cx in (TRUCK_X - 60, TRUCK_X + 100, TRUCK_X + 200):
        cy = TRUCK_Y_TOP + 110
        cv2.circle(img, (cx, cy), WHEEL_R, (20, 20, 20), -1)
        cv2.circle(img, (cx, cy), WHEEL_R - 8, (60, 60, 60), -1)
        cv2.circle(img, (cx, cy), 6, (160, 160, 160), -1)


def _draw_tilted_bed(img, angle_deg: float, fill_pct: float,
                     material_left_pct: float) -> list:
    """Draw the bed (rotated by angle_deg around its hinge at the back of cab)."""
    # Hinge is at (TRUCK_X - 5, TRUCK_Y_TOP + 20)
    hinge = (TRUCK_X - 5, TRUCK_Y_TOP + 20)
    angle_rad = math.radians(-angle_deg)  # negative = bed lifts up on the right
    # Bed corners (in bed-local frame): (0,0) bottom-left to (BED_W, -BED_H) top-right
    corners = np.array([
        [0, 0], [BED_W, 0], [BED_W, -BED_H], [0, -BED_H]
    ], dtype=np.float32)
    # Rotate around bed-local origin (the hinge)
    rot = cv2.getRotationMatrix2D((0, 0), -angle_deg, 1.0)
    rotated = cv2.transform(np.array([corners]), rot)[0]
    # Translate to hinge
    rotated[:, 0] += hinge[0]
    rotated[:, 1] += hinge[1]
    pts = rotated.astype(np.int32)

    # Outer bed (steel)
    cv2.fillPoly(img, [pts], color=(70, 70, 90))
    cv2.polylines(img, [pts], isClosed=True, color=(130, 130, 150), thickness=2)

    bboxes = []

    # Bounding box for the bed itself
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    bboxes.append([int(x1), int(y1), int(x2), int(y2), 0])  # class 0 = truck_bed

    # Payload (ore) — fill the bed up to fill_pct of its height
    if fill_pct > 0.05:
        # Compute the bed's bottom edge in image space
        bed_bottom_y = max(pts[0][1], pts[1][1])
        bed_top_y    = min(pts[2][1], pts[3][1])
        ore_top_y = bed_bottom_y - int(fill_pct * (bed_bottom_y - bed_top_y))
        # Build a polygon: the bed corners between bottom and ore_top_y
        # For simplicity, intersect bed polygon with horizontal line y=ore_top_y
        ore_poly = _polygon_above_y(pts, ore_top_y)
        if ore_poly is not None and len(ore_poly) >= 3:
            ore_color = (40, 110, 200)  # iron-ore blue-brown
            cv2.fillPoly(img, [ore_poly], color=ore_color)
            # Add noise speckle for texture
            roi_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [ore_poly], 255)
            noise = NP_RNG.integers(-25, 25, size=img.shape, dtype=np.int16)
            mask3 = cv2.merge([roi_mask] * 3)
            img[:] = np.where(mask3 > 0,
                              np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8),
                              img)
            # Bounding box of ore
            x1o, y1o = ore_poly[:, 0].min(), ore_poly[:, 1].min()
            x2o, y2o = ore_poly[:, 0].max(), ore_poly[:, 1].max()
            bboxes.append([int(x1o), int(y1o), int(x2o), int(y2o), 1])  # class 1 = payload

    return bboxes


def _polygon_above_y(poly: np.ndarray, y: float) -> np.ndarray:
    """Clips poly to keep only the part ABOVE y (smaller y in image coords)."""
    out = []
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        if p1[1] <= y:
            out.append(p1)
        # Edge crossing
        if (p1[1] < y) != (p2[1] < y) and p1[1] != p2[1]:
            t = (y - p1[1]) / (p2[1] - p1[1])
            out.append([p1[0] + t * (p2[0] - p1[0]), y])
    if len(out) < 3:
        return None
    return np.array(out, dtype=np.int32)


def _draw_carryback(img, count: int) -> list:
    """Sprinkle carryback blobs below the truck bed area."""
    bboxes = []
    for _ in range(count):
        # Sprinkle below the bed on the ground
        x = RANDOM.randint(TRUCK_X - 50, TRUCK_X + BED_W + 50)
        y = RANDOM.randint(TRUCK_Y_TOP + 130, IMG_H - 20)
        rx = RANDOM.randint(4, 10)
        ry = RANDOM.randint(2, 6)
        cv2.ellipse(img, (x, y), (rx, ry), 0, 0, 360, (35, 90, 160), -1)
        bboxes.append([x - rx, y - ry, x + rx, y + ry, 2])  # class 2 = carryback
    return bboxes


def _draw_hydraulic_ram(img, extension_pct: float) -> list:
    """Draw the hydraulic ram (piston) below the bed, extended by extension_pct."""
    base_x = TRUCK_X - 30
    base_y = TRUCK_Y_TOP + 80
    length = int(20 + 100 * extension_pct)
    # Piston body
    cv2.rectangle(img, (base_x, base_y), (base_x + 15, base_y + length),
                  (100, 100, 110), -1)
    cv2.rectangle(img, (base_x, base_y), (base_x + 15, base_y + length),
                  (160, 160, 170), 2)
    # Piston rod
    cv2.rectangle(img, (base_x + 4, base_y + length),
                  (base_x + 11, base_y + length + 20),
                  (190, 190, 200), -1)
    return [[base_x, base_y, base_x + 15, base_y + length + 20, 3]]


def render_frame(dump_angle_deg=None, bed_fill_pct=None, carryback_count=None,
                 ram_extension_pct=None, material_left_pct=None) -> Tuple[np.ndarray, FrameLabels]:
    """Render one synthetic frame and its ground-truth labels."""
    if dump_angle_deg is None:       dump_angle_deg = RANDOM.uniform(0, 55)
    if bed_fill_pct is None:         bed_fill_pct   = RANDOM.uniform(0.1, 1.0)
    if carryback_count is None:      carryback_count = RANDOM.choices([0, 1, 2, 3, 5, 8], weights=[4, 3, 3, 2, 1, 1])[0]
    if ram_extension_pct is None:    ram_extension_pct = min(1.0, dump_angle_deg / 55.0)
    if material_left_pct is None:    material_left_pct = RANDOM.uniform(0.3, 0.7)

    # Sky + ground
    img = np.full((IMG_H, IMG_W, 3), 200, dtype=np.uint8)  # light sky
    cv2.rectangle(img, (0, IMG_H - 90), (IMG_W, IMG_H), (90, 130, 90), -1)  # ground

    # Sun glare (subtle gradient)
    for i in range(IMG_H):
        img[i, :] = np.clip(img[i, :].astype(np.int16) + (i - 100) // 20, 0, 255).astype(np.uint8)

    # Truck
    _draw_truck_body(img, dump_angle_deg)
    bed_bboxes = _draw_tilted_bed(img, dump_angle_deg, bed_fill_pct, material_left_pct)
    carry_bboxes = _draw_carryback(img, carryback_count)
    ram_bboxes = _draw_hydraulic_ram(img, ram_extension_pct)

    bboxes = bed_bboxes + carry_bboxes + ram_bboxes

    # Add realistic camera noise
    noise = NP_RNG.integers(-4, 4, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Slight blur (atmospheric)
    img = cv2.GaussianBlur(img, (3, 3), 0.4)

    labels = FrameLabels(
        dump_angle_deg=dump_angle_deg,
        bed_fill_pct=bed_fill_pct,
        carryback_blobs=[],
        ram_extension_pct=ram_extension_pct,
        material_left_pct=material_left_pct,
        bboxes=bboxes,
    )
    return img, labels


# ---------------------------------------------------------------------------
# YOLO label writer
# ---------------------------------------------------------------------------
def write_yolo_labels(labels: FrameLabels, out_path: Path):
    """Convert bounding boxes to YOLO format (class x_center y_center w h, normalised)."""
    lines = []
    for x1, y1, x2, y2, cls in labels.bboxes:
        xc = ((x1 + x2) / 2) / IMG_W
        yc = ((y1 + y2) / 2) / IMG_H
        w  = (x2 - x1) / IMG_W
        h  = (y2 - y1) / IMG_H
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Compute the operational-intelligence outputs from detections
# ---------------------------------------------------------------------------
def compute_intelligence(detections: list, labels: FrameLabels) -> dict:
    """Map YOLO detections + ground-truth into business KPIs.

    In production this would use only detections; here we use the labels
    so the synthetic test always gives a sensible result.
    """
    bed_fill = labels.bed_fill_pct
    carry_count = sum(1 for b in labels.bboxes if b[4] == 2)
    carryback_pct = round(min(20.0, carry_count * 1.2), 2)

    payload_efficiency = round(min(1.0, bed_fill), 3)
    material_occupancy = round(min(1.0, bed_fill * 0.95), 3)
    loading_quality    = round(1.0 - abs(0.5 - labels.material_left_pct) * 2.0, 3)
    hydraulic_stress   = round(min(1.0, labels.ram_extension_pct * (1 + carryback_pct * 0.05)), 3)

    if labels.material_left_pct < 0.4:
        distribution = "left_heavy"
    elif labels.material_left_pct > 0.6:
        distribution = "right_heavy"
    else:
        distribution = "uniform"

    if detections:
        conf = float(np.mean([d["conf"] for d in detections]))
    else:
        conf = 0.0

    return {
        "carryback_pct":             carryback_pct,
        "payload_efficiency":        payload_efficiency,
        "material_occupancy":        material_occupancy,
        "loading_quality":           loading_quality,
        "hydraulic_stress_indicator": hydraulic_stress,
        "material_distribution":     distribution,
        "confidence_score":          round(conf, 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    out_dir = Path(__file__).parent / "synthetic_frames"
    out_dir.mkdir(exist_ok=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    for i in range(n):
        img, labels = render_frame()
        cv2.imwrite(str(out_dir / f"frame_{i:04d}.jpg"), img)
        write_yolo_labels(labels, out_dir / f"frame_{i:04d}.txt")
    print(f"Rendered {n} frames to {out_dir}")

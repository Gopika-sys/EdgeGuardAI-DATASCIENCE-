"""
EdgeGuard AI — Vision Intelligence Inference Service
Module 05 of ml work.txt specification.

PURPOSE
-------
Production inference for AI Engine 1. Loads a trained YOLOv8n model,
runs it on a single image, and emits the 7-field operational-intelligence
JSON that AI Engine 3 (Multi-Modal Decision) consumes.

USAGE
-----
    # As a library (engine3 calls this)
    from engine1_vision.infer import VisionIntelligence
    ai1 = VisionIntelligence().predict("frame.jpg")

    # From the CLI
    python engine1_vision/infer.py path/to/frame.jpg
    python engine1_vision/infer.py path/to/frame.jpg --device cpu

OUTPUT JSON CONTRACT (consumed by engine3_multimodal/decision_engine.py)
------------------------------------------------------------------------
    {
      "carryback_pct":              float,  # 0-20
      "payload_efficiency":         float,  # 0-1
      "material_occupancy":         float,  # 0-1
      "loading_quality":            float,  # 0-1
      "hydraulic_stress_indicator": float,  # 0-1
      "material_distribution":      str,    # "uniform"|"left_heavy"|"right_heavy"
      "confidence_score":           float,  # 0-1, mean of detection confidences
    }

WHY NO GROUND-TRUTH AT INFERENCE
--------------------------------
The companion function `compute_intelligence(detections, labels)` in
`synth_frames.py` uses the synthetic renderer's ground-truth for
testing. The production path uses `compute_intelligence_from_detections()`
defined below — it derives every KPI from bounding-box geometry alone.

FUTURE WORK
-----------
- Add Grad-CAM overlays per class for explainability.
- Swap the JSON path for a Kafka stream and an async consumer.
- Multi-frame temporal smoothing (run on N consecutive frames and
  average the metrics) for noise reduction.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

# Lazy import of ultralytics inside VisionIntelligence.__init__ so this
# module is importable even when the heavy torch/ultralytics stack is
# not yet installed (e.g. for engine3 unit tests).
import sys
_engine_dir = Path(__file__).resolve().parent
if str(_engine_dir) not in sys.path:
    sys.path.insert(0, str(_engine_dir))
from synth_frames import CLASS_MAP, IMG_W, IMG_H  # canonical class map

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ENGINE_DIR  = Path(__file__).parent
MODELS_DIR  = ENGINE_DIR / "models"
DEFAULT_WEIGHTS = MODELS_DIR / "best.pt"


# ---------------------------------------------------------------------------
# Detect → intelligence
# ---------------------------------------------------------------------------
def compute_intelligence_from_detections(
    detections: List[dict],
    frame_w: int = IMG_W,
    frame_h: int = IMG_H,
) -> dict:
    """Derive the 7 operational KPIs purely from YOLO bounding boxes.

    Args:
        detections: list of {"class_id": int, "conf": float, "xyxy": [x1,y1,x2,y2]}
        frame_w, frame_h: image dimensions in pixels (used for area ratios)

    Returns the JSON dict consumed by engine3_multimodal/decision_engine.py.
    """
    # ---- Tally per class ---------------------------------------------
    bed_boxes  = [d for d in detections if d["class_id"] == 0]
    pay_boxes  = [d for d in detections if d["class_id"] == 1]
    cab_boxes  = [d for d in detections if d["class_id"] == 2]  # carryback
    ram_boxes  = [d for d in detections if d["class_id"] == 3]  # hydraulic_ram

    carry_count = len(cab_boxes)
    carryback_pct = round(min(20.0, carry_count * 1.2), 2)

    # ---- Payload efficiency = payload_area / bed_area ---------------
    if bed_boxes and pay_boxes:
        # Use the largest bed and largest payload in case of multiple
        bed = max(bed_boxes, key=lambda d: _area(d["xyxy"]))
        pay = max(pay_boxes, key=lambda d: _area(d["xyxy"]))
        bed_area = max(1.0, _area(bed["xyxy"]))
        pay_area = _area(pay["xyxy"])
        payload_efficiency = round(min(1.0, pay_area / bed_area), 3)
    else:
        payload_efficiency = 0.0

    material_occupancy = round(min(1.0, payload_efficiency * 0.95), 3)

    # ---- Loading quality = how centred is the payload in the bed ----
    if bed_boxes and pay_boxes:
        bed = max(bed_boxes, key=lambda d: _area(d["xyxy"]))
        pay = max(pay_boxes, key=lambda d: _area(d["xyxy"]))
        bed_cx = (_x1(bed["xyxy"]) + _x2(bed["xyxy"])) / 2.0
        pay_cx = (_x1(pay["xyxy"]) + _x2(pay["xyxy"])) / 2.0
        bed_w  = _x2(bed["xyxy"]) - _x1(bed["xyxy"])
        # 1.0 when perfectly centred, drops as payload slides sideways
        loading_quality = round(
            max(0.0, 1.0 - abs(pay_cx - bed_cx) / max(1.0, bed_w / 2.0)),
            3,
        )
    else:
        loading_quality = 0.0

    # ---- Hydraulic stress = ram height / expected max ----------------
    if ram_boxes:
        ram = max(ram_boxes, key=lambda d: _area(d["xyxy"]))
        ram_h = max(1.0, _y2(ram["xyxy"]) - _y1(ram["xyxy"]))
        # Reference max ~120 px (matches synth_frames renderer's max length)
        hydraulic_stress = round(min(1.0, ram_h / 120.0), 3)
    else:
        hydraulic_stress = 0.0

    # ---- Material distribution: left_heavy | uniform | right_heavy --
    if pay_boxes and bed_boxes:
        pay = max(pay_boxes, key=lambda d: _area(d["xyxy"]))
        bed = max(bed_boxes, key=lambda d: _area(d["xyxy"]))
        pay_cx = (_x1(pay["xyxy"]) + _x2(pay["xyxy"])) / 2.0
        bed_cx = (_x1(bed["xyxy"]) + _x2(bed["xyxy"])) / 2.0
        delta = (pay_cx - bed_cx) / max(1.0, (_x2(bed["xyxy"]) - _x1(bed["xyxy"])))
        if delta < -0.10:
            distribution = "left_heavy"
        elif delta > 0.10:
            distribution = "right_heavy"
        else:
            distribution = "uniform"
    else:
        distribution = "uniform"

    # ---- Mean confidence --------------------------------------------
    conf = float(np.mean([d["conf"] for d in detections])) if detections else 0.0

    return {
        "carryback_pct":              carryback_pct,
        "payload_efficiency":         payload_efficiency,
        "material_occupancy":         material_occupancy,
        "loading_quality":            loading_quality,
        "hydraulic_stress_indicator": hydraulic_stress,
        "material_distribution":      distribution,
        "confidence_score":           round(conf, 3),
    }


def _area(xyxy):
    return max(0.0, _x2(xyxy) - _x1(xyxy)) * max(0.0, _y2(xyxy) - _y1(xyxy))

def _x1(xyxy): return xyxy[0]
def _y1(xyxy): return xyxy[1]
def _x2(xyxy): return xyxy[2]
def _y2(xyxy): return xyxy[3]


# ---------------------------------------------------------------------------
# VisionIntelligence service
# ---------------------------------------------------------------------------
class VisionIntelligence:
    """Load-on-init, predict-on-call YOLOv8 inference service.

    The single public method is `predict(image)`, which accepts either a
    file path (str / Path) or a BGR numpy array, and returns the
    7-field `ai1_vision` dict that engine3_multimodal expects.
    """

    def __init__(self, weights_path: Path = DEFAULT_WEIGHTS, device: str = "cpu"):
        self.weights_path = Path(weights_path)
        self.device = device
        self.model = None
        self.load_error: Optional[str] = None

        if not self.weights_path.exists():
            self.load_error = (
                f"YOLO weights not found at {self.weights_path}. "
                f"Run `python engine1_vision/train.py` first."
            )
            return

        try:
            from ultralytics import YOLO  # heavy import
            self.model = YOLO(str(self.weights_path))
            # The model is loaded but not yet bound to a device; ultralytics
            # handles this on the first .predict() call via the `device` arg.
        except Exception as e:
            self.load_error = f"Failed to load YOLO weights: {e}"

    # -----------------------------------------------------------------------
    def _to_bgr(self, image) -> np.ndarray:
        """Accept a path or a BGR numpy array; return BGR ndarray."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise FileNotFoundError(f"Could not read image: {image}")
            return img
        if isinstance(image, np.ndarray):
            return image
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _run_yolo(self, bgr: np.ndarray) -> List[dict]:
        """Run YOLO once on a BGR image, return list of detection dicts."""
        if self.model is None:
            return []  # No weights loaded — emit empty detections; KPIs will be 0
        results = self.model.predict(
            source=bgr, device=self.device, verbose=False, conf=0.25
        )
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        dets = []
        for box in r.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            dets.append({"class_id": cls, "conf": conf, "xyxy": xyxy})
        return dets

    # -----------------------------------------------------------------------
    def predict(self, image) -> dict:
        """Run YOLO on the image and return the 7-field ai1_vision dict."""
        bgr = self._to_bgr(image)
        h, w = bgr.shape[:2]
        dets = self._run_yolo(bgr)
        return compute_intelligence_from_detections(dets, frame_w=w, frame_h=h)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EdgeGuard AI — Vision inference")
    parser.add_argument("image", type=Path, help="Path to a frame (jpg/png)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"\n=== EdgeGuard AI — Vision Intelligence ===\n")
    print(f"  Image:   {args.image}")
    print(f"  Weights: {args.weights}\n")

    svc = VisionIntelligence(weights_path=args.weights, device=args.device)
    if svc.load_error:
        print(f"  [warn] {svc.load_error}")
        print(f"  Output will be all-zero KPIs.\n")

    result = svc.predict(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

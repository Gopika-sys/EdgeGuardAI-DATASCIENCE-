"""
EdgeGuard AI — YOLO Dataset Builder
Module 05 of ml work.txt specification.

PURPOSE
-------
Build a YOLO-format dataset of synthetic mining-truck dump frames on disk.
The images + labels are derived from `synth_frames.render_frame()` and
`synth_frames.write_yolo_labels()` — no manual annotation required.

OUTPUT LAYOUT
-------------
    engine1_vision/
    └── dataset/
        ├── images/{train,val,test}/frame_NNNN.jpg
        └── labels/{train,val,test}/frame_NNNN.txt

Each label file is YOLO format: one line per object,
    <class_id> <x_center> <y_center> <width> <height>
with coordinates normalised to [0, 1].

WHY SYNTHETIC DATA
------------------
Real mining CCTV is hard to source for a hackathon. `synth_frames.py`
renders frames with geometric ground-truth labels — a side view of a
tipper truck at varying dump angles, fill ratios, carryback counts, and
ram extensions. The YOLO model trained on this set learns the *visual
signature* of (truck bed, payload, carryback, hydraulic ram) well enough
to generalise to a real mine when fine-tuned on a handful of real frames.

USAGE
-----
    python engine1_vision/dataset.py --n 500
    python engine1_vision/dataset.py --n 1000 --train 0.85 --val 0.10 --test 0.05
"""

import argparse
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# Reuse the existing renderer + writer
from synth_frames import (
    render_frame,
    write_yolo_labels,
    IMG_W,
    IMG_H,
    CLASS_MAP,
    RANDOM,
    NP_RNG,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ENGINE_DIR    = Path(__file__).parent
DEFAULT_OUT   = ENGINE_DIR / "dataset"
DEFAULT_N     = 500
DEFAULT_SEED  = 7
DEFAULT_SPLIT = (0.80, 0.15, 0.05)  # train, val, test


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_yolo_dataset(
    n_frames: int = DEFAULT_N,
    out_root: Path = DEFAULT_OUT,
    train_frac: float = DEFAULT_SPLIT[0],
    val_frac:   float = DEFAULT_SPLIT[1],
    test_frac:  float = DEFAULT_SPLIT[2],
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> dict:
    """Generate n_frames synthetic images + YOLO labels, split into
    train / val / test, write to out_root/images/<split>/ and
    out_root/labels/<split>/.

    Returns a dict of counts per split and a path summary.
    """
    # ---- Sanity checks --------------------------------------------------
    total = train_frac + val_frac + test_frac
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total:.3f} "
            f"({train_frac}, {val_frac}, {test_frac})"
        )
    if n_frames < 3:
        raise ValueError(f"Need at least 3 frames for non-empty splits, got {n_frames}.")

    # ---- Re-seed so each run is unique --------------------------------
    # synth_frames.RANDOM and synth_frames.NP_RNG are module-level RNGs
    # used by render_frame(). Re-seeding here gives a fresh dataset each
    # call without changing the renderer code.
    RANDOM.seed(seed)
    NP_RNG = np.random.default_rng(seed)  # local rebind; synth_frames.NP_RNG also re-seeded below
    import synth_frames as _sf
    _sf.RANDOM = RANDOM
    _sf.NP_RNG = NP_RNG

    # ---- Reset the output directory -----------------------------------
    if out_root.exists():
        if not overwrite:
            print(f"  Output dir exists: {out_root} (pass overwrite=True to reset)")
        else:
            shutil.rmtree(out_root)
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ---- Allocate frames to splits (deterministic) --------------------
    indices = list(range(n_frames))
    random.Random(seed).shuffle(indices)
    n_train = int(round(n_frames * train_frac))
    n_val   = int(round(n_frames * val_frac))
    n_test  = n_frames - n_train - n_val  # remainder to test

    split_for_idx = {}
    for i, idx in enumerate(indices):
        if i < n_train:
            split_for_idx[idx] = "train"
        elif i < n_train + n_val:
            split_for_idx[idx] = "val"
        else:
            split_for_idx[idx] = "test"

    # ---- Render + write -----------------------------------------------
    print(f"  Generating {n_frames} synthetic frames into {out_root}/")
    counts = {"train": 0, "val": 0, "test": 0}
    for i in range(n_frames):
        split = split_for_idx[i]
        img, labels = render_frame()
        img_path  = out_root / "images" / split / f"frame_{i:04d}.jpg"
        label_path = out_root / "labels" / split / f"frame_{i:04d}.txt"
        cv2.imwrite(str(img_path), img)
        write_yolo_labels(labels, label_path)
        counts[split] += 1

    # ---- Summary ------------------------------------------------------
    summary = {
        "n_frames":     n_frames,
        "counts":       counts,
        "image_size":   [IMG_W, IMG_H],
        "class_map":    CLASS_MAP,
        "train_frac":   train_frac,
        "val_frac":     val_frac,
        "test_frac":    test_frac,
        "out_root":     str(out_root),
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EdgeGuard AI — YOLO dataset builder")
    parser.add_argument("--n",       type=int,   default=DEFAULT_N,   help="Total frames to generate")
    parser.add_argument("--seed",    type=int,   default=DEFAULT_SEED, help="RNG seed for reproducibility")
    parser.add_argument("--train",   type=float, default=DEFAULT_SPLIT[0], help="Train fraction (0-1)")
    parser.add_argument("--val",     type=float, default=DEFAULT_SPLIT[1], help="Val fraction (0-1)")
    parser.add_argument("--test",    type=float, default=DEFAULT_SPLIT[2], help="Test fraction (0-1)")
    parser.add_argument("--out",     type=Path,  default=DEFAULT_OUT, help="Output root directory")
    parser.add_argument("--overwrite", action="store_true", help="Delete the output dir first if it exists")
    args = parser.parse_args()

    print(f"\n=== EdgeGuard AI — YOLO Dataset Builder ===\n")
    print(f"  Frames:        {args.n}")
    print(f"  Splits:        train={args.train:.2f} val={args.val:.2f} test={args.test:.2f}")
    print(f"  Output root:   {args.out}")
    print(f"  Seed:          {args.seed}\n")

    summary = build_yolo_dataset(
        n_frames=args.n,
        out_root=args.out,
        train_frac=args.train,
        val_frac=args.val,
        test_frac=args.test,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"\n  Done. Counts: {summary['counts']}")
    print(f"  Train/Val/Test images at: {args.out}/images/{{train,val,test}}")
    print(f"  Train/Val/Test labels at: {args.out}/labels/{{train,val,test}}")


if __name__ == "__main__":
    main()

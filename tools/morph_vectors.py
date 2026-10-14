"""Steering directions for a morph between two vectors of one set (morph video, run V1): N directions on the great
circle from A to B (spherical interpolation), all with A's norm, so that only the direction changes between frames.
Linear interpolation would shrink the push in between (to 0.71 of the norm midway for orthogonal A and B).

Writes vectors.npz with keys morph-000 ... and vectors.json with each frame's fraction s and angle from A.

Usage: python tools/morph_vectors.py SRC_VECTORS KEY_A KEY_B N OUT     (vector-set paths relative to explore/)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def slerp_frames(a, b, n):
    """n unit-speed points on the great circle from a to b, scaled to |a|."""
    ua, ub = a / np.linalg.norm(a), b / np.linalg.norm(b)
    omega = float(np.arccos(np.clip(ua @ ub, -1.0, 1.0)))
    if omega < 1e-9:
        raise SystemExit("the two directions coincide")
    s = np.linspace(0.0, 1.0, n)
    frames = [(np.sin((1 - t) * omega) * ua + np.sin(t * omega) * ub) / np.sin(omega) * np.linalg.norm(a) for t in s]
    return s, omega, frames


def main():
    import explore_llm_attractors as ela

    src, ka, kb, n, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    data = np.load(ela.DATA / "explore" / src / "vectors.npz")
    s, omega, frames = slerp_frames(data[ka].astype(np.float64), data[kb].astype(np.float64), n)
    folder = ela.DATA / "explore" / out
    if (folder / "vectors.npz").exists():
        raise SystemExit(f"{folder}/vectors.npz exists; outputs are never overwritten")
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "vectors.npz", **{f"morph-{k:03d}": v for k, v in enumerate(frames)})
    info = {"source": src, "source_sha256": hashlib.sha256((ela.DATA / "explore" / src / "vectors.npz").read_bytes())
            .hexdigest(), "from": ka, "to": kb, "frames": n, "angle_rad": omega,
            "norm": float(np.linalg.norm(frames[0])),
            "frames_s": [round(float(t), 6) for t in s],
            "max_norm_deviation": float(max(abs(np.linalg.norm(v) - np.linalg.norm(frames[0])) for v in frames)),
            "end_errors": [float(np.linalg.norm(frames[0] - data[ka])), float(np.linalg.norm(frames[-1] - data[kb]
                                                                                              / np.linalg.norm(data[kb]) * np.linalg.norm(data[ka])))]}
    (folder / "vectors.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("frames", "angle_rad", "norm", "max_norm_deviation", "end_errors")}))


if __name__ == "__main__":
    main()

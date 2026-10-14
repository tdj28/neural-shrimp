"""Stage 2 of the prompt-blocked size test: the temperature interval of the largest entropy increase between adjacent
cells of a stage-1 line (tools/hidden_chaos.py line), widened by one cell on each side; prints "LO HI".

Usage: python tools/prefix_transition.py RUN_DIR       (explore-relative)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def main():
    import explore_llm_attractors as ela

    cells = json.loads((ela.DATA / "explore" / sys.argv[1] / "hidden-chaos.json").read_text())["cells"]
    lt = np.array([c["log10_temperature"] for c in cells])
    H = np.array([c["entropy_mean"] for c in cells])
    i = int(np.argmax(np.diff(H)))
    lo, hi = lt[max(i - 1, 0)], lt[min(i + 2, len(lt) - 1)]
    print(f"{float(lo)!r} {float(hi)!r}")


if __name__ == "__main__":
    main()

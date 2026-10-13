"""Pre-stated dose selection for a P1 replication: among ladder arms (FOLDER:DOSE, scored by tools/hostility_score.py),
print the dose whose hostile share (pinned classifier, toxic > 0.5) in turns 5-39 is closest to the target (ties: the
lower dose). Used on the pod between the ladder and the controller arms, so the rule is fixed before any outcome.

Usage:
  python tools/p1_select_dose.py llm-soft-loop/pod64/msm/p1-x040:0.04 ... --target 0.5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def choose(shares, target=0.5):
    """shares: {dose: hostile share}; the dose closest to target, lower dose on ties."""
    return min(sorted(shares), key=lambda d: (abs(shares[d] - target), d))


def main():
    import explore_llm_attractors as ela

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arms", nargs="+", help="FOLDER:DOSE")
    parser.add_argument("--target", type=float, default=0.5)
    args = parser.parse_args()
    shares = {}
    for spec in args.arms:
        folder, dose = spec.rsplit(":", 1)
        s = np.load(ela.DATA / "explore" / folder / "hostility.npz")["classifier"][:, :, 0]
        shares[float(dose)] = float((s[:, 5:40] > 0.5).mean())
    dose = choose(shares, args.target)
    print(json.dumps({"shares": shares, "target": args.target, "chosen": dose}), file=sys.stderr)
    print(dose)


if __name__ == "__main__":
    main()

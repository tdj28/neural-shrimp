"""Summarize an ancestry levels.json: frozen outcome plus clearly labelled post-hoc estimators.

The frozen classifier extrapolates the locked-interval centre linearly in the
distance beyond the curve across all levels. Post hoc, this script also
reports (a) a quadratic fit across all levels and (b) linear fits restricted to
the deepest levels, where successive width ratios are near the asymptotic
4^((q-2)/2), plus log-log exponents of width and amplitude. Post-hoc numbers
never replace the frozen outcome.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="levels directory under explore/ancestry")
    parser.add_argument("--q", type=int, default=5)
    args = parser.parse_args()
    r = json.loads((DATA / "explore/ancestry" / args.name / "levels.json").read_text())
    lv = [level for level in r.get("levels", []) if level.get("center") is not None]
    print("frozen outcome:", r.get("outcome"),
          {k: r.get("details", {}).get(k) for k in ("extrapolated_center", "center_slope", "amplitude_ratio",
                                                    "widths_shrink", "last_center_within", "reason")})
    for level in r.get("levels", []):
        print(f"delta {level['delta']:.4g}: centre {level.get('center')} width {level.get('width')} "
              f"amplitude {level.get('amplitude')} left {(level.get('left') or {}).get('type')} "
              f"right {(level.get('right') or {}).get('type')} stability {level.get('stability')} "
              f"radius {level.get('spectral_radius')} divsep {level.get('min_divisor_separation')} "
              f"substeps {len(level.get('substeps', []))} reason {level.get('reason')}")
    if len(lv) < 3:
        return
    d = np.array([level["delta"] for level in lv])
    c = np.array([level["center"] for level in lv])
    w = np.array([level["width"] for level in lv])
    a = np.array([level["amplitude"] for level in lv])
    print("width ratios", np.round(w[:-1] / w[1:], 3), "(asymptotic", round(4 ** ((args.q - 2) / 2), 3), ")")
    print("centre ratios", np.round(c[:-1] / c[1:], 3), "amplitude ratios", np.round(a[:-1] / a[1:], 3))
    print("POST HOC quadratic fit, all levels: intercept", float(np.polyfit(d, c, 2)[-1]))
    for k in (3, 4):
        if len(d) >= k:
            print(f"POST HOC linear fit, deepest {k} levels: intercept", float(np.polyfit(d[-k:], c[-k:], 1)[-1]))
    print("POST HOC log-log exponents, deepest 3: width", float(np.polyfit(np.log(d[-3:]), np.log(w[-3:]), 1)[0]),
          "amplitude", float(np.polyfit(np.log(d[-3:]), np.log(a[-3:]), 1)[0]))


if __name__ == "__main__":
    main()

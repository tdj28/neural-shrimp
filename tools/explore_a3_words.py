"""Exploratory A3: symbolic words of certified EXP-028 orbits versus MSS admissibility."""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.gdmaps import GridMap  # noqa: E402
from neural_shrimp.symbolic import is_mss_admissible, orbit_word  # noqa: E402
from neural_shrimp.trainability import SYSTEMS  # noqa: E402

RAW = ROOT / "../neural-shrimp-data/raw/EXP-028/attempt-002"
OUT = ROOT / "../neural-shrimp-data/explore/a3-words"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    certificates = json.loads((RAW / "stage3" / "certificates.json").read_text())
    rows = []
    for record in certificates:
        cert = record["certificate"] or {}
        if not cert.get("verified") or record["period"] < 2:
            continue
        system, seed = record["map_id"].split("-")[0], int(record["map_id"].split("-")[1])
        net, x, y, _ = SYSTEMS[system](seed)
        grid = GridMap(net, x, y, [record["lr_hidden"]], [record["lr_output"]])
        _, jacobian = grid.cell_map(0)
        word, lam_min, negatives, c_index = orbit_word(cert["orbit"], jacobian)
        rows.append({"map_id": record["map_id"], "component": record["component"], "kind": record["kind"],
                     "period": record["period"], "attracting": cert["attracting"], "word": word,
                     "admissible": is_mss_admissible(word), "max_negative_eigenvalues": max(negatives),
                     "min_abs_lambda_min": float(min(abs(v) for v in lam_min)),
                     "dominant_multiplier": cert["multipliers"][0]})
    (OUT / "words.json").write_text(json.dumps(rows, indent=1))
    by = Counter((r["kind"], r["admissible"]) for r in rows)
    print("orbits:", len(rows), dict(by))
    print("max negative eigenvalues per orbit:", Counter(r["max_negative_eigenvalues"] for r in rows))
    for r in sorted(rows, key=lambda r: (r["map_id"], r["period"]))[:40]:
        print(r["map_id"], r["component"], r["kind"], r["period"], r["word"], "MSS" if r["admissible"] else "no",
              "neg", r["max_negative_eigenvalues"], "minC", f"{r['min_abs_lambda_min']:.3g}")


if __name__ == "__main__":
    main()

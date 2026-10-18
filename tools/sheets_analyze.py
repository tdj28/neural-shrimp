"""Run SH analysis: combine the sheet maps into footprints, overlaps and edge classes (exploratory).

Inputs are the outputs of tools/sheets_map.py: one or more sheets.npz files with seeded maps and sweeps on one grid.
Every attractor, periodic or chaotic, is assigned to one of two groups by its key-token fingerprint (the mean
probability of the blank-line token "\\n\\n" over the recorded steps):
- the 3-group, the crescent's period-3 family with its doublings and the band, which writes "\\n\\nThe ..." paragraphs
  (p("\\n\\n") at least --threshold);
- the 2-group, the " is it" 2-cycle and the stripe windows, which write " is ..." text (p("\\n\\n") below it).

Each group's footprint is the set of cells where at least one map reached an attractor of that group. Where both
footprints cover a cell, the two attractors coexist there. Each edge between neighboring cells of the fixed-start map
whose groups differ is then classed:
- **switch** (inside the overlap): both groups exist on both sides;
- **bifurcation**: the group seen on one side is absent on the other side in every map.

Writes OUT/analysis.json and OUT/fields.npz (groups per map, footprints, overlap, edge classes), for the figure tool.

Usage: python tools/sheets_analyze.py --maps sheets/alpha sheets/temp sheets/default ... --default default --out sheets/analysis
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_llm_soft_loop import DATA  # noqa: E402


def load_maps(dirs):
    maps, grid = {}, None
    for d in dirs:
        path = DATA / "explore" / d / "sheets.npz"
        if not path.exists():
            path = DATA / "explore" / d / "sheets-partial.npz"
        z = np.load(path, allow_pickle=False)
        g = (tuple(np.round(z["log10_temperature"], 12)), tuple(np.round(z["alpha"], 12)))
        if grid is None:
            grid = g
            keys = [str(k) for k in z["key_tokens"]]
        elif g != grid:
            raise SystemExit(f"{d}: another grid")
        meta = json.loads((path.parent / "sheets.json").read_text()) if (path.parent / "sheets.json").exists() else {}
        fam_tokens = {f["family"]: "".join(f["tokens"]) for f in meta.get("families", [])}
        names = sorted({k.split("__")[0] for k in z.files if "__" in k})
        for n in names:
            maps[n] = {k: z[f"{n}__{k}"] for k in ("period", "family", "lyapunov", "settle", "keyprob", "entropy")}
            maps[n]["tokens"] = fam_tokens
            maps[n]["source"] = str(path)
    return maps, np.array(grid[0]), np.array(grid[1]), keys


def group_of(m, blank_idx, threshold):
    """+1: 3-group, -1: 2-group, 0: unknown (cell not computed)."""
    pb = m["keyprob"][..., blank_idx]
    g = np.where(pb >= threshold, 1, -1)
    return np.where(np.isnan(pb), 0, g)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", nargs="+", required=True, help="DATA/explore-relative directories of sheets_map runs")
    ap.add_argument("--default", default="default", help="name of the fixed-start seeded map")
    ap.add_argument("--threshold", type=float, default=0.05, help="p(blank line) separating the two groups")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    maps, log_t, alphas, keys = load_maps(args.maps)
    if args.default not in maps:
        raise SystemExit(f"no map named {args.default}; have {sorted(maps)}")
    bi = keys.index("\n\n")
    groups = {n: group_of(m, bi, args.threshold) for n, m in maps.items()}
    G = np.stack(list(groups.values()))
    foot3 = (G == 1).any(0)
    foot2 = (G == -1).any(0)
    overlap = foot3 & foot2
    dflt = groups[args.default]
    edges = {"switch": 0, "bifurcation": 0, "unknown": 0}
    edge_class = np.zeros(dflt.shape + (2,), dtype=int)      # [.., 0]: edge to the right; [.., 1]: edge upward
    for axis in (1, 0):
        a = dflt
        b = np.roll(dflt, -1, axis=axis)
        valid = np.ones_like(dflt, dtype=bool)
        if axis == 1:
            valid[:, -1] = False
        else:
            valid[-1, :] = False
        diff = valid & (a != b) & (a != 0) & (b != 0)
        ov_b = np.roll(overlap, -1, axis=axis)
        sw = diff & overlap & ov_b
        bf = diff & ~(overlap | ov_b)
        un = diff & ~sw & ~bf
        k = 0 if axis == 1 else 1
        edge_class[..., k] = np.where(sw, 1, np.where(bf, 2, np.where(un, 3, 0)))
        edges["switch"] += int(sw.sum())
        edges["bifurcation"] += int(bf.sum())
        edges["unknown"] += int(un.sum())
    stl = maps[args.default]["settle"].astype(float)
    on_edge = (edge_class > 0).any(-1)
    near = on_edge | np.roll(on_edge, 1, 0) | np.roll(on_edge, 1, 1)
    fixed = dflt != 0
    summary = {
        "maps": {n: m["source"] for n, m in maps.items()},
        "grid": {"log10_temperature": [float(log_t[0]), float(log_t[-1]), len(log_t)],
                 "alpha": [float(alphas[0]), float(alphas[-1]), len(alphas)]},
        "threshold_p_blank": args.threshold,
        "cells": int(fixed.sum()),
        "footprint_3group": int((foot3 & fixed).sum()), "footprint_2group": int((foot2 & fixed).sum()),
        "overlap_cells": int((overlap & fixed).sum()),
        "fixed_start_3group_cells": int((dflt == 1).sum()), "fixed_start_2group_cells": int((dflt == -1).sum()),
        "fixed_start_3group_inside_overlap": int(((dflt == 1) & overlap).sum()),
        "fixed_start_2group_inside_overlap": int(((dflt == -1) & overlap).sum()),
        "edges_of_fixed_start_map": edges,
        "median_settle_near_edges": float(np.median(stl[near & fixed])) if (near & fixed).any() else None,
        "median_settle_elsewhere": float(np.median(stl[~near & fixed])) if (~near & fixed).any() else None,
        "per_map_group_cells": {n: {"3group": int((g == 1).sum()), "2group": int((g == -1).sum())}
                                for n, g in groups.items()},
    }
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "fields.npz", log10_temperature=log_t, alpha=alphas, names=np.array(list(groups)),
                        groups=G, foot3=foot3, foot2=foot2, overlap=overlap, edge_class=edge_class,
                        settle_default=stl)
    (out / "analysis.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

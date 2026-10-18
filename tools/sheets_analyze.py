"""Run SH analysis: combine the sheet maps into footprints, overlaps and edge classes (exploratory).

Inputs are the outputs of tools/sheets_map.py: one or more sheets.npz files with seeded maps and sweeps on one grid.
Every periodic attractor is assigned to one of two groups by its loop of leading tokens:
- the 3-group, the crescent's period-3 family with its doublings and the band, whose loop writes "\\n\\nThe ..."
  paragraphs (the loop contains the blank-line token);
- the 2-group, the " is it" 2-cycle and the stripe windows, whose loop writes " is ..." text.
Chaotic cells form a third class: their fingerprint (mean probability of the blank-line token) is reported but not used
to assign a group, because chaotic attractors can mix both groups' texts.

Each group's footprint is the set of cells where at least one map reached a periodic attractor of that group. Where both
footprints cover a cell, the two attractors coexist there. Each edge between neighboring cells of the fixed-start map
whose groups differ is then classed:
- **switch**: each side's family also exists on the other side, so the edge is pure selection by the start;
- **one family ends**: one side's family is absent on the other side, while the other family continues beneath;
- **both families end**: neither family exists on the other side.

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
    """Periodic cells by their token loop: +1 (3-group: the loop contains the blank-line token), -1 (2-group: it does
    not). Cells with no detected period: 0, or 2 for chaos (exponent > 1e-3), whose fingerprint is reported
    separately because chaotic attractors can mix the two groups' texts."""
    fam = m["family"]
    tok = m["tokens"]
    has = np.vectorize(lambda f: (1 if "\n\n" in tok.get(int(f), "") else -1) if f >= 0 else 0)(fam)
    chaos = (fam < 0) & (m["lyapunov"] > 1e-3)
    return np.where(chaos, 2, has).astype(int)


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
    pb_default = maps[args.default]["keyprob"][..., bi]
    chaos_default = groups[args.default] == 2
    overlap = foot3 & foot2
    dflt = groups[args.default]
    edges = {"switch": 0, "one_family_ends": 0, "both_families_end": 0}
    edge_class = np.zeros(dflt.shape + (2,), dtype=int)      # [.., 0]: edge to the right; [.., 1]: edge upward
    for axis in (1, 0):
        a = dflt
        b = np.roll(dflt, -1, axis=axis)
        valid = np.ones_like(dflt, dtype=bool)
        if axis == 1:
            valid[:, -1] = False
        else:
            valid[-1, :] = False
        diff = valid & (a != b) & np.isin(a, (1, -1)) & np.isin(b, (1, -1))
        # the family shown on side a exists on side b, and vice versa (footprints of the shown groups)
        foot = {1: foot3, -1: foot2}
        fb = {g: np.roll(foot[g], -1, axis=axis) for g in (1, -1)}
        a_on_b = np.where(a == 1, fb[1], fb[-1])            # a's group exists at b
        b_on_a = np.where(b == 1, foot3, foot2)             # b's group exists at a
        sw = diff & a_on_b & b_on_a
        both = diff & ~a_on_b & ~b_on_a
        one = diff & ~sw & ~both
        k = 0 if axis == 1 else 1
        edge_class[..., k] = np.where(sw, 1, np.where(one, 3, np.where(both, 2, 0)))
        edges["switch"] += int(sw.sum())
        edges["one_family_ends"] += int(one.sum())
        edges["both_families_end"] += int(both.sum())
    # The footprints include the fixed-start map, so "the fixed start never shows a group absent at that cell" holds by
    # construction. The meaningful check uses footprints from the other maps only (18 Oct 2026, after Codex Task 31).
    oth = np.array([g for n, g in groups.items() if n != args.default])
    o3, o2 = (oth == 1).any(0), (oth == -1).any(0)
    leave_default_out = {}
    for lab, m in (("other_maps_find_3group_only", o3 & ~o2), ("other_maps_find_2group_only", o2 & ~o3),
                   ("other_maps_find_both", o3 & o2), ("other_maps_find_neither", ~o3 & ~o2)):
        leave_default_out[lab] = {"cells": int(m.sum()), "fixed_start_3group": int(((dflt == 1) & m).sum()),
                                  "fixed_start_2group": int(((dflt == -1) & m).sum()),
                                  "fixed_start_chaos": int(((dflt == 2) & m).sum()),
                                  "fixed_start_no_period": int(((dflt == 0) & m).sum())}
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
        "edge_class_note": ("switch: each side's group was found on the other side by some map. one_family_ends and "
                            "both_families_end mean that one or both groups were NOT FOUND on the other side by any "
                            "map: a finite search, not a certified bifurcation or disappearance (Codex Task 31)."),
        "leave_default_out": leave_default_out,
        "median_settle_near_edges": float(np.median(stl[near & fixed])) if (near & fixed).any() else None,
        "median_settle_elsewhere": float(np.median(stl[~near & fixed])) if (~near & fixed).any() else None,
        "per_map_group_cells": {n: {"3group": int((g == 1).sum()), "2group": int((g == -1).sum()),
                                    "chaos": int((g == 2).sum()), "no_period": int((g == 0).sum())}
                                for n, g in groups.items()},
        "chaos_in_fixed_start_map": {"cells": int(chaos_default.sum()),
                                     "p_blank_histogram": {"bins": [0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 1.0],
                                                           "counts": np.histogram(pb_default[chaos_default],
                                                                                  bins=[0, 0.01, 0.03, 0.05, 0.08, 0.12,
                                                                                        0.2, 0.3, 1.0])[0].tolist()},
                                     "chaos_cells_inside_overlap": int((chaos_default & overlap).sum())},
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

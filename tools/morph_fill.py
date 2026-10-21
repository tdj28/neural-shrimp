"""Gap fill of the rotation morph (run V3): check the frames that the morphfill pods computed and install them beside
the V1b frames, never overwriting a file.

For each frame of the given pod runs (each run folder's result.json names its job arguments; the frames are read from
its extracted archive):
  1. completeness: all cells of the 296 x 512 grid, finite exponents, periods within the scan's range;
  2. arguments: the frame's scan arguments equal those of the V1b frames except the start states, the output folder
     and the direction (which must be morph-NNN of this frame); options added since V1b must be off;
  3. reproduction: where a V1b pod left a partial result of the same frame from the same start states, the cells it
     finished are compared, first bit for bit (exponents, periods, entropies, token loops), then by periods at
     periodic cells, chaotic labels and exponent differences.
With --install, scan.npz, scan.json and scan.png (and scan-partial.npz into an empty folder) are copied into
morph/frame-NNN, refusing any existing file, a note is appended to morph/EXTRACTION-NOTES.txt, and the receipt lists
the installed files' hashes.

Usage: python tools/morph_fill.py RUN-FOLDER... --out docs/morph-fill-receipt.json [--install]
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
MORPH = DATA / "explore" / "llm-soft-loop" / "pod64" / "morph"
SHAPE = (296, 512)
CHAOS = 1e-3                       # the scan's chaos level: exponent above it and no period
OFF = {"prefix": None, "return_period": 0, "start_vector": None, "weights": "native", "word_period": 0}
V1_PARTIALS = {80, 220}            # partial files left by the 13-14 Oct V1 attempt (other settings): not compared


def rel(path):
    """A data path as the repository cites it (../neural-shrimp-data/...)."""
    path = Path(path)
    return "../neural-shrimp-data/" + str(path.relative_to(DATA)) if path.is_relative_to(DATA) else str(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def loops(z, keys_json):
    """Token loop of every cell as a tuple (empty where the cell has no period)."""
    keys = [tuple(k) for k in json.loads(str(keys_json))]
    fam = np.asarray(z).ravel()
    return [keys[f] if f >= 0 else () for f in fam]


def frames_of(run):
    res = json.loads((run / "result.json").read_text())
    args = res["upload"]["jobset_args"].split()
    init, first, last, step = args[0], int(args[1]), int(args[2]), int(args[3])
    base = run / "extracted" if res.get("extract_clash") else DATA
    src = base / "explore" / "llm-soft-loop" / "pod64" / "morph"
    return res, init, list(range(first, last + 1, step)), src


def check_frame(k, src, ref_args, init_rel, first):
    d = src / f"frame-{k:03d}"
    out = {"frame": k, "source": rel(d)}
    if not (d / "scan.npz").exists() or not (d / "scan.json").exists():
        out["status"] = "missing"
        return out
    z = np.load(d / "scan.npz")
    js = json.loads((d / "scan.json").read_text())
    lyap, per, ent = z["lyapunov"], z["period"], z["entropy"]
    a = js["args"]
    expect_init = init_rel if k == first else f"llm-soft-loop/pod64/morph/frame-{k - 2:03d}"
    diff = {key: [ref_args[key], a.get(key)] for key in ref_args
            if key not in ("init_states", "out", "concept") and ref_args[key] != a.get(key)}
    added = {key: a[key] for key in a if key not in ref_args}
    out.update(cells=int(js["cells"]), shape=list(lyap.shape), finite=bool(np.isfinite(lyap).all()),
               period_range=[int(per.min()), int(per.max())], seconds=js["seconds"],
               chaotic_cells=int(((lyap > CHAOS) & (per == 0)).sum()),
               lyapunov_positive=int(js["lyapunov_positive"]), concept=a.get("concept"),
               init_states=a.get("init_states"), argument_differences=diff, added_options=added)
    ok = (lyap.shape == SHAPE and out["cells"] == SHAPE[0] * SHAPE[1] and out["finite"]
          and 0 <= per.min() and per.max() <= ref_args["max_period"] and not diff
          and a.get("concept") == f"morph-{k:03d}" and a.get("init_states") == expect_init
          and all(key in OFF and added[key] == OFF[key] for key in added))
    out["checks_pass"] = bool(ok)
    old = MORPH / f"frame-{k:03d}" / "scan-partial.npz"
    if old.exists() and not (MORPH / f"frame-{k:03d}" / "scan.npz").exists():
        if k in V1_PARTIALS:
            out["reproduction"] = {"partial": rel(old), "compared": False,
                                   "reason": "partial file of the 13-14 Oct V1 attempt (other settings)"}
        else:
            p = np.load(old)
            n = int(p["done"])
            grid_equal = bool(np.array_equal(p["log10_temperature"], z["log10_temperature"])
                              and np.array_equal(p["alpha"], z["alpha"]))
            lo, po, eo = p["lyapunov"][:n], p["period"][:n], p["entropy"][:n]
            ln, pn, en = lyap.ravel()[:n], per.ravel()[:n], ent.ravel()[:n]
            new_keys = json.dumps([f["token_ids"] for f in sorted(js["families"], key=lambda f: f["family"])])
            fo, fn = loops(p["family"][:n], p["family_keys"]), loops(z["family"].ravel()[:n], new_keys)
            periodic = po > 0
            cho, chn = (lo > CHAOS) & (po == 0), (ln > CHAOS) & (pn == 0)
            dl = np.abs(lo - ln)
            out["reproduction"] = {
                "partial": rel(old), "partial_sha256": sha256(old), "cells_compared": n, "grid_equal": grid_equal,
                "bitwise_lyapunov": bool(np.array_equal(lo, ln)), "bitwise_period": bool(np.array_equal(po, pn)),
                "bitwise_entropy": bool(np.array_equal(eo, en)), "token_loops_equal": fo == fn,
                "cells_lyapunov_differ": int((lo != ln).sum()), "cells_period_differ": int((po != pn).sum()),
                "periodic_cells": int(periodic.sum()),
                "periodic_cells_same_period": int((periodic & (po == pn)).sum()),
                "chaotic_cells_partial": int(cho.sum()), "chaotic_cells_new": int(chn.sum()),
                "chaotic_label_agree": int((cho == chn).sum()),
                "max_abs_lyapunov_difference": float(dl.max()), "median_abs_lyapunov_difference": float(np.median(dl))}
            r = out["reproduction"]
            # installed only when the partial cells are reproduced: bit for bit, or (another rounding) the same
            # period at nearly all periodic cells; otherwise the states or the direction would not be what they seem
            r["reproduced"] = bool(grid_equal and (r["bitwise_period"] or
                                                   r["periodic_cells_same_period"] >= 0.9 * max(r["periodic_cells"], 1)))
            out["checks_pass"] = bool(out["checks_pass"] and r["reproduced"])
    return out


def install(k, src):
    d, dst = src / f"frame-{k:03d}", MORPH / f"frame-{k:03d}"
    if d.resolve() == dst.resolve():                  # archive extracted in place (no clash): nothing to copy
        return {name: sha256(dst / name) for name in ("scan.npz", "scan.json", "scan.png")}
    dst.mkdir(exist_ok=True)
    names = ["scan.npz", "scan.json", "scan.png"]
    if not any(dst.iterdir()):
        names.append("scan-partial.npz")
    copied = {}
    for name in names:
        if (dst / name).exists():
            raise SystemExit(f"{dst / name} exists; nothing is overwritten")
    for name in names:
        shutil.copy2(d / name, dst / name)
        if sha256(dst / name) != sha256(d / name):
            raise SystemExit(f"copy of {d / name} differs")
        copied[name] = sha256(dst / name)
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="pod run folders (DATA/runpod/soft64/STAMP)")
    parser.add_argument("--out", required=True, help="receipt (JSON)")
    parser.add_argument("--install", action="store_true", help="copy checked frames into morph/frame-NNN")
    args = parser.parse_args()
    receipt = {"tool": "tools/morph_fill.py", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "runs": [], "frames": []}
    todo = []
    for run in map(Path, args.runs):
        res, init, frames, src = frames_of(run)
        tgz = run / "results.tgz"
        receipt["runs"].append({"run": run.name, "pod_id": res["pod_id"], "gpu_type": res["gpu_type"],
                                "commit": res["commit"], "jobset": res["jobset"], "jobset_args": res["upload"]["jobset_args"],
                                "data_sha256": res["upload"]["data_sha256"], "runner_returncode": res.get("runner_returncode"),
                                "hash_verified": res.get("hash_verified"), "results_sha256": res.get("local_sha256"),
                                "results_sha256_recomputed": sha256(tgz) if tgz.exists() else None,
                                "extract_clash": bool(res.get("extract_clash")), "wall_seconds": res.get("wall_seconds")})
        init_k = int(init.rstrip("/").rsplit("-", 1)[1])
        ref_args = json.loads((MORPH / f"frame-{init_k:03d}" / "scan.json").read_text())["args"]
        for k in frames:
            row = check_frame(k, src, ref_args, init, frames[0])
            row["run"] = run.name
            receipt["frames"].append(row)
            if row.get("checks_pass") and res.get("hash_verified"):
                todo.append((row, k, src))
    if args.install:
        for row, k, src in todo:
            row["installed"] = install(k, src)
        note = MORPH / "EXTRACTION-NOTES.txt"
        with open(note, "a") as handle:
            for row, k, src in todo:
                handle.write(f"{receipt['created_utc']} V3 fill (run {row['run']}): frame-{k:03d} files "
                             f"{', '.join(sorted(row['installed']))} copied from {rel(src)} without overwriting"
                             + ("; the scan-partial.npz beside them is the V1b (or V1) pod's unfinished attempt\n"
                                if "scan-partial.npz" not in row["installed"] else "\n"))
    receipt["frames"].sort(key=lambda r: r["frame"])
    receipt["summary"] = {"frames": len(receipt["frames"]),
                          "checks_pass": sum(bool(r.get("checks_pass")) for r in receipt["frames"]),
                          "installed": sum("installed" in r for r in receipt["frames"])}
    Path(args.out).write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt["summary"]))


if __name__ == "__main__":
    main()

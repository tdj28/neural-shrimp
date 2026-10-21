"""Gap fill of the rotation morph (run V3): check the frames that the morphfill pods computed and install them beside
the V1b frames, never overwriting a file.

Supply all six V3 pod runs. The 17 expected frames must occur exactly once. Before loading numerical outputs,
each archive must match its recorded remote and local hashes with hash_verified=true, and each extracted source
file must match its unique regular archive member. For every frame:
  1. completeness: all cells of the 296 x 512 grid, finite exponents, periods within the scan's range;
  2. arguments: the frame's scan arguments equal those of the V1b frames except the start states, the output folder
     and the direction (which must be morph-NNN of this frame); options added since V1b must be off;
  3. reproduction: where a V1b pod left a partial result of the same frame from the same start states, the cells it
     finished are compared, first bit for bit (exponents, periods, entropies, token loops), then by periods at
     periodic cells, chaotic labels and exponent differences.
With --install, scan.npz, scan.json and scan.png (and scan-partial.npz into an empty folder) are copied into
morph/frame-NNN, refusing any existing file, a note is appended to morph/EXTRACTION-NOTES.txt, and the receipt lists
the installed files' hashes. Every row and every copy destination is checked before the first copy. Failed or
incomplete checks return a nonzero status and retain a receipt; existing receipt paths are never overwritten.
Use different --out paths for the read-only check and the eventual installation.

Usage: python tools/morph_fill.py RUN-FOLDER... --out docs/morph-fill-receipt.json [--install]
"""
import argparse
from collections import Counter
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
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
CHAINS = {14: (16, 18), 76: (78, 80), 94: (96, 98), 190: (192, 194, 196, 198),
          210: (212, 214, 216, 218, 220), 234: (236, 238)}
EXPECTED_FRAMES = frozenset(k for chain in CHAINS.values() for k in chain)
FRAME_FILES = ("scan.npz", "scan.json", "scan.png")


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
    init_k = int(init.rstrip("/").rsplit("-", 1)[1])
    if (res.get("jobset") != "morphfill" or init_k not in CHAINS or step != 2
            or init != f"llm-soft-loop/morph-fill-init/frame-{init_k:03d}"
            or (first, last) != (CHAINS[init_k][0], CHAINS[init_k][-1])):
        raise ValueError("run does not describe one of the six V3 warm chains")
    base = run / "extracted" if res.get("extract_clash") else DATA
    src = base / "explore" / "llm-soft-loop" / "pod64" / "morph"
    return res, init, list(range(first, last + 1, step)), src


def archive_sources(run, frames, src):
    """Bind extracted files to unique regular members of the authenticated archive before loading arrays."""
    wanted = {f"explore/llm-soft-loop/pod64/morph/frame-{k:03d}/{name}": (k, name)
              for k in frames for name in (*FRAME_FILES, "scan-partial.npz")}
    bound = {k: {} for k in frames}
    with tarfile.open(run / "results.tgz") as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            k, name = wanted[member.name]
            if not member.isfile() or name in bound[k]:
                raise ValueError(f"non-regular or duplicate archive member: {member.name}")
            with archive.extractfile(member) as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            path = src / f"frame-{k:03d}" / name
            if not path.is_file() or path.is_symlink() or sha256(path) != digest:
                raise ValueError(f"extracted file missing or differs from archive: {member.name}")
            bound[k][name] = digest
    for k in frames:
        missing = set(FRAME_FILES) - bound[k].keys()
        if missing:
            raise ValueError(f"frame-{k:03d} missing archive files: {', '.join(sorted(missing))}")
    return bound


def check_frame(k, src, ref_args, init_rel, first, ref_grid):
    d = src / f"frame-{k:03d}"
    out = {"frame": k, "source": rel(d)}
    if not (d / "scan.npz").exists() or not (d / "scan.json").exists():
        out["status"] = "missing"
        return out
    with np.load(d / "scan.npz", allow_pickle=False) as loaded:
        z = {name: loaded[name] for name in ("lyapunov", "period", "entropy", "family",
                                           "log10_temperature", "alpha")}
    js = json.loads((d / "scan.json").read_text())
    lyap, per, ent, fam = (z[name] for name in ("lyapunov", "period", "entropy", "family"))
    shapes = {name: list(z[name].shape) for name in ("lyapunov", "period", "entropy", "family")}
    grid_equal = all(np.array_equal(z[name], ref_grid[name]) for name in ref_grid)
    period_valid = bool(np.issubdtype(per.dtype, np.integer) and per.size
                        and (per >= 0).all() and (per <= ref_args["max_period"]).all())
    family_ids = sorted(f["family"] for f in js["families"])
    family_valid = bool(np.issubdtype(fam.dtype, np.integer)
                        and family_ids == list(range(len(family_ids)))
                        and np.isin(fam, [-1, *family_ids]).all())
    shape_valid = all(tuple(shape) == SHAPE for shape in shapes.values())
    if not shape_valid or not period_valid or not family_valid:
        out.update(checks_pass=False, status="invalid_arrays", shapes=shapes, period_valid=period_valid,
                   family_valid=family_valid, grid_equal=grid_equal)
        return out
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
    out.update(shapes=shapes, entropy_finite=bool(np.isfinite(ent).all()), grid_equal=grid_equal,
               period_valid=period_valid, family_valid=family_valid)
    ok = (lyap.shape == SHAPE and out["cells"] == SHAPE[0] * SHAPE[1] and out["finite"]
          and out["entropy_finite"] and grid_equal
          and 0 <= per.min() and per.max() <= ref_args["max_period"] and not diff
          and a.get("concept") == f"morph-{k:03d}" and a.get("init_states") == expect_init
          and all(key in OFF and added[key] == OFF[key] for key in added))
    out["checks_pass"] = bool(ok)
    old = MORPH / f"frame-{k:03d}" / "scan-partial.npz"
    if old.exists() and old.resolve() != (d / "scan-partial.npz").resolve():
        if k in V1_PARTIALS:
            out["reproduction"] = {"partial": rel(old), "compared": False,
                                   "reason": "partial file of the 13-14 Oct V1 attempt (other settings)"}
        else:
            with np.load(old, allow_pickle=False) as loaded:
                p = {name: loaded[name] for name in loaded.files}
            n = int(p["done"])
            if not 0 < n <= SHAPE[0] * SHAPE[1]:
                raise ValueError(f"invalid completed-cell count in {old}")
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
            r["acceptance_rule"] = "equal grid and bitwise periods, or at least 90% of old periodic cells retain their period"
            r["periodic_agreement_threshold"] = 0.9
            # installed only when the partial cells are reproduced: bit for bit, or (another rounding) the same
            # period at nearly all periodic cells; otherwise the states or the direction would not be what they seem
            r["reproduced"] = bool(grid_equal and (r["bitwise_period"] or
                                                   r["periodic_cells_same_period"] >= 0.9 * max(r["periodic_cells"], 1)))
            out["checks_pass"] = bool(out["checks_pass"] and r["reproduced"])
    return out


def install_plan(k, src, source_hashes):
    """Preflight all files and collisions without changing any frame directory."""
    d, dst = src / f"frame-{k:03d}", MORPH / f"frame-{k:03d}"
    names = list(FRAME_FILES)
    same = d.resolve() == dst.resolve()
    if not same and (not dst.exists() or not any(dst.iterdir())):
        names.append("scan-partial.npz")
    for name in names:
        path = d / name
        if name not in source_hashes or not path.is_file() or sha256(path) != source_hashes[name]:
            raise ValueError(f"install source missing or changed: {path}")
        if not same and (dst / name).exists():
            raise ValueError(f"{dst / name} exists; nothing is overwritten")
    return names


def install(row, k, src, names):
    d, dst = src / f"frame-{k:03d}", MORPH / f"frame-{k:03d}"
    # Recheck bound source bytes immediately before copying; exclusive creation also protects against new collisions.
    install_plan(k, src, row["source_sha256"])
    row["installed"] = {}
    if d.resolve() == dst.resolve():
        row["installed"].update({name: row["source_sha256"][name] for name in names})
        return
    dst.mkdir(exist_ok=True)
    for name in names:
        with open(d / name, "rb") as source, open(dst / name, "xb") as target:
            shutil.copyfileobj(source, target)
        if sha256(dst / name) != row["source_sha256"][name]:
            raise ValueError(f"copy of {d / name} differs")
        row["installed"][name] = row["source_sha256"][name]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="all six V3 pod run folders (DATA/runpod/soft64/STAMP)")
    parser.add_argument("--out", required=True, help="new receipt path (existing receipts are never overwritten)")
    parser.add_argument("--install", action="store_true", help="install only if the complete 17-frame set passes")
    args = parser.parse_args(argv)
    output = Path(args.out)
    # Reserve the receipt before any installation. Refuse reuse so a failed audit remains available.
    with output.open("x") as handle:
        receipt = {"tool": "tools/morph_fill.py", "tool_sha256": sha256(Path(__file__)),
                   "tool_bytes": Path(__file__).stat().st_size,
                   "analysis_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                                     capture_output=True, text=True).stdout.strip(),
                   "analysis_worktree_dirty": bool(subprocess.run(
                       ["git", "status", "--porcelain"], cwd=ROOT, check=True,
                       capture_output=True, text=True).stdout.strip()),
                   "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "expected_frames": sorted(EXPECTED_FRAMES), "runs": [], "frames": [], "errors": []}
        todo = []
        for run in map(Path, args.runs):
            record = {"run": run.name, "source": rel(run), "checks_pass": False}
            receipt["runs"].append(record)
            frames = []
            try:
                res, init, frames, src = frames_of(run)
                record.update({key: res.get(key) for key in
                               ("pod_id", "gpu_type", "commit", "jobset", "runner_returncode", "hash_verified",
                                "remote_sha256", "wall_seconds")})
                record.update(jobset_args=res["upload"]["jobset_args"],
                              data_sha256=res["upload"]["data_sha256"],
                              result_json_sha256=sha256(run / "result.json"),
                              result_json_bytes=(run / "result.json").stat().st_size,
                              results_sha256=res.get("local_sha256"),
                              results_bytes=(run / "results.tgz").stat().st_size,
                              extract_clash=bool(res.get("extract_clash")),
                              results_sha256_recomputed=sha256(run / "results.tgz"))
                actual = record["results_sha256_recomputed"]
                if (res.get("hash_verified") is not True
                        or actual != res.get("local_sha256") or actual != res.get("remote_sha256")):
                    raise ValueError("archive must match both recorded hashes and hash_verified must be true")
                if res.get("runner_returncode") != 0:
                    raise ValueError("runner did not complete successfully")
                bound = archive_sources(run, frames, src)
                init_k = int(init.rstrip("/").rsplit("-", 1)[1])
                reference = MORPH / f"frame-{init_k:03d}"
                ref_args = json.loads((reference / "scan.json").read_text())["args"]
                with np.load(reference / "scan.npz", allow_pickle=False) as z:
                    ref_grid = {name: z[name] for name in ("log10_temperature", "alpha")}
                if (ref_grid["log10_temperature"].shape != (SHAPE[0],)
                        or ref_grid["alpha"].shape != (SHAPE[1],)
                        or not all(np.isfinite(grid).all() for grid in ref_grid.values())):
                    raise ValueError("reference grid is not finite or has the wrong dimensions")
                record["reference_sha256"] = {name: sha256(reference / name) for name in ("scan.json", "scan.npz")}
                record["reference_bytes"] = {name: (reference / name).stat().st_size for name in ("scan.json", "scan.npz")}
                record["checks_pass"] = True
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
                receipt["errors"].append(f"{run.name}: {record['error']}")
                receipt["frames"].extend({"frame": k, "run": run.name, "checks_pass": False,
                                          "status": "run_validation_failed", "error": record["error"]} for k in frames)
                continue
            for k in frames:
                try:
                    row = check_frame(k, src, ref_args, init, frames[0], ref_grid)
                except Exception as error:
                    row = {"frame": k, "checks_pass": False, "status": "invalid_frame",
                           "error": f"{type(error).__name__}: {error}"}
                row.update(run=run.name, source=rel(src / f"frame-{k:03d}"), source_sha256=bound[k])
                row["source_bytes"] = {name: (src / f"frame-{k:03d}" / name).stat().st_size for name in bound[k]}
                receipt["frames"].append(row)
                if row.get("checks_pass"):
                    todo.append((row, k, src))
        counts = Counter(row["frame"] for row in receipt["frames"])
        coverage = {"missing": sorted(EXPECTED_FRAMES - counts.keys()),
                    "unexpected": sorted(counts.keys() - EXPECTED_FRAMES),
                    "duplicates": sorted(k for k, count in counts.items() if count != 1)}
        receipt["coverage"] = coverage
        valid = (not receipt["errors"] and not any(coverage.values())
                 and len(todo) == len(EXPECTED_FRAMES))
        if not valid:
            receipt["errors"].append("the complete 17-frame V3 set did not pass; no frames installed")
        if args.install and valid:
            try:
                # Every row, required copy, and destination must pass before the first mutation.
                plans = [(row, k, src, install_plan(k, src, row["source_sha256"])) for row, k, src in todo]
                for row, k, src, names in plans:
                    install(row, k, src, names)
                with (MORPH / "EXTRACTION-NOTES.txt").open("a") as note:
                    for row, k, src, names in plans:
                        note.write(f"{receipt['created_utc']} V3 fill (run {row['run']}): frame-{k:03d} files "
                                   f"{', '.join(names)} copied from {rel(src)} without overwriting; "
                                   "existing partial evidence retained\n")
            except Exception as error:
                valid = False
                receipt["errors"].append(f"installation failed: {type(error).__name__}: {error}")
        receipt["frames"].sort(key=lambda row: row["frame"])
        receipt["summary"] = {"frames": len(receipt["frames"]), "checks_pass": len(todo),
                              "installed": sum(bool(row.get("installed")) for row in receipt["frames"]),
                              "all_checks_pass": valid}
        json.dump(receipt, handle, indent=1)
        handle.write("\n")
    print(json.dumps(receipt["summary"]))
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())

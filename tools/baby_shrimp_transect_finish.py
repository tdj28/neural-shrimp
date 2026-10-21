"""Audit every Jones-transect cell and show finite-cut window organization.

Recurrences remain candidates. Direction-sign words are compared with the MSS
catalogue without promoting an ambiguous last symbol to exact superstability.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "8"
import numpy as np
import baby_shrimp_finish as audit
from baby_shrimp_trajectories import recurrence
from baby_shrimp_returnmaps import phase_map
import symbolic_dynamics as sd


def stream_digest(handle):
    h = hashlib.sha256()
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def verify_tar_files(archive, expected):
    """Verify each requested member in one forward decompression pass."""
    seen, extras = set(), {}
    names = ("explore/llm-soft-loop/pod64/logs/hardware.txt", "explore/llm-soft-loop/pod64/logs/versions.txt")
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.name in expected:
                audit.require(member.isfile() and member.name not in seen, "nonregular or duplicate raw archive member")
                audit.require(stream_digest(tar.extractfile(member)) == audit.digest(expected[member.name]),
                              "local raw file differs from retrieved archive")
                seen.add(member.name)
            elif member.name in names:
                audit.require(member.isfile() and member.name not in extras, "invalid hardware archive member")
                extras[member.name] = tar.extractfile(member).read()
    audit.require(seen == set(expected), "raw file missing from archive")
    audit.require(set(extras) == set(names), "hardware or versions missing from archive")
    return extras[names[0]], extras[names[1]]


def verify_worker(worker):
    """Bind retrieved files to the archive and verify exact resource shutdown."""
    import runpod_pod as rp
    result = json.loads((worker / "result.json").read_text())
    archive = worker / "results.tgz"
    audit.require(result["hash_verified"] and audit.digest(archive) == result["local_sha256"]
                  == result["remote_sha256"], "archive hash mismatch")
    argv = shlex.split(result["upload"]["jobset_args"])
    folder = audit.DATA / "explore" / argv[argv.index("--out") + 1]
    expected = {str(p.relative_to(audit.DATA)): p for p in sorted(folder.rglob("*")) if p.is_file()}
    hardware, versions = verify_tar_files(archive, expected)
    files = [audit.artifact(p) for p in expected.values()]
    ledger = [json.loads(s) for s in rp.LEDGER.read_text().splitlines()]
    pid = result["pod_id"]
    audit.require(any(e.get("event") == "terminated_verified" and e.get("pod_id") == pid
                      for e in ledger), "termination record missing")
    audit.require(pid not in {p["id"] for p in rp.inventory()["pods"]}
                  and rp.pod_status(pid) is None, "pod remains present")
    evidence = {"pod_id": pid, "source_commit": result["commit"],
                "archive": audit.artifact(archive), "termination_verified": True,
                "runner_returncode": result["runner_returncode"],
                "hardware": hardware.decode().strip(),
                "hardware_sha256": hashlib.sha256(hardware).hexdigest(),
                "versions_sha256": hashlib.sha256(versions).hexdigest()}
    return result, folder, evidence, files


def word_record(cosines, period, text_ok):
    if not text_ok or not period or period % 6:
        return {"status": "ineligible text or recurrence", "eligible": False}
    k = period // 6
    attempts = []
    # The method is unchanged; repeated tail cycles check temporal consistency.
    for offset in (0, period, 2 * period):
        tail = cosines if offset == 0 else cosines[:-offset]
        word, status, details = sd.cell_word(tail, 6, k, .5)
        attempts.append({"word": word, "status": status, "details": details})
    eligible = all(r["status"] == "ok" for r in attempts)
    eligible = eligible and len({r["word"][:-1] for r in attempts}) == 1
    word = attempts[0]["word"] if eligible else None
    W = word[:-1] if word is not None else None
    catalogue = sd.mss_words(7)
    return {"status": "consistent phase-majority window word" if eligible else "word gate failed",
            "eligible": eligible, "return_period": k, "word": word, "window_word": W,
            "last_symbol": word[-1] if word else None,
            "last_symbol_scope": "C denotes split phase votes, not a proven critical orbit or zero multiplier",
            "mss_catalogue_through_7_match": bool(eligible and k <= 7 and W in catalogue),
            "mss_rank": catalogue.index(W) if eligible and k <= 7 and W in catalogue else None,
            "tail_cycles": attempts}


def rank_order(ranks):
    """State both possible orientations; never infer order from an empty list."""
    if len(ranks) < 2:
        return {"status": "fewer than two distinct sampled windows", "increasing": None, "decreasing": None}
    return {"status": "sampled ranks only", "increasing": all(a <= b for a, b in zip(ranks, ranks[1:])),
            "decreasing": all(a >= b for a, b in zip(ranks, ranks[1:]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result, folder, worker, manifest = verify_worker(args.worker)
    status = json.loads((folder / "status.json").read_text())
    info = json.loads((folder / "trajectories.json").read_text())
    protocol = json.loads((folder / "protocol.json").read_text())
    audit.require(status["status"] == "complete" and result["runner_returncode"] == 0,
                  "transect worker not complete")
    plan_path = "docs/baby-shrimp-zoom/jones-transect-plan.json"
    blob = subprocess.check_output(["git", "show", result["commit"] + ":" + plan_path], cwd=audit.ROOT)
    plan = json.loads(blob)
    audit.require(hashlib.sha256(blob).hexdigest() == status["plan_sha256"]
                  and status["source_commit"] == result["commit"], "plan/source binding differs")
    audit.require(info["protocol"] == protocol and protocol["cells"] == plan["cells"]
                  and protocol["record"] == protocol["transient"] == 3000
                  and len(info["cells"]) == 65 and protocol["model_revision"] == plan["model_revision"]
                  and protocol["rotary_float64"] and protocol["dtype"] == "torch.float64"
                  and 0 < protocol["norm_sensitivity_to_1e_12"] < 1e-9, "protocol mismatch")
    states = np.load(folder / "states.npy", mmap_mode="r", allow_pickle=False)
    with np.load(folder / "diagnostics.npz", allow_pickle=False) as z:
        diag = {k: z[k] for k in z.files}
    audit.require(states.shape == (3000, 65, 896) and states.dtype == np.float64,
                  "state record shape/precision mismatch")
    audit.require(all(a.shape == (3000, 65) for a in diag.values())
                  and np.isfinite(diag["log_growth"]).all() and np.isfinite(diag["word_cos"][6:]).all(),
                  "diagnostic record incomplete")
    reference_receipt = json.loads((audit.ROOT / "docs/baby-shrimp-zoom/probe-receipt.json").read_text())
    reference_path = args.reference / "states.npy"
    audit.require(any(f["path"] == str(reference_path.relative_to(audit.DATA))
                      and f["sha256"] == audit.digest(reference_path) for f in reference_receipt["files"]),
                  "reference state file not bound by prior audit")
    reference_info = json.loads((args.reference / "trajectories.json").read_text())
    index = next(i for i, c in enumerate(reference_info["cells"]) if c["name"] == "baby-chaos-below")
    R = np.array(np.load(reference_path, mmap_mode="r", allow_pickle=False)[::6, index])
    mean = R.mean(0)
    _, _, vt = np.linalg.svd(R - mean, full_matrices=False)
    axis = vt[0]
    if axis[np.argmax(abs(axis))] < 0:
        axis *= -1
    span = float(np.ptp((R - mean) @ axis))
    rows, cloud, windows = [], [], []
    for i, cell in enumerate(info["cells"]):
        audit.require(all(cell[k] == plan["cells"][i][k] for k in plan["cells"][i]), "cell parameters differ")
        S = np.array(states[:, i])
        rec = recurrence(S)
        audit.require(rec["period_candidate"] == cell["period_candidate"]
                      and np.allclose(rec["max_relative_return_by_lag"], cell["max_relative_return_by_lag"], rtol=1e-12, atol=1e-15),
                      "raw recurrence differs")
        growth = diag["log_growth"][:, i]
        halves = [float(v.mean()) for v in np.array_split(growth, 2)]
        audit.require(float(growth.mean()) == cell["lyapunov"] and halves == cell["lyapunov_halves"],
                      "finite-time exponent differs")
        tokens = diag["tokens"][:, i]
        text_period = next((q for q in range(1, 193) if np.array_equal(tokens[q:], tokens[:-q])), 0)
        audit.require(text_period == cell["text_period"], "text recurrence differs")
        text_ok = text_period == 6 and sd.rotmin_key(tokens[:6]) == sd.rotmin_key([271, 785, 3405, 271, 785, 13885])
        if text_period:
            audit.require(tokens[:text_period].tolist() == cell["text_token_ids"], "text loop summary differs")
        nonperiodic = rec["period_candidate"] == 0 and min(halves) > .001
        phases = []
        for phase in range(6):
            p, x, y, colors = phase_map(S, diag["word_cos"][:, i], phase)
            if not nonperiodic:
                p["robust_turning_points"] = None
            phases.append(p)
        usable = [p for p in phases if p.get("usable")]
        pooled = sum(p["errors"] for p in usable) / sum(p["minority"] for p in usable) if usable else None
        word = word_record(diag["word_cos"][:, i], rec["period_candidate"], text_ok)
        distances = [float(np.linalg.norm(S[p::6].mean(0) - mean)) for p in range(6)]
        ordered = np.argsort(distances)
        alignment = distances[ordered[0]] < .1 * distances[ordered[1]]
        phase = int(ordered[0])
        projection = ((S[phase::6] - mean) @ axis) / span
        cloud.append(projection if alignment else np.full(500, np.nan))
        row = {"index": i, **cell, "same_cyclic_text": text_ok, "phases": phases,
               "nonperiodic_expanding_both_halves": nonperiodic, "usable_phases": len(usable),
               "pooled_errors_over_minority": pooled,
               "mother_coding_gate_passed": bool(text_ok and nonperiodic and usable and pooled < .05),
               "direction_word": word, "projection_phase": phase, "projection_alignment_passed": bool(alignment),
               "phase_mean_distances_to_reference": distances,
               "verified_primitive_orbit": False}
        rows.append(row)
        if word.get("mss_rank") is not None:
            if windows and windows[-1]["last_index"] == i - 1 and windows[-1]["word"] == word["window_word"]:
                windows[-1].update(last_index=i, last_alpha=cell["alpha"])
            else:
                windows.append({"word": word["window_word"], "return_period": word["return_period"],
                                "rank": word["mss_rank"], "first_index": i, "last_index": i,
                                "first_alpha": cell["alpha"], "last_alpha": cell["alpha"]})
        print(json.dumps({"cell": i, "period_candidate": rec["period_candidate"],
                          "word": word.get("word"), "coding_passed": row["mother_coding_gate_passed"]}), flush=True)
    alpha = np.array([r["alpha"] for r in rows])
    cloud = np.asarray(cloud)
    np.savez_compressed(args.out / "projection.npz", alpha=alpha, coordinates=cloud,
                        reference_mean=mean, reference_axis=axis, reference_span=span)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1.1]}, layout="constrained")
    palette = {18: "#d09318", 36: "#315bc4", 72: "#a444aa", 144: "#00878a"}
    for i, row in enumerate(rows):
        axes[0].scatter(np.full(500, alpha[i]), cloud[i], s=1.6, linewidths=0, alpha=.4,
                        color=palette.get(row["period_candidate"], "#526079"))
    axes[0].set(ylabel="Position along a fixed state-space axis", title="The same six-token text hides different state cycles")
    for period, color in palette.items():
        axes[0].scatter([], [], color=color, s=25, label=f"{period} steps")
    axes[0].legend(loc="upper right", ncol=4, fontsize=9, frameon=False)
    lyap = np.array([r["lyapunov"] for r in rows])
    halves = np.array([r["lyapunov_halves"] for r in rows])
    axes[1].fill_between(alpha, halves.min(1), halves.max(1), color="#c6ced9", alpha=.8,
                         label="Difference between recorded halves")
    axes[1].plot(alpha, lyap, "o-", ms=3, lw=1, color="#26334d")
    axes[1].axhline(0, color="#8791a2", lw=.8)
    axes[1].set(ylabel="Finite-time exponent\n(per model step)")
    axes[1].legend(fontsize=8, loc="lower left", frameon=False)
    per = np.array([r["period_candidate"] / 6 for r in rows])
    axes[2].scatter(alpha, per, c=[palette.get(r["period_candidate"], "#526079") for r in rows], s=27)
    for n, w in enumerate(windows):
        center = (w["first_alpha"] + w["last_alpha"]) / 2
        label = (w["word"] or "empty") + f"  (k={w['return_period']})"
        axes[2].annotate(label, (center, w["return_period"]), (0, 13 + (n % 2) * 13),
                         textcoords="offset points", ha="center", fontsize=8,
                         arrowprops={"arrowstyle": "-", "lw": .5, "color": "#8791a2"})
    axes[2].set(xlabel="Steering strength", ylabel="Candidate return period k\n(one return = six steps)",
                ylim=(-2, max(per.max() + 7, 15)))
    axes[2].text(.01, .94, "0: no recurrence detected through 192 steps", transform=axes[2].transAxes, fontsize=8)
    axes[2].xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    fig.suptitle("A fixed cut through the baby window", fontsize=23, fontweight="bold")
    fig.supxlabel("65 fixed parameter cells · 3000 settling + 3000 recorded steps · one common, phase-aligned projection\n"
                  "Labels are consistent derivative-sign window words matching the quadratic MSS catalogue through k=7.\n"
                  "These are recurrence candidates, not newly Newton-verified orbits; the cut does not establish global MSS order.", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"jones-transect.{ext}", dpi=170)
    plt.close(fig)
    summary = {"cells": len(rows), "recurrence_counts": dict(Counter(r["period_candidate"] for r in rows)),
               "same_text_cells": sum(r["same_cyclic_text"] for r in rows),
               "expanding_nonperiodic_both_halves": sum(r["nonperiodic_expanding_both_halves"] for r in rows),
               "mother_coding_gate_passed": sum(r["mother_coding_gate_passed"] for r in rows),
               "consistent_direction_word_cells": sum(r["direction_word"]["eligible"] for r in rows),
               "mss_match_cells": sum(r["direction_word"].get("mss_catalogue_through_7_match", False) for r in rows),
               "windows": windows, "sampled_rank_order": rank_order([w["rank"] for w in windows]),
               "phase_alignment_passed": sum(r["projection_alignment_passed"] for r in rows)}
    (args.out / "rows.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    receipt = {"id": "BSJ1-transect-audit", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "analysis_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=audit.ROOT, text=True).strip(),
               "source_sha256": audit.digest(__file__), "worker": worker, "plan_sha256": status["plan_sha256"],
               "reference": audit.artifact(reference_path), "summary": summary,
               "files": [audit.artifact(p) for p in args.out.iterdir() if p.is_file()],
               "limits": "Finite exploratory cut with recurrence candidates. Direction-sign words do not verify critical-point itineraries, exact superstability, global conjugacy, a complete window inventory or a homoclinic periodicity hub. Phase selection for the common projection uses proximity to the already fixed lower-neighbor phase; ambiguous alignments are masked. C in a final word denotes split phase votes only.",
               "cost": "see private ledger"}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

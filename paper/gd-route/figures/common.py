"""Shared constants and provenance helpers for the gd-route manuscript figures.

Every figure script reads only saved evidence (frozen EXP-028 files or saved
exploratory outputs) unless it states otherwise, writes ``<name>.pdf`` and
``<name>.png`` next to itself, and records a ``<name>.provenance.json`` with
the SHA-256 of every input file, the script's own hash and the git commit.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))
RAW028 = DATA / "raw/EXP-028/attempt-002"
EXPLORE = DATA / "explore"

# Repository palette (validated categorical slots 1-3; see tools/plot_a2_overlay.py).
FOLD, FLIP, NS = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, MUTED, NEUTRAL, SECONDARY = "#fcfcfb", "#0b0b0b", "#898781", "#f0efec", "#52514e"
TYPE_COLORS = {"fold": FOLD, "flip": FLIP, "neimark_sacker": NS}
TYPE_MARKERS = {"fold": "o", "flip": "^", "neimark_sacker": "s"}
TYPE_LABELS = {"fold": "fold (+1)", "flip": "flip (-1)", "neimark_sacker": "complex pair (Neimark-Sacker candidate)"}
# Period ramp (sequential, one hue light to dark), as in tools/plot_a3_ns_tongues.py.
PERIOD_RAMP = ["#b7d3f6", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
               "#184f95", "#104281", "#0d366b"]
# EXP-028 dynamical-class colors (tools/analyze_exp028.py).
STATUS = {"divergent": 0, "fixed": 1, "periodic": 2, "chaotic": 3, "unresolved": 4, "drift": 5}
CLASS_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 0: "#52514e", 4: "#e1e0d9", 5: "#a3a29b"}
CLASS_LABELS = {1: "fixed point", 2: "periodic (p >= 2)", 3: "chaotic", 0: "divergent",
                4: "unresolved", 5: "drift (norm growth)"}


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    })


def portable(path):
    """Checkout-independent input label: repo:<relative> or data:<relative>."""
    resolved = Path(path).resolve()
    for prefix, base in (("repo:", ROOT), ("data:", DATA)):
        try:
            return prefix + str(resolved.relative_to(Path(base).resolve()))
        except ValueError:
            continue
    return str(resolved)


def located(label):
    """Filesystem path for a portable input label (absolute labels pass through)."""
    for prefix, base in (("repo:", ROOT), ("data:", DATA)):
        if str(label).startswith(prefix):
            return Path(base) / str(label)[len(prefix):]
    return Path(label)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip())
    return head, dirty


VERIFY = "--verify" in sys.argv


def _write(fig, directory, name):
    pdf, png = Path(directory) / f"{name}.pdf", Path(directory) / f"{name}.png"
    # Fixed metadata keeps regeneration byte-identical, which --verify relies on.
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=200, metadata={"Software": None})
    return pdf, png


def save(fig, name, inputs, script, evidence, alt, derived=None, notes=""):
    """Write PDF, PNG and the figure's provenance receipt, or verify them with --verify.

    The receipt holds the evidence label, alt text (the figure's text
    equivalent), every input with its SHA-256, the generator's hash, output
    hashes and the derived values the figure displays. In verify mode the
    figure is regenerated in a temporary directory and the run fails unless the
    inputs, generator, derived values and both output files match the receipt
    byte for byte.
    """
    import tempfile

    receipt_path = HERE / f"{name}.provenance.json"
    inputs = [Path(p) for p in inputs]
    derived = json.loads(json.dumps(derived or {}, default=float))
    if VERIFY:
        recorded = json.loads(receipt_path.read_text())
        problems = []
        for entry in recorded["inputs"]:
            if sha256(located(entry["path"])) != entry["sha256"]:
                problems.append(f"input changed: {entry['path']}")
        if sha256(script) != recorded["script_sha256"]:
            problems.append("generator changed since the receipt was written")
        if derived != recorded["derived"]:
            problems.append("derived values differ from the receipt")
        with tempfile.TemporaryDirectory() as tmp:
            pdf, png = _write(fig, tmp, name)
            for kind, path in (("pdf", pdf), ("png", png)):
                if sha256(path) != recorded["outputs"][f"{kind}_sha256"]:
                    problems.append(f"regenerated {kind} differs from the receipt")
                if sha256(HERE / path.name) != recorded["outputs"][f"{kind}_sha256"]:
                    problems.append(f"{kind} on disk differs from the receipt")
        plt.close(fig)
        if problems:
            raise SystemExit(f"{name}: VERIFY FAILED: " + "; ".join(problems))
        print(json.dumps({"figure": name, "verified": True}))
        return
    pdf, png = _write(fig, HERE, name)
    head, dirty = git_commit()
    record = {
        "figure": name, "evidence_label": evidence, "alt_text": alt, "notes": notes,
        "script": str(Path(script).resolve().relative_to(ROOT)), "script_sha256": sha256(script),
        "git_commit_at_generation": head, "worktree_dirty_at_generation": dirty,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "inputs": [{"path": portable(p), "sha256": sha256(p), "bytes": Path(p).stat().st_size} for p in inputs],
        "derived": derived,
        "accessibility": {"color_only_channel": False,
                          "note": "boundary types also differ by marker shape; classes and series are labeled"},
        "outputs": {"pdf": pdf.name, "pdf_sha256": sha256(pdf), "png": png.name, "png_sha256": sha256(png)},
    }
    receipt_path.write_text(json.dumps(record, indent=1) + "\n")
    plt.close(fig)
    print(json.dumps({"figure": name, "pdf": str(pdf)}))


def confirmed_windows():
    """Windows qualifying in both the EXP-028 frozen analysis and its independent audit."""
    summary = json.loads((ROOT / "docs/EXP-028-summary.json").read_text())
    audit = json.loads((DATA / "audits/EXP-028/validation-001/audit.json").read_text())
    producer = {(q["map_id"], q["component"]) for q in summary["gate"]["qualifying"]}
    audited = {(q["map_id"], q["component"]) for q in audit["audit_gate"]["qualifying"]}
    return sorted(producer & audited)


def ns_rows(path):
    rows = [r for r in json.loads(Path(path).read_text())["rows"] if r.get("type") == "neimark_sacker"]
    return sorted(rows, key=lambda r: r["log10_output"])


def ns_segments(path):
    """Runs of consecutive traced slices that lost stability by Neimark-Sacker (gaps break runs)."""
    rows = sorted(json.loads(Path(path).read_text())["rows"], key=lambda r: r["log10_output"])
    runs, current = [], []
    for r in rows:
        if r.get("type") == "neimark_sacker":
            current.append(r)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def ns_roots(runs, max_den=13, min_den=3):
    """Points of the curve where the crossing angle theta / 2 pi equals m / q (linear interpolation)."""
    from fractions import Fraction

    found = []
    for run in runs:
        for a, b in zip(run[:-1], run[1:]):
            lo_r, hi_r = sorted((a["theta_over_2pi"], b["theta_over_2pi"]))
            for q in range(min_den, max_den + 1):
                for m in range(1, q):
                    f = Fraction(m, q)
                    if f.denominator != q or not (lo_r < float(f) <= hi_r):
                        continue
                    t = (float(f) - a["theta_over_2pi"]) / (b["theta_over_2pi"] - a["theta_over_2pi"])
                    found.append({"fraction": f,
                                  "log10_output": a["log10_output"] + t * (b["log10_output"] - a["log10_output"]),
                                  "log10_hidden": a["log10_hidden_boundary"]
                                  + t * (b["log10_hidden_boundary"] - a["log10_hidden_boundary"])})
    return found


def plateaus(rows, min_cells=3, tol=6e-4):
    """Maximal runs of consecutive periodic cells locked at rotation m/q with period 2q.

    ``rows`` are staircase rows (log10_varied, rho, status, period). A cell is
    locked when it is periodic (status 2) with even period p >= 4 and its
    rotation number is within ``tol`` of m / q with p = 2 q 2^k (k >= 0, so
    period-doubled orbits inside the m/q tongue count) and gcd(m, q) = 1.
    The default tolerance exceeds the estimator's resolution for a periodic
    orbit (at most one turn over the 1,999 two-step samples of a staircase
    cell) and is below half the spacing 1 / (q q') of distinct rationals with
    denominators up to 29.
    """
    from math import gcd

    def lock(r):
        p = r["period"]
        if r["status"] != 2 or p < 4 or p % 2 or r["rho"] is None:
            return None
        big_q = p // 2
        m = round(r["rho"] * big_q)
        if m <= 0 or abs(r["rho"] - m / big_q) > tol:
            return None
        g = gcd(m, big_q)
        # rotation m/q with period 2q (g = 1) or a period-doubled orbit of it (period 2q 2^k, g = 2^k)
        return (m // g, big_q // g) if g & (g - 1) == 0 else None

    out, current = [], None
    for r in rows:
        key = lock(r)
        if current and key == current["key"]:
            current["end"] = r["log10_varied"]
            current["cells"] += 1
            continue
        if current and current["cells"] >= min_cells:
            out.append(current)
        current = {"key": key, "start": r["log10_varied"], "end": r["log10_varied"], "cells": 1} if key else None
    if current and current["cells"] >= min_cells:
        out.append(current)
    for p in out:
        p["m"], p["q"] = p.pop("key")
        p["base_period"] = 2 * p["q"]
    return out


def merged_plateaus(rows, merge_gap=0.003, **options):
    """Plateaus of the same m/q separated by at most ``merge_gap`` (log10 rate) merged into one span."""
    merged = []
    options.setdefault("min_cells", 1)
    for p in sorted(plateaus(rows, **options), key=lambda p: p["start"]):
        same = [m for m in merged if (m["m"], m["q"]) == (p["m"], p["q"]) and p["start"] - m["end"] <= merge_gap]
        if same:
            same[-1]["end"] = p["end"]
            same[-1]["cells"] += p["cells"]
            same[-1]["runs"] += 1
        else:
            merged.append({**p, "runs": 1})
    return sorted(merged, key=lambda p: p["start"])


def quasi_periodic(r, chaos=5e-3):
    """Non-periodic, non-divergent cell with |FTLE| below the EXP-028 chaos threshold."""
    return r["status"] in (4, 5) and r["ftle"] is not None and abs(r["ftle"]) < chaos and r["rho"] is not None

"""Readable BSC6 summary from the existing strictly validated audit receipt.

This separate renderer does not change the collector's frozen plotter. Every
target, actual trace and conditional feature comes from its load_receipt API.
Scientific rendering requires both this source and that dependency committed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

import plot_baby_shrimp_mixed_junction as original

ROOT = original.ROOT
SOURCE = "tools/plot_baby_shrimp_mixed_junction_summary.py"
SOURCES = [SOURCE, "tools/plot_baby_shrimp_mixed_junction.py"]
SCOPE = original.SCOPE
LABELS = {
    "resolved_positive": "Resolved positive",
    "resolved_negative": "Resolved negative",
    "residual_qualified_near_zero": "Small nominal residual",
    "unresolved_sign": "Unresolved sign",
    "not_admitted": "Not admitted",
    "unmeasured": "Unmeasured",
    "diagnostic only; admission incomplete": "Diagnostic only",
}


def frozen_sources():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = {path: original.digest(ROOT/path) for path in SOURCES}
    for path, sha in sources.items():
        original.require(hashlib.sha256(original.git_bytes(commit, path)).hexdigest() == sha,
                         "commit summary renderer and validation dependency before scientific rendering: "+path)
    return commit, sources


def build_summary(receipt, data, *, mobile=False, synthetic=False):
    """Lay out already validated values; direct use is for synthetic QA only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    cfg, rows = data["plan"]["mixed_junction"], data["rows"]
    width, height = (4.8, 11.8) if mobile else (6.5, 10.4)
    with plt.rc_context({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(width, height), facecolor="white")
        grid = fig.add_gridspec(3, 1, height_ratios=[2.8, 2.7, 1.5],
                               left=.19 if mobile else .16, right=.96,
                               top=.84, bottom=.22, hspace=.72)
        feature, trace, table = [fig.add_subplot(cell) for cell in grid]
        for ax in (feature, trace):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.16)
            ax.axhline(0, color=".45", lw=.8, zorder=0)
            ax.set_xlim(-4.4, 4.4); ax.set_xticks([-3, 0, 3])
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.set_xlabel("Memory offset (×10⁻⁹)")
        feature.set_title("A  Conditional Hessian feature", loc="left", pad=12)
        feature.set_ylabel("f₋ (×10⁻⁷)")
        trace.set_title("B  Actual signed trace residuals", loc="left", pad=12)
        trace.set_ylabel("τ (×10⁻¹²)")
        phase_markers = {0: "o", 6: "s", 12: "^"}
        measured_traces = []
        for row in rows:
            i, x = row["case_index"], row["beta_offset"]*1e9
            color = original.COLORS[i]
            if row["feature"] is not None:
                error = None if row["feature_error"] is None else row["feature_error"]*1e7
                artist = feature.errorbar(x, row["feature"]*1e7, yerr=error,
                    fmt="o" if row["admitted"] else "D", color=color, ms=8,
                    mfc=color if row["admitted"] else "white", capsize=5, ls="none", zorder=3)
                artist.lines[0].set_gid(f"feature-case-{i}")
            else:
                feature.text(x, .12, "Unmeasured", transform=feature.get_xaxis_transform(),
                             ha="center", fontsize=9, rotation=90 if mobile else 0, color=".4")
            traces = row["actual_traces"]
            for j, value in enumerate(traces):
                y = value["tau"]*1e12; measured_traces.append(y)
                marker = "x" if value["source"] == "primary center" else phase_markers[value["phase"]]
                artist = trace.scatter(x+(j-(len(traces)-1)/2)*.16, y,
                                       marker=marker, color=color, s=43, zorder=3)
                artist.set_gid(f"trace-case-{i}-sample-{j}")
            if not traces:
                trace.text(x, .12, "Unmeasured", transform=trace.get_xaxis_transform(),
                           ha="center", fontsize=9, rotation=90 if mobile else 0, color=".4")
        if not any(row["feature"] is not None for row in rows):
            feature.set_yticks([])
        # Limits follow actual measurements, so a wide admission gate cannot
        # flatten small, nonzero residuals into an apparent exact zero.
        extent = max(.02, max((abs(v) for v in measured_traces), default=0.)*1.35)
        trace.set_ylim(-extent, extent)
        trace.legend(handles=[Line2D([], [], marker=m, color=".35", ls="none", label=label)
            for m, label in (("x", "Primary"), ("o", "Phase 0"), ("s", "Phase 6"), ("^", "Phase 12"))],
            ncol=2, frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(.5, -.32))
        table.axis("off")
        table_rows = []
        for row in sorted(rows, key=lambda r: r["beta_offset"]):
            label = LABELS.get(row["feature_status"])
            original.require(label is not None, "unknown validated feature status")
            table_rows.append([f"{row['beta_offset']*1e9:+g}",
                               textwrap.fill(row["status"], 14 if mobile else 20),
                               textwrap.fill(label, 18 if mobile else 24)])
        tab = table.table(cellText=table_rows,
                          colLabels=["Offset\n×10⁻⁹", "Case status", "Feature status"],
                          colWidths=[.17, .35, .48], cellLoc="left", bbox=[-.08, -.1, 1.08, 1.15])
        tab.auto_set_font_size(False); tab.set_fontsize(9.5 if mobile else 10)
        for (r, c), cell in tab.get_celld().items():
            cell.set_edgecolor("#e2e5e8"); cell.set_linewidth(.5); cell.PAD=.06
            if r == 0: cell.set_facecolor("#f0f3f5"); cell.set_text_props(weight="bold")
        title = "Conditional contraction-feature test"
        if mobile: title = "Conditional contraction-\nfeature test"
        fig.suptitle(title, x=.08, y=.985, ha="left", fontsize=15 if mobile else 16)
        status = "All three targets admitted" if receipt["complete"] else "Audited partial pilot"
        if synthetic: status = "SYNTHETIC QA · "+status
        fig.text(.08, .92, status, fontsize=10 if synthetic else 11, weight="bold")
        fig.text(.08, .895,
                 f"Offsets relative to β = {cfg['target']['beta']:.15f}", fontsize=9.5, color=".35")
        gate = cfg["trace_tolerance"]
        caption = ("Bars: empirical feature + location envelope, not confidence intervals. "
                   "Filled dots: admitted; open diamonds: diagnostic only. "
                   f"Trace gate: |τ| ≤ {gate:.0e}. Horizontal jitter separates phase markers only. "
                   "Targets are not joined. Conditional Hessian features do not certify an exact zero or a contraction-valley crossing.")
        fig.text(.08, .16, textwrap.fill(caption, 57 if mobile else 80), fontsize=10, va="top", linespacing=1.35)
        inventory = (f"Retained: {len(data['pending'])} incomplete/unqualified attempts; "
                     f"{len(data['untested_or_incomplete_stages'])} incomplete stages. "
                     +("Terminal failure retained." if data["terminal_failure"] else "No terminal failure recorded."))
        fig.text(.08, .035, textwrap.fill(inventory, 57 if mobile else 80), fontsize=9, va="bottom", color=".4")
        return fig


def render(receipt, data, out, *, synthetic=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    for mobile in (False, True):
        fig = build_summary(receipt, data, mobile=mobile, synthetic=synthetic)
        name = "mixed-junction-mobile-summary" if mobile else "mixed-junction-readable-summary"
        for ext in (("png",) if mobile else ("png", "pdf")):
            fig.savefig(out/f"{name}.{ext}", dpi=200, facecolor="white")
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    commit, sources = frozen_sources()
    receipt, data = original.load_receipt(args.receipt, args.receipt_sha256, allow_partial=args.allow_partial)
    out, raw = args.out.resolve(), Path(receipt["raw_directory"]).resolve()
    original.require(not out.exists() and not out.is_relative_to(ROOT.resolve())
                     and not out.is_relative_to(raw) and not raw.is_relative_to(out),
                     "new output outside repository and raw run required")
    out.mkdir(parents=True); render(receipt, data, out)
    result = {"id": "BSC6-mixed-junction-readable-summary", "receipt": str(args.receipt.resolve()),
              "receipt_sha256": args.receipt_sha256, "source": SOURCE, "source_commit": commit,
              "source_sha256": sources[SOURCE], "source_hashes": sources, "threads": 1,
              "scientific_complete": receipt["complete"], "partial_explicitly_allowed": args.allow_partial,
              "displayed_cases": data["rows"], "unqualified_or_incomplete_points": data["pending"],
              "point_admissions": receipt["point_admissions"],
              "untested_or_incomplete_stages": data["untested_or_incomplete_stages"],
              "terminal_failure": data["terminal_failure"], "scope": SCOPE,
              "trace_axis": "Signed measured residuals; data-dependent limits; no trace projection or zero substitution.",
              "feature_bars": "Empirical pointwise + intersection-location envelope; not confidence intervals.",
              "visual_review": "required; successful rendering is not visual approval",
              "figures": [{"path": p.name, "bytes": p.stat().st_size, "sha256": original.digest(p)}
                          for p in sorted(out.iterdir()) if p.is_file()]}
    (out/"receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"out": str(out), "receipt_sha256": original.digest(out/"receipt.json")}))


if __name__ == "__main__":
    main()

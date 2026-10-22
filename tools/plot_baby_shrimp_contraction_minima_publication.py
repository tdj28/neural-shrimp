"""Publication layout of the audited BSC5 center zoom; no numerical recomputation.

The scientific artists are constructed by the diagnostic plotter unchanged.
Only layout, typography, and diagnostic metadata placement differ. The receipt
records the caption semantics that must accompany this compact figure.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import plot_baby_shrimp_contraction_minima as diagnostic

ROOT = diagnostic.ROOT
SOURCE = "tools/plot_baby_shrimp_contraction_minima_publication.py"
SOURCES = (SOURCE, diagnostic.SOURCE)


def known_diagnostic_labels(receipt, data, *, synthetic=False):
    """Exact metadata contract; a new diagnostic warning must never be hidden."""
    distance = diagnostic.center_zoom_data(data)["distance"]
    sx, sy = data["plan"]["coordinate_scales"]
    status = "Audited innermost sections" if receipt["complete"] else "PARTIAL / UNRESOLVED — missing sections remain untested"
    return [
        "Section-minima pilot — center zoom"+(" — SYNTHETIC QA" if synthetic else ""),
        f"{status}. Planned longitudinal distance = {distance:g}.\n"
        "Equal axes in the declared scaled Euclidean metric; the display includes the full section supports.",
        f"Scaled coordinates: (steering − center) / {sx:g} and (log₁₀ temperature − center) / {sy:g}; both panels show ×10⁻⁴ units.\n"
        "Black bars are measured final bracket widths, not confidence intervals. Red × / + retain unadmitted samples / unfinished callbacks.\n"
        "No bracket means no strict minimum bracket on the sampled support. These sections do not establish connected valleys or an exact crossing.",
    ]


def caption_semantics(receipt, data):
    """Disclosures moved from the diagnostic header/footer to the caption."""
    zoom = diagnostic.center_zoom_data(data)
    sx, sy = data["plan"]["coordinate_scales"]
    return {
        "scientific_complete": receipt["complete"],
        "planned_longitudinal_distance": zoom["distance"],
        "coordinate_scales": [sx, sy],
        "display_multiplier": 1e4,
        "metric": "Equal axes in the declared scaled Euclidean metric: steering displacement divided by the first scale, log10-temperature displacement divided by the second scale.",
        "frame": "Left: conditional trace-Hessian predictor rays and full planned transverse section supports, not measured valley connections.",
        "measurements": "Right: every observed sample on the four innermost planned sections; independently qualified minima are not joined.",
        "markers": "Dots are admitted samples; stars and black bars show qualified minima and measured final bracket widths. Red crosses and pluses retain unadmitted samples and unfinished callbacks.",
        "brackets": "Black bars are not confidence intervals or exact extremum enclosures.",
        "status": "All four section statuses and sample/minimum counts are retained. No bracket means no strict minimum bracket on the sampled mesh; missing sections remain untested.",
        "limits": "Finite section minima do not establish connected intrinsic valleys or an exact crossing. Conditional frame geometry is separate from measured section evidence.",
        "diagnostic_labels_replaced_or_moved": known_diagnostic_labels(receipt, data),
    }


def make_publication(receipt, data, *, synthetic=False):
    """Reflow the same artists into a seven-inch-wide publication figure."""
    fig = diagnostic.make_center_zoom(receipt, data, synthetic=synthetic)
    fig.set_size_inches(7., 5.)
    cards = [text for text in fig.texts if text.get_text().startswith("Arm ")]
    diagnostic.require(len(cards) == 4, "four section status cards required")
    removed = [text.get_text() for text in fig.texts if text not in cards]
    diagnostic.require(sorted(removed) == sorted(known_diagnostic_labels(receipt, data, synthetic=synthetic)),
                       "unrecognized diagnostic metadata; refuse to hide a label")
    diagnostic.require([text.get_text().split(":", 1)[0] for text in cards] == [f"Arm {i}" for i in range(1, 5)],
                       "unrecognized section status cards")
    for text in list(fig.texts):
        if text not in cards:
            text.remove()
    fig._suptitle = None
    for i, text in enumerate(cards):
        text.set_position((.085 + (i % 2)*.49, .165 - (i // 2)*.10))
        text.set_fontsize(8.5)
    old_legend = fig.legends[0]
    handles = old_legend.legend_handles
    labels = [text.get_text() for text in old_legend.get_texts()]
    old_legend.remove()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(.535, .235),
               ncol=2, frameon=False, fontsize=8.5, columnspacing=1.6,
               handlelength=1.9, handletextpad=.55)
    titles = ("A  Conditional trace frame", "B  Measured inner sections")
    for ax, title in zip(fig.axes, titles):
        ax.set_title(title, loc="left", fontsize=9.5, pad=7)
        ax.set_xlabel("Steering displacement\n(scaled units ×10⁻⁴)", fontsize=9)
        ax.set_ylabel("Log-temperature displacement\n(scaled units ×10⁻⁴)", fontsize=9)
        ax.tick_params(axis="both", labelsize=8.5)
        for text in ax.texts:
            text.set_fontsize(8.5 if text.get_text().isdigit() else 9)
    notice = "SYNTHETIC QA — not neural measurements" if synthetic else ""
    if not receipt["complete"]:
        notice = ("SYNTHETIC QA — " if synthetic else "")+"PARTIAL / UNRESOLVED; missing sections untested"
    if notice:
        fig.text(.5, .965, notice, ha="center", va="top", fontsize=8, color="#9c4c1a")
    fig.subplots_adjust(left=.09, right=.985, top=.88, bottom=.415, wspace=.37)
    return fig


def render(receipt, data, out, *, synthetic=False):
    import matplotlib.pyplot as plt
    figure = make_publication(receipt, data, synthetic=synthetic)
    try:
        for extension in ("png", "pdf"):
            figure.savefig(Path(out)/f"section-minima-center-zoom-publication.{extension}", dpi=300)
    finally:
        plt.close(figure)


def checked_sources(commit):
    sources = {path: diagnostic.digest(ROOT/path) for path in SOURCES}
    for path, sha in sources.items():
        diagnostic.require(hashlib.sha256(diagnostic.git_bytes(commit, path)).hexdigest() == sha,
                           "commit publication and diagnostic sources before rendering scientific evidence: "+path)
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = checked_sources(commit)
    receipt, data = diagnostic.load_receipt(args.receipt, args.receipt_sha256, allow_partial=args.allow_partial)
    out = args.out.resolve()
    diagnostic.require(not out.exists() and not out.is_relative_to(ROOT.resolve())
                       and not out.is_relative_to(Path(receipt["raw_directory"]).resolve()),
                       "new output outside repository and raw run required")
    out.mkdir(parents=True)
    render(receipt, data, out)
    result = {
        "id": "BSC5-center-zoom-publication-figures", "receipt": str(args.receipt.resolve()),
        "receipt_sha256": args.receipt_sha256, "scientific_complete": receipt["complete"],
        "partial_explicitly_allowed": args.allow_partial, "source_commit": commit,
        "source_sha256": sources, "figure_size_inches": [7., 5.], "threads": 1,
        "caption_semantics": caption_semantics(receipt, data),
        "figures": [{"path": p.name, "bytes": p.stat().st_size, "sha256": diagnostic.digest(p)}
                    for p in sorted(out.iterdir()) if p.is_file()],
        "visual_review": "required; not established by successful rendering",
        "scope": "Layout-only reuse of audited center-zoom artists. No new numerical evaluation or interpolation.",
    }
    (out/"receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"out": str(out), "receipt_sha256": diagnostic.digest(out/"receipt.json")}))


if __name__ == "__main__":
    main()

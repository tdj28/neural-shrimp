"""Stack the audited BSC5 center-zoom panels without changing plotted evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import textwrap

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import plot_baby_shrimp_contraction_minima as base

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "tools/plot_baby_shrimp_contraction_minima_mobile.py"
SOURCES = [SOURCE, "tools/plot_baby_shrimp_contraction_minima.py"]


def stacked_center_zoom(receipt, data, *, synthetic=False):
    """Change layout only; every panel artist and scientific label is retained."""
    fig = base.make_center_zoom(receipt, data, synthetic=synthetic)
    base.require(len(fig.axes) == 2 and len(fig.legends) == 1,
                 "expected two existing panels and one complete legend")
    fig.set_size_inches(5.8, 18)
    fig.axes[0].set_position([.15, .605, .80, .258])
    fig.axes[1].set_position([.15, .28, .80, .258])
    statuses = [t for t in fig.texts if t.get_text().startswith("Arm ")]
    headers = [t for t in fig.texts if t.get_text().startswith(("Audited innermost", "PARTIAL / UNRESOLVED"))]
    footers = [t for t in fig.texts if t.get_text().startswith("Scaled coordinates:")]
    base.require(len(statuses) == 4 and len(headers) == len(footers) == 1
                 and len(fig.texts) == 7, "unexpected source labels; refuse to omit them")
    for i, status in enumerate(statuses):
        status.set_position((.08+.47*(i % 2), .215-.045*(i//2)))
        status.set_fontsize(9)
        status.set_text("\n".join(textwrap.fill(line, 29) for line in status.get_text().splitlines()))
    fig._suptitle.set_position((.08, .985))
    fig._suptitle.set_fontsize(14)
    if synthetic:
        fig._suptitle.set_text(fig._suptitle.get_text().replace(" — SYNTHETIC QA", "\nSYNTHETIC QA"))
    headers[0].set_position((.08, .94))
    headers[0].set_verticalalignment("top")
    headers[0].set_fontsize(8.5)
    headers[0].set_text("\n".join(textwrap.fill(line, 78) for line in headers[0].get_text().splitlines()))
    footers[0].set_position((.08, .025))
    footers[0].set_fontsize(8)
    footers[0].set_text("\n".join(textwrap.fill(line, 80) for line in footers[0].get_text().splitlines()))
    legend = fig.legends[0]
    handles, labels = legend.legend_handles, [t.get_text() for t in legend.get_texts()]
    legend.remove()
    fig.legend(handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(.53, .115),
               ncol=2, frameon=False, fontsize=8.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = {}
    for path in SOURCES:
        sha = base.digest(ROOT/path)
        frozen = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
        base.require(hashlib.sha256(frozen).hexdigest() == sha, "commit plot sources before scientific rendering")
        sources[path] = sha
    receipt, data = base.load_receipt(args.receipt, args.receipt_sha256, allow_partial=args.allow_partial)
    out = args.out.resolve(); raw = Path(receipt["raw_directory"]).resolve()
    base.require(not out.exists() and not out.is_relative_to(ROOT.resolve()) and not out.is_relative_to(raw)
                 and not raw.is_relative_to(out), "new output outside repository and raw run required")
    out.mkdir(parents=True)
    fig = stacked_center_zoom(receipt, data)
    for extension in ("png", "pdf"):
        fig.savefig(out/f"section-minima-center-zoom-mobile.{extension}", dpi=180)
    import matplotlib.pyplot as plt
    plt.close(fig)
    record = {"id": "BSC5-section-minima-mobile-figures", "receipt": str(args.receipt.resolve()),
              "receipt_sha256": args.receipt_sha256, "scientific_complete": receipt["complete"],
              "partial_explicitly_allowed": args.allow_partial, "source_commit": commit,
              "source_sha256": sources, "threads": 1,
              "figures": [{"path": p.name, "bytes": p.stat().st_size, "sha256": base.digest(p)}
                          for p in sorted(out.iterdir()) if p.is_file()],
              "scope": "Layout-only stacking of both original center-zoom panels, with all measurements, brackets, statuses and labels retained.",
              "visual_review": "required; rendering does not establish visual approval"}
    (out/"receipt.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"out": str(out), "receipt_sha256": base.digest(out/"receipt.json")}))


if __name__ == "__main__":
    main()

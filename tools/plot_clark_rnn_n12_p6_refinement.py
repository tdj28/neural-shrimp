"""Render all saved N12 p6 refinement maps; no model evaluations."""
import os
os.environ["MPLBACKEND"] = "Agg"
import argparse, hashlib, json, resource, subprocess, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter
RAW = Path("/data2/neural-shrimp-data/explore/clark-rnn/n12-p6-refinement-001")
PIN = "ed58f64bbfa4513775f2ed2e9e0935da6902b7b6f151e8d72ca176210b6e7674"
REPO = Path(__file__).resolve().parents[1]
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()

def main(out):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        raise ValueError("clean source required")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10 or any(os.environ.get(k) != "1" for k in THREADS):
        raise ValueError("CPU1 nice10 required")
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"status": "incomplete", "input_outer": PIN, "pid": os.getpid(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "source_sha256": sha(Path(__file__)), "scope": "Saved-data rendering only; no model or derivative evaluations."}
    try:
        assert sha(RAW / "manifest.json") == PIN
        inv = json.loads((RAW / "manifest.json").read_text())["files"]
        names = ["run/batch.npz", "run/inputs.npz", "run/result.json", "run/components.json"]
        for name in names:
            path = RAW / name
            assert not path.is_symlink() and sha(path) == inv[name]["sha256"] and path.stat().st_size == inv[name]["bytes"]
        with np.load(RAW / "run/inputs.npz") as z:
            gain, delta = z["gains"], z["deltas"]
        with np.load(RAW / "run/batch.npz") as z:
            period, growth = z["period_proposals"], z["lambda_per_step"]
        result = json.loads((RAW / "run/result.json").read_text())
        assert result["status"] == "completed" and period.shape == growth.shape == (5202,)
        components = json.loads((RAW / "run/components.json").read_text())["components"]
        leading = max((c for c in components if c["period"] == 6), key=lambda c: c["count"])
        fig = plt.figure(figsize=(13, 8.8))
        layout = fig.add_gridspec(3, 3, height_ratios=[1, 1, .05], left=.06, right=.97, bottom=.17, top=.92, hspace=.32, wspace=.24)
        axes = np.array([[fig.add_subplot(layout[row, col]) for col in range(3)] for row in range(2)])
        color_axes = [fig.add_subplot(layout[2, col]) for col in range(3)]
        cmap = plt.get_cmap("turbo").copy()
        cmap.set_bad("#dedede")
        selected_cmap = ListedColormap(["#dedede", "#f7f7f7", "#969696", "#276caa", "#f49b42", "#922c85"])
        norm = BoundaryNorm(np.arange(-.5, 6), selected_cmap.N)
        for initial in [0, 1]:
            p = period[initial::2].reshape(51, 51).T
            images = []
            images.append(axes[initial, 0].pcolormesh(gain, delta, np.ma.masked_equal(p, 0), cmap=cmap, vmin=1, vmax=64, shading="nearest", rasterized=True))
            category = np.ones_like(p)
            for value, label in [(0, 0), (2, 2), (6, 3), (12, 4), (24, 5)]:
                category[p == value] = label
            images.append(axes[initial, 1].pcolormesh(gain, delta, category, cmap=selected_cmap, norm=norm, shading="nearest", rasterized=True))
            images.append(axes[initial, 2].pcolormesh(gain, delta, growth[initial::2].reshape(51, 51).T, cmap="RdBu_r", vmin=-.03, vmax=.03, shading="nearest", rasterized=True))
            for col, title in enumerate(["All proposed recurrence periods", "Period-six and doubled-period context", "Finite mean tangent growth"]):
                axes[initial, col].set_title(f"Start {initial}: {title}", fontsize=10)
                axes[initial, col].set_xscale("log")
                axes[initial, col].xaxis.set_major_locator(FixedLocator([3, 5, 8, 12]))
                axes[initial, col].xaxis.set_major_formatter(FixedFormatter(["3", "5", "8", "12"]))
                axes[initial, col].xaxis.set_minor_locator(FixedLocator([]))
                if initial == 1:
                    axes[initial, col].set_xlabel("Gain g")
            axes[initial, 0].set_ylabel("Discrete leak / Euler parameter δ")
            for q in result["records"]:
                if q["initial"] == initial:
                    axes[initial, 1].plot(q["gain"], q["delta"], "k.", markersize=2)
        fig.colorbar(images[0], cax=color_axes[0], orientation="horizontal", label="Proposed period; gray = unresolved")
        cb = fig.colorbar(images[1], cax=color_axes[1], orientation="horizontal", ticks=range(6))
        cb.ax.set_xticklabels(["Unresolved", "Other", "2", "6", "12", "24"], fontsize=8)
        fig.colorbar(images[2], cax=color_axes[2], orientation="horizontal", label="Growth per step (clipped)")
        fig.suptitle("Fixed random N=12, seed28192: complete 51 × 51 maps, both starts", y=.99)
        fig.text(.06, .025, f"Leading consensus p6 component: {leading['count']} cells, crop margin {leading['grid_edge_margin']}. "
                 f"All {result['qualification_attempts']} selected native cases qualified (black dots).\n"
                 f"{result['two_start_disagreements']}/2601 cells disagree between starts. Recurrence and finite growth do not establish shrimp anatomy or chaos.", fontsize=9)
        for suffix in ["png", "pdf"]:
            fig.savefig(out / f"maps.{suffix}", dpi=160)
        plt.close(fig)
        if time.monotonic() - started > 10 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 448 * 1024:
            raise RuntimeError("render resource cap")
        if sum(p.stat().st_size for p in out.iterdir()) > 2 * 1024**2:
            raise RuntimeError("render byte cap")
        record["status"] = "completed"
    except BaseException as exc:
        record.update(status="failed_or_interrupted", error=type(exc).__name__ + ": " + str(exc))
    finally:
        record.update(elapsed_seconds=time.monotonic() - started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      files={p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()})
        (out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        for path in out.iterdir():
            path.chmod(0o444)
        out.chmod(0o555)
    print(json.dumps(record))
    return 0 if record["status"] == "completed" else 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    raise SystemExit(main(parser.parse_args().out))

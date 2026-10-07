"""Plot an exploratory real-data learning-rate scan (tools/explore_realdata.py scan).

Three panels sharing the learning-rate axis:
  A  loss values visited on the attractor (last ``--tail`` recorded updates),
     colored by regime: fixed point, periodic, quasi-periodic candidate
     (unresolved by the screen, |FTLE| below the chaos threshold and forward
     angle steps on at least 99% of two-step iterations), chaotic candidate;
  B  rotation number of the two-step map for non-fixed, non-period-2 cells;
  C  loss power spectrum per learning rate (log10 relative power).
Also prints the learning-rate extent of each regime.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
REGIMES = {"fixed": "#a3a29b", "periodic": "#2a78d6", "quasi-periodic": "#1baf7a", "chaotic": "#eb6834",
           "unresolved": "#d4d3cc"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#b7d3f6", "#5598e7", "#256abf", "#0d366b"])


def regime(row, chaos_threshold=5e-3):
    status = row["status"]
    if status == 1:
        return "fixed"
    if status == 2:
        return "periodic"
    if status == 3:
        return "chaotic"
    if status in (4, 5) and row.get("ftle") is not None and abs(row["ftle"]) < chaos_threshold \
            and row.get("rho2_forward", 0) >= 0.99:
        return "quasi-periodic"
    return "unresolved" if status in (4, 5) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True, help="scan directory")
    parser.add_argument("--tail", type=int, default=512)
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    scan = Path(args.scan)
    rows = json.loads((scan / "scan.json").read_text())["rows"]
    saved = np.load(scan / "scan-losses.npz")
    log_eta, losses = saved["log10_eta"], saved["losses"].astype(np.float64)
    labels = [regime(r) for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]})
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]
    for name, color in REGIMES.items():
        idx = [i for i, label in enumerate(labels) if label == name]
        if not idx:
            continue
        xs = np.repeat(log_eta[idx], args.tail)
        ys = losses[idx, -args.tail:].ravel()
        ax.plot(xs, ys, ",", color=color, alpha=0.6, rasterized=True)
        ax.plot([], [], "o", color=color, label=name, markersize=5)
    ax.set_yscale("log")
    ax.set_ylabel("training loss on the attractor", fontsize=9, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=5)
    ax.set_title(args.title or f"{scan.parent.name}: one global learning rate, fixed initialization (exploratory)",
                 loc="left", fontsize=10, color=INK)
    ax = axes[1]
    for name in ("periodic", "quasi-periodic", "chaotic", "unresolved"):
        pts = [(r["log10_eta"], r["rho2"]) for r, label in zip(rows, labels)
               if label == name and r.get("rho2") is not None]
        if pts:
            a = np.array(pts)
            ax.plot(a[:, 0], a[:, 1], "o", markersize=3, color=REGIMES[name], markeredgewidth=0)
    for frac in (1 / 6, 1 / 5, 2 / 9, 1 / 4, 2 / 7, 1 / 3, 2 / 5, 1 / 2):
        ax.axhline(frac, color=GRID, linewidth=0.6, zorder=0)
    ax.set_ylabel("rotation number of the two-step map", fontsize=9, color=MUTED)
    ax = axes[2]
    power = []
    for series in losses:
        if not np.all(np.isfinite(series)) or not np.any(series - series.mean()):
            power.append(np.full(len(series) // 2 + 1, np.nan))
            continue
        p = np.abs(np.fft.rfft((series - series.mean()) * np.hanning(len(series)))) ** 2
        power.append(np.log10(p / p[1:].sum() + 1e-14))
    power = np.array(power)
    freqs = np.fft.rfftfreq(losses.shape[1])
    image = ax.imshow(power.T, origin="lower", aspect="auto", cmap=BLUES, vmin=-8, vmax=0,
                      extent=[log_eta[0], log_eta[-1], freqs[0], freqs[-1]], interpolation="nearest")
    bar = fig.colorbar(image, ax=ax, shrink=0.9)
    bar.set_label("log10 relative power", fontsize=8, color=MUTED)
    ax.set_ylabel("loss frequency (cycles per update)", fontsize=9, color=MUTED)
    ax.set_xlabel("log10 global learning rate", fontsize=9, color=MUTED)
    for a in axes:
        a.tick_params(labelsize=8, colors=MUTED)
        a.set_facecolor(SURFACE)
    fig.savefig(scan / "scan.png", dpi=160, facecolor=SURFACE)
    for name in REGIMES:
        xs = [r["log10_eta"] for r, label in zip(rows, labels) if label == name]
        if xs:
            print(f"{name:15s} n={len(xs):4d} log10 eta {min(xs):+.4f} .. {max(xs):+.4f}")
    periods = {}
    for r, label in zip(rows, labels):
        if label == "periodic":
            periods.setdefault(r["period"], []).append(r["log10_eta"])
    for p in sorted(periods):
        print(f"period {p:3d}: n={len(periods[p]):3d} log10 eta {min(periods[p]):+.4f} .. {max(periods[p]):+.4f}")
    for r, label in zip(rows, labels):
        if label == "periodic" and r["period"] > 2 and r.get("rho2") is not None:
            q = r["period"] // 2 if r["period"] % 2 == 0 else r["period"]
            m = round(r["rho2"] * q)
            print(f"  window log10 eta {r['log10_eta']:+.4f} period {r['period']:3d} rho2 {r['rho2']:.5f} "
                  f"~ {m}/{q} (residual {abs(r['rho2'] - m / q):.1e}, forward {r['rho2_forward']:.2f})")
    qp = [(r["log10_eta"], r["rho2"]) for r, label in zip(rows, labels) if label == "quasi-periodic"]
    if qp:
        a = np.array(qp)
        print(f"quasi-periodic rho2 range {a[:, 1].min():.4f} .. {a[:, 1].max():.4f}; "
              f"Spearman-like monotonicity {np.corrcoef(np.argsort(np.argsort(a[:, 0])), np.argsort(np.argsort(a[:, 1])))[0, 1]:+.3f}")
    diverged = [r["log10_eta"] for r in rows if r["status"] == 0]
    if diverged:
        print(f"divergent       n={len(diverged):4d} from log10 eta {min(diverged):+.4f}")


if __name__ == "__main__":
    main()

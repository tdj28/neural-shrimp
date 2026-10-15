"""Quick-look rendering of a two-block zoom (run TBZ): exponent map with periodic cells in grey by contraction, and the period map (period_colors). Usage: python tools/plot_tb_zoom.py RUN OUT.png [A1LO A1HI A2LO A2HI]"""
import sys, numpy as np, matplotlib
sys.path.insert(0, "/data2/neural-shrimp/tools")
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import period_colors as pc
run, out = sys.argv[1], sys.argv[2]
box = [float(x) for x in sys.argv[3:7]] if len(sys.argv) > 6 else None
z = np.load(f"/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/tb/{run}/tb-t-0.100.npz", allow_pickle=True)
a1, a2, L, P = z["alpha1"], z["alpha2"], z["lyapunov"], z["period"]
if box:
    i = (a1 >= box[0]) & (a1 <= box[1]); j = (a2 >= box[2]) & (a2 <= box[3])
    a1, a2, L, P = a1[i], a2[j], L[np.ix_(j, i)], P[np.ix_(j, i)]
ext = [a1[0], a1[-1], a2[0], a2[-1]]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), dpi=150)
per = P > 0
ax = axes[0]
ax.imshow(np.where(per, np.clip(L, -2, 0), np.nan), origin="lower", extent=ext, aspect="auto", cmap="Greys_r", vmin=-1.2, vmax=0, interpolation="nearest")
im = ax.imshow(np.where(per, np.nan, L), origin="lower", extent=ext, aspect="auto", cmap="inferno", vmin=0, vmax=0.45, interpolation="nearest")
fig.colorbar(im, ax=ax, fraction=0.046, label="exponent (chaotic cells)")
ax.set_title(f"{run}: periodic cells gray (dark = strongly contracting)", fontsize=8); ax.set_xlabel("alpha1"); ax.set_ylabel("alpha2")
ax = axes[1]
Pm = np.where(per, P, np.nan).astype(float)
cats = pc.categorize([Pm], max_blocks=16)
ax.set_facecolor("0.12")
im2 = ax.imshow(cats.index(Pm), origin="lower", extent=ext, aspect="auto", cmap=cats.cmap, norm=cats.norm, interpolation="nearest")
cats.colorbar(im2, ax=ax, label="period (dark: chaotic or none)")
ax.set_title("periods", fontsize=8); ax.set_xlabel("alpha1")
fig.tight_layout(); fig.savefig(out)

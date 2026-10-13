"""Plot a partial soft-loop scan (scan-partial.npz of a timed-out run; cells not done shown gray): plot_scan_partial.py NPZ OUT."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
src, out = sys.argv[1], sys.argv[2]
z = np.load(src)
done = int(z["done"]); lt, al = z["log10_temperature"], z["alpha"]
nT, nA = len(lt), len(al)
P = z["period"].astype(float); L = z["lyapunov"].astype(float)
P[done:] = np.nan; L[done:] = np.nan
P = P.reshape(nT, nA); L = L.reshape(nT, nA)
ext = [al[0], al[-1], lt[0], lt[-1]]
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
im = ax[0].imshow(np.clip(L, -0.5, 0.5), origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", interpolation="nearest")
plt.colorbar(im, ax=ax[0]); ax[0].set_title(f"largest Lyapunov exponent ({done} of {nT*nA} cells; gray: not done)")
Pm = np.where(P > 0, P, np.nan)
cmap = plt.get_cmap("tab20").copy(); cmap.set_bad("white")
im2 = ax[1].imshow(Pm, origin="lower", extent=ext, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0.5, vmax=40.5)
plt.colorbar(im2, ax=ax[1]); ax[1].set_title("period (white: none or not done)")
for a in ax:
    a.set_xlabel("steering along 'hostility-vs-kindness'"); a.set_ylabel("log10 temperature")
    a.set_facecolor("0.85")
fig.tight_layout(); fig.savefig(out, dpi=140)

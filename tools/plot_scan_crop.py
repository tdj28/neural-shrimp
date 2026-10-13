"""Crop of a soft-loop scan (Lyapunov exponent and period panels) over an alpha x log10 T window: plot_scan_crop.py SRC A0 A1 T0 T1 OUT."""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
src, a0, a1, t0, t1, out = sys.argv[1], *map(float, sys.argv[2:6]), sys.argv[6]
z = np.load(f"/data2/neural-shrimp-data/explore/{src}/scan.npz")
lt, al = z["log10_temperature"], z["alpha"]
L, P = z["lyapunov"], z["period"]
# axes: rows = log10 T, columns = alpha (as in the scan image)
ti = np.flatnonzero((lt >= t0) & (lt <= t1)); ai = np.flatnonzero((al >= a0) & (al <= a1))
Lc = L[np.ix_(ti, ai)] if L.shape == (len(lt), len(al)) else L.T[np.ix_(ti, ai)]
Pc = P[np.ix_(ti, ai)] if P.shape == (len(lt), len(al)) else P.T[np.ix_(ti, ai)]
ext = [al[ai[0]], al[ai[-1]], lt[ti[0]], lt[ti[-1]]]
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
im = ax[0].imshow(np.clip(Lc, -0.5, 0.5), origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", interpolation="nearest")
plt.colorbar(im, ax=ax[0]); ax[0].set_title("largest Lyapunov exponent")
Pm = np.where(Pc > 0, Pc, np.nan)
im2 = ax[1].imshow(Pm, origin="lower", extent=ext, aspect="auto", cmap="tab20", interpolation="nearest", vmin=0.5, vmax=40.5)
plt.colorbar(im2, ax=ax[1]); ax[1].set_title("period (white: none)")
for a in ax: a.set_xlabel("alpha"); a.set_ylabel("log10 T")
fig.tight_layout(); fig.savefig(out, dpi=150)
u, c = np.unique(Pc, return_counts=True); print(dict(zip(u.tolist(), c.tolist())))
print(L.shape, len(lt), len(al))

"""Exploratory Jones-style critical-point analysis of resolved six-step maps.

This validates a scalar reconstruction against held-out returns before using
its turning points. Iterates outside the observed domain are stopped. Local
branch symbols do not establish a Rossler hub, conjugacy or true superstability.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "8"
import numpy as np
from numpy.polynomial import Chebyshev
from numpy.polynomial.chebyshev import chebvander

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("mother-chaos", "baby-chaos-left", "baby-chaos-right", "baby-chaos-below")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights(x):
    bins = np.clip((np.asarray(x)*30).astype(int), 0, 29)
    count = np.bincount(bins, minlength=30)
    return 1 / np.sqrt(count[bins])


def fit(x, y, degree):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 60 or len(np.unique(x)) < 40 or not np.isfinite([x, y]).all():
        raise ValueError("insufficient distinct finite return pairs")
    return Chebyshev.fit(x, y, degree, domain=[0, 1], w=weights(x))


def critical(model):
    roots = model.deriv().roots()
    return np.array(sorted(float(r.real) for r in roots
                           if abs(r.imag) < 1e-10 and .02 < r.real < .98
                           and model.deriv()(r.real-.001)*model.deriv()(r.real+.001) < 0))


def itinerary(model, roots, index, steps=7, ambiguity=.002):
    x = float(roots[index]); values = [x]; word = ["CD"[index] if index < 2 else f"K{index}"]
    reason = "requested iterates completed"
    for _ in range(steps):
        y = float(model(x))
        if not np.isfinite(y) or not 0 <= y <= 1:
            reason = "stopped: next iterate outside sampled input domain"
            break
        values.append(y)
        word.append("?" if np.min(abs(roots-y)) < ambiguity else str(int(np.searchsorted(roots, y))))
        x = y
    return {"critical_index": index, "values": values, "word": " ".join(word), "reason": reason}


def reconstruct(samples, degree=7):
    if degree not in (7, 11):
        raise ValueError("declared reconstruction degree required")
    # The coordinate is fixed using all input states; response fits are checked
    # in both directions between consecutive halves of the time record.
    mean = samples.mean(0)
    _, singular, vt = np.linalg.svd(samples-mean, full_matrices=False)
    axis = vt[0].copy()
    if axis[np.argmax(abs(axis))] < 0:
        axis *= -1
    raw = (samples-mean) @ axis
    span = float(np.ptp(raw))
    if not np.isfinite(span) or span < 1e-10:
        raise ValueError("collapsed state visits cannot define a return curve")
    origin = mean + raw.min()*axis
    u = (raw-raw.min())/span
    x, y = u[:-1], u[1:]
    mid = len(x)//2
    halves = [np.arange(mid), np.arange(mid, len(x))]
    models = {d: fit(x, y, d) for d in (degree-2, degree, degree+2)}
    cv = []
    for train, test in (halves, halves[::-1]):
        m = fit(x[train], y[train], degree)
        err = m(x[test])-y[test]
        cv.append({"rms": float(np.sqrt(np.mean(err**2))), "max": float(abs(err).max())})
    roots = {d: critical(m) for d, m in models.items()}
    r = roots[degree]
    agreement = bool(1 <= len(r) <= 2 and all(len(v) == len(r) and np.max(abs(v-r)) < .01 for v in roots.values()))
    density = [[int(np.sum((x > v-.1) & (x < v))), int(np.sum((x > v) & (x < v+.1)))] for v in r]
    scalar = all(e["rms"] < .001 and e["max"] < .01 for e in cv)
    stable = bool(scalar and agreement and all(min(n) >= 5 for n in density))
    V = chebvander(2*u-1, degree)
    w = weights(u)
    coefficients = np.linalg.lstsq(V*w[:, None], (samples-origin)*w[:, None], rcond=None)[0]
    lift_cv = []
    half_visits = [np.arange(len(u)//2), np.arange(len(u)//2, len(u))]
    for train, test in (half_visits, half_visits[::-1]):
        coef = np.linalg.lstsq(V[train]*w[train, None], (samples[train]-origin)*w[train, None], rcond=None)[0]
        error = np.linalg.norm(V[test] @ coef+origin-samples[test], axis=1)/span
        lift_cv.append({"rms": float(np.sqrt(np.mean(error**2))), "max": float(error.max())})
    lift_ok = all(e["rms"] < .01 and e["max"] < .05 for e in lift_cv)
    full_error = np.linalg.norm(chebvander(2*models[degree](x)-1, degree) @ coefficients+origin-samples[1:], axis=1)/span
    result = {"degree": degree, "visits": len(samples), "coordinate_span": span,
              "pc1_explained": float(singular[0]**2/np.sum(singular**2)),
              "scalar_cross_validation": cv, "scalar_passed": bool(scalar),
              "critical_points_by_degree": {str(d): v.tolist() for d,v in roots.items()},
              "critical_density_left_right": density, "critical_points_qualified": stable,
              "lift_cross_validation": lift_cv, "lift_passed": bool(lift_ok),
              "scalar_plus_lift_next_state_relative_rms": float(np.sqrt(np.mean(full_error**2))),
              "critical_iterates": [itinerary(models[degree], r, k) for k in range(len(r))] if stable else [],
              "fit_scope": "Shared coordinate from all visits; consecutive-half response/lift validation in both directions. Errors normalized by the observed state-coordinate span."}
    arrays = {"origin": origin, "axis": axis, "span": np.array(span), "lift_coefficients": coefficients,
              "scalar_coefficients": models[degree].coef, "coordinate": u, "critical_points": r}
    return result, arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--degree", type=int, choices=(7,11), default=7)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    info = json.loads((args.run/"trajectories.json").read_text())
    if json.loads((args.run/"status.json").read_text())["status"] != "complete":
        raise ValueError("complete source trajectories required")
    states = np.load(args.run/"states.npy", mmap_mode="r", allow_pickle=False)
    if states.shape != (3000, 12, 896) or not np.isfinite(states).all():
        raise ValueError("unexpected or nonfinite trajectory archive")
    records, arrays = [], {}
    for name in NAMES:
        matches = [(i,c) for i,c in enumerate(info["cells"]) if c["name"] == name]
        if len(matches) != 1:
            raise ValueError("missing or duplicate source cell")
        index, cell = matches[0]
        if cell["period_candidate"] or min(cell["lyapunov_halves"]) <= 0:
            raise ValueError("resolved nonperiodic expanding source required")
        for phase in range(6):
            result, data = reconstruct(states[phase::6, index, :], args.degree)
            result.update(name=name, phase=phase, alpha=cell["alpha"], log10_temperature=cell["log10_temperature"])
            records.append(result)
            for key, value in data.items():
                arrays[f"{name}__{phase}__{key}"] = value
    np.savez_compressed(args.out/"reconstructions.npz", **arrays)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(11,9),layout="constrained")
    titles = ("Mother neighbor", "Baby: left neighbor", "Baby: right neighbor", "Baby: lower neighbor")
    for ax, name, title in zip(axes.flat,NAMES,titles):
        phase = 3 if name == "mother-chaos" else 0  # same disclosed comparison as BSP1
        row = next(r for r in records if r["name"] == name and r["phase"] == phase)
        key = f"{name}__{phase}__"
        u = arrays[key+"coordinate"]
        model = Chebyshev(arrays[key+"scalar_coefficients"], domain=[0,1])
        xx = np.linspace(0,1,600)
        ax.scatter(u[:-1],u[1:],s=8,c="#64748b",alpha=.5,label="Measured returns")
        ax.plot(xx,model(xx),color="#152f49",lw=1.5,label="Validated fit" if row["scalar_passed"] else "Unqualified fit")
        roots = arrays[key+"critical_points"]
        for k,root in enumerate(roots):
            color = ("#c65a1e", "#007d87")[k % 2]
            ax.scatter([root],[model(root)],s=70,color=color,zorder=4)
            ax.annotate("CD"[k] if k<2 else f"K{k}",(root,model(root)),xytext=(8,12 if model.deriv(2)(root)>0 else -18),textcoords="offset points",color=color,fontweight="bold")
        ax.plot([0,1],[0,1],color="#cbd5e1",lw=.7)
        errors = max(r["rms"] for r in row["scalar_cross_validation"])
        ax.set(title=title+f" · phase {phase+1}",xlabel="Position now",ylabel="Position six steps later",xlim=(-.03,1.03),ylim=(-.03,1.03))
        ax.text(.43,.96,f"Held-out RMS: {errors:.2g} of span\nCritical-point check: {'passes' if row['critical_points_qualified'] else 'unresolved'}",transform=ax.transAxes,va="top",fontsize=8,bbox={"facecolor":"white","edgecolor":"none","alpha":.85})
        ax.set_aspect("equal")
    fig.suptitle("Testing Jones's return-map construction",fontsize=22,fontweight="bold")
    fig.supxlabel("C and D name left/right turning points locally; these are projected return curves.\nFit degrees 5, 7 and 9 and both time halves are checked. No hub or exact superstable crossing is inferred from these fits.",fontsize=9)
    for suffix in ("png","pdf"):
        fig.savefig(args.out/f"jones-returnmaps.{suffix}",dpi=170)
    plt.close(fig)
    record = {"source_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "source_sha256": sha(__file__), "trajectory_metadata_sha256": sha(args.run/"trajectories.json"),
              "states_sha256": sha(args.run/"states.npy"), "reference": "https://arxiv.org/abs/1201.4343",
              "status": "exploratory; adapted after BSP1 return maps", "rows": records,
              "artifacts": {p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in args.out.iterdir() if p.is_file()},
              "limits": "Continuous maps inferred only from nonperiodic visits. Critical iterates are fitted-model diagnostics, stopped outside sampled input support. Local C/D and 0/1/2 symbols need continuation before comparison with Jones's persistent branches. No global hub, conjugacy or nilpotent full-state orbit is established."}
    (args.out/"jones.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"rows":len(records),"scalar_passed":sum(r["scalar_passed"] for r in records),
                      "critical_points_qualified":sum(r["critical_points_qualified"] for r in records)}))


if __name__ == "__main__":
    main()

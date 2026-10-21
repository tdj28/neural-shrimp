"""Phase-resolved BSP1 return maps, using the mother's published diagnostics.

Full-state recurrence, text period, finite-time exponent, projection straightness
and derivative-sign coding are reported separately. Periodic point sets are
drawn as discrete visits; they are never fit as continuous return curves.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

for thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[thread_variable] = "8"
import numpy as np

import symbolic_dynamics as sd


def phase_map(states, cosines, phase, lag=6):
    samples = np.asarray(states[phase::lag], dtype=float)
    centered = samples-samples.mean(0)
    relative_spread = float(np.linalg.norm(centered, axis=1).max() / max(np.linalg.norm(samples.mean(0)), 1.0))
    if relative_spread < 1e-10:
        xy = np.zeros((len(samples), 2))
        explained = None
    else:
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:2].copy()
        for vector in basis:
            if vector[np.argmax(np.abs(vector))] < 0:
                vector *= -1
        xy = centered @ basis.T
        explained = float(singular[0]**2 / (singular**2).sum())
    x,y = xy[:-1,0], xy[1:,0]
    record = {"phase": phase, "lag": lag, "relative_spread": relative_spread,
              "collapsed": relative_spread < 1e-10, "explained_1": explained}
    colors = np.full(len(x), "#26334d", dtype="<U7")
    if relative_spread >= 1e-10:
        record["robust_turning_points"] = int(sd.robust_turning_points(x,y)[0])
        record["bend_degrees"] = sd.bend(xy)
    if lag == 6 and relative_spread >= 1e-10:
        cos = cosines[phase+6::6][:len(x)]
        weak = (np.abs(cos)<.5) | ~np.isfinite(cos)
        symbols = np.where(cos>0, "L", "R")
        agreement, threshold, orientation, used = sd.separability(x, symbols, weak)
        left = int(np.sum((symbols=="L") & ~weak))
        minority = min(left, used-left)
        errors = int(round((1-agreement)*used)) if agreement is not None else None
        bend = record["bend_degrees"]
        usable = bool(weak.mean() <= .1 and minority >= 5 and bend is not None and bend < 45)
        record.update(weak_fraction=float(weak.mean()), usable=usable, minority=minority, errors=errors,
                      errors_over_minority=errors/minority if minority and errors is not None else None,
                      separability=agreement, threshold=threshold, left_below=orientation)
        colors[(symbols=="L") & ~weak] = "#0072b2"
        colors[(symbols=="R") & ~weak] = "#d55e00"
        colors[weak] = "#aab1bf"
    return record, x, y, colors


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--orbits", type=Path)
    args=ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    info=json.loads((args.run/"trajectories.json").read_text())
    status=json.loads((args.run/"status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("trajectory attempt is not complete")
    states=np.load(args.run/"states.npy", mmap_mode="r", allow_pickle=False)
    diag=np.load(args.run/"diagnostics.npz", allow_pickle=False)
    if states.shape[:2] != (info["protocol"]["record"],len(info["cells"])) or not np.isfinite(states).all():
        raise ValueError("incomplete or nonfinite state record")
    if not np.isfinite(diag["log_growth"]).all() or not np.isfinite(diag["word_cos"][6:]).all():
        raise ValueError("nonfinite diagnostic record")
    verified={}
    if args.orbits:
        raw=json.loads(args.orbits.read_text())
        if raw.get("complete") is not True:
            raise ValueError("orbit verification is not complete")
        verified={c["name"]:c for c in raw["cells"]}
        expected={c["name"]:c for c in info["cells"]}
        if set(verified) != set(expected):
            raise ValueError("orbit and trajectory cells differ")
        for name,cell in expected.items():
            if any(verified[name][key] != cell[key] for key in
                   ("alpha", "log10_temperature", "period_candidate")):
                raise ValueError(f"orbit parameters differ for {name}")
        provenance=raw["protocol"]["seed_provenance"]
        if provenance["trajectory_metadata_sha256"] != hashlib.sha256((args.run/"trajectories.json").read_bytes()).hexdigest():
            raise ValueError("orbit seeds came from different trajectories")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                         "axes.spines.right":False, "axes.titlesize":12})
    results=[]; maps={}; indices={c["name"]:i for i,c in enumerate(info["cells"])}
    for i,cell in enumerate(info["cells"]):
        phases=[]
        for phase in range(6):
            record,x,y,colors=phase_map(states[:,i,:],diag["word_cos"][:,i],phase)
            if cell["period_candidate"] or cell["lyapunov"] <= .001:
                record["robust_turning_points"] = None
                record["turning_count_scope"] = "not interpreted for periodic or nonexpanding samples"
            phases.append(record);maps[cell["name"],6,phase]=(x,y,colors)
        usable=[p for p in phases if p.get("usable")]
        eligible=cell["period_candidate"]==0 and cell["lyapunov"]>.001
        pooled=(sum(p["errors"] for p in usable)/sum(p["minority"] for p in usable)) if usable else None
        best=min(usable or phases,key=lambda p:(p.get("bend_degrees") if p.get("bend_degrees") is not None else 999))
        v=verified.get(cell["name"],{})
        row={**cell,"phases":phases,"selected_phase":best["phase"],"usable_phases":len(usable),
             "chaos_returnmap_eligible":eligible,"pooled_errors_over_minority":pooled,
             "mother_coding_gate_passed":bool(eligible and usable and pooled<.05),
             "verified_primitive_orbit":v.get("verified_primitive_orbit",False),
             "spectral_radius":v.get("spectral_radius"),"attracting":v.get("attracting")}
        phase18=best["phase"]
        rec18,x,y,colors=phase_map(states[:,i,:],diag["word_cos"][:,i],phase18,18)
        if not eligible:
            rec18["robust_turning_points"] = None
            rec18["turning_count_scope"] = "not interpreted for periodic or nonexpanding samples"
        maps[cell["name"],18,phase18]=(x,y,colors);row["eighteen_step_projection"]=rec18
        results.append(row)
    by_name={r["name"]:r for r in results}

    def panel(ax,name,lag,title):
        row=by_name[name];phase=row["selected_phase"]
        x,y,colors=maps[name,lag,phase]
        low=min(float(x.min()),float(y.min()))
        high=max(float(x.max()),float(y.max()))
        spread=max(high-low,1e-15)
        center=(low+high)/2
        u,v=(x-center)/spread,(y-center)/spread
        periodic=row["period_candidate"]>0
        ax.scatter(u,v,c=colors,s=22 if periodic else 6,alpha=.65 if periodic else .55,linewidths=0)
        ax.plot([-.65,.65],[-.65,.65],color="#c5c9d0",lw=.6,zorder=-1)
        ax.set(xlim=(-.65,.65),ylim=(-.65,.65),xlabel="Position at this visit",ylabel="Position at next visit",title=title)
        detail=(f"Period {row['period_candidate']}" + (" · verified" if row["verified_primitive_orbit"] else " · recurrence candidate")) if periodic else f"Phase {phase+1} · {lag}-step return"
        ax.text(.04,.92,detail,transform=ax.transAxes,color="#526079",fontsize=8)
        ax.set_aspect("equal",adjustable="box")

    fig,axes=plt.subplots(2,3,figsize=(13.5,8.8),layout="constrained")
    panel(axes[0,0],"mother-core",6,"Mother: the core")
    panel(axes[0,1],"baby-core",6,"Small island: the core")
    panel(axes[0,2],"baby-double-left",6,"Small island: the left side")
    # Fix the neighboring cell prospectively, rather than pick the prettiest curve.
    panel(axes[1,0],"mother-chaos",6,"Near the mother")
    panel(axes[1,1],"baby-chaos-left",6,"Near the small island")
    panel(axes[1,2],"baby-chaos-left",18,"Same nearby point, every 18 steps")
    fig.suptitle("Returning to the same band",fontsize=25,fontweight="bold")
    fig.supxlabel("Each dot links two visits along a phase's principal axis; coordinates scaled within each panel.\n"
                  "Six-step maps: blue = positive direction sign, orange = negative, gray = weak. The 18-step map is uncolored.\n"
                  "1000 transient + 3000 recorded steps · periodic points are discrete; projections alone do not establish a scalar map.",fontsize=9)
    fig.savefig(args.out/"baby-mother-returnmaps.png",dpi=170);fig.savefig(args.out/"baby-mother-returnmaps.pdf")
    plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(13.5,8.8),layout="constrained")
    ladder=[("mother-core","Mother: period 6"),
            ("mother-double","Mother: period 12"),
            ("mother-double-twice","Mother: period 24"),
            ("baby-core","Small island: period 18"),
            ("baby-double-left","Small island: period 36"),
            ("baby-unresolved","Small island: period 72")]
    for ax,(name,title) in zip(axes.ravel(),ladder):
        panel(ax,name,6,title)
        visits=by_name[name]["period_candidate"]//6
        ax.text(.04,.05,f"{visits} visit{'s' if visits!=1 else ''} per band before repeating",
                transform=ax.transAxes,color="#526079",fontsize=9)
    fig.suptitle("Longer cycles behind the same text",fontsize=25,fontweight="bold")
    fig.supxlabel("A dot links positions six model steps apart. The visible text repeats every six steps at every location.\n"
                  "The mother's sampled states repeat after 6, 12 or 24 steps; the small island's after 18, 36 or 72.\n"
                  "Distinct parameter locations; these samples alone do not trace a bifurcation sequence. Panel coordinates scaled separately.",fontsize=9)
    fig.savefig(args.out/"periodic-returnmaps.png",dpi=170);fig.savefig(args.out/"periodic-returnmaps.pdf");plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(10,9),layout="constrained")
    neighbors=[("mother-chaos","Near the mother"),
               ("baby-chaos-left","Left of the small island"),
               ("baby-chaos-right","Right of the small island"),
               ("baby-chaos-below","Below the small island")]
    for ax,(name,title) in zip(axes.ravel(),neighbors):
        panel(ax,name,6,title)
        row=by_name[name]
        ax.text(.04,.05,"Paper's coding test: " + ("passes" if row["mother_coding_gate_passed"] else "fails"),
                transform=ax.transAxes,color="#526079",fontsize=9)
    fig.suptitle("The surrounding return curves",fontsize=24,fontweight="bold")
    fig.supxlabel("Six-step returns along a phase's first principal axis; same phase-selection rule at all four fixed locations.\n"
                  "Blue = positive direction sign; orange = negative; gray = weak. Coding test pools all usable phases.\n"
                  "All four have positive finite-time exponents over 3000 recorded steps; long-term fate is unresolved.",fontsize=9)
    fig.savefig(args.out/"neighbor-returnmaps.png",dpi=170);fig.savefig(args.out/"neighbor-returnmaps.pdf");plt.close(fig)

    fig,axes=plt.subplots(3,4,figsize=(13.5,11),layout="constrained")
    for ax,cell in zip(axes.ravel(),info["cells"]):
        panel(ax,cell["name"],6,cell["name"].replace("-"," "))
    fig.suptitle("All twelve probe locations",fontsize=22,fontweight="bold")
    fig.supxlabel("Six-step returns · same selection and diagnostic rules for every location · scaled phase coordinates",fontsize=10)
    fig.savefig(args.out/"all-probe-returnmaps.png",dpi=160);plt.close(fig)

    np.savez_compressed(args.out/"returnmap-projections.npz",**{
        f"{name}_lag{lag}_phase{phase}":np.array([x,y]) for (name,lag,phase),(x,y,_) in maps.items()})
    root=Path(__file__).resolve().parents[1]
    receipt={"analysis_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
             "analysis_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "symbolic_diagnostics_source_sha256":hashlib.sha256(Path(sd.__file__).read_bytes()).hexdigest(),
             "orbit_metadata_sha256":hashlib.sha256(args.orbits.read_bytes()).hexdigest() if args.orbits else None,
             "source_trajectory_commit":status["source_commit"],"trajectory_metadata_sha256":hashlib.sha256((args.run/"trajectories.json").read_bytes()).hexdigest(),
             "states_sha256":hashlib.sha256((args.run/"states.npy").read_bytes()).hexdigest(),"cells":results,
             "diagnostics":"Paper SD definitions: six-step cosine threshold0.5, weak fraction<=0.1, bend<45deg, minority>=5; pooled error/minority<0.05.",
             "scope":"Representative points. No exhaustive periodic-orbit inventory or shrimp boundary classification."}
    (args.out/"returnmaps.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"out":str(args.out),"cells":len(results),"coding_passes":sum(r["mother_coding_gate_passed"] for r in results)}))


if __name__=="__main__":
    main()

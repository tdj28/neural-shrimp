"""Audit and render full-network critical trajectories and the period-18 orbit."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="8"
import numpy as np
import baby_shrimp_finish as audit
from baby_shrimp_jones_probe import lift,brackets


def precision(row):
    steps=np.array([f["h"] for f in row["finite_differences"]]);errors=np.array([f["relative_errors"] for f in row["finite_differences"]])
    audit.require(np.isfinite(errors).all() and (errors>0).all() and (np.diff(steps)<0).all(),"invalid finite-difference record")
    orders=np.log(errors[:-1]/errors[1:])/np.log(steps[:-1]/steps[1:])[:,None]
    audit.require(np.allclose(orders,row["orders"],rtol=1e-12,atol=1e-12),"finite-difference orders differ")
    passed=bool(row["plain_vs_ad_primal_error"]<1e-11 and np.all(errors.min(axis=0)<1e-5)
                and np.all(np.any((orders[:-1]>1.5)&(orders[:-1]<2.5)&(orders[1:]>1.5)&(orders[1:]<2.5),axis=0)))
    audit.require(passed==row["derivative_qualified"],"derivative gate label differs")
    return passed


def projected_word(coordinates,criticals):
    roots=np.sort([c["coordinate"] for c in criticals])
    out=[]
    for x in coordinates:
        if not 0<=x<=1:out.append("outside")
        elif np.min(abs(roots-x))<1e-6:out.append("turn")
        else:out.append(str(int(np.searchsorted(roots,x))))
    return out


def orbit_phase(points,curve):
    if points.shape!=(18,896):raise ValueError("verified period-18 full-state points required")
    distances=[]
    for phase in range(6):
        P=points[phase::6]
        distances.append(float(np.mean([np.linalg.norm(curve-v,axis=1).min() for v in P])))
    return int(np.argmin(distances)),distances


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ("raw","precision","reconstruction","trajectory","core","out"):ap.add_argument("--"+key,type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    raw=json.loads((args.raw/"results.json").read_text());final=json.loads((args.precision/"results.json").read_text())
    rec=json.loads((args.reconstruction/"jones.json").read_text())
    audit.require(raw["complete"] and final["complete"] and len(raw["cells"])==len(final["cells"])==6,"incomplete critical probe")
    audit.require(final["prior_results_sha256"]==audit.digest(args.raw/"results.json"),"precision extension input differs")
    audit.require(rec["states_sha256"]==audit.digest(args.trajectory/"states.npy"),"trajectory input differs")
    audit.require(rec["artifacts"]["reconstructions.npz"]["sha256"]==audit.digest(args.reconstruction/"reconstructions.npz"),"lift archive differs")
    for folder in (args.raw,args.precision):
        status=json.loads((folder/"status.json").read_text())
        audit.require(status["status"]=="complete","local worker incomplete")
        for source,digest in status["sources"].items():
            blob=subprocess.check_output(["git","show",status["source_commit"]+":"+source],cwd=audit.ROOT)
            import hashlib
            audit.require(hashlib.sha256(blob).hexdigest()==digest,"source snapshot differs")
    arrays=np.load(args.reconstruction/"reconstructions.npz",allow_pickle=False)
    core=np.load(args.core/"orbit-points.npy",allow_pickle=False)
    probe_receipt=json.loads((audit.ROOT/"docs/baby-shrimp-zoom/probe-receipt.json").read_text())
    audit.require(any(f["path"]==str((args.core/"orbit-points.npy").relative_to(audit.DATA)) and f["sha256"]==audit.digest(args.core/"orbit-points.npy") for f in probe_receipt["files"]),"core points not bound by original audit")
    core_row=next(r for r in probe_receipt["orbit_rows"] if r["name"]=="baby-core")
    audit.require(core_row["verified_primitive_orbit"] and core_row["period"]==18,"core orbit not qualified")
    rows=[];plotdata={}
    for old,new in zip(raw["cells"],final["cells"]):
        audit.require(old["name"]==new["name"],"row order differs")
        original_pass=precision(old)
        if original_pass:audit.require(old==new,"already qualified row changed")
        else:
            expected={**old,**{k:new[k] for k in ("derivative_qualified","plain_vs_ad_primal_error","finite_differences","orders","precision_extension")}}
            audit.require(expected==new,"precision extension changed scientific observations")
            with np.load(args.precision/old["name"]/"precision.npz",allow_pickle=False) as p:
                errors=abs(p["finite_differences"]-p["ad"])/np.maximum(abs(p["ad"]),1e-12)
                audit.require(np.allclose(errors,p["relative_errors"],rtol=1e-12,atol=1e-16),"extension raw finite differences differ")
                audit.require(np.allclose(errors,[f["relative_errors"] for f in new["finite_differences"]],rtol=1e-12,atol=1e-16),"extension summary differs")
        audit.require(precision(new),"final projected derivative failed")
        path=args.raw/old["name"]
        audit.require(json.loads((path/"summary.json").read_text())==old,"original summary differs")
        with np.load(path/"map.npz",allow_pickle=False) as m:mp={k:m[k] for k in m.files}
        audit.require(all(np.isfinite(v).all() for v in mp.values()),"nonfinite map arrays")
        prefix=f"{old['source']}__{old['phase']}__"
        coef=arrays[prefix+"lift_coefficients"];origin=mp["origin"];axis=mp["axis"];span=float(mp["span"])
        audit.require(np.array_equal(mp["u"],np.linspace(0,1,201)),"map sampling differs")
        audit.require(np.allclose(mp["input_states"],lift(coef,origin,mp["u"]),rtol=0,atol=1e-13),"input lift differs")
        g=(mp["output_states"]-origin)@axis/span
        audit.require(np.allclose(g,mp["image"],rtol=0,atol=1e-12),"projected map differs")
        inside=(g>=0)&(g<=1);off=np.linalg.norm(mp["output_states"][inside]-lift(coef,origin,g[inside]),axis=1)/span
        audit.require(np.isclose(np.sqrt(np.mean(off**2)),old["lift_invariance_rms_over_span"],rtol=1e-10),"invariance RMS differs")
        b=brackets(mp["u"],mp["derivative"])
        audit.require(len(b)==len(old["critical_points"])==len(list(path.glob("critical-*.npz"))),"critical-point coverage differs")
        candidates=[]
        for index,c in enumerate(old["critical_points"]):
            with np.load(path/f"critical-{index}.npz",allow_pickle=False) as f:crit={k:f[k] for k in f.files}
            audit.require(crit["orbit"].shape==(8,896) and all(np.isfinite(v).all() for v in crit.values()),"critical orbit shape/values")
            audit.require(np.allclose(crit["orbit"][0],lift(coef,origin,[c["coordinate"]])[0],rtol=0,atol=1e-13),"critical initial state differs")
            u=(crit["orbit"]-origin)@axis/span
            returns=np.linalg.norm(crit["orbit"][1:]-crit["orbit"][0],axis=1)/max(np.linalg.norm(crit["orbit"][0]),1)
            audit.require(np.allclose(u,c["coordinates"],rtol=0,atol=1e-12) and np.allclose(returns,c["full_state_returns_six_steps"],rtol=1e-12,atol=1e-15),"critical trajectory summary differs")
            audit.require(np.array_equal(crit["tokens"][6:],crit["tokens"][:-6])==c["exact_six_token_loop"],"critical token loop differs")
            audit.require(b[index][0]<c["coordinate"]<b[index][1] and c["bracket_width"]<1e-9 and abs(c["projected_derivative"])<1e-7,"critical root resolution failed")
            candidates.append({**c,"projected_branch_word":projected_word(u,old["critical_points"]),
                               "full_state_p18_closure_passed":bool(returns[2]<1e-11)})
        record={**new,"critical_points":candidates}
        if old["name"].startswith("baby-core-from-"):
            phase,distances=orbit_phase(core,mp["input_states"])
            P=core[phase::6];coordinate=(P-origin)@axis/span
            record["verified_core_projection"]={"phase":phase,"coordinates":coordinate.tolist(),"phase_alignment_distances":distances}
        rows.append(record);plotdata[old["name"]]=(mp,record)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(11,9),layout="constrained")
    titles=("Mother neighbor","Baby: left neighbor","Baby: right neighbor","Baby: lower neighbor")
    for ax,row,title in zip(axes.flat,rows[:4],titles):
        mp,_=plotdata[row["name"]];ax.plot(mp["u"],mp["image"],color="#213e55",lw=2)
        for c in row["critical_points"]:
            color="#00878a" if c["label"].startswith("C") else "#d26021"
            ax.scatter([c["coordinate"]],[c["projected_image"]],s=65,color=color,zorder=5)
            ax.annotate(c["label"][0],(c["coordinate"],c["projected_image"]),xytext=(10,14 if c["label"].startswith("C") else -22),textcoords="offset points",color=color,fontweight="bold")
        ax.plot([0,1],[0,1],color="#cbd5e1",lw=.7)
        ax.set(title=title,xlabel="Position now",ylabel="Position six steps later",xlim=(-.04,1.04),ylim=(-.04,1.07));ax.set_aspect("equal")
        ax.text(.4,.96,f"{len(row['critical_points'])} projected turning point(s)\nFull-network derivative check passes",transform=ax.transAxes,va="top",fontsize=8,bbox={"facecolor":"white","edgecolor":"none","alpha":.9})
    fig.suptitle("One turn becomes two around the baby window",fontsize=22,fontweight="bold")
    fig.supxlabel("Actual six-step network evaluations along fitted state curves; C = minimum, D = maximum.\nThe curves are approximate sections: a zero projected slope does not establish a zero full-state multiplier.",fontsize=9)
    for ext in ("png","pdf"):fig.savefig(args.out/f"jones-native-returnmaps.{ext}",dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout="constrained")
    for ax,row in zip(axes,rows[4:]):
        mp,_=plotdata[row["name"]];q=np.array(row["verified_core_projection"]["coordinates"])
        ax.plot(mp["u"],mp["image"],color="#64748b",lw=2,label="Network evaluated along initial curve")
        ax.scatter(q,np.roll(q,-1),s=80,color="#315bc4",zorder=5,label="Verified period-18 orbit")
        for c in row["critical_points"]:
            color="#00878a" if c["label"].startswith("C") else "#d26021"
            ax.scatter(c["coordinate"],c["projected_image"],s=70,color=color,zorder=6)
            ax.annotate(c["label"][0],(c["coordinate"],c["projected_image"]),xytext=(10,12 if c["label"].startswith("C") else -20),textcoords="offset points",color=color,fontweight="bold")
        ax.plot([-.05,1.1],[-.05,1.1],color="#cbd5e1",lw=.7)
        ax.set(title="Curve initialized from "+("below" if "below" in row["name"] else "the right"),xlabel="Position now",ylabel="Position six steps later",xlim=(-.06,1.13),ylim=(-.06,1.13));ax.set_aspect("equal")
        ax.text(.05,.78,"Three distinct visits before repeating",transform=ax.transAxes,fontsize=9)
    axes[0].legend(loc="center right",fontsize=8,frameon=False)
    fig.suptitle("The baby is a three-cycle of the six-step return",fontsize=21,fontweight="bold")
    fig.supxlabel("Same verified core parameters; two independently initialized state curves. Their critical candidates are not exact closed orbits.\nOffsets between the blue points and gray curve measure the limits of the scalar approximation.",fontsize=9)
    for ext in ("png","pdf"):fig.savefig(args.out/f"jones-core-three-cycle.{ext}",dpi=170)
    plt.close(fig)
    # Actual 896-dimensional trajectories, projected onto a common PCA frame
    # for the lower-neighbor phase and the aligned verified baby orbit.
    trajectory=json.loads((args.trajectory/"trajectories.json").read_text())
    states=np.load(args.trajectory/"states.npy",mmap_mode="r",allow_pickle=False)
    ci=next(i for i,c in enumerate(trajectory["cells"]) if c["name"]=="baby-chaos-below")
    S=np.array(states[::6,ci]);mean=S.mean(0);_,singular,vt=np.linalg.svd(S-mean,full_matrices=False);B=vt[:3].T
    row=rows[4];P=core[row["verified_core_projection"]["phase"]::6]
    fig=plt.figure(figsize=(11,7),layout="constrained");ax=fig.add_subplot(projection="3d")
    V=(S-mean)@B;ax.scatter(*V.T,s=6,color="#98a6b4",alpha=.25,label="Neighbor's observed states")
    V=(P-mean)@B;ax.scatter(*V.T,s=80,color="#315bc4",label="Verified baby cycle")
    ax.plot(*np.vstack([V,V[:1]]).T,color="#315bc4",lw=1.6)
    for k,c in enumerate(row["critical_points"]):
        with np.load(args.raw/row["name"]/f"critical-{k}.npz",allow_pickle=False) as f:V=(f["orbit"]-mean)@B
        color="#00878a" if c["label"].startswith("C") else "#d26021"
        ax.plot(*V.T,"o-",ms=4,color=color,alpha=.8,label=c["label"]+" trajectory")
        ax.text(*V[0],c["label"][0],color=color)
    explained=singular[:3]**2/np.sum(singular**2)
    ax.set(xlabel=f"PC1 ({100*explained[0]:.2f}% variance)",ylabel=f"PC2 ({100*explained[1]:.3f}%)",zlabel="PC3")
    ax.xaxis.labelpad=10;ax.yaxis.labelpad=10;ax.zaxis.labelpad=12
    ax.text2D(.02,.10,f"PC3 explains {100*explained[2]:.3f}% of observed variance",transform=ax.transAxes,fontsize=9)
    from matplotlib.ticker import MaxNLocator
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.set_major_locator(MaxNLocator(4))
    ax.view_init(elev=24,azim=-55);ax.legend(loc="upper left",fontsize=9)
    fig.suptitle("Following the critical candidates in state space",fontsize=21,fontweight="bold")
    fig.supxlabel("Every marker is an actual full-network state six steps later; the axes project 896 state coordinates.\nC/D begin on the lower-neighbor fitted curve at the baby-core parameters. Independent axis scales expose small transverse motion.",fontsize=9)
    for ext in ("png","pdf"):fig.savefig(args.out/f"jones-state-space.{ext}",dpi=170)
    plt.close(fig)
    manifest=[audit.artifact(p) for root in (args.raw,args.precision,args.reconstruction) for p in sorted(root.rglob("*")) if p.is_file()]
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    receipt={"id":"Jones-full-state-diagnostic","generated_utc":datetime.now(timezone.utc).isoformat(),
             "analysis_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=audit.ROOT,text=True).strip(),
             "source_sha256":audit.digest(__file__),"all_six_final_derivative_gates_passed":True,
             "raw_results":audit.artifact(args.raw/"results.json"),"precision_results":audit.artifact(args.precision/"results.json"),
             "core_audit_sha256":audit.digest(audit.ROOT/"docs/baby-shrimp-zoom/probe-receipt.json"),"rows":rows,
             "files":[audit.artifact(p) for p in args.out.iterdir() if p.is_file()],
             "limits":"Exploratory numerical diagnostics on fitted state curves, not invariant-manifold or conjugacy proofs. Original five baby precision rows are checked from saved error records; the supplemental mother check additionally retains raw finite differences. Projected branch words are not verified critical-point kneading sequences. No fixed-point homoclinic contact or periodicity hub is established."}
    (args.out/"receipt.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"rows":len(rows),"critical_candidates":sum(len(r["critical_points"]) for r in rows),"all_derivative_gates_passed":True}))


if __name__=="__main__":main()

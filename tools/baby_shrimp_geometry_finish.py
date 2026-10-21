"""Audit every continued orbit, then track only qualified spectral subspaces.

Zero-cell and exhausted-frontier outputs are preserved with their actual
coverage. Missing and spectrally failed cells cannot bridge contour curves.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "8"
import numpy as np
import baby_shrimp_finish as audit
from baby_shrimp_geometry import canonical, qualified_spectral_count
from baby_shrimp_probe_audit import precision_gate
from delta2d_analysis import critical_points
from baby_shrimp_transect_finish import verify_tar_files


def completion_evidence(folder, grid, result):
    """Explicit operator interruption preserves partial coverage, never completion."""
    status_path = folder / "status.json"
    if status_path.exists():
        return json.loads(status_path.read_text()), None
    note = json.loads((folder / "operator-stop.json").read_text())
    audit.require(note["pod_id"] == result["pod_id"] and note["source_commit"] == result["commit"]
                  and note["signal"] == "SIGTERM" and note["saved_grid_attempts"] == len(grid["rows"])
                  and note["grid_sha256_before_signal"] == audit.digest(folder / "grid.json"),
                  "interruption identity or last checkpoint differs")
    status = {k: grid[k] for k in ("attempts", "solved", "orbit_solved", "planned_cells", "remaining_frontier")}
    status.update(complete=False, reason="operator stopped the broad continuation; partial output preserved")
    return status, note


def gate_row(row, folder):
    """Recheck saved raw arrays. A failed row is retained, never promoted."""
    reasons = []
    history = row.get("newton_residuals", [])
    numeric = bool(history and all(isinstance(v,(int,float)) and np.isfinite(v) for v in history))
    if not numeric or history[-1] >= 1e-11:
        reasons.append("Newton closure")
    primitive = row.get("primitive", {})
    returns = np.asarray(primitive.get("all_returns", []))
    orbit_passed = bool(len(returns)==18 and np.isfinite(returns).all()
                        and returns[-1]<1e-11 and (returns[:-1]>1e-8).all() and not reasons)
    if not orbit_passed:
        reasons.append("primitive orbit")
    path = folder/"spectral.npz"
    arrays = None
    if path.exists():
        with np.load(path,allow_pickle=False) as a:
            arrays = {k:a[k] for k in a.files}
        points, values, vectors = (arrays[k] for k in ("points","multipliers","vectors"))
        audit.require(points.shape==(18,896) and values.shape==(896,) and vectors.shape==(896,6),"spectral array shape")
        audit.require(all(np.isfinite(a).all() for a in arrays.values()),"nonfinite saved spectral arrays")
        audit.require(np.all(np.diff(abs(values))<=1e-15),"unordered multipliers")
        state=np.load(folder/"state.npy",allow_pickle=False)
        audit.require(np.array_equal(state,points[0]),"state and orbit origin differ")
        distances=np.linalg.norm(points[1:]-points[0],axis=1)/max(np.linalg.norm(points[0]),1)
        audit.require(len(returns)==18 and np.allclose(distances,returns[:-1],rtol=1e-8,atol=1e-14),"lower-lag return mismatch")
        watched=qualified_spectral_count(values)
        audit.require(watched==row["qualified_spectral_count"],"watched spectral count mismatch")
        dense,direct=arrays["dense_jvp"],arrays["direct_jvp"]
        errors=np.linalg.norm(dense-direct,axis=0)/np.maximum(np.maximum(np.linalg.norm(dense,axis=0),np.linalg.norm(direct,axis=0)),1e-300)
        audit.require(np.allclose(errors,arrays["jvp_relative_error"],rtol=1e-12,atol=1e-16),"operator discrepancy mismatch")
        audit.require(np.allclose(errors,row["jvp_relative_error"],rtol=1e-12,atol=1e-16),"operator summary mismatch")
        audit.require(np.allclose(arrays["eigen_relative_residual"],row["eigen_relative_residual"],rtol=1e-12,atol=1e-16),"eigen-residual summary mismatch")
        audit.require(float(abs(values[0]))==row["spectral_radius"],"spectral radius mismatch")
        stored=np.array(row["multipliers"])
        audit.require(np.array_equal(stored[:,0]+1j*stored[:,1],values[:6]),"leading multiplier summary mismatch")
        if errors.max()>=1e-7:
            reasons.append("composed-JVP precision")
        if arrays["eigen_relative_residual"][:watched].max()>=1e-6:
            reasons.append("eigen-residual precision")
        same=canonical(arrays["tokens"].tolist())==canonical([271,785,3405,271,785,13885]*3)
        audit.require(same==row["same_cyclic_text"],"text gate mismatch")
        if not same:
            reasons.append("cyclic text")
    else:
        reasons.append("no saved spectral arrays")
    passed=not reasons
    audit.require(passed==row["passed"],"stored qualification disagrees with recomputed gates")
    return {"row":row["row"],"column":row["column"],"attempt":row["attempt"],
            "passed":passed,"orbit_passed":orbit_passed,"failure_reasons":reasons},arrays


def candidate_spans(values, vectors, count):
    candidates=[]
    for a,b in combinations(range(count),2):
        la,lb=values[a],values[b]
        complex_pair=abs(la.imag)>1e-14 or abs(lb.imag)>1e-14
        if complex_pair:
            if abs(la-lb.conjugate())>1e-9*max(abs(la),abs(lb),1e-20) or not (la.imag*lb.imag<0):
                continue
            v=vectors[:,a]
            matrix=np.column_stack([v.real,v.imag])
        else:
            columns=[]
            for k in (a,b):
                v=vectors[:,k]
                columns.append((v*np.exp(-1j*np.angle(v[np.argmax(abs(v))]))).real)
            matrix=np.column_stack(columns)
        s=np.linalg.svd(matrix,compute_uv=False)
        if s[-1]<=1e-10*s[0]:
            continue
        candidates.append(((a,b),np.linalg.qr(matrix)[0]))
    return candidates


def track(cells, shape, alpha, temp, seed):
    if not cells:
        return None,{"status":"no admissible spectral seed","tracked":0}
    eligible={c:candidate_spans(a["multipliers"],a["vectors"],qualified_spectral_count(a["multipliers"])) for c,a in cells.items()}
    starts=[c for c,cs in eligible.items() if any(pair==(0,1) for pair,q in cs)]
    if not starts:
        return None,{"status":"no admissible real two-dimensional seed","tracked":0}
    start=min(starts,key=lambda c:((alpha[c[1]]-seed[0])/np.ptp(alpha))**2+((temp[c[0]]-seed[1])/np.ptp(temp))**2)
    q0=next(q for pair,q in eligible[start] if pair==(0,1))
    basis={start:q0}; pairs={start:(0,1)}; overlap={start:1.0}
    queue=deque([start]); rejected=[]
    while queue:
        c=queue.popleft()
        for d in ((c[0]+1,c[1]),(c[0]-1,c[1]),(c[0],c[1]+1),(c[0],c[1]-1)):
            if d not in cells or d in basis or not eligible[d]:
                continue
            distances=np.linalg.norm(cells[d]["points"]-cells[c]["points"][0],axis=1)
            phase=int(distances.argmin())
            if phase!=0:
                rejected.append({"parent":c,"child":d,"reason":"cyclic phase changes","phase":phase})
                continue
            scores=[float(np.mean(np.linalg.svd(basis[c].T@q,compute_uv=False)**2)) for pair,q in eligible[d]]
            k=int(np.argmax(scores))
            if scores[k]<.9:
                rejected.append({"parent":c,"child":d,"reason":"subspace overlap","overlap":scores[k]})
                continue
            pairs[d],basis[d]=eligible[d][k];overlap[d]=scores[k];queue.append(d)
    fields={k:np.full(shape,np.nan) for k in ("tau","delta","rho","overlap","leading_real","leading_imag")}
    for c,pair in pairs.items():
        values=cells[c]["multipliers"]
        a,b=values[list(pair)]
        fields["tau"][c]=(a+b).real;fields["delta"][c]=(a*b).real
        fields["rho"][c]=abs(values[0]);fields["overlap"][c]=overlap[c]
        fields["leading_real"][c]=values[0].real;fields["leading_imag"][c]=values[0].imag
    valid=np.isfinite(fields["tau"])
    fits={key:critical_points(fields[key],valid,alpha,temp) for key in ("tau","delta")}
    for rows in fits.values():
        for row in rows:
            row["fit_quality_passed"]=row["field_scale"]>0 and row["residual_rms"]<=.05*row["field_scale"]
            row["value_exceeds_five_fit_rms"]=abs(row["value"])>5*row["residual_rms"]
    return fields,{"status":"tracked seed-connected qualified cells","start_cell":start,"tracked":len(basis),
                   "qualified_but_untracked":len(cells)-len(basis),"rejected_edges":rejected,
                   "minimum_overlap":min(overlap.values()),"stationary_fits":fits,
                   "stationary_status":"local fits retained" if any(fits.values()) else "no admissible complete 5x5 stationary fit"}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers",nargs="+",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    receipts=[];cells={};manifest=[];all_rows=[];first_plan=None
    import runpod_pod as rp
    ledger=[json.loads(s) for s in rp.LEDGER.read_text().splitlines()]
    inventory={p["id"] for p in rp.inventory()["pods"]}
    for worker in args.workers:
        result=json.loads((worker/"result.json").read_text());archive=worker/"results.tgz"
        audit.require(result["hash_verified"] and audit.digest(archive)==result["local_sha256"]==result["remote_sha256"],"archive mismatch")
        argv=shlex.split(result["upload"]["jobset_args"])
        folder=audit.DATA/"explore"/argv[argv.index("--out")+1]
        protocol=json.loads((folder/"protocol.json").read_text());plan=protocol["plan"]
        if first_plan is None:
            first_plan=plan
        audit.require(all(plan[k]==first_plan[k] for k in ("alpha","log10_temperature","resolution","period","seed","model_revision")),"incompatible geometry grids")
        plan_path=protocol["args"]["plan"]
        blob=subprocess.check_output(["git","show",result["commit"]+":"+plan_path],cwd=audit.ROOT)
        audit.require(hashlib.sha256(blob).hexdigest()==protocol["plan_sha256"] and json.loads(blob)==plan,"plan/source binding mismatch")
        audit.require(precision_gate({"derivative_precision":protocol["seed_precision"]}),"seed precision failed")
        expected={str(path.relative_to(audit.DATA)):path for path in sorted(folder.rglob("*")) if path.is_file()}
        hardware,versions=verify_tar_files(archive,expected)
        manifest.extend(audit.artifact(path) for path in expected.values())
        pid=result["pod_id"]
        audit.require(any(e.get("event")=="terminated_verified" and e.get("pod_id")==pid for e in ledger),"termination record absent")
        audit.require(pid not in inventory and rp.pod_status(pid) is None,"pod still present")
        grid=json.loads((folder/"grid.json").read_text());status,interruption=completion_evidence(folder,grid,result)
        audit.require(status["attempts"]==len(grid["rows"])==len(list((folder/"cells").glob("*/cell.json"))),"attempt coverage mismatch")
        rows=[]
        for row in grid["rows"]:
            row_lo,row_hi=plan["bands"][protocol["args"]["band"]]["rows"]
            audit.require(row_lo<=row["row"]<row_hi and 0<=row["column"]<plan["resolution"][1],"row outside declared band")
            audit.require(row["alpha"]==float(np.linspace(*plan["alpha"],plan["resolution"][1])[row["column"]])
                          and row["log10_temperature"]==float(np.linspace(*plan["log10_temperature"],plan["resolution"][0])[row["row"]]),"row coordinate mismatch")
            dest=folder/"cells"/f"r{row['row']:02d}-c{row['column']:02d}-a{row['attempt']:02d}"
            audit.require(json.loads((dest/"cell.json").read_text())==row,"grid/raw row mismatch")
            checked,arrays=gate_row(row,dest);rows.append(checked)
            if checked["passed"]:
                cell=(row["row"],row["column"])
                audit.require(cell not in cells,"duplicate qualified grid location across supplied workers")
                cells[cell]=arrays
        passed=sum(r["passed"] for r in rows)
        audit.require(passed==status["solved"],"qualified-cell count mismatch")
        incomplete=[p.name for p in sorted((folder/"cells").iterdir()) if p.is_dir() and not (p/"cell.json").exists()]
        audit.require(not incomplete or interruption is not None,"unexplained incomplete cell")
        receipts.append({"pod_id":pid,"source_commit":result["commit"],"plan_sha256":protocol["plan_sha256"],
                         "archive":audit.artifact(archive),"raw_status":status,"qualified_cells":passed,
                         "operator_interruption":interruption,"incomplete_cell_directories":incomplete,
                         "scientific_status":"failed setup: no qualified spectral cells" if not passed else "partial spatial coverage" if passed<status["planned_cells"] else "full qualified coverage",
                         "termination_verified":True,"versions_sha256":hashlib.sha256(versions).hexdigest(),
                         "hardware_sha256":hashlib.sha256(hardware).hexdigest(),"hardware":hardware.decode().strip()})
        all_rows.extend(rows)
    alpha=np.linspace(*plan["alpha"],plan["resolution"][1]);temp=np.linspace(*plan["log10_temperature"],plan["resolution"][0])
    fields,summary=track(cells,tuple(plan["resolution"]),alpha,temp,plan["seed"])
    if fields is not None:
        np.savez_compressed(args.out/"fields.npz",alpha=alpha,log10_temperature=temp,**fields)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm
        fig,axes=plt.subplots(1,3,figsize=(16,5.5),layout="constrained")
        for ax,key,title in zip(axes,("rho","tau","delta"),("Contraction per 18-step cycle","Trace of the tracked pair","Determinant of the tracked pair")):
            data=np.log10(fields[key]) if key=="rho" else fields[key]
            finite=data[np.isfinite(data)]
            scale=float(max(abs(finite).max(),1e-20))
            norm=None if key=="rho" else SymLogNorm(linthresh=scale*.001,vmin=-scale,vmax=scale)
            im=ax.pcolormesh(alpha,temp,np.ma.masked_invalid(data),cmap="viridis" if key=="rho" else "RdBu_r",norm=norm,shading="nearest",rasterized=True)
            fig.colorbar(im,ax=ax,label="log10 spectral radius" if key=="rho" else key)
            if key!="rho" and finite.min()<0<finite.max():
                ax.contour(alpha,temp,np.ma.masked_invalid(data),levels=[0],colors="black",linewidths=1)
            ax.set(title=title,xlabel="Steering strength",ylabel="log10 temperature")
            ax.ticklabel_format(useOffset=False)
        fig.suptitle("The baby orbit's contraction geometry",fontsize=21,fontweight="bold")
        fig.supxlabel("Only qualified, phase-aligned, seed-connected cells are shown. White cells remain unqualified or unvisited.\nBlack curves interpolate zero levels between adjacent qualified cells; no exact intersection follows from this grid alone.",fontsize=9)
        for suffix in ("png","pdf"):fig.savefig(args.out/f"baby-geometry.{suffix}",dpi=170)
        plt.close(fig)
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (args.out/"rows.json").write_text(json.dumps(all_rows,indent=2)+"\n")
    receipt={"id":"BSG-geometry-audit","generated_utc":datetime.now(timezone.utc).isoformat(),
             "analysis_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=audit.ROOT,text=True).strip(),
             "analysis_source_sha256":audit.digest(__file__),"workers":receipts,"analysis":summary,
             "files":[audit.artifact(p) for p in args.out.iterdir() if p.is_file()],
             "limits":"Numerical, partial-coverage analysis. Saved eigen-residuals are checked against their recorded gates, not independently recomputed without a saved dense matrix. Full-state closure uses the recorded final return. A five-fit-RMS comparison is a sensitivity diagnostic, not a statistical confidence bound. No exact nilpotent orbit or complete boundary organization is certified.","cost":"see private ledger"}
    (args.out/"receipt.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"workers":len(receipts),"attempts":len(all_rows),"qualified_cells":len(cells),"analysis":summary}))


if __name__=="__main__":
    main()

"""Bind BSP1/BSP2 trajectories, every orbit gate, and return-map figures to raw evidence."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="8"
import numpy as np
import baby_shrimp_finish as audit
from baby_shrimp_trajectories import recurrence
from symbolic_dynamics import rotmin_key


def precision_gate(row):
    p=row["derivative_precision"]
    steps=np.array(p["central_difference_relative_steps"])
    errors=np.array(p["central_difference_relative_errors"])
    audit.require(np.isfinite(errors).all() and (errors>0).all(),"invalid finite-difference errors")
    orders=np.log(errors[:-1]/errors[1:])/np.log(steps[:-1]/steps[1:])
    quadratic=bool(np.any((orders[:-1]>=1.5)&(orders[:-1]<=2.5)&(orders[1:]>=1.5)&(orders[1:]<=2.5)))
    passed=bool(p["dense_vs_composed_jvp_relative_error"]<1e-7 and quadratic and errors.min()<1e-5
                and p["spectral_radius_relative_difference"]<1e-5 and p["refined_leading_relative_residual"]<1e-6)
    audit.require(p["passed"]==passed and p["quadratic_regime_found"]==quadratic,"stored derivative gate differs")
    return passed


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectory",type=Path,required=True)
    ap.add_argument("--worker",type=Path,required=True)
    ap.add_argument("--extension",type=Path,required=True)
    ap.add_argument("--returnmaps",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=audit.ROOT,text=True).strip()
        for name in ("tools/baby_shrimp_probe_audit.py","tools/baby_shrimp_trajectories.py","tools/baby_shrimp_finish.py","tools/symbolic_dynamics.py"):
            blob=subprocess.check_output(["git","show",commit+":"+name],cwd=audit.ROOT)
            audit.require(hashlib.sha256(blob).hexdigest()==audit.digest(audit.ROOT/name),f"uncommitted analysis source {name}")
        info=json.loads((args.trajectory/"trajectories.json").read_text())
        status=json.loads((args.trajectory/"status.json").read_text())
        selection=audit.ROOT/"docs/baby-shrimp-zoom/probe-cells.json"
        plan=json.loads(selection.read_text())
        audit.require(status["status"]=="complete" and status["cells_sha256"]==audit.digest(selection),"trajectory status/selection mismatch")
        for name,sha in status["source_sha256"].items():
            blob=subprocess.check_output(["git","show",status["source_commit"]+":"+name],cwd=audit.ROOT)
            audit.require(hashlib.sha256(blob).hexdigest()==sha,"trajectory source mismatch")
        protocol=info["protocol"]
        for key,value in (("transient",1000),("record",3000),("eps",1e-6),("phase_period",6),("max_period",192),("batch_cells",12)):
            audit.require(protocol[key]==value,f"trajectory setting mismatch: {key}")
        audit.require(protocol["cells"]==plan["cells"] and protocol["dtype"]=="torch.float64" and protocol["rotary_float64"],"trajectory protocol mismatch")
        audit.require(protocol["model_revision"]=="7ae557604adf67be50417f59c2c2f167def9a775" and 0<protocol["norm_sensitivity_to_1e_12"]<1e-9,"trajectory effective precision mismatch")
        states=np.load(args.trajectory/"states.npy",mmap_mode="r",allow_pickle=False)
        diag=np.load(args.trajectory/"diagnostics.npz",allow_pickle=False)
        partial=np.load(args.trajectory/"diagnostics-partial.npz",allow_pickle=False)
        audit.require(states.shape==(3000,12,896) and states.dtype==np.float64 and np.isfinite(states).all(),"invalid full-state record")
        audit.require(int(partial["recorded"])==3000,"diagnostic checkpoint incomplete")
        for key in ("tokens","log_growth","word_cos"):
            audit.require(diag[key].shape==(3000,12) and np.array_equal(diag[key],partial[key],equal_nan=True),f"diagnostic checkpoint mismatch: {key}")
        audit.require(np.isfinite(diag["log_growth"]).all() and np.isfinite(diag["word_cos"][6:]).all(),"nonfinite diagnostics")
        traj_rows=[]
        for i,cell in enumerate(info["cells"]):
            audit.require(all(cell[k]==plan["cells"][i][k] for k in plan["cells"][i]),"trajectory cell ordering mismatch")
            observed=recurrence(states[:,i,:])
            audit.require(observed["period_candidate"]==cell["period_candidate"] and np.array_equal(observed["max_relative_return_by_lag"],cell["max_relative_return_by_lag"]),"recurrence summary mismatch")
            growth=diag["log_growth"][:,i]
            audit.require(float(growth.mean())==cell["lyapunov"],"exponent summary mismatch")
            halves=[float(v.mean()) for v in np.array_split(growth,2)]
            audit.require(halves==cell["lyapunov_halves"],"exponent half-window summary mismatch")
            text=diag["tokens"][:,i]
            tp=next((q for q in range(1,193) if np.array_equal(text[q:],text[:-q])),0)
            audit.require(tp==cell["text_period"]==6 and text[:tp].tolist()==cell["text_token_ids"],"text loop summary mismatch")
            traj_rows.append({k:cell[k] for k in ("name","alpha","log10_temperature","period_candidate","lyapunov","lyapunov_halves","text_period")})
        text_loops={tuple(rotmin_key(c["text_token_ids"])) for c in info["cells"]}
        audit.require(len(text_loops)==1,"text loops are not the same cyclic word")
        seed=audit.DATA/"explore/llm-soft-loop/bsz1-probe-input"
        seed_info=json.loads((seed/"seeds.json").read_text())
        audit.require(seed_info["states_sha256"]==audit.digest(seed/"seeds.npz") and seed_info["cells"]==info["cells"],"orbit seed metadata mismatch")
        audit.require(seed_info["provenance"]["trajectory_states_sha256"]==audit.digest(args.trajectory/"states.npy"),"seed trajectory hash mismatch")
        audit.require(np.array_equal(np.load(seed/"seeds.npz",allow_pickle=False)["states"],states[-1]),"seed state mismatch")
        worker=json.loads((args.worker/"result.json").read_text())
        archive=args.worker/"results.tgz"
        audit.require(worker["hash_verified"] and worker["runner_returncode"]==0 and not worker.get("extract_clash"),"orbit transfer failed")
        audit.require(audit.digest(archive)==worker["local_sha256"]==worker["remote_sha256"],"orbit archive hash mismatch")
        audit.require("end bsp2-orbits status 0" in (args.worker/"runner-stdout.txt").read_text(),"orbit numerical command failed")
        orbit_dir=audit.DATA/"explore/llm-soft-loop/pod64/bsz1-probes/orbits-01"
        with tarfile.open(archive) as tar:
            for path in sorted(orbit_dir.rglob("*")):
                if path.is_file():
                    raw=tar.extractfile(str(path.relative_to(audit.DATA))).read()
                    audit.require(hashlib.sha256(raw).hexdigest()==audit.digest(path),f"orbit raw/local mismatch: {path.name}")
            hardware=tar.extractfile("explore/llm-soft-loop/pod64/logs/hardware.txt").read()
            versions=tar.extractfile("explore/llm-soft-loop/pod64/logs/versions.txt").read()
        original=json.loads((orbit_dir/"orbits.json").read_text())
        updated=json.loads((args.extension/"orbits.json").read_text())
        ext=json.loads((args.extension/"extension.json").read_text())
        audit.require(json.loads((args.extension/"status.json").read_text())["status"]=="complete","precision extension incomplete")
        audit.require(ext["original_orbits_sha256"]==audit.digest(orbit_dir/"orbits.json"),"extension input hash mismatch")
        prior_extension=args.extension.parent/"20261021-precision-extension-01"
        prior=json.loads((prior_extension/"extension.json").read_text())
        audit.require(prior["original_orbits_sha256"]==ext["original_orbits_sha256"] and not prior["derivative_precision"]["passed"],"prior failed precision attempt missing")
        audit.require(original["complete"] and updated["complete"] and len(original["cells"])==len(updated["cells"])==12,"orbit rows incomplete")
        orbit_rows=[]
        for old,new,cell in zip(original["cells"],updated["cells"],info["cells"]):
            audit.require(old["name"]==new["name"]==cell["name"],"orbit row ordering mismatch")
            audit.require(all(old[k]==cell[k] for k in cell),"orbit seed parameters differ")
            if old["name"]!="baby-unresolved": audit.require(old==new,"unselected orbit changed")
            p=cell["period_candidate"]
            if not p:
                audit.require("no recurrence candidate" in old["status"],"noncandidate was silently substituted")
                continue
            folder=orbit_dir/old["name"]
            points=np.load(folder/"orbit-points.npy",allow_pickle=False)
            M=np.load(folder/"monodromy.npy",allow_pickle=False)
            vals=np.load(folder/"multipliers.npy",allow_pickle=False)
            audit.require(points.shape==(p,896) and M.shape==(896,896) and vals.shape==(896,),"orbit array shape mismatch")
            audit.require(all(np.isfinite(a).all() for a in (points,M,vals)),"nonfinite orbit arrays")
            audit.require(float(abs(vals).max())==old["spectral_radius"],"spectral-radius mismatch")
            primitive=old["primitive_gate"]
            returns=np.array(primitive["all_returns"])
            audit.require(len(returns)==p and returns[-1]<1e-11 and np.all(returns[:-1]>1e-8),"primitive gate failed")
            audit.require(old["newton_residuals"][-1]<1e-11,"closure failed")
            old_pass=precision_gate(old); new_pass=precision_gate(new)
            audit.require(old["verified_primitive_orbit"]==old_pass and new["verified_primitive_orbit"]==new_pass,"verified label differs from gates")
            audit.require(new_pass and new["spectral_radius"]<1-1e-8 and new["attracting"],"final numerical orbit or attraction gate failed")
            orbit_rows.append({"name":old["name"],"period":p,"alpha":cell["alpha"],"log10_temperature":cell["log10_temperature"],
                               "closure":float(returns[-1]),"minimum_lower_return":float(returns[:-1].min()),"spectral_radius":old["spectral_radius"],
                               "leading_multiplier":old["multipliers"][0],"asymptotic_log_contraction_per_step":old["asymptotic_log_contraction_per_step"],
                               "original_precision_passed":old_pass,"final_precision_passed":new_pass,"verified_primitive_orbit":True,"attracting":True})
        maps=json.loads((args.returnmaps/"returnmaps.json").read_text())
        audit.require(maps["trajectory_metadata_sha256"]==audit.digest(args.trajectory/"trajectories.json") and maps["orbit_metadata_sha256"]==audit.digest(args.extension/"orbits.json"),"return-map input mismatch")
        audit.require(len(maps["cells"])==12,"return-map rows incomplete")
        import runpod_pod as rp
        ledger=[json.loads(s) for s in rp.LEDGER.read_text().splitlines()]
        pid=worker["pod_id"]
        audit.require(any(e.get("event")=="terminated_verified" and e.get("pod_id")==pid for e in ledger),"orbit pod termination not recorded")
        audit.require(pid not in {p["id"] for p in rp.inventory()["pods"]} and rp.pod_status(pid) is None,"orbit pod still present")
        files=[audit.artifact(p) for folder in (args.trajectory,orbit_dir,args.extension,args.returnmaps) for p in sorted(folder.rglob("*")) if p.is_file()]
        receipt={"id":"BSP1-BSP2","generated_utc":datetime.now(timezone.utc).isoformat(),"status":"all trajectory, orbit, linkage and lifecycle gates passed",
                 "analysis_commit":commit,"analysis_source_sha256":audit.digest(__file__),"trajectory_commit":status["source_commit"],
                 "orbit_commit":worker["commit"],"extension_commit":ext["source_commit"],"pod_id":pid,"termination_verified":True,
                 "hardware":hardware.decode().strip(),"versions":versions.decode().strip(),
                 "hardware_sha256":hashlib.sha256(hardware).hexdigest(),"versions_sha256":hashlib.sha256(versions).hexdigest(),
                 "archive":audit.artifact(archive),"files":files,"trajectory_rows":traj_rows,"orbit_rows":orbit_rows,
                 "preserved_failed_extension":[audit.artifact(prior_extension/name) for name in ("extension.json","status.json","derivative-check.npz","orbits.json")],
                 "canonical_token_loop":list(next(iter(text_loops))),"all_twelve_share_text_loop":True,
                 "return_map_rows":[{k:r[k] for k in ("name","selected_phase","usable_phases","pooled_errors_over_minority","mother_coding_gate_passed")} for r in maps["cells"]],
                 "limits":["Numerical checks, not interval certificates or full boundary organization.","Selected examples, not an exhaustive periodic-orbit inventory.",
                           "Neighbor expansion is measured for3000steps; long-term attraction or eventual capture is unresolved.",
                           "The period72 precision gate failed in two preserved attempts before the disclosed extension."],"cost":"see private ledger"}
        (args.out/"receipt.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
        print(json.dumps({"out":str(args.out),"status":receipt["status"],"verified_orbits":len(orbit_rows)}))
    except BaseException as error:
        (args.out/"failed.json").write_text(json.dumps({"type":type(error).__name__,"error":str(error)})+"\n")
        raise


if __name__=="__main__":
    main()

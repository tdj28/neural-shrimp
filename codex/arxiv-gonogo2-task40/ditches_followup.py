"""Disclosed follow-up to failed exact-axis check; retain both grids' coordinates.

No original trajectory, Newton or model replay. Corresponding lattice cells
are compared only after recording the coordinate differences, with a maximum
one-ULP correspondence gate. This is not an identical-input device benchmark.
"""
import io
import json
import time
import numpy as np
from delta_ditch_audit import BASE, OUT, PINS, read, grid, combine, stats


def main():
    tic = time.monotonic()
    result = {"prior_attempt": "ditches-001", "prior_outcome": "exact axes differed",
              "limits": ["Near-identical parameter cells are not identical binary64 inputs.",
                         "HR10 nearest-cell periods do not establish full orbit identity.",
                         "No model, trajectory, Newton or original controls rerun."]}
    runs = ["pod64/dss/p6-body", "p6s/ditch-top-a", "p6s/ditch-main-b"]
    groups = {name: grid(BASE/name) for name in runs}
    groups["ditch2d"] = combine([BASE/f"pod64/dss/ditch2d-b{k}" for k in range(4)])
    for name, z in groups.items():
        result[name] = stats(z)
        valid = z["ok"]
        result[name]["max_leading_modulus"] = float(np.max(np.hypot(z["mu"], z["mu_imag"])[valid]))
    cpu_folders = [BASE/f"p6s/ditch-fine-b{k}" for k in range(3)]
    gpu_folders = [BASE/f"pod64/dss/ditchfine-g{k}" for k in range(2)]
    result["fine_metadata"] = [json.loads(read(f/"grid.json")) for f in cpu_folders+gpu_folders]
    cpu, gpu = combine(cpu_folders), combine(gpu_folders)
    for label,z in (("cpu",cpu),("gpu",gpu)):
        if not z["ok"].all() or not all(np.isfinite(z[k]).all() for k in ("mu","mu_imag","next_modulus","newton")):
            raise ValueError(f"Fine grid {label}: incomplete or nonfinite; no unmasked minimum")
    result["fine_cpu"], result["fine_gpu"] = stats(cpu), stats(gpu)
    axes = {}
    correspondence = True
    for key in ("alpha", "log10_temperature"):
        a,b = cpu[key],gpu[key]
        if a.shape != b.shape:
            raise ValueError("Axis shape differs: no cell comparison")
        distance = np.abs(a-b)
        ulp = np.abs(np.spacing(b))
        axis_pass = bool(np.all(distance <= ulp))
        correspondence &= axis_pass
        axes[key] = {"cpu_hex": [float(x).hex() for x in a],
                     "gpu_hex": [float(x).hex() for x in b],
                     "exact_equal": bool(np.array_equal(a,b)),
                     "max_absolute_difference": float(distance.max()),
                     "max_ulp_difference": float((distance/ulp).max()),
                     "differing_indices": np.flatnonzero(a!=b).tolist(),
                     "within_one_ulp": axis_pass}
    result["axis_comparison"] = axes
    result["one_ulp_correspondence"] = correspondence
    if correspondence:
        valid=cpu["ok"] & gpu["ok"]
        if not valid.any():
            raise ValueError("No common solved cells")
        dif=np.abs(cpu["mu"]+1j*cpu["mu_imag"]-gpu["mu"]-1j*gpu["mu_imag"])
        result["fine_comparison"] = {
            "common_solved":int(valid.sum()),
            "mismatched_ok":int(np.sum(cpu["ok"]!=gpu["ok"])),
            "max_leading_complex_abs_diff":float(dif[valid].max()),
            "median_leading_complex_abs_diff":float(np.median(dif[valid])),
            "max_relative_leading_diff":float(np.max(dif[valid]/np.hypot(gpu["mu"],gpu["mu_imag"])[valid])),
            "max_next_abs_diff":float(np.max(np.abs(cpu["next_modulus"]-gpu["next_modulus"])[valid])),
            "same_tokens_mismatches":int(np.sum(cpu["same_tokens"][valid]!=gpu["same_tokens"][valid]))}
    for label,z in (("fine_cpu",cpu),("fine_gpu",gpu)):
        col=int(np.argmin(abs(z["alpha"]-.07318)))
        result[label]["column_alpha"]=float(z["alpha"][col])
        result[label]["column_min"]=float(z["mu"][:,col].min())
        result[label]["column_min_log10T"]=float(z["log10_temperature"][np.argmin(z["mu"][:,col])])
        result[label]["row_minimum_columns"]=np.argmin(z["mu"],axis=1).tolist()
    # Crosscheck the blanket fixed-start assertion against the saved HR10 map.
    with np.load(io.BytesIO(read(BASE/"pod64/hr10/scan.npz")),allow_pickle=False) as scan:
        aa,tt,pp=scan["alpha"],scan["log10_temperature"],scan["period"]
        comparisons={}
        for name,z in groups.items():
            rows=[]
            maximum=0.0
            for i,j in np.argwhere(z["ok"]):
                ai=int(np.argmin(abs(aa-z["alpha"][j])))
                ti=int(np.argmin(abs(tt-z["log10_temperature"][i])))
                separation=max(abs(aa[ai]-z["alpha"][j])/abs(aa[1]-aa[0]),
                               abs(tt[ti]-z["log10_temperature"][i])/abs(tt[1]-tt[0]))
                maximum=max(maximum,float(separation))
                if int(pp[ti,ai])!=6:
                    rows.append({"continued_index":[int(i),int(j)],
                                 "continued_parameters":[float(z["alpha"][j]),float(z["log10_temperature"][i])],
                                 "nearest_scan_index":[ti,ai],
                                 "nearest_scan_parameters":[float(aa[ai]),float(tt[ti])],
                                 "scan_period":int(pp[ti,ai]),"distance_in_scan_steps":float(separation)})
            comparisons[name]={"solved":int(z["ok"].sum()),
                               "nearest_cell_max_distance_steps":maximum,
                               "nearest_period_not_six":rows}
        result["fixed_start_nearest_cell_comparison"]=comparisons
    result["inputs"]=PINS
    result["seconds"]=time.monotonic()-tic
    with (OUT/"ditches-followup-evidence.json").open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({"seconds":result["seconds"],"one_ulp_correspondence":correspondence,
                      "fine_comparison":result.get("fine_comparison")}))


if __name__=="__main__":
    main()

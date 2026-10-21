"""Audit and plot the complete BSP3 longer-period survey, with raw-cell gates."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="8"
import numpy as np
import baby_shrimp_finish as audit


def render(folder,out,cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    z=np.load(folder/"scan.npz",allow_pickle=False)
    periods=z["period"]
    labels=sorted(int(p) for p in np.unique(periods) if p>0)
    colors=["#d9dee7"]+[plt.get_cmap("tab20")(i%20) for i in range(len(labels))]
    preferred={18:"#edb120",36:"#0072b2",72:"#cc79a7",144:"#009e73"}
    for i,p in enumerate(labels,1):
        colors[i]=preferred.get(p,colors[i])
    image=np.zeros(periods.shape,dtype=int)
    for i,p in enumerate(labels,1): image[periods==p]=i
    a,t=z["alpha"],z["log10_temperature"]
    da,dt=(a[1]-a[0])/2,(t[1]-t[0])/2
    extent=[a[0]-da,a[-1]+da,t[0]-dt,t[-1]+dt]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(10,9),layout="constrained")
    ax.imshow(image,origin="lower",extent=extent,interpolation="nearest",aspect="auto",
              cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,len(colors)+.5),len(colors)))
    marked=[]
    for row in cells:
        if row["name"].startswith("baby-"):
            ax.scatter(row["alpha"],row["log10_temperature"],s=60,facecolors="none",edgecolors="black",lw=1.1)
            short={"baby-core":"18", "baby-arch":"18", "baby-double-left":"36", "baby-double-right":"36",
                   "baby-unresolved":"72", "baby-chaos-left":"L", "baby-chaos-right":"R", "baby-chaos-below":"B"}[row["name"]]
            ax.annotate(short,(row["alpha"],row["log10_temperature"]),xytext=(6,5),textcoords="offset points",fontsize=9)
            marked.append({"name":row["name"],"alpha":row["alpha"],"log10_temperature":row["log10_temperature"],"annotation":short})
    ax.set(xlabel="Steering strength α",ylabel="log₁₀ temperature")
    ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(6));ax.ticklabel_format(useOffset=False)
    handles=[Patch(color=colors[0],label="No recurrence ≤192")]+[Patch(color=colors[i],label=f"Period {p}") for i,p in enumerate(labels,1)]
    ax.legend(handles=handles,loc="upper left",bbox_to_anchor=(1.01,1),frameon=False,fontsize=9)
    fig.suptitle("How long before the hidden state repeats?",fontsize=23,fontweight="bold")
    fig.supxlabel("128 × 128 cells · 1000 settling + 768 measured steps · full-state recurrence candidates through period192\n"
                  "Circles locate the separate trajectory probes. L/R/B are expanding neighbors; numbers are candidate periods.\n"
                  "Gray includes both irregular and insufficiently settled trajectories. Selected orbits are verified separately.",fontsize=9)
    fig.savefig(out/"baby-shrimp-periods.png",dpi=180);fig.savefig(out/"baby-shrimp-periods.pdf");plt.close(fig)
    return {"period_classes":labels,"probe_markers":marked,"interpolation":"nearest"}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        for source in ("tools/baby_shrimp_finish.py","tools/baby_shrimp_period_survey_finish.py"):
            blob=subprocess.check_output(["git","show","HEAD:"+source],cwd=audit.ROOT)
            audit.require(audit.hashlib.sha256(blob).hexdigest()==audit.digest(audit.ROOT/source),"commit audit source before execution")
        plan_path=audit.ROOT/"docs/baby-shrimp-zoom/period-survey-plan.json"
        plan=json.loads(plan_path.read_text())
        plan["weights"]="native"
        band={"name":"bsz1-periods","resolution":plan["resolution"],"rows":[0,plan["resolution"][0]],
              "log10_temperature":plan["log10_temperature"]}
        row=audit.gate_band(plan,band,args.worker)
        import runpod_pod as rp
        ledger=[json.loads(s) for s in rp.LEDGER.read_text().splitlines()]
        pid=row["pod_id"]
        audit.require(any(e.get("event")=="terminated_verified" and e.get("pod_id")==pid for e in ledger),"termination not recorded")
        audit.require(pid not in {p["id"] for p in rp.inventory()["pods"]} and rp.pod_status(pid) is None,"pod is still present")
        row["termination_verified"]=True
        folder=audit.DATA/plan["output"]
        cells=json.loads((audit.ROOT/"docs/baby-shrimp-zoom/probe-cells.json").read_text())["cells"]
        plot=render(folder,args.out,cells)
        receipt={"id":"BSP3","generated_utc":datetime.now(timezone.utc).isoformat(),"status":"all numerical and lifecycle gates passed",
                 "analysis_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=audit.ROOT,text=True).strip(),
                 "analysis_source_sha256":{source:audit.digest(audit.ROOT/source) for source in
                                           ("tools/baby_shrimp_finish.py","tools/baby_shrimp_period_survey_finish.py")},
                 "plan_sha256":audit.digest(plan_path),"band":row,"plot":plot,
                 "figures":[audit.artifact(p) for p in sorted(args.out.glob("*.png"))]+[audit.artifact(p) for p in args.out.glob("*.pdf")],
                 "limits":["Recurrence candidates only; selected full-state orbits checked separately.",
                           "Grid, batch layout and horizon differ from BSZ1; basin capture is not assumed equivalent.",
                           "Finite-horizon irregular or undetected cells are not automatically classified as chaotic."],"cost":"see private ledger"}
        (args.out/"receipt.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
        print(json.dumps({"status":receipt["status"],"counts":row["gates"]["period_counts"],"out":str(args.out)}))
    except BaseException as error:
        (args.out/"failed.json").write_text(json.dumps({"error":str(error),"type":type(error).__name__})+"\n")
        raise


if __name__=="__main__":
    main()

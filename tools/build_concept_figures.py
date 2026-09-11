#!/usr/bin/env python3
"""Construct explanatory diagrams and an exact analytical training-cycle plate.

These are schematics/algebraic examples, never observations from the pilots.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "paper/figures"
EPOCH = datetime(2026, 9, 11, tzinfo=timezone.utc)
COLORS = dict(ink="#233247", blue="#2468A0", orange="#C96819", green="#287C64", red="#AC3B40", gray="#707883")
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.titlesize":10,
    "axes.labelsize":9, "xtick.labelsize":8, "ytick.labelsize":8, "svg.fonttype":"none",
    "pdf.fonttype":42, "svg.hashsalt":"neural-shrimp-concepts-v1", "axes.spines.top":False,
    "axes.spines.right":False, "axes.edgecolor":COLORS["ink"], "text.color":COLORS["ink"]})


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cycle_values(eta):
    """Exact same-sign two-cycle of GD on (theta²-1)²/2; eta>1/2."""
    eta=np.asarray(eta,dtype=float)
    if np.any(~np.isfinite(eta)) or np.any(eta<=.5): raise ValueError("eta must be finite and >1/2")
    p=1/(2*eta)
    low=(np.sqrt(1+3*p)-np.sqrt(1-p))/2
    high=(np.sqrt(1+3*p)+np.sqrt(1-p))/2
    return dict(eta=eta, low=low, high=high, low_prediction=low**2, high_prediction=high**2,
        multiplier=9-2*(1+2*eta)**2, mean_phase_mse=(1-p**2)/2,
        mean_prediction_mse=(p-1)**2/4, mean_weight_mse=9*(p-1)**2/16,
        mean_prediction=(1+p)/2)


def box(axis, x,y,w,h,heading,body,color):
    patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.018,rounding_size=0.015",
                         facecolor="white",edgecolor=color,linewidth=1.5)
    axis.add_patch(patch)
    axis.text(x+w/2,y+h-.055,heading,ha="center",va="top",fontsize=10,weight="bold",color=color)
    axis.text(x+w/2,y+h/2-.025,body,ha="center",va="center",fontsize=9,linespacing=1.7)


def arrow(axis,start,end,label=None):
    axis.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=12,lw=1.4,color=COLORS["gray"]))
    if label: axis.text((start[0]+end[0])/2,(start[1]+end[1])/2+.025,label,ha="center",fontsize=8)


def spaces():
    fig,ax=plt.subplots(figsize=(7,3.6)); ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    box(ax,.035,.40,.26,.44,"A  Controls", "Layer learning rates\n"+r"$(\eta_1,\eta_2)$"+"\nDefine the update map",COLORS["blue"])
    box(ax,.37,.40,.26,.44,"B  State",r"$\theta_{t+1}=F_\eta(\theta_t)$"+"\nFull weights\nRecurrence, attraction",COLORS["orange"])
    box(ax,.705,.40,.26,.44,"C  Readout",r"$\hat y_t=f_{\theta_t}(x)$"+"\nPredictions\nQuality, sensitivity",COLORS["green"])
    arrow(ax,(.30,.62),(.35,.62)); arrow(ax,(.635,.62),(.69,.62))
    ax.text(.165,.29,"Shrimp windows live here\nif branches have that organization",ha="center",fontsize=8.5)
    ax.text(.50,.29,"Trajectories live here\nfor one fixed map",ha="center",fontsize=8.5)
    ax.text(.835,.29,"Readout can preserve\nor hide state motion",ha="center",fontsize=8.5)
    ax.text(.5,.07,"Useful control requires better task outcomes after charging the cost of observing and acting.",ha="center",fontsize=9,weight="bold")
    fig.suptitle("Where is the pattern, and where should its consequences appear?",y=.98,fontsize=11)
    return fig,{},"conceptual-spaces","SCHEMATIC", "Control parameters define the update; its state is decoded into task outcomes. Internal activity and computation remain observable even when the readout hides motion."


def readout():
    t=np.arange(8); h=(-1.)**t
    fig,axes=plt.subplots(3,1,figsize=(7,4.8),sharex=True,layout="constrained")
    configs=[(h,"A  Internal state",r"$h_t=(-1)^t$",COLORS["orange"],"o"),
             (h,"B  A readout that preserves the sign",r"$g_1(h)=h$",COLORS["blue"],"s"),
             (h*h,"C  A readout invariant to the sign",r"$g_2(h)=h^2$",COLORS["green"],"D")]
    for ax,(v,title,ylabel,c,m) in zip(axes,configs):
        ax.plot(t,v,marker=m,color=c,lw=1.3);ax.set(ylim=(-1.35,1.4),yticks=[-1,0,1],ylabel=ylabel)
        ax.set_title(title,loc="left",pad=5);ax.grid(axis="y",alpha=.18)
    axes[-1].set_xlabel("Iteration t (constructed example)")
    fig.suptitle("Large internal oscillation can vanish under a many-to-one readout",fontsize=11)
    return fig,{"t":t.tolist(),"h":h.tolist(),"visible":h.tolist(),"invariant":(h*h).tolist()},"conceptual-readout","ALGEBRAIC EXAMPLE", "The exact neutral map F(h)=-h started at one alternates sign; identity readout oscillates while squared readout is constant. This example claims neither attraction nor shrimp geometry."


def roadmap():
    fig,ax=plt.subplots(figsize=(7,5.1));ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    stages=[("A  Establish a neural periodic regime", "Full-state orbit closure + multiplier stability", "If absent: refine or relabel the search",COLORS["blue"]),
            ("B  Measure what survives the readout", "Phase predictions, gradients, averaging bias", "If output-invisible: test sensitivity and cost",COLORS["orange"]),
            ("C  Cross a verified boundary", "Equal-dose normal / tangent / random; zero-dose sham", "If no added prediction: favor the simpler diagnostic",COLORS["green"]),
            ("D  Test a useful decision", "Quality, robustness, or compute to a target", "Charge mapping, diagnostics, tuning and failures",COLORS["red"])]
    for i,(title,observable,failure,c) in enumerate(stages):
        y=.78-i*.235
        patch=FancyBboxPatch((.06,y),.88,.185,boxstyle="round,pad=0.012",facecolor="white",edgecolor=c,lw=1.5)
        ax.add_patch(patch)
        ax.text(.085,y+.151,title,va="top",fontsize=10.2,weight="bold",color=c)
        ax.text(.085,y+.096,observable,va="top",fontsize=9.4)
        ax.text(.085,y+.043,failure,va="top",fontsize=8.4,color=COLORS["gray"])
        if i<3: arrow(ax,(.5,y-.017),(.5,y-.045))
    fig.suptitle("Proposed tests: from a dynamical object to an observable advantage",fontsize=11,y=1.01)
    return fig,{"stages":[{"step":s[0],"observable":s[1],"decision":s[2]} for s in stages]},"conceptual-roadmap","PROPOSED EXPERIMENTS", "Four gated stages identify a cycle, measure phase-dependent observables, cross a verified boundary with matched interventions, and test quality or cost against simpler controls. This is a plan, not a completed result."


def analytic():
    eta=np.linspace(.5005,.65,360)
    v=cycle_values(eta);critical=(np.sqrt(5)-1)/2
    fig,axes=plt.subplots(3,1,figsize=(7,6.2),sharex=True,layout="constrained")
    ax=axes[0]
    ax.plot(eta,v["low_prediction"],color=COLORS["blue"],label=r"Phase prediction $u^2$")
    ax.plot(eta,v["high_prediction"],color=COLORS["orange"],ls="--",label=r"Phase prediction $v^2$")
    ax.axhline(1,color=COLORS["gray"],ls=":",label="Target = 1")
    ax.set(ylabel="Prediction at x = 1",ylim=(.42,1.75));ax.set_title("A  A verified algebraic cycle produces alternating predictions",loc="left")
    ax.legend(ncol=3,fontsize=8,loc="upper left")
    ax=axes[1]
    for name,label,color,ls in [("mean_phase_mse","Mean phase MSE",COLORS["orange"],"-"),
          ("mean_prediction_mse","MSE of mean prediction",COLORS["green"],"--"),
          ("mean_weight_mse","MSE of mean weight's prediction",COLORS["blue"],":")]:
        ax.plot(eta,v[name],label=label,color=color,ls=ls)
    ax.set(yscale="log",ylabel="Squared error (log)")
    ax.set_title("B  Averaging reduces error but retains a calculable bias",loc="left")
    ax.legend(fontsize=8,loc="lower right")
    ax=axes[2];ax.plot(eta,v["multiplier"],color=COLORS["ink"])
    ax.axhline(-1,color=COLORS["red"],ls="--");ax.axhline(1,color=COLORS["gray"],ls=":")
    ax.set(ylabel=r"Cycle multiplier $\mu$",xlabel=r"Learning rate $\eta$")
    ax.set_title("C  Attraction has a precise interval; no shrimp shape is claimed",loc="left")
    for ax in axes:
        ax.axvspan(.5,critical,color=COLORS["green"],alpha=.07)
        ax.axvline(critical,color=COLORS["red"],ls="--",lw=1)
        ax.grid(alpha=.18)
        ax.set_xlim(.5,.65)
    axes[2].text(.553,-.84,"Attracting two-cycle",ha="center",fontsize=8.5,color=COLORS["green"])
    axes[2].text(.635,.58,"Unstable\nbranch",ha="center",fontsize=8.5,color=COLORS["red"])
    fig.suptitle(r"Exact toy neural training: $\hat y=\theta^2x$, one sample $(x,y)=(1,1)$",fontsize=11)
    data={k:a.tolist() for k,a in v.items()};data["attracting_upper_eta"]=float(critical)
    return fig,data,"analytic-training-cycle","EXACT TOY MODEL", "Algebraic two-cycle predictions, three squared-error readouts, and cycle multiplier for a tied-weight depth-two scalar network. Green interval denotes attraction. Beyond its boundary the algebraic branch remains but is unstable, not an observed attractor. Fixed-point predictions attain zero error, omitted from the log axis."


def build(destination):
    destination.mkdir(parents=True,exist_ok=True)
    entries=[]
    for make in [spaces,readout,roadmap,analytic]:
        fig,data,stem,kind,description=make()
        outputs={}
        for extension in ("pdf","svg","png"):
            path=destination/f"{stem}.{extension}"
            metadata=({"Title":stem,"Subject":description,"CreationDate":EPOCH,"ModDate":EPOCH}
                      if extension=="pdf" else {"Title":stem,"Description":description,"Date":EPOCH.isoformat()}
                      if extension=="svg" else {"Title":stem,"Description":description})
            fig.savefig(path,dpi=300,bbox_inches="tight",metadata=metadata)
            if extension=="svg": ET.parse(path)
            outputs[path.name]=digest(path)
        plt.close(fig)
        receipt={"figure_id":stem,"kind":kind,"title":stem.replace("-"," "),"description":description,"alt_text":description,
                 "empirical":False,"source":"Constructed diagram or exact stated algebra; no pilot outcome or new target experiment",
                 "generator":{"path":"tools/build_concept_figures.py","sha256":digest(__file__)},"matplotlib":matplotlib.__version__,
                 "derived_values":data,"outputs":outputs,"claim_scope":"No modern neural shrimp or practical advantage demonstrated by this illustration.",
                 "accessibility":"Labeled panels, line styles and markers supplement color; exact construction in receipt."}
        path=destination/f"{stem}.receipt.json"
        path.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n")
        entries.append({"path":path.name,"sha256":digest(path)})
    (destination/"conceptual-provenance.json").write_text(json.dumps({"figures":entries},indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--check",action="store_true");parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    if args.check or args.verify:
        index=json.loads((DEST/"conceptual-provenance.json").read_text())
        for item in index["figures"]:
            path=DEST/item["path"]
            if digest(path)!=item["sha256"]:raise ValueError("Receipt hash drift")
            receipt=json.loads(path.read_text())
            if digest(ROOT/receipt["generator"]["path"])!=receipt["generator"]["sha256"]:raise ValueError("Generator drift")
            for name,expected in receipt["outputs"].items():
                if digest(DEST/name)!=expected:raise ValueError("Output drift")
        if args.verify:
            with tempfile.TemporaryDirectory() as tmp:
                temp=Path(tmp);build(temp)
                for path in temp.iterdir():
                    if path.read_bytes()!=(DEST/path.name).read_bytes():raise ValueError(f"Rebuild drift: {path.name}")
        print("Four conceptual/analytic figure receipts verified")
    else:build(DEST)


if __name__=="__main__":main()

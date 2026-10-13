"""Cusp check on a fold curve: the fold normal form u' = u + a eps + b u^2 (float64, CPU) at points of the curve around
each parameter-tangent reversal; at a cusp b passes through zero (and the cubic term is nonzero). The signs of a and b
both flip with the orientation of the critical eigenvector (r -> -r, l -> -l), so the reported b_oriented = b sign(a)
is the quantity whose sign change marks a cusp (a must stay away from zero). With a fourth argument "cubic" each
point also gets the cubic coefficient c of bifurcation_curves.fold_cubic (orientation-independent; nonzero c at the
sign change of b is the cusp's nondegeneracy condition).

Usage: python tools/cusp_check.py CURVE_RUN INDICES [DEVICE [cubic]]"""
import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import bifurcation_curves as bc  # noqa: E402
import explore_llm_phase_map as elpm  # noqa: E402
import explore_llm_soft_loop as esl  # noqa: E402

run = sys.argv[1]
indices = [int(i) for i in sys.argv[2].split(",")]
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"           # e.g. cuda on a pod with fast float64
cubic = len(sys.argv) > 4 and sys.argv[4] == "cubic"
src = esl.DATA / "explore" / run
cv = json.loads((src / "curve.json").read_text())
states = np.load(src / "states.npy")
start = json.loads((esl.DATA / "explore" / cv["args"]["start"] / "branch.json").read_text())["args"]
loop = esl.SoftLoop(start["model"], start["layer"], 12, start["prompt"], start["opener"], start["window"], "float64", device)
v = elpm.load_vectors(start["vectors"])[start["concept"]]
fam = esl.soft_family(loop, v, int(cv["period"]), 10.0, 512)
for i in indices:
    p = cv["points"][i]
    theta = np.array([p["log10_temperature"], p["alpha"] * 10.0])
    x = states[i].ravel()
    nf = bc.fold_normal_form(fam, x, theta, param=1, param_scale=10.0, steps=(1e-3, 3e-4, 1e-4))
    rec = {"index": i, "log10_T": p["log10_temperature"], "alpha": p["alpha"], "cusp_flag": p["cusp_candidate"],
           "multiplier": nf["multiplier"], "a": nf["a"], "b": nf["b"],
           "b_oriented": float(np.sign(nf["a"]) * nf["b"]), "b_steps": [t["b"] for t in nf["b_steps"]],
           "tangent": p["parameter_tangent"]}
    if cubic:
        cu = bc.fold_cubic(fam, x, theta)
        rec.update(c=cu["c"], c_plateau_pair=cu["c_plateau_pair"], c_plateau_relative_spread=cu["c_plateau_relative_spread"],
                   c_steps=cu["c_steps"], second_modulus=cu["second_modulus"])
    print(json.dumps(rec), flush=True)

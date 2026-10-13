"""Cusp check on a period-4 fold curve: the fold normal form u' = u + a eps + b u^2 (float64, CPU) at points of the
curve around each parameter-tangent reversal; at a cusp b passes through zero (and the cubic term is nonzero)."""
import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import bifurcation_curves as bc  # noqa: E402
import explore_llm_phase_map as elpm  # noqa: E402
import explore_llm_soft_loop as esl  # noqa: E402

run = sys.argv[1]
indices = [int(i) for i in sys.argv[2].split(",")]
src = esl.DATA / "explore" / run
cv = json.loads((src / "curve.json").read_text())
states = np.load(src / "states.npy")
start = json.loads((esl.DATA / "explore" / cv["args"]["start"] / "branch.json").read_text())["args"]
loop = esl.SoftLoop(start["model"], start["layer"], 12, start["prompt"], start["opener"], start["window"], "float64", "cpu")
v = elpm.load_vectors(start["vectors"])[start["concept"]]
fam = esl.soft_family(loop, v, int(cv["period"]), 10.0, 512)
for i in indices:
    p = cv["points"][i]
    theta = np.array([p["log10_temperature"], p["alpha"] * 10.0])
    x = states[i].ravel()
    nf = bc.fold_normal_form(fam, x, theta, param=1, param_scale=10.0, steps=(1e-3, 3e-4, 1e-4))
    print(json.dumps({"index": i, "log10_T": p["log10_temperature"], "alpha": p["alpha"], "cusp_flag": p["cusp_candidate"],
                      "multiplier": nf["multiplier"], "a": nf["a"], "b": nf["b"], "b_steps": [t["b"] for t in nf["b_steps"]],
                      "tangent": p["parameter_tangent"]}), flush=True)

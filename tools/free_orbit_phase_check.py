"""Run FW3: is a flat stretch of the fold figure's panel (d) a passage through the same bottleneck, one step out of phase?

Applies one step of the soft-token map, at the trajectory's parameters, to every saved state of the published free
trajectory (eps = 1e-5, natural-3), and reads the critical coordinate of the result with the same x_c and l. Saves both
readings. Usage: HF_HUB_OFFLINE=1 python tools/free_orbit_phase_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_llm_attractors as ela  # noqa: E402
import explore_llm_phase_map as elpm  # noqa: E402
import explore_llm_soft_loop as esl  # noqa: E402

RUN = ela.DATA / "explore/llm-soft-loop/passage-natural-words-p2-t-0.115"
NF = ela.DATA / "explore/llm-soft-loop/normal-form-p2-t-0.115"
PICK, EPS = 7, 1e-5


def main():
    nf = json.loads((NF / "normal-form.json").read_text())
    vec = np.load(NF / "vectors.npz")
    xc, l = vec["x_c"], vec["l"]
    so = nf["source"]
    sign = 1.0 if nf["a"] * nf["b"] > 0 and nf["a"] > 0 else -1.0     # the orientation of passage()
    l = sign * l
    states = np.load(RUN / "top.npz")["states"][:, PICK]
    u0 = np.load(RUN / "u.npy")[:, PICK]
    assert np.abs((states - xc) @ l - u0).max() < 1e-12
    loop = esl.SoftLoop(so["model"], so["layer"], 8, so["prompt"], so["opener"], so["window"], "float64", "cpu")
    torch = loop.torch
    v = np.asarray(elpm.load_vectors(so["vectors"])[so["concept"]], dtype=np.float64)
    alpha = so["alpha"] + EPS
    T = 10.0 ** so["log10_temperature"]
    out = []
    for s in range(0, len(states), 32):
        batch = states[s:s + 32]
        X = loop.prepare(np.repeat((alpha * v)[None], len(batch), axis=0)).clone()
        X[:, -1] = torch.as_tensor(batch, dtype=X.dtype, device=X.device)
        with torch.no_grad():
            X, _ = loop.step(X, np.full(len(batch), T))
        out.append(X[:, -1].double().cpu().numpy())
    next_states = np.concatenate(out)
    u1 = (next_states - xc) @ l
    np.savez_compressed(RUN / "phase-check.npz", u_same_phase=u0, u_one_step_later=u1, next_states=next_states)
    print(json.dumps({"states": len(states), "alpha": alpha, "U": 0.1 / abs(nf["b"])}, indent=1))


if __name__ == "__main__":
    main()

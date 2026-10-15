"""Check (15 Oct 2026, latent-bug fix): exact tangents of a two-block cell (hostility@4 + gratitude@16, the bimodal cell) against central
differences, with the fix, and with the second block dropped (the old behaviour) for contrast."""
import sys
sys.path.insert(0, "/data2/neural-shrimp/tools")
import numpy as np
import torch
import explore_llm_attractors as ela
import explore_llm_phase_map as elpm
import explore_llm_soft_loop as esl

loop = esl.SoftLoop("qwen0.5b", 4, 16, "open", ela.OPENER, 1, "float64", "cpu")
loop.sm.add_steer_layer(16)
u = np.asarray(elpm.load_vectors("llm-phase-map/vectors-l4")["hostility"], dtype=np.float64)
w = np.asarray(elpm.load_vectors("llm-phase-map/vectors-l16")["gratitude"], dtype=np.float64)
a1, a2, lt = 0.0568421052631579, -0.018947368421052602, -0.1
T = np.array([10 ** lt])
loop.extra = (16, np.array([a2 * w]))
X = loop.prepare(np.array([a1 * u]))
for _ in range(60):
    X, _ = loop.step(X, T)
g = torch.Generator().manual_seed(0)
V = torch.randn(X.shape, generator=g, dtype=X.dtype)
V /= V.norm()
h = 1e-6 * float(X.norm())
Yp, _ = loop.step(X + h * V, T)
Ym, _ = loop.step(X - h * V, T)
fd = ((Yp - Ym) / (2 * h))[:, -1]
_, _, JV = loop.step_jvp(X, T, V)
print("SoftLoop.step_jvp vs central difference, relative error: %.2e" % float((JV[:, -1] - fd).norm() / fd.norm()))
keep = loop.extra_rows
loop.extra_rows = None                               # the old behaviour: second block absent from the tangent pass
_, _, JV0 = loop.step_jvp(X, T, V)
loop.extra_rows = keep
print("old behaviour (second block dropped), relative error: %.2e" % float((JV0[:, -1] - fd).norm() / fd.norm()))
orb = esl.SoftOrbit(loop, a1 * u, 10 ** lt)
Xo = orb.iterate(orb.start, 60)
jv = orb.jvp_rows(Xo, V)[:, -1]
fd2 = ((orb.f(Xo + h * V) - orb.f(Xo - h * V)) / (2 * h))[:, -1]
print("SoftOrbit.jvp_rows vs central difference, relative error: %.2e" % float((jv - fd2).norm() / fd2.norm()))
print("orbit start identical to the loop's trajectory: %.2e" % float((Xo - X).abs().max()))

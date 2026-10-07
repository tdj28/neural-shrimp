# Adjudication: mechanism research-strategy consult, 7 October 2026

- **Request:** `brief.md` in this folder, review kind `research-strategy`, model
  `gpt-6-astra` (Pro mode, high effort), response `resp_041696ed48f741a4016ac6c0af241487d09034ab2bfc3f198a`,
  status completed. Usage: 31,129 input and 24,797 output tokens (16,449
  reasoning). Conservative cost $1.55. Raw request, response, extracted review
  and hashed manifest: `../neural-shrimp-data/consults/2026-10-07-mechanism-strategy/execute-001/`.
- **Standing:** advisory design input from one frontier model, not independent
  verification (integrity skill, section 7A). The human researcher remains the
  independent channel.

## Findings and decisions

| # | Consult finding | Decision |
|---|---|---|
| 1 | Q1: test odd windows with an effective one-dimensional return-map hypothesis (held-out closure of a scalar coordinate, one reproducible critical point, contracting transverse fibers under matched-state perturbations, held-out kneading predictions); spirals alone are insufficient; "doubly superstable" needs a multimodal reduction or a full-map critical set | Accepted. Pilot on one odd window with a logistic-plus-contracting-coordinates positive control and a Henon-like negative control. Flagged for the human (author-level expertise on the hub picture) |
| 2 | Q2: the boundary census is consistent with torus breakdown but not diagnostic; the decisive test is resonance-anchored ancestry: both fold edges of a tongue converge to the independently predicted rational point on the parent Neimark-Sacker curve while the orbit collapses onto the parent 2-cycle | Accepted as the top-priority experiment. It replaces the map-based tongue-root test, which was limited by tip resolution |
| 3 | Q3: two coordinates suffice in principle (engineered example S0 = diag(1, -1), S1 = r R_phi S0); for the factorization model check D = (tr M)^2 - 4 det M along the 2-cycle (a definite step derivative forces real eigenvalues); natural candidate: a coupled four-weight two-mode model with C = [[1, c], [c, 1]] | Accepted: run the coupled-model scan (c in {0, 0.3, 0.6, 0.9}, two ridges, eight rate ratios) and the D diagnostic on the factorization model |
| 4 | Replication units are independently generated losses; initializations, rays and cells are nested | Accepted. The reserved seeds 29000-29999 generate data, targets and initial weights together, so each seed is an independent loss instance; the confirmatory design will count losses |
| 5 | Validate circles by solving K(theta + rho) = F^2(K(theta)) (Fourier parameterization) rather than relying on near-zero exponents | Accepted as a follow-on after ancestry |
| 6 | Bound the real-data test: CIFAR-10 subsets of 256-512 images, one 32-unit hidden layer, weight decay, float64, two subsets (second reserved), matrix-free Arnoldi multipliers; profile locally and rent only after profiling | Accepted; the paid-compute plan is revised accordingly |
| 7 | Target selection rule: lowest denominator among undoubled, well-ordered windows whose ratio is bracketed by the traced birth rotation, ties by distance to the curve | Accepted. 1/6 is not bracketed (birth-rotation minimum 0.16699), so the target is S2-28011:139 (period 10, rotation 1/5, bracketed on both branches); fallback S2-28011-zoom:587 (2/9) |

No finding was rejected. Nuance kept for the ancestry test: Neimark-Sacker
boundaries arbitrarily close to a putative generic tip, without locking
branches, would count against the tongue identification.

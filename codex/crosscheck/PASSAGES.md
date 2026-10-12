# Independent controlled-passage cross-check

The independently implemented transformer reproduces all 18 controlled escape times exactly. The soft-token state lingers near the destroyed two-cycle for up to 144 two-step iterations; the derivative-based quadratic model predicts these controlled passages within one iteration. This supports the local bottleneck mechanism in the LLM paper.

The test holds the original centre, critical modes, coefficients, entrance width and initial states fixed. Independent coefficient and fold computations are separate runs. These are already-consumed cases, not a new holdout.

| Epsilon | Entrance | Independent / reference | Quadratic prediction | Difference from prediction |
|---:|:---|---:|---:|---:|
| 5e-05 | full | 24 / 24 | 24 | +0 |
| 5e-05 | midpoint | 12 / 12 | 13 | -1 |
| 5e-05 | full+transverse | 23 / 23 | 24 | -1 |
| 0.0005 | full | 3 / 3 | 4 | -1 |
| 0.0005 | midpoint | 2 / 2 | 2 | +0 |
| 0.0005 | full+transverse | 3 / 3 | 4 | -1 |
| 3e-06 | full | 144 / 144 | 144 | +0 |
| 3e-06 | midpoint | 73 / 73 | 74 | -1 |
| 3e-06 | full+transverse | 143 / 143 | 144 | -1 |
| 1e-05 | full | 71 / 71 | 71 | +0 |
| 1e-05 | midpoint | 36 / 36 | 37 | -1 |
| 1e-05 | full+transverse | 70 / 70 | 71 | -1 |
| 3e-05 | full | 34 / 34 | 34 | +0 |
| 3e-05 | midpoint | 17 / 17 | 18 | -1 |
| 3e-05 | full+transverse | 33 / 33 | 34 | -1 |
| 0.0001 | full | 14 / 14 | 14 | +0 |
| 0.0001 | midpoint | 7 / 7 | 7 | +0 |
| 0.0001 | full+transverse | 13 / 13 | 14 | -1 |

Maximum coordinate discrepancy before the first exit: 1.660879767e-11, or 4.22337e-09 times the fixed channel half-width. The smallest recorded pre-exit/crossing margin is 8.885276665e-06. All first-exit inequalities were checked directly in the saved trajectories.

The quadratic approximation matches seven counts and overestimates eleven by one iteration. Its largest relative discrepancy is 25% for the short development passage of 3 versus 4 iterations at epsilon 0.0005. The zero count differences in the table concern independent transformer versus reference transformer, not quadratic versus transformer.

The independent audit also checked the exact entrances and RNG-0 transverse vector, sign orientation, all 18 retained coordinate traces, numerical quadrature of the continuous passage formula, all source and reference hashes, and the complete output inventory.

## Reproducibility and scope

- Worker source: `fe0c26e49b0163dee31181c48e71bdda337c6490`.
- Raw root: `/data2/neural-shrimp-data/codex/task2-independent-crosscheck/passages-001`.
- Hash-bound inventory: `receipts/passages-001.json`.
- Separate read-only audit: `audit_passages.py` and `audits/passages-001.json`.
- Four CPU threads, no GPU, 316.38 seconds including process startup and shutdown.

This is numerical cross-validation of the fixed-prefix, one-token expected-embedding map of Qwen2.5-0.5B-Instruct. The manual transformer forward pass uses the same pinned pretrained weights and PyTorch tensor primitives, but imports no repository research tool or Transformers model. It is not an interval certificate, an ordinary sampled-generation experiment, or a check of global shrimp boundaries.

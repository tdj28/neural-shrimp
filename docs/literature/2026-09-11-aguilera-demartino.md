# Aguilera–De Martino preprint: bounded relevance and checks

Source: [arXiv:2609.07341v1](https://arxiv.org/abs/2609.07341v1), submitted 7 September 2026. Full [PDF](https://arxiv.org/pdf/2609.07341v1) and [HTML](https://arxiv.org/html/2609.07341v1) checked 11 September 2026; no journal reference identified. Treat as unreviewed.

**Authors' model and evidence.** Binary spins undergo stochastic partial refresh with probability 0.1. Random-pattern coupling blocks encode dynamical memories. “Infinite” denotes extensive load as network size tends to infinity, not unlimited finite-network storage. Dispersing block eigenphases reportedly reduces interference. Capacity uses target-overlap RMS above half its zero-load baseline, starting aligned with the target (PDF pp. 1–3, 6–7). This measures neither waveform preservation nor basin-wide retrieval. The cited [code repository](https://github.com/MiguelAguilera/DynamicalMemoryCapacity) and its public API returned 404 during review; implementation could not be checked.

**Our analytical audit.** For independent zero-field spins, the stated update law implies

\[
q_{s+\ell,s}=0.9^\ell,\qquad
\chi_{s+\ell,s}=0.1\beta\,0.9^{\ell-1},\quad\ell\ge1.
\]

Printed SM Eqs. S40–S41 (PDF p. 15) instead imply zero off-diagonal correlation and response \(\beta\,0.9^{\ell-1}\). Other equations retain refresh factors, so this establishes an internal inconsistency, not an implementation failure. Main Eq. 3 also differs from the correctly indexed SM Eq. S3.

Every finite strictly causal response matrix is nilpotent, with spectral radius zero. The paper's nonzero response-radius threshold therefore requires an explicitly defined infinite-time operator or other diagnostic; finite-window amplification remains possible. This caveat does not disprove its intended asymptotic mechanism.

Our [synthetic audit](../../tools/audit_preprint_identities.py) derives exact rational lag values; [tests](../../tests/test_preprint_identities.py) independently enumerate the Markov chain. Neither reproduces paper numerics or runs neural targets.

**Implication for this project.** This motivates a future test of interference among multiple dynamical memories. It establishes neither shrimp geometry nor controller/training utility. Encoding-block eigenphases must not be confused with orbit Floquet multipliers. EXP-006's distinction between eventual nine-peak retention and immediate waveform quality remains essential; held-out, cost-matched comparisons remain necessary.

Reviewed PDF SHA-256: `0c3afa02e8cd3d4dc7abf1513de63da4abd481d2fd2fcde1ef0c662e518051b7`. No paid calls or new target outcomes.

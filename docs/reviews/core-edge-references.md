# Core EoS reference content audit

Audited 2026-10-08 UTC, independently of manuscript production. **No blocking identity or citation-content error was found for the three keys below.** The current Chen--Bruna and Song--Yun citations only call them further analyses. The suggested changes make the actual antecedents and their limits clearer.

The companion [receipt](core-edge-references-receipt.json) binds this report to manuscript commit `16168ba91b749e4607c6f5cc69783f023046c7f7`, exact file bytes, and the three downloaded primary PDFs. No manuscript, bibliography, or reference ledger was edited. No experiments or paid calls were made. This is a source-content audit, not an independent proof verification or an exhaustive novelty review.

## cohen2021

Identity passes: author list, title, and ICLR 2021 agree with the primary PDF/header and [conference listing](https://iclr.cc/Downloads/2021). Content inspected in [arXiv v3](https://arxiv.org/pdf/2103.00065v3): Sections 2--4, Appendix A, I.4 and L.1/Figure 82. This is a later author version, not a byte-equivalence check of the conference PDF.

The introduction's sharpness/loss statement is supported as an empirical full-batch-GD observation. The threshold comes from quadratic linear stability; it is not a universal convergence guarantee for nonlinear training. Appendix A records exceptions. Appendix L.1 already reports a two-layer Transformer on WikiText-2; I.4 estimates its sharpness on a 2,500-example subset. That is relevant prior art for our architectural pilot.

Suggested clarification, replacing the introduction's first two sentences up to this citation:

```latex
Full-batch gradient descent can continue reducing neural-network training loss
beyond the local quadratic stability threshold. In the edge-of-stability
regime observed by \citet{cohen2021}, sharpness stays near $2/\eta$ while
short-term loss fluctuations coexist with longer-term decrease;
```

This avoids leaving “common practice” apparently established for every optimizer by this citation. The current related-work sentence is accurate.

## chenbruna2023

Identity passes against [PMLR 202:4330--4391](https://proceedings.mlr.press/v202/chen23b.html): Lei Chen and Joan Bruna, ICML 2023. Inspected [published PDF](https://proceedings.mlr.press/v202/chen23b/chen23b.pdf), Sections 4--6, Appendix B.3.2 and C.

Theorem 1 gives local two-step fixed-point **existence**, with nonzero third derivative and `3(f''')² - f'' f'''' > 0` at a minimum. Convergence is established in additional restricted settings: the scalar quartic, a positive two-variable factorization trajectory, and a small balanced-initialization single-ReLU teacher/student model under spherical population loss. Theorem 2's scalar range is `1 < eta*mu < sqrt(4.5)-1`. Higher-dimensional matrix-factorization and MNIST evidence does not establish a universal attracting-period-two theorem. The MNIST experiment checks directional derivative conditions and alignment.

Suggested replacement contribution to the related-work paragraph:

```latex
\citet{chenbruna2023} analyze two-step GD, proving period-2 convergence
in specified scalar and single-neuron settings and testing related
conditions in larger models.
```

No correction is mandatory for the current generic citation. Avoid describing Theorem 1 alone as an attraction theorem.

## song2023

Identity passes against the [NeurIPS 2023 proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e2a9256bd816ab9e082dfaa22f1f62a2-Abstract-Conference.html): Minhak Song and Chulhee Yun. Inspected the [published PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/e2a9256bd816ab9e082dfaa22f1f62a2-Paper-Conference.pdf), Sections 2--5, Theorems 4.2--4.4, and Appendices A.4/C.1.

The proofs concern a single example: two-layer linear networks with constrained convex Lipschitz losses, or a scalar nonlinear model with squared loss, subject to derivative, step-size and initialization assumptions. Empirical extensions include tanh MLP/CNN training on 50 CIFAR-10 images. The bifurcation diagram freezes a reduced control coordinate; full dynamics track it while that coordinate changes. Theorem 4.3 ultimately converges to a minimizer. This supports trajectory alignment, not an enduring full-state period-two orbit, NS bifurcation, or shrimp geometry.

Suggested related-work sentence:

```latex
\citet{song2023} study alignment with a reduced bifurcation diagram,
with proofs for specified single-example models and neural-network
experiments. Our orbit tests concern closure and stability of the
full parameter state.
```

After adding these sentences, retain Kalra, Chen et al. and Cohen et al. (2025) in the remaining “Further analyses” list.

## Integration and verification

The receipt's `ledger_updates` can replace only the three matching entries in `paper/gd-route/data/references-ledger.json`; `open: []` applies to their current cited claims, not to all possible uses of these papers. Add primary URLs from the receipt to the corresponding BibTeX entries if desired.

The receipt lists PDF SHA-256 hashes and the immutable local archive. Recompute those hashes with `sha256sum`, and compare the manuscript hashes before applying wording suggestions. The citation contexts were checked against the recorded source snapshot. The inspection did not audit every proof step, replicate a paper's experiments, or establish that no related prior work exists.

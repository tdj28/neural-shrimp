# Further primary-reference content checks

Coordinator content check, 8 October 2026, against manuscript snapshot
`5e716f6a191e58022e17052277d4de90c44091c7`. This checks the scope of current
citations; it is not an independent replication of these papers or an exhaustive
novelty search. The companion JSON binds the snapshot, report and retained
primary PDFs. No numerical research experiment was run.

## damian2023

[Author version v2](https://arxiv.org/html/2209.15594v2), sections 3--6 and
assumptions/theorem 1, supports the self-stabilization citation. The analysis
assumes progressive sharpening, an eigengap leaving one unstable direction,
and derivative-control conditions. It does not establish all optimizer
dynamics or a universal sharpness cap. ICLR 2023 identity is confirmed by
the [authors' institution](https://collaborate.princeton.edu/en/publications/self-stabilization-the-implicit-bias-of-gradient-descent-at-the-e/).
The current related-work statement is supported; the introduction should
identify this as an analysis of self-stabilization rather than imply an
unconditional explanation of all training.

## agarwala2023

The [published ICML/PMLR paper](https://proceedings.mlr.press/v202/agarwala23b/agarwala23b.pdf),
sections 2--5 and appendix I, supports progressive sharpening in quadratic
regression models. Appendix I compares local quadratic/cubic approximations
with a nonlinear network and records remaining quantitative differences.
The related-work wording already scopes the model class. Carry that scope
into the introduction; do not promote a local Taylor model to a universal
description of the later training attractor. Publication identity agrees
with the [PMLR record](https://proceedings.mlr.press/v202/agarwala23b.html).

## lewkowycz2020

[arXiv v1](https://arxiv.org/html/2003.02218v1), sections 2--3, supports initial
loss growth followed by convergence in the studied catapult regime, with
lower final curvature than the small-rate comparison. Practical experiments
include SGD; the solvable model has its own assumptions. This is not a
universal stage that must occur after an edge-of-stability period-two orbit.
Remove the introduction's ambiguous suggestion that catapult is simply what
happens at rates above the regime just described. No universal generalization
or optimal-rate theorem is claimed in our manuscript.

## cohen2025central

[arXiv v2](https://arxiv.org/html/2410.24206v2), sections 3--8, describes
time-averaged deterministic GD/RMSProp trajectories. Section 8 explicitly
limits the approach's rigor, stochastic/momentum scope and computational
feasibility; section 6 records failure modes. The [authors' project site](https://centralflows.github.io/)
confirms ICLR 2025. Replace the generic further-work citation with its actual
connection: averaged trajectories complement our tests of full-state orbit
closure. Predicting training evolution is already established prior work;
our proposed LM diagnostic needs additional predictive or intervention value.

## kong2020

The [NeurIPS paper](https://papers.nips.cc/paper/2020/file/1b9a80606d74d3da6db2f1274557e644-Paper.pdf),
sections 1--2, studies deterministic GD for specified multiscale objectives,
including statistical limiting behavior under stated conditions. The
introduction leaves generalization outside its scope. Identity agrees with
the official proceedings and PDF header. Refine our broad chaos citation to
name the multiscale-objective setting; it is not a general neural-network
chaos theorem.

## jimenez2026

The [published APS PDF](https://journals.aps.org/prresearch/pdf/10.1103/t5p9-kv5w),
pages 2--4 and appendix B, relates finite-time perturbation expansion to
epochs needed to reach task-specific test-accuracy thresholds in full-batch
MLP/CNN experiments. The concluding limitations leave minibatch noise and
more elaborate schedules open. This is relevant prior utility evidence, not
merely a report of oscillations. Our geometry-specific utility still needs
its own comparison. The primary site confirms publication on 18 May 2026.
Browser full-text access succeeded; direct PDF archival returned HTTP 403,
which is retained in the retrieval manifest. No access barrier was bypassed.

## kuznetsov2019

Identity (authors, title, publisher and year) is confirmed by the
[publisher's chapter/contents record](https://www.cambridge.org/core/books/numerical-bifurcation-analysis-of-maps/analytical-methods/6758F4EF03E3148DDE1035B834F85130)
and the [author's institutional record](https://research-portal.uu.nl/en/publications/numerical-bifurcation-analysis-of-maps-from-theory-to-software/).
Detailed chapter formulas were not accessible for this check. Retain that
open ledger item. Cite the already inspected Kuznetsov--Sacker Scholarpedia
article for the specific local NS coefficient/theorem, and the book for the
general numerical-bifurcation subject.

## Changes and remaining limits

No numerical result changes. The manuscript should add a short explicit
training/inference distinction, narrow the catapult and multiscale citations,
and acknowledge existing transient-growth utility and averaged-trajectory
prediction. Other unreviewed reference entries remain open. Five primary PDFs
were archived with hashes; the APS archival failure remains recorded alongside
the source-content inspection. No claim of complete bibliography verification
is warranted yet.

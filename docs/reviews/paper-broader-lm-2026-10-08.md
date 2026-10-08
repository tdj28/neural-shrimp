# Paper review: broader-data LM and introductory citation scope

8 October 2026 UTC. Independent read-only reviewer `certificate_context` found
no material accuracy or scope issue in the new broader-data paragraph, paired
macros or introductory citation narrowings. It checked producer receipt
9b2495fb and independent audit a1bf4c13, integrated at43227c41.

Paired late expanded-arm normalized MSE in seed28161/28162 order: B3
0.2999/0.5026; B2 0.3051/0.5153; tune-selected ordinary B1/AR1 0.1027/0.0949.
The draft preserves both failed forecast gates, the single fixed pool/two
initializations, equal-update versus equal-target comparisons, absent wall-time
matching, causal fit/tune/late separation, probe nonselection and absent
independent probe reevaluation. Its linked audit explicitly identifies shared
model/feature primitives and limits independent feature evaluation to eight
fixed checkpoints. No training, inference or geometry-specific benefit is claimed.

Reviewed main.tex SHA256:
47ee74f31b497378968ded6cc6801ec896e7cc5ce53d7b7e61079858afc5fe36.
Reviewed numerical generator SHA256:
b5090cd89e74c65369e53b61176d4475b6dfb018837520f28c6e8030d555e334.
The introduction now separates initialization-space convergence regions from
selected numerical learning-rate boundary maps, and empirical retraining
sensitivity from rigorous basin or frozen-inference claims, consistent with
the already archived primary-source content review.

After that read-only review, root made one further documentation correction:
the reproducibility section and README now identify which older pilots the
aggregate verify.py actually replays. The new S1R/CIFAR/LM pilots retain their
separate cited source-bound audit commands; the aggregate tool does not claim
to execute all of them. No numerical source, threshold or evidence changed.

259 generated numerical macros verify. The multi-file paper compiles without
warnings or layout errors; the 28-page rendered reading copy was visually
inspected, including the new LM paragraph at full-page scale. Nine figure
sources and receipts are unchanged. The 29 integrated audit/progress controls
and independent target audit had already passed; no unchanged experimental
trajectory, figure or certificate was rerun solely for this prose edit.

Final layout repair: the longer reproducibility paragraph exposed an overfull
line in its existing path list. Loading the already installed xurl package
adds legal path-breaking opportunities. The source and failed render are
preserved by the tool log; the current PDF is recompiled and rerendered below
with that typography-only repair.

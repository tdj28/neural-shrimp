# 10 October 2026 UTC — saved diagnosis of first march failure

Unit005 retained twelve qualified charts, then one failed coordinate inclusion
at chart012 (contraction <0.794, primitive and full anchor gates passed).
There are 94 joined charts through exact gauge 187/64 and hidden rate near
74.310322. This failure does not diagnose instability or a bifurcation.

One saved-only analysis, no model/derivative/trajectory/root acquisition:
reconstruct each row's forcing |C R|, propagated error, inclusion ratio and
old-anchor containment requirement from the failed chart. B=|I-CJ| comes from
its saved original box. A floating linear solve suggests radii
`max(2*(I-B)^-1*|C R|, 2*anchor_requirement, min(old radii)/16)`, rounded
upward to binary64 and retained as exact dyadics. Reject negative/nonfinite
solutions; normalize with the smallest enclosing power-of-two radius. This is
ONLY a proposal: any new box requires a freshly enclosed Jacobian on that
whole box, strict coordinate inclusion, rigorous weighted contraction and
full common-endpoint anchor containment. No float Perron/radius proof.

Also retain all eighteen cyclic alignments of the existing certified fold's
full 234-state enclosure plus hidden-rate box against the last qualified
chart. Use the entire correlated gauge interval n.(U_fold-origin), never its
midpoint; check strict gauge and correction-domain containment. Native cycles
only, no neuron/sign identifications. This expected distant failure is useful
for choosing a later reanchoring direction, not evidence against a connection.
Bind exact model/data/fixed output equality and the canonical fold inventory.

Source: `tools/analyze_s2_period18_march_failure.py`, its owned launcher and
synthetic controls. Input unit005 manifest
b09b179266c7880fd2480e213e2ce1d489e847fd689bafb0e2b1d5f4b21e1ffa,
execution 0c978692eff507dc1ea06ec044c2855e2176367f; chart011 is the last
qualified predecessor, chart012 the first failure. Canonical fold receipt
`certificates/s2-period18-fold-receipt.json` binds all original inputs.

Run once after source freeze/tests and root review:
`tools/launch_s2_period18_march_failure.py --out
/data2/neural-shrimp-data/explore/certificates/s2-period18-march-diagnosis-2026-10-10/attempt-001`.
CPU/BLAS1, nice10; 20s internal,30+5s external; 1GiB RSS,2MiB worker +1MiB
launcher. Previous comparable saved two-chart reduction <1s supports this
conservative cap; no fresh model profile. Preserve any failure, all rows and
all alignments, exact source/runtime/manifest/owned PID exit. No new proof
claim or further target launch follows automatically from this diagnosis.

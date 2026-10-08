# Independent utility-dose source review — 8 October 2026 UTC

**TLDR:** The reviewed code can test whether the proposed uniform and two
layer-specific rate reductions have equal displacement relative to sham without
requiring either layer's rate reduction to exceed25%. No dose has yet been
measured at a saved fork. The missing control/support evidence still prevents a
pulse comparison or any geometry-specific utility claim.

**Verdict:** pass for the fixed four-gradient gate at source
`075a76ecddddaeeae92f714d8ebbf170812b0500`, subject to root's separate launch
authorization. The ten executable/card files match that commit byte-for-byte;
the companion JSON binds them, the input inventory and independent controls.
Producer files were not edited in this review checkout.

The original60e5d82a source omitted controls.py/models.py imported by package
initialization and accepted dirty unbound source. Two independent controls
reproduced those gaps. Root's17453ecc amendment binds those imports and rejects
a dirty checkout. A later synthetic clock control found that the last
prediction write could cross5s yet report feasibility. Root's075a76ec adds the
final cooperative check before classification. All20 producer and8 independent
controls now pass in0.45s, CPU/BLAS1,nice10. The pre-fix failures remain recorded
in the receipt; no threshold, fork, rate or dose formula changed.

The complete nine-file input inventory independently matches manifest
`2f962ced4b6f32aca588f1012985591acc670437ff944e8e15375acf4fd2e03f`.
Standard-library byte/hex checks reproduce all four13-parameter state hashes,
row4100/origin order and8202 update counters. Both origins at point0 use
(78.71189928623194,0.3626495805545865); point1 uses
(47.72960569225131,0.5032628489431), with exact stored rate bits and log-rate
exponentiation checked. The pinned model is tanh2x3x1, biases included, half-MSE
plus0.001/2 times the squared norm of all13 parameters. Training-only RNG
reconstruction agrees with both frozen train hashes; validation/test streams
are not instantiated. The historical engineering screen recomputes to
1.2066534822693324s; this is an allowance based on batched history, not a newly
measured single-gradient runtime.

At a shared unchanged state, let s=P_recovery*g. Sham's predicted increment is
-s. Every treatment's difference from sham is its block reduction times s,
with norm R=.05||s||. The hidden/output fractions R/||s_block|| and<=.25 gate
are implemented over0:9 and9:13, including both bias groups. Independent signed
vector calculations agree with every predicted increment and total step norm.
Equal treatment-minus-sham displacement does **not** equate total step lengths;
those remain recorded competing explanations. The exact cap, next-float
rejection, zero/nonfinite groups and all-layer ridge gradient have controls.

Only one analytic gradient call is made per fixed fork, at most four. Native
states are copied, checked unchanged and never advanced; no map/HVP/Hessian or
new-state loss is called. A failed dose does not skip later forks. Independent
synthetic cases retain all four bound inputs and every returned gradient when
a gradient is nonfinite, a later call throws, mutation is detected, or SIGTERM
arrives immediately after a return. Post-return and final-loop deadline stops
leave feasibility unset. Atomic/fsynced Store writes, reserved finalization,
complete manifests and read-only outputs preserve these cases within the
768KiB store allocation. The wrapper adds256KiB, caps stdout at128KiB, retains
source/command/PIDs/exit and applies10s TERM plus5s KILL with an inherited-pipe
supervisor. A forced kill can leave an explicitly incomplete attempt; there is
no automatic retry or claim of crash-proof completion.

This review used synthetic gradients and read-only existing input identities.
It evaluated zero target gradients, profiles, trajectories, Hessian/HVPs or
pulse outcomes and no protected data. Root must authorize the exact launch;
its eventual saved output still needs an independent outcome audit.

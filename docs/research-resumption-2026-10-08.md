# Research progress — evening of 8 October 2026

We have advanced from an attracting long-cycle example to a proved local
mechanism at one of its numerical window boundaries. The strongest shrimp
candidate, S2-28011-zoom:587, now has an independently reviewed **primitive
period-18 attracting orbit** and an independently reviewed **local saddle-node
of primitive period-18 cycles** nearby. Both concern all 13 trainable parameters
of the exact-real tanh GD map. They do not yet prove a continuous connection
between the certified interior orbit and the fold, or the complete shrimp.

## What is proved

The interior certificate excludes every proper divisor of 18 and proves
full-state attraction with `||M^16||inf < 0.726`. Its independent review includes
separate objective differentiation and a stronger nested interval proof.

At fixed exact output rate `1848862009810323/2251799813685248`, the new fold is
near actual hidden rate 75.76429203111452. The 27-variable augmented system includes the 13
weights, one rate and a normalized null vector; 27 is not the network size.
A validated root, simple +1 multiplier, nonzero unfolding/quadratic coefficients
and attracting complement establish a local nondegenerate saddle-node. The
fold itself is neutral. Locally, an attracting 18-cycle and a saddle 18-cycle
meet; the nearby pair lies on the lower-hidden-rate side. No finite-offset
neighborhood or connection to the earlier interior certificate is enclosed.

The first radius 2^-80 attempt could not prove the complementary stability
bound, although the other six gates passed. We retained it and disclosed a
single same-target radius amendment to 2^-100. That attempt passed, with
`||(M-vw^T)^32||inf < 0.035629`. Independent exact replay, saved interval
arithmetic and a 384-bit/radius 2^-110 proof in a strictly nested box all passed.
The new derivative kernel received an independent mathematical source review;
the replay shares its analytic implementation and Arb/FLINT, not a proof
assistant or a separate higher-derivative implementation.

See the [orbit review](reviews/s2-period18-independent-review.md),
[fold receipt](../certificates/s2-period18-fold-receipt.json) and
[independent fold review](reviews/s2-period18-fold-certificate-independent-review.md).
The earlier two 13-dimensional period-10 certificates and 72-dimensional period-2 certificate remain
valid. Those period-10 examples are our private results, not an external
13-parameter precedent. We make no largest-network or bibliographic priority
claim; scalable analytic theorem families are a different comparison.

## How the shrimp search stands

The independently cross-checked census contains 61 periodic windows, with
periods 4–58. Two pass our exploratory classical-shrimp anatomy criteria. The
clearest period-18 candidate has 46 numerical fold rays, 15 flip rays and 3
complex-pair rays; 10 of its 15 flip rays lead to period 36 labels in the adjacent
coarse map. The second candidate is period 16. These counts remain numerical
anatomy evidence; the new local fold proof certifies one selected point.

The broader phenomenon deserves explanation too: 41 of 61 windows have at least
8 complex-pair boundary candidates. Calling every periodic island a classical
shrimp would erase an important distinction in the results.

Next, qualify the selected flip ray 13 and its own primitive period-36 branch, then
prove a connecting tube from the certified interior cycle to the boundary.
The 10/15 aggregate doubling count does not certify ray 13 specifically. A flip
proof needs higher derivatives than the new fold kernel, and ray 13 varies
both learning rates. The [concrete scientific note](plans/s2-period18-window-next.md)
spells out those gaps and separates local boundary proofs from cusp and whole
window organization. No new flip or branch-connection computation has run.

## Scaling, utility and preservation

Training and inference utility remain unestablished. The previous six-reference
curvature approximation failed at every selected point; the saved-plane audit
explains that it omitted the actual dominant directions, especially in the
real-mode controls. The next scaling question is cheap identification of the
actual multi-step dynamical directions through matrix-vector actions, followed
by a cost/accuracy comparison. The negative LM pilots and unresolved CIFAR
correction remain unchanged; protected holdouts have not been opened.

The current 33-page reading draft is [paper/gd-route/gd-route-working-draft.pdf](../paper/gd-route/gd-route-working-draft.pdf).
All new scientific work used local CPU. All owned scientific processes exited;
no paid run, GPU job, external message or submission occurred. Both successful
and failed raw attempts have verified copies on a separate physical disk:
[orbit/numerical-fold backup](backups/2026-10-09-period18-orbit-and-numerical-fold.json)
and [rigorous-fold backup](backups/2026-10-09-period18-rigorous-fold.json).
These complement regular private Git pushes; the disk copy is on the same
machine, not an offsite backup. Main and the original Claude-owned dirty log
remain untouched. This is an interactive research checkpoint, not an extension
of the expired overnight automation or completion of the whole research goal.

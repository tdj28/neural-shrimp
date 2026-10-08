# Independent saved CIFAR restart outcome audit

8 October 2026 UTC. **Passed within the saved-data scope.** One authorized
execution of frozen auditor `4de644d379884ba85ce71bf39611c7be6ab90d10`, reviewed
at `fa4fafffa394f82701233fd4d7e2a4371eac3c17`, checked producer execution
`6fea29cda810ebb72c234dbd0fc1533bbca4e5cb`. Its twelve execution files match
reviewed producer `8aa146fdbeca2f7d1cd4957995ced9b8bda719ee` and independent
producer review `1f4f2abe8b4d4d4b8ad6e50cf1c9416d568d65f9`. The audit source
was frozen before producer execution; its independent reviewer remained
outcome-blind. Root separately authorized this audit at
`4057d1d71c137f3cc72d6e8bc65e5a273aaec00a`. No source changed and no retry ran.

The full35-file producer inventory,5-file launcher inventory, three old input
inventories, source bytes, exact initial replay and normalized data identity
pass. Gradient/HVP/two-step-JVP parity has zero saved-vector discrepancy. The
first50 callbacks match the saved restart50/four-cycle prefix under the
unchanged tolerance. All200 callbacks and202 A /203 J2VP /407 HVP /11 direct
gradient attempted and returned events have consistent causal order, with no
pending operations. No operation was independently regenerated in this audit.

Independent `math.fsum` reductions reproduce the true forcing decrease from
0.8843136324964807 to0.661988519072711 at the same saved development state.
SciPy returns info1; neither requested true tolerance1e-10 nor the unchanged
looser forcing<=0.5 gate passes. R dot Ap is negative
(-1.292705731796484e-10), but none of the four fixed trial steps decreases the
absolute residual merit or passes Armijo. All scientific negative results
remain in the immutable evidence:

| Alpha | Merit/original merit | Relative closure | Candidate |
|---:|---:|---:|:---|
| 1 | 665.1136522 | 1.279921299e-4 | no |
| 1/2 | 58.40686159 | 3.799696065e-5 | no |
| 1/4 | 4.099472089 | 1.006974976e-5 | no |
| 1/8 | 1.060834176 | 5.122522790e-6 | no |

Every trial fails closure<1e-8. Its separation screen passes, which does not
establish primitive periodicity without closure. No accepted orbit, applied
continuation step, spectrum or additional damping trial follows. This is a
qualified comparison with negative acceptance results, not an incomplete
execution or a proof that a periodic orbit cannot exist.

The audit worker took0.668624s and its launcher0.840708s, with process and
wrapper exit0. It retained6+6 files totaling46,553 bytes, below4MiB; every
manifest entry and read-only permission was checked after completion. All
observed audit and producer PIDs, plus the audit launcher PID, were absent.
There was no deadline, memory, storage, prefix, parity or audit failure. The
producer ran12.686797s (launcher14.160857s), with sampled group peak932,532,224
bytes and worker peak903,364KiB, within its declared monitored2GiB limit.

Audit raw manifest:
`0a6e42f5bd926504cbe6ed3f7f81f9fe4d33c6e634f17a4a193ed8d3c9bc9c1c`.
Audit launcher manifest:
`f79eb594494f3730e61d6e852b36ed71ea43b4c327012b12faed7279e4943049`.
The companion JSON records exact paths, input/producer manifests, source
hashes, authorization, timing, counters and independent numerical reductions.

This independently checks saved-array arithmetic and provenance; saved Ap,
map images and derivative outputs remain source-bound inputs. The unsaved
Arnoldi basis and complete solver path cannot be replayed. There were no new
model, gradient, HVP, JVP, QR, SVD, solver or trajectory evaluations. The result
supports only this matched development-state diagnostic, without a general
causal claim about restart, convergence, orbit existence or attraction.

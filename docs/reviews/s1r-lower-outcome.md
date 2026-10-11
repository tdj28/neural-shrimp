# Lower S1R comparison — numerical gap retained

8 October 2026 UTC. Independent outcome audit pending; the exact source passed
independent review before this one bounded collection.

**TLDR:** We can follow the same numerical period-two orbit through 54 of 75
planned parameter steps in the 72-parameter network. The next step fails the
corrector checks. We therefore still do not know whether the two saved endpoint
orbits belong to the same family. This adds no training or inference benefit.

The hidden log-rate moved from 1.10 to 0.992 at fixed output log-rate
0.6271428571428572. Every accepted link retained direct, two-half-step and
reverse witnesses, with phase and secant checks. The collection used 63 link
attempts, 473 phase-corrector calls and retained 56 witnessed edges; seven
attempts were rejected. It completed in 38.103 seconds internally and
38.755 seconds including the wrapper, below the fixed resource ceilings.
Raw collection and launcher evidence totals 120,264,102 bytes; all files and
manifests were checked and made read-only. No GPU or paid compute was used.

At the next nominal target 0.990, all five proposals (initial plus four actual
halvings) stopped with first-phase line-search failure. The final segment's
historical field `halvings=5` counts failed levels, not five applied halvings.
There was no accepted subpath in that segment, and the lower endpoint at 0.95
was never reached by the comparison. A numerical gap is not evidence of a fold,
nonexistence, disconnection or coexistence. The source allows unstable orbits;
it did not reject a link merely for instability.

The compact receipt binds the exact source, review, raw inventories and last
accepted pair. Numerical outcome audit is being implemented separately from
the producer; its evaluator is frozen before reading these results. No retry
or opposite-direction collection is authorized by this result. A next method
change would require a separate disclosed design, informed first by the saved
failure evidence. Existing exact-real point certificates remain unchanged.

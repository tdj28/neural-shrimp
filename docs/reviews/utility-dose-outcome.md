# Four-gradient utility gate — fixed dose infeasible

8 October 2026 UTC. Independent analytic saved-outcome audit pending.

**TLDR:** The proposed fair-dose comparison is too large to implement as a
modest output-layer rate reduction at any of the four saved starts. We reject
that dose definition before running a pulse experiment. This says nothing yet
about whether geometry can improve training.

Exactly four analytic gradients were evaluated at the unchanged, hash-bound
S2-28011 states after 8,202 updates, using their fixed recovery rates. Hidden
and output blocks include all biases. At R=0.05||P*g||, the output-block
reduction fractions needed to match treatment-minus-sham displacement are
1.412134, 1.444129, 0.766117 and 0.766069, all above the preset 0.25 cap.
Hidden fractions are approximately 0.05003, 0.05003, 0.05011 and 0.05011.
The first two required output fractions even exceed one, so their purely
algebraic predicted steps would reverse that block's update direction. No
such update was applied. All four gradients and all algebraic predictions,
including unequal total step lengths, are retained.

The fixed candidate is infeasible. No fork was dropped, dose clipped, threshold
widened, pulse applied or control outcome acquired. A smaller dose is a possible
new exploratory design, not a retroactive success of this one. Qualified distant
periodic and chaos controls are still absent, independently of this rejection.

Producer computation used one CPU thread, four gradient returns, 0.0703 seconds
internally and 0.5079 seconds including the wrapper. It advanced no state and
used no GPU, paid resource or protected outcome. The receipt binds immutable
raw/launcher manifests, all four native inputs, the exact reviewed source and
the fixed cap. Independent review will reconstruct the gradients separately
and check every dose/step decision before manuscript integration.

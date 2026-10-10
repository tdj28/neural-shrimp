# 10 October 2026 — saved forcing split after reverse-chart failure

The first reverse chart failed q=2.4755/inclusion while primitive and full-fold
anchor passed. Bind its exact worker manifest9aee1856145ff51a5f76ab809118a406ef40d478344217dddef0aa5472432f71
and execution136b77b4. No new model/derivative/solve/eigen/SVD here.

One fixed saved reduction at256bits: separate weighted C*H0,C*H1*d,
C*H2*d²/2,C*H3*d³/6 and old D*E. Evaluate residual on ONE proposed halfwidth2^-10
(old width/8), retaining the entire old valid broad H3 enclosure and all
center defects. Propose ONE dyadic radius >=8 times the resulting weighted
forcing upper bound, with the same strictly positive weights. This is only a
proposal: new derivative domains, contraction/inclusion/primitive and full
correlated anchors must be separately reacquired and checked. No inferred
linear scaling is itself a proof. If the radius is outside[-100,-10], stop.

CPU1/nice10,30s internal/40+5 external,1GiB RSS,3MiB worker+1MiB launcher.
One235D saved matrix-vector calculation per term; no chart search. Exclusive
attempt, exact source/input hashes and immutable output; preserve failure.

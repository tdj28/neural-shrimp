# One retained stationary direction: actual gradient-merit trial

2026-10-11 exploratory adaptation after the complete stationary-seed attempt001.
That attempt failed its frozen .5 linear-forcing gate at .7651434545501713;
no iterate was accepted. Preserve that failed result. The mathematical threshold
for descent of phi(x)=||grad L(x)||^2/2 is eta<1: with r=Hp+g,
g^T Hp=-||g||^2+g^T r <= -(1-eta)||g||^2. The .5 threshold is a solver-quality
choice. A separate exploratory pilot may test actual decrease without rewriting
that old gate. No interval claim is made from floating Cauchy diagnostics.

Bind the exact original98666D midpoint, saved final MINRES256 direction and its
true Hessian action (native HVP259), full source archive and immutable inventories.
Replay one gradient and one HVP byte-for-byte before any trial. Require fixed
eta<=.9 and negative measured slope. Test alpha=1,1/2,...,1/4096, in order,
within unchanged5% relative original-state trust radius; skip out-of-radius
proposals before derivatives. Accept first measured gradient-merit Armijo decrease
with c=1e-4. Save every input, actual gradient and trial, stop at first acceptance
or13exhausted proposals. No new Krylov solve, power method, optimizer run, modified
objective, radius search, data, holdout or promoted orbit.

At most1HVP and14gradients including replay. Old measured maxima and exact same
runtime/native-source closure give2*(HVPmax+14*gradientmax)+20s, required below90s.
CPU1nice10/BLAS1;90s cooperative,105+5s owned external;2GiB RSS/8GiB AS;
128MiB worker evidence with4MiB closing reserve/131MiB combined. Original failed
attempt archived. Exact source/card and independent review before target; exclusive
explore/cifar/stationary-seed/saved-merit001. Accepted point is a local numerical
stationary-seed improvement only; stationarity, eigenpair and cycle each remain
separate gates. Preserve unsuccessful direction/line search without automatic retry.

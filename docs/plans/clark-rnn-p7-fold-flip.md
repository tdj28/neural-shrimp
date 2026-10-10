# P7 two fixed fold--flip localizations

Both existing fold branches meet a real minus-one complement near(g,delta)
=(1.802,.7866). The plus-one remains simple and foldc nonzero. These are
specifically the lower130–131 and upper28–29 brackets in immutable
p7-lowgain-extension-001, manifest01297f353bdcc889bee8ffe497fa3cea28186836f45596b120ddc8d12a088d93.
Use only the earlier native node in each bracket, and its saved closest−1
right eigenvector. This is an exploratory post-outcome target choice.

Unknown z=(x4,g,delta,q4,r4), fourteen equations:
H=(F^7(x)-x,(M-I)q,(M+I)r,(q^Tq-1)/2,(r^Tr-1)/2).
The full14x14 Jacobian uses existing analytic second endpoint derivatives.
One maximum12-Newton solve per fixed seed, no grid or alternate root search.
Retain full endpoint/derivative/cycle/Jacobian/linear/eigen/SVD arrays before
downstream decisions. Require primitive7, simple+1 and−1, remaining modes
attracting, nonzero foldc, nonsingular scaled augmented Jacobian and native
phase/sign association to the starting cycle. Numeric gates: residual<1e-10,
criticalerrors<1e-7,gaps>1e-4,remainingradius<1-1e-5,divisor1>1e-6,
scaled smin>1e-10 max(1,smax),|c|>1e-5,phase/sign relative distance<.05.
These support local numerical fold--flip nominations, not interval theorems.
Close parameter coordinates alone do not imply a common orbit.

Fixed scales[x20×4,g10,delta.1,q1×4,r1×4], residualscale[.05×4,1×10].
Stepguard scaledmax.1; gain stays within.05 of seed,delta within.01,
1<g<5/.5<delta<.95. All failures retained, no automatic retry or halving.
Only after completed outcome interpretation may these roots nominate flip
continuation. No automatic certificate or full-shrimp claim.

CPU1nice10,45s internal60+5external,512MiB,16+1MiBraw,128augmentedreturns.
Ten non-target synthetic14D evaluations first;64max+5<=45s costgate before
input loading. One exclusive p7-fold-flip-localization-001. Constant-time
prewrite reservation and full owned-process-group cleanup unchanged.
Command: `nice -n10 /data2/neural-shrimp/.venv/bin/python tools/launch_clark_rnn_fold_flip.py --out /data2/neural-shrimp-data/explore/clark-rnn/p7-fold-flip-localization-001`.

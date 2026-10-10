# Short native RNN fold/flip curves, adapted after numerical slices

Start exactly the gain10.02 period5 fold and flip returned by
p5-boundary-slices-001 (manifest20d1171d8ccfcc41743d85b8757c5c8affbd15b1c393d6f5209af971b363f77f), same byte-bound N4 J0.
This asks whether the local fold/flip pair continues and turns in the plane.
No new random instance, parameter grid, paid/GPU resource or protected data.

Unknowns(x4,g,delta,q4); nine equations(R,(M-signI)q,(q^Tq-1)/2).
Analytic first/second derivatives of the exact mathematical tanh map, evaluated
in binary64. Column scales(20,20,20,20,10,.1,1,1,1,1); row scales(.05 four times,
1 five times). Null tangent from full9×10 SVD, smallest nonzero singular value
>1e-8, null residual<1e-10; orient initially toward increasing gain and later by
previous tangent. Save every full U/s/Vh, native derivative return and solve.

Four traces: two boundary signs, both arclength directions. Fixed step.015,
at most48 accepted steps each, eight Newton evaluations per correction. Keep
1<g<40,.05<delta<1, scaled Newton step<=.1. Scaled system closure<1e-10 and
physical cycle closure<1e-9. Every accepted edge must reverse-correct from its
child back to the original parent's hyperplane within1e-7 scaled coordinates.
Retain all failed forward/reverse attempts and stop that direction; no halving
or retry in this unit. Node primitive separation>1e-6, critical error<1e-7 and
isolated multiplier distance>1e-4. Complement stability and fold coefficient
sign are retained even if changed; no favorable-node filter. No closed-loop,
whole geometry, flip cubic or interval claim from this numerical trace.

Ten non-target synthetic family returns (including writes) first. Resource
projection3500*maximum_unit_seconds+20<=120s, before any target input loading.
Hard stop CPU/BLAS1,nice10,120sinternal135+5external,4000 attempted family calls,
448MiBworker+64MiBheadroom,64MiB scientific raw+1MiBlauncher. A conservative
sequential prewrite reservation replaces repeated full-directory walks; every
file and final manifest stays hashed/read-only, all owned processes checked.
At least1GiB fresh MemAvailable; leave foreign processes untouched.

Command `nice -n10 /data2/neural-shrimp/.venv/bin/python tools/launch_clark_rnn_continuation.py --out /data2/neural-shrimp-data/explore/clark-rnn/p5-boundary-curves-001`.
The fixed short traces can be incomplete. A curve endpoint/correction failure
is not absence of a shrimp. Further extension or rigorous enclosure requires a
dated source-bound unit, and strong claims await independent verification.

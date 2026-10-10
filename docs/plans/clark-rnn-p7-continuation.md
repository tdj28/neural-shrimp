# Two native p7 fold branches: fixed short2D geometry

Adapted after partial p7-boundary-slices-001, source69e2c0a5, outermanifest
56b0b759634412447a18700602f75230aa2b4613a84376ffd64cefdd69dc3103.
The first two gain slices completed before the90s deadline. Use only the two
qualified middle-gain roots g3-dir-1-fold and g3-dir1-fold at gain4.1279512551,
leak.7462969796/.8257282717. Both have primitive7 closure, simple+1,
nonzero fold coefficients, contracting complements and native identity passes.
No unfinished third-slice result or rejected association enters selection.

Exactly96 fixed .015 scaled-arclength steps in each direction from each fold:
four traces,384 new nodes maximum. Native9eq/10unknown(x4,g,delta,q4), same
analytic family Jacobian/fullSVD tangent and preconditioning as p5 PAL.
All native derivatives/SVDs/linear solves/eigensystems/coefficient returns saved.
At each node record foldc and full signed spectrum; retain nonattracting
complement nodes. Primitive1-divisor/simple gates unchanged. Every edge must
retain nearest native phase/sign and reverse-correction error<1e-7. Stop on
domain1<g<10/.5<delta<.95, rank/primitive/identity/correction failure or cap.
No automatic halving/retry, no scalar coefficient sign jump declared cusp.
Only a c-zero with simple+1 and appropriate two-parameter nondegeneracy can
nominate a cusp; curve projection alone is insufficient. No connecting orbit
is inferred from close parameter projections. This unit discovers actual
anatomy; it does not trigger another certificate automatically.

The old slice attempt remains immutable and failed/partial. Its repeated
filesystem byte scans underestimated late cost. This new source uses the
already established constant-time conservative prewrite byte reservation;
no old source is repaired and no trajectory prefix is rerun.
CPU1nice10,90s internal105+5 external,448MiBworker+64MiBheadroom,
128MiBraw+1MiBlauncher,7000familycalls. Ten non-target synthetic period7
family evaluations precede target input access;6500*max(time)+20<=90s.
One exclusive p7-boundary-curves-001. All failed traces/partial current trace
retained in result plus full immutable records. No paid/GPU/protecteddata.

Command: `nice -n10 /data2/neural-shrimp/.venv/bin/python tools/launch_clark_rnn_p7_continuation.py --out /data2/neural-shrimp-data/explore/clark-rnn/p7-boundary-curves-001`.

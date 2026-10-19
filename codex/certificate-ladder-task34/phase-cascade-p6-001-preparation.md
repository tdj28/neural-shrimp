# Cascade-P6 phase proposal preparation

2026-10-19T01:42Z. Prepared only while the original dense P6 attempt runs. No
partial dense result is read or interpreted. Numerical qualification, current
source/input review and a fresh serial capacity admission remain required.

The unchanged phase_preconditioner.py and derivative_profile.validate_cells
already accept the shared five-cell card. This new allocation uses exactly
--cell cascade-p6, period6, binary64alpha.065693/log10T-.14798 and its own original
513x896 history. The same-period band-P6 history and P3 history are not substituted.
The shared phase-cascade-p3-001-card.json is preserved under its original name.
No caller, manual map, evaluator or active P6 source is modified. Earlier generic
source/entry controls remain completed; no model or old control was rerun.

After complete P6 dense qualification, the proposed worker computes all six
one-step896-column Jacobians, saves every phase image/residual/matrix, and saves
the seven exact-binary64 proposal arrays (A,B,centers,parameters,ordered product,
left/right inverse defects). Phase closure must pass the existing finite gate.
These are numerical/exact-dyadic proposals, not interval derivative or inverse
proofs. Subsequent saved audit must verify all phases, ordered product and every
proposal array under this exact named-cell/source authority.

## Measured prospective envelope

Original accepted three-phase run took 1144.603614164051seconds
externally. Scaling its six-versus-three phase count gives 2289.207228328101seconds
only as a forecast. The own P6 composed-column forecast2160.076149seconds is a
separate consistency check; neither is a guaranteed one-step-phase duration.
The new limits are 5080CPU/5180wall,12GiB AS,one CPU plus supervisor,
128MiB retained output,28GiB serial startup and16GiB runtime available-RAM floors.
The output budget covers two copies of six896-square matrices plus four square
proposal arrays (102760448 raw matrix bytes), phase vectors and metadata.
No old attempted run is extended or retried. The P3 initial40GiB concurrent
preflight refusal remains preserved under its own allocation; this is the first
new, serial P6 phase allocation. Source, all original inputs, exact dense saved
qualification/closure and any intervening owned job closure must be pinned again
before private execution checkpoint and fresh capacity admission. No automatic
launch, retry, extension, source substitution or remote provisioning.

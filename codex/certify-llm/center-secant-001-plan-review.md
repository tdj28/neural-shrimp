# Concrete center-secant-001 plan admission

13 October 2026. Reviewed plan SHA256:
`ee56cd6aad3a7beae563eddb60f50751c3dba197c70aa7e4c6a581597eeb206a`.
Driver remains the reviewed
`c828ebc744eb95795e9051d5dbe0b8361828ebdfb13688afae5c1f60f7694132`.

**Admitted for the stated one-candidate exploratory attempt**, conditional on
its run card, private push and bounded launcher. No secant replay, candidate
construction or model call was performed during this audit. Those entry points
were replaced by refusing stubs in the audit process. Only the authenticated
saved-data readers and exact residual reconstruction ran.

The concrete plan pins complete parent receipt
`84ddb13b223720da8b29b46a9705129391d6f9850c9a29030c36bf737affaf5d`
and all current importer/kernel/map dependencies correctly. Full admission
checked the 32-artifact prefix, 56-artifact fixed-R seed, 77-artifact original
refinement, and 98-artifact completed resume. The inherited exact map parameters,
assets, prefix, preconditioner, source hashes and backend agree.

All four chronological tuples are present: original before, original after,
resume correction 1 after, and resume correction 2 after. Each has all eight
896-coordinate exact centres, full image balls and recomputed residual balls;
32 prior phase evaluations are accounted for. No tuple was selected or omitted
based on improvement. The final inherited residual upper-bound diagnostic is
approximately `7.15720237376593e-52`; importing it neither proves an orbit nor
predicts a secant improvement.

The plan allows exactly one candidate, with all three historical secants replayed
under history/signal/update/step caps `4/128/1024/4`, then exactly one fresh
eight-phase pass. It specifies CPU1, nice 10, 18 GiB, 1800 seconds, no paid/GPU
work. The nested 3600-second legacy plan is admission metadata only and does not
increase the new runtime budget. Every rejection, baseline fallback, failed or
worsened result must remain recorded.

Timing fields agree with completed evidence. The recent mean `144.308102` seconds
is the mean of **all 16 resume phases**, not only the final eight (whose mean is
`145.821275906`). The recent maximum is `167.565919809`; the largest prior phase
across both completed attempts is `190.798208741`. Eight times that prior maximum
is `1526.385669928` seconds, leaving `273.614330072` seconds for admission,
preparation, restore and persistence. The completed saved-data admission in
this audit took 36.950 seconds after imports, peak 759,924 KiB. The large-target
secant replay/conversion cost is still unmeasured; completion and acceleration
are explicitly not guaranteed by the planning allowance. The launcher retains
and terminates a timeout rather than extending the budget.

An initial audit-only timing assertion mistakenly interpreted “recent mean” as
the last eight phases. It failed after full history admission; checking all 16
confirmed the plan's value. The corrected bounded audit passed. No producer
files or completed evidence changed, and no kernel replay occurred in either
invocation.

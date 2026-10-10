# Native eigenpair-locus prototype: synthetic cost screen

TLDR: the full 470D branch representation appears affordable on one local CPU;
this has not yet extended a training-cycle boundary or connected any certificate.

Source23b0a4629a62d91cbf5953e98a5a8dd2f571dac1 passed the single committed
synthetic profile. Two full polynomial chart evaluations took4.791/4.486s;
12 synthetic13D/24-example GD MixedJet calls had maximum0.01046s. Total10.910s,
peak919,820KiB,194 ticks. Both uniform eigenpair-locus controls passed and
both correctly rejected primitive18 because the fixture repeats a primitive2.

The fixed first-two-target-chart cost heuristic projects32.695s,1,970,712KiB
(~1.88GiB) and21.6MB compressed evidence. The memory estimate is close to the
proposed2GiB cap; exact target constants, conditioning and serialization can
change it. The future target supervisor must monitor its own whole process
group and retain a resource failure. This is an admission estimate, not a
promise that the target bound or its resource limits will pass.

All223 files/3,347,152bytes are immutable and hash-verified; launcher/timeout/
worker PIDs1077928/1078041/1078042 are absent. The companion JSON binds the
complete source/input/runtime/output manifests. No target model/state was
read by this profile. The prototype makes no simplicity, genericity,
complementary-stability, projected-curve or full-shrimp claim.

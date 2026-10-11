# EXP-014 operational amendment, 17 September 2026

Attempt 001 is incomplete and remains immutable. It ran from privately pushed
`d1ca6043c2476f34b2314d88309ff732b34822a2`, with plan SHA-256
`c6d82aa6efbef7da39f2fe6c3b1b38f7d11577589dd3d80561c3a582189a8062`.
Both reference jobs and all sixteen phase propagators completed. Seven traces
completed; three were interrupted and 902 never started. Every one of the
930 planned identities is retained. There is no qualified setting average.

The original 2 GiB combined owned-process RSS ceiling stopped trace dispatch.
The measured peak was 6,062,612,480 bytes across the parent and three children;
this RSS sum deliberately counts shared pages once per process. The parent
later retained about 1.53 GB during finalization. Releasing the prepared Python
object did not reduce the resident allocation enough before forking. This is
an allocation/accounting limit, not exhaustion of server RAM: available host
memory remained at least 76,119,867,392 bytes. Raw bytes were below the unchanged
2 GiB limit. All registered worker identities were joined and are absent.

The final receipt has 112 artifacts, 316,844,371 bytes, SHA-256
`565527cec80c64dd7584123cf04e576cddd52f330fae52c746f071837d561053`.
The raw directory is `/data2/neural-shrimp-data/raw/EXP-014/attempt-001/`.
A separate operational audit verifies 307 inventory, identity, chronology,
resource and terminal-ledger checks. Saved-array replay is recorded separately
in the compact receipt; failure preservation does not convert this attempt to
scientific completion. The complete elapsed producer/finalization scope was
18.074 seconds. No paid compute was used.

## Prospective resource-only recovery

Keep the original frozen sources, protocol and plan unchanged. Add an explicit
recovery controller and machine plan. Raise combined owned-process RSS from
2 to **12 GiB**, require **16 GiB** host RAM available before each stage, and
retain an **8 GiB** host reserve during work. At the measured parent footprint,
parent plus four forked workers gives roughly 7.7 GB of summed RSS; 12 GiB
allows working overhead while retaining substantial headroom on this server.
This is a conservative RSS accounting ceiling, not a claim that all counted
pages are independently allocated physical memory. Check availability afresh.

Retain four workers, one numerical thread each, the 1,800-second producer
ceiling, 120-second prerequisite jobs, 60-second traces, 120-second sealing
grace and 2 GiB raw budget. Change no mathematical function, solver, setting,
initialization rule, pulse, readout, prediction, qualification gate, estimand,
analysis denominator or interpretation. Bind the recovery source/test and
this amendment in addition to every original source. Saved verification must
read the corresponding committed recovery plan when auditing recovery data.

A new private execution freeze and independent review precede all recovery
work. Rerun the complete two references, sixteen propagators and 912 traces in
a new exclusive attempt; do not reuse the seven completed pulse outcomes or
resume the old directory. Repeat reference-only prediction sealing before new
pulses. This operational change responds to the measured resource failure;
it neither selects favorable outcomes nor changes the scientific target.

# LCM1: paired maximum-context and short-context exploratory maps

Approved 2026-10-24 following the completed LCP1 pilot and its measured quote.
One rental; cost and ceiling are in the private ledger and conversation.
This is an exploratory map, not a confirmatory chaos claim.

Qwen/Qwen2.5-0.5B-Instruct revision
7ae557604adf67be50417f59c2c2f167def9a775, original soft-token feedback,
layer-12 philosophy steering. Each arm uses 16 evenly spaced log10 temperatures
from -0.3 to 0.3 and 16 steering strengths from 0 to 0.1. Steering is the outer
loop; temperature is the inner loop. Maximum-context arm runs first, followed
by the identical short-prefix control grid. All 512 cells start fresh.

Maximum prefix: 32767 tokens plus one feedback position. The exact LCP1
construction repeats tokenized prose and appends the unchanged short chat
prefix (37 tokens in the pilot). Archive IDs and decoded text. This confounds
prompt length with added repeated content; it is not a pure length effect.
Each cell uses 256 settling and 512 measured updates; base embedding states
and per-update twin growth are saved. Relative perturbation radii 0.001 and
0.0003 share a base, random direction seeded by 7 + grid index. Float32,
TF32 disabled, prefix chunks of 256. Retain the pilot's short-input chunked
prefill agreement and repeatability gates and short/maximum timing probes.
Do not interpret contraction or absence of detected recurrence as proof of
absence of chaos; positive finite-window growth is only a candidate.

Pilot maximum-context cells took about 106.4 seconds each. Extrapolated
maximum arm is 7.57 hours, with about one hour for the short control plus
setup, checkpoint verification and retrieval. Expected total 9–11 hours.
Absolute resource lifetime 780 minutes; numerical guard 690 minutes, tool
limit 700 minutes and runner limit 720 minutes. Independent watchdog retains
60 minutes of retrieval reserve. No automatic rerental or numerical retry.

Raw output: `llm-soft-loop/pod64/lcm1/map-01`, outside Git. A dedicated local
collector starts before rental, discovers only the exact creation name,
source and approval record, and verifies every checkpoint manifest against
local files before sending an acknowledgment. The worker waits at startup,
after every completed cell, and at final manifest publication; a missing
acknowledgment stops numerical execution after five minutes. Transfer failure
retains partial files. Collector death therefore cannot silently allow the
entire run to proceed without retrieval. Final normal completion requires
all manifest files verified locally. The small final archive is supplemental.
The collector has the same absolute deadline as the owned pod. Never adopt
another resource. Keep all partial outcomes if the deadline wins.

Before delivery: verify source, exact prompt and direction, all 512 cells,
coordinate coverage, array shape and finiteness, twin/half growth, recurrence
through lag 64 with precision-aware labels, final manifest, archive if present,
and exact pod absence by direct lookup and inventory. Verify a separate-disk
backup. Render comparable long/short scientific maps and visually inspect.
Store compact receipts and update research/hardware logs; push main.

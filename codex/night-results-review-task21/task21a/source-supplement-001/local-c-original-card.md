## 2026-10-15 02:02 UTC: run (exploratory): FB stage 1c, the deeper cascade levels on the local GPU (human: "Do it")

- **Question:** Do the later ratios of the crescent's cascade at log10 T -0.156 approach 4.669, now that the first ratio there is 14.93 and the frozen prediction P-F1 has failed (scored in the next entry)? And what is the first ratio at -0.152?
- **Setup:** Local RTX 2000 Ada, float64, tools/llm_feigenbaum.py cascade with the settings of the pod runs (philosophy, layer 12, free alpha, transient 900, k 6, tol 1e-12), but starting from the last level that ran cleanly, so that the first switch uses a known point:
  - (A) T -0.156: period 12 at alpha 0.06507 (inside the measured period-12 band 0.0650494-0.0650877), next 0.0650907 (inside the period-24 band), 5 flips, ratio guess 4.669, 96 steps per period after a switch.
  - (B) T -0.152: period 6 at alpha 0.06505, next 0.06535 (start of the period-12 band on the cross-512 map), 5 flips, ratio guess 4.0, 96 steps per period.
- **Reading:** The ratios are scored against P-F2 (later ratios within 5 % of 4.669) and reported as they come out. A failed switch is reported with its message.
- **Output:** `../neural-shrimp-data/explore/llm-soft-loop/pod64/cascade/crescent-t-0.156-c`, `crescent-t-0.152-c`

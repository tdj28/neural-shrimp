# Consult 24 adjudication: the synchronization section (Sec. VIII of the article)

One call, executed 20 Oct 2026 02:01-02:07 UTC (brief at c533a12a, dry-001 checked first; completed, 56,400 tokens;
artifacts in neural-shrimp-data/consults/2026-10-20-sync-section/execute-001). Verdict: "READY AFTER SPECIFIED FIXES".
Advice is advisory; each finding was checked against the receipts (docs/sync/sync-summary.json, the SYNC log entries).

| finding | decision | what was done (article cee14ee; report mirrored) |
|---|---|---|
| B01 typical-orbit stability presented as a general prediction | accepted | Opening and condition paragraph say c* is the typical-orbit stability threshold of the synchronized chaotic state; the symmetric range up to 1-c* is noted; other invariant sets and attractor choice are stated; cell-1 multistability (period-2 synchronization from far starts) and the two above-c* joint periodic orbits are disclosed. |
| B02 thresholds pooled across protocols | accepted | New Table VII: every bracket recomputed from the receipt with the analysis rule, for near (3000 and 10,000 steps) and far starts, both criteria; non-monotone and period-2 cases flagged. The text quotes protocol-bound brackets only. |
| B03 burst descriptions beyond the measured mechanism | accepted | Counted bursts described by their thresholds (below 1e-8 to above 1e-3); "intermittency of the on-off type"; bubbling "consistent with", candidate orbits not computed; rounding "plausibly supplies the tiny differences, causal role not isolated"; the 8.9e-16 figure stated for identical inputs. |
| S02 "same text" needs a readout | accepted | Text defined as the most probable token at each step, as elsewhere in the article. |
| B04 approximate agreement stated as exact | accepted | "approximately followed", largest discrepancy 0.051 per step, up to four twin standard errors, uncertainty of lambda_1 excluded; takeaway and conclusion bounded. |
| S01 protocol inventory | accepted in short form | One sentence lists what was not tested (prefixes, models, directions, larger start ensembles, asymmetric or delayed coupling, precision changes); the table carries the varied protocols. |
| S03 brief's "-0.02 to -0.05" | accepted as an error in the brief | From lambda_1 = 0.196 the predicted transverse exponents at c = 0.105, 0.11, 0.115 are -0.040, -0.052, -0.065. The wrong range came from the SYNC results log entry; a correction is logged. Neither paper quoted it. |
| F01 plotted mean versus tail-maximum classification | accepted | Caption: the dotted line is a reference level; classification uses the last 500 steps and its thresholds are in Table VII. |
| R01 citation roles | accepted | Fujisaka and Yamada cited for the condition; Pecora and Carroll for synchronization of chaotic systems more broadly. Both entries were verified when added. |

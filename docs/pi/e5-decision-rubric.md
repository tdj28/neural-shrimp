# E5 decision rubric for the development choices (written 2026-10-16T14:58:56Z, before any development outcome)

No development batch or sample file exists at the time of writing (dev loop running since 14:57 UTC, first batch
not finished; dev sampling not started). Tool commit 425716a8.

1. Retirement in the held-out loop: use --retire if and only if the development loop has zero virtual-retirement
   mismatches (a cell virtually retired whose full label is not periodic with the period read at retirement) and no
   gate failures. Otherwise run the held-out loop without --retire and scale down elsewhere if needed.
2. Repetition rule: keep the card's rule (last 96 tokens periodic, period <= 32) unless it is degenerate on the
   development prompts, defined as T_rep censored (below or above the grid) for at least 4 of the 8 prompts. If it
   is: when the period <= 48 diagnostic shows long loops, use period <= 48 with window 96; when loops start late
   (the window-64 diagnostic repeats but the window-96 rule does not), use window 64 with period <= 32; adopt the
   first alternative that leaves at most 3 of 8 prompts censored; otherwise keep the card's rule and report.
3. Entropy level: keep 2 nats unless at least 4 of 8 development prompts do not reach 2 nats within the grid
   (T_H outside [0.1, 3.16]). If so, use the median over development prompts of the first-step entropy at
   log10 T = -0.25 (grid midpoint), rounded to 0.25 nats, a level that does not depend on T_rep.
4. Held-out samples: keep 16 per cell unless the projected held-out duration (loop with retirement projected from the
   development virtual-retirement steps and the measured throughput; sampling at the development rate) exceeds 7 h;
   then 12 per cell (8 if the projection exceeds 9.5 h). All 24 temperatures and all 24 held-out prompts are kept.

# Weighted quadratic continuation outcome — 10 October 2026 UTC

The one fixed pilot passes both rigorous shooting charts for the original
S2-28011-zoom:587 primitive-period-18 orbit. The connected actual hidden-rate
interval is exactly
[1254710224381623/17592186044416, 1255809736009399/17592186044416], width1/16,
at the unchanged exact output rate1848862009810323/2251799813685248.
Each chart passes all234 strict coordinate inclusions, all234 native anchor
inclusions, and exclusions of proper divisors1,2,3,6,9. The second anchor is
the first chart's enclosed root at their identical exact endpoint. Weighted
contraction upper bounds are below0.429375 and0.426822; these are rigorous
bounds on the newly acquired full-domain Jacobian, not extrapolated bounds.

Source0534e0fd67b15dff349f1fb452b0cb4114bb8279; execution
 d183f0cca438e24a61256cb5beeff3290f2d706b. Fixed settings were256 bits,
quadratic predictor, halfwidth2^-6, maximum correction radius2^-24, the same
exact saved positive weights in both charts, and two Newton proposals/chart.
The worker used10.1286s,761268KiB,266ticks,144 wrapped/288 unit derivatives
and108 path jets. Launcher10.5930s,exit0; owned PIDs absent. All raw inventories,
source hashes and decompressed payload identities were checked;29,060,541
bytes are retained read-only. Exact hashes and complete stage/runtime records
are bound in `s2-period18-weighted-outcome.json`.

This extends the same connected primitive branch fourfold relative to the
previous quadratic pilot. It does not evaluate attraction on the wider
interval, connect to a fold/flip, prove whole shrimp geometry, or add a new
attracting-orbit instance. Independent mathematical outcome review is pending.
A receipt-only script initially addressed `charts` at the envelope level and
raised KeyError before writing; correcting the envelope lookup read the same
saved result, without proof replay or target acquisition.

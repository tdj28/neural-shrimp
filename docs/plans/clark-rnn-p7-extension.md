# P7 low-g endpoint resolution, fixed step adaptation

The two-fold p7 geometry unit completed with258 accepted nodes; high-g arms
reach the fixedgain10 limit. All accepted complements attract. Lower-lowg
stopped because its next .015 step exceeded the fixed.05 full-state identity
threshold; upper-lowg stopped at96nodes. Neither result is a cusp or shrimp.
Use only immutable p7-boundary-curves-001 endpoints lower:-1:36 and upper:-1:96,
manifest107003b5ddd108b636701a9cd7083fea9e06b2d4b0a4d9f6818a119611b0b098.

One disclosed resolution change:192 new steps per endpoint, fixed.003 instead
of.015. Do not weaken identity/rank/closure/reverse gates, repeat prefixes, or
choose new seeds. Domain1<g<5/.5<delta<.95. Keep all spectra and foldc/coupling;
a projected turn or sign jump does not establish cusp, especially near a
second+1. A resonance tongue (possibly terminating at small-amplitude dynamics)
remains a competing explanation. The aim is a finite anatomy decision, not
another proof count. No automatic retry/halving. No parameter intersection
is asserted to identify the same orbit without full-state phase/sign evidence.

CPU1nice10/90s internal105+5 external/512MiB/128+1MiB/6000familycalls.
Ten synthetic native period7 family timings,5500max+20<=90, before target
inputs. Constant-time conservative prewrite reservation with final reserve.
One exclusive p7-lowgain-extension-001, all returns/failures immutable.
Command: `nice -n10 /data2/neural-shrimp/.venv/bin/python tools/launch_clark_rnn_p7_extension.py --out /data2/neural-shrimp-data/explore/clark-rnn/p7-lowgain-extension-001`.

# Saved Taylor/defect split before a wider pilot — 10 October 2026

Adapted after tangent unit002 passed. Fixed input is that completed unit's last
full chart at gauge19/128, execution5f32fccb, immutable manifests
`60c2b2d52d6a19a9e68d11d62978d9e148e63866818a5f6d00ea1f827740f7ed` /
`0eb0261857d5b2361c7907a4e66af7e2b25399203d80de461b6ab48823e6dd9d`.
No new map, derivative, model, eigensolve, SVD or trajectory acquisition.

Restore the saved balls conservatively at256bits. Independently reduce
D0=I−C*J_center and the four coefficient actions −C*r_k, using center
coefficients for degrees0/1/2 and the saved whole-path cubic coefficient for3.
Retain returned D0 and all four actions before subsequent calculations.
Report coordinate-radius-normalized upper bounds of these forcing terms,
D0, observed D−D0, and the complete old D. The degree contributions scale with
delta^k. The same-width triangle proxy and a *heuristic* doubled-width proxy
are retained for every row: D0+2*observed_variation plus degree multipliers
1/2/4/8. New derivative variation need not scale linearly, so this is explicitly
not a bound or proof on a wider domain. A proxy below1 is only a reason to try
one newly acquired widened-domain proof; it is never an inclusion verdict.

If a wider pilot is justified, it must keep the entire authenticated old endpoint
and native order/sign, reacquire all whole-domain C1/path coefficients and strict
root/primitive/anchor gates. No old derivative enclosure is extrapolated. This
unit merely informs that choice; no target pilot is launched by the script.

One exclusive saved-data run:
`/data2/neural-shrimp/.venv/bin/python tools/launch_eigenpair_width_diagnostic.py`.
Output `explore/eigenpair-locus-prototype/attempt-001-width-diagnosis`.
CPU/BLAS1nice10,45s/internal60+5external,1.5GiB monitoredRSS/4GiB virtual address,
64MiB worker raw (60 ordinary+4 final reserve),1MiB launcher/log128KiB. Read
and verify full parent/source inventories before reduction; preserve all returned
arrays, input/source pins, failure and process evidence; immutable manifests.
The prior large saved-array audits fit these bounds; this needs one470D matrix
product and four matrix-vector products, with no expensive native derivatives.

Synthetic controls use a fixed polynomial with exact degree scaling, a zero
forcing/identity case, strict unproved-parent rejection and source inventory
checks. One initial test mistakenly requested that an exact rounded *upper bound*
contain a separately rounded rational Arb ball; it failed as expected. The
correct test compares its exact dyadic value with the rational truth. No target
outcome or scientific threshold was involved in this test repair.

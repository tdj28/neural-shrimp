# S1R bootstrap association: fixed-state resource profile

TLDR: no new network behavior was measured. Timing the exact historical
single-cell update path supports attempting the unchanged bounded collection;
that collection is still unrun and requires separate root authorization.

On 8 October 2026 at09:16 UTC, root authorized one resource-only profile on
source `e0fccd73df025f81130d07dfbe60dfe170ef4840`, after independent review
`ff2ec8dc133d4fabf766ec5d776e19af1143f2f7` passed52 synthetic/regression controls.
The profile reset every call to saved S1R node0004 phase zero and a fixed e0
JVP direction. No trajectory, bootstrap, Newton correction, spectrum, bridge,
association outcome, GPU or paid call was generated.

Exactly2,000 gradient-with-loss update calls took0.0881711980 seconds; exactly128
JVP calls took0.00685544894 seconds. The declared calculation
`2*(80000*map_seconds/2000 + 2400*jvp_seconds/128) + 30` gives37.3107751779 seconds,
passing the unchanged100-second collection gate. Profile elapsed time was
0.185546659 seconds; the complete launcher took0.737090859 seconds and exited0.
Peak child RSS was77,904 KiB. CPU/BLAS1 and nice10 were enforced. The launch
resource snapshot had load1.44/1.99/2.45 and44 GiB available RAM; the GPU was idle
and unused. Root could concurrently run its separate one-CPU Arb profile;
foreign processes were untouched.

The projection is heuristic: the factor of two estimates unmeasured warm-loop,
classification and durability overhead, while the30-second allowance uses
existing local corrector timings. It does not guarantee that full bootstraps
or more difficult bridge corrections will finish. Any timeout or failure must
remain preserved under the unchanged scientific and resource limits.

The9 raw files total15,942 bytes under
`/data2/neural-shrimp-data/explore/s1r-bootstrap-association/attempt-001-profile`.
Manifest SHA-256:
`9a9fe3ff9db30bf7a37cae8848e51c5ff1b7cb49e5761072e622bf6440c88599`.
The exclusive launcher source, initial resource snapshot, complete output log,
exit receipt and manifest are under
`/data2/neural-shrimp-data/logs/s1r-bootstrap-association-attempt-001-profile`.
Every manifest entry was hash-verified, and all evidence files are read-only.
Launcher PID110258 and observed descendants110350/110351/110356 exited. There
was no retry or incomplete attempt. The compact full bindings are in
`s1r-bootstrap-profile-receipt.json`. Root handles separate-disk backup.

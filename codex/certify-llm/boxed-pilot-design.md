# Bounded 32-column whole-box derivative pilot

This is one engineering measurement at phase zero of the original
hash-pinned nine-state proposal. No target run was performed while writing
this driver and design. The parent must commit/private-push the source and
plan before launch through the reviewed resource-limited launcher.

The scope is fixed: 1024-bit Arb arithmetic; a full 896-coordinate nominal
box of radius `2^-800` around the exact binary64 phase-zero center; signed
coordinate directions `e_0,...,e_31`; and all 151,936 vocabulary entries.
The radius is many orders below the measured roughly `1e-14` prototype
residual and is **not proposed as an isolating orbit box**. There is no root
inclusion gate or assertion that this box contains or excludes the unknown
exact orbit. The other 864 derivative columns and seven orbit phases are
explicitly omitted.

## Provenance and input-domain construction

The strict committed plan schema is `codex-llm-box-derivative-pilot-v1`.
It fixes all parameters, counts, the CPU1/nice10/18-GiB/3600-second envelope,
and these original evidence bindings:

| Evidence | SHA-256 |
| --- | --- |
| pilot002 receipt | `565ecc98b37b592bb1da16a58ad4109e1dd2ab5f2eac71682f2af860531f33b9` |
| pilot002 map source | `3250d8b262e4a940b83de84005ad6f3155bc70635bfc623b0a000804cfc28c3e` |
| original proposal | `d8c57bff1d8a986c7c6a0f7d083caa77b4f6373c560d1f60ce0c57461caf668b` |
| signed-pilot001 receipt | `32b73defdf9672e2593ad33370a37fce2535ccb7c9ddc04769d2c16cd70add7d` |
| signed-pilot001 step | `6bf1bb80e2a662adeb5f87880091ae27a75ff299319f82ef0578d0d4d0f4d666` |

The driver authenticates every inventoried artifact and captured source against
its receipt and source manifest. The pinned pilot002 attempt remains a failed
later-box diagnostic; its complete prefix is admitted through the existing
reviewed cache loader. Its failed status is not erased or converted to success.
The comparison point must match the same proposal, parameters, phase and prefix.

`phase_domain` constructs a one-selected-phase artifact from the exact saved
center and nominal radius. The domain artifact is fresh output of the committed
constructor applied to the hash-pinned proposal/plan, not an incoming artifact
trusted through its own claimed hash. Its content hash is independently
recomputed and its endpoint witnesses verified. The selected domain's
`period=1` means one evaluated phase; provenance separately records source
orbit period eight and source phase zero.

The driver then actually reconstructs the receiver balls with `unpack`, keeps
any further outward magnitude-radius expansion, and runs exact endpoint
`containment_witnesses` again. Both the serialized artifact and the actual
receiver-ball records/hashes are saved before evaluation. The evaluator uses
those full balls, never their midpoints. All 32 exact basis directions are
saved and hash-bound as well.

## Computation and complete comparison

`BallQwenJet` supplies the inherited transformer architecture and signed
multi-direction arithmetic. The driver does not copy the transformer or
perform any new floating map/JVP comparison. The pinned model loader is used
only to authenticate and load immutable constants. The existing 1024-bit
prefix cache is receiver-bound through `cache_io`.

After its single forward call, the driver requires all 896 primal entries and
all 32-by-896 derivative entries to be finite and saves them in
`boxed-step.json` **before** comparison. The output binds the actual input and
direction hashes. If a comparison fails, the complete output and every failed
comparison row remain available.

For each of the first eight columns, every coordinate is compared with the
saved signed point enclosure using the original **exact dyadic interval
endpoints**. Overlap is required; containment is not. Both computations should
contain the same true point derivative if their evaluator contracts hold.
The 896 primal entries are compared similarly. All eight rows are checked;
the remaining 24 columns still receive complete finite-entry and maximum-radius
summaries. Outward Arb decoding is checked not to narrow data, but decoder
widening is not used to fabricate overlap of originally disjoint saved intervals.

A successful status is an engineering setup result and always retains
`certificate_passed=false`. It does not establish the missing columns, a full
monodromy, existence, primitive period, token labels or attraction. A failure
is retained with its phase, traceback, source-bound inputs and any completed
output; a launcher timeout remains an incomplete setup.

## Cost basis and validation

The measured eight-column point pilot took 261.808 seconds on CPU1. Multiplying
by four gives 1047.233 seconds as a planning reference, not a throughput or
completion guarantee: whole-box arithmetic, 32-column temporary memory and
setup overhead must be measured. The attempt retains the hard CPU1/18-GiB/
3600-second envelope, uses no GPU or paid resources, and records actual elapsed
forward time and peak RSS. No larger derivative run is implied by this pilot.

The implemented driver passed 43 CPU1 synthetic tests, including strict plan
pins/scope, complete inventory/source checks, nonfinite and missing-coordinate
rejection, all eight comparison rows, all 32 radius summaries, exact-endpoint
comparison despite decoder radius widening, actual-domain containment and
mocked successful/failed orchestration. The mock checks that full output is
saved before comparison and evaluates no target model. A separate read-only
admission against the actual pinned upstream attempts checked 22 prefix and
27 signed-pilot artifacts without importing Torch or loading a checkpoint.

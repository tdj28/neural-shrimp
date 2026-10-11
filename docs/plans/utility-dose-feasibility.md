# Four saved-fork gradient feasibility gate — preparation only

8 October 2026 UTC. After the independent saved-input audit42d901c0 passed,
root authorized preparation of this smallest next gate, **not execution**.
The utility comparison remains no-go: this gate can reject its fixed dose
before acquiring missing controls; passing cannot authorize a pulse or prove
common support, useful direction, geometry benefit or an outcome difference.

Inputs are exactly the four row4100 S2-28011 development forks, after8202
updates, and their saved recovery rates from manifest
2f962ced4b6f32aca588f1012985591acc670437ff944e8e15375acf4fd2e03f.
The source pins all four state hashes and checks every manifest member, state
bit string, recovery-rate bit string and control exponentiation. S2 has13 flat
parameters: W0[0:6], hidden bias[6:9], output weights[9:12], output bias[12:13].
Hidden and output groups are therefore0:9 and9:13, including both bias groups.
Use the original analytic full-batch gradient of half-MSE plus ridge0.001/2
on every parameter. The fixed training split alone is reconstructed using the
original teacher/training RNG streams; both exact plan data hashes must match.
Validation and test splits are neither generated nor accessed.

Evaluate exactly four gradients, one at each unchanged native state. Do not
call a map, Hessian, HVP, orbit corrector or loss/trajectory evaluation at new
states. Do not form theta plus/minus any predicted increment. Retain each
returned gradient before further checks, exact group packing, scaled gradient,
group/full norms, dose, fractions, predicted treatment-minus-sham increments
and predicted total update lengths. A failed feasibility result does not skip
other fixed forks; a resource/fatal failure preserves the evaluated subset and
all four bound inputs. No data-dependent target selection is allowed.

At each fork define s=P_recovery*g, R=0.05||s||2. Uniform reduction is0.05;
hidden/output reductions are R/||s_hidden||2 and R/||s_output||2. Both groups
must have finite nonzero norm, R must be finite positive, and every group
fraction must be<=0.25. Check predicted treatment-minus-sham dose using
absolute tolerance max(1e-14,1e-12*R). Any failure makes the fixed candidate
infeasible. Do not replace zero/nonfinite values or widen thresholds. Total
first-update step lengths remain unequal and are retained as competing generic
explanations. These are algebraic predictions at one shared state, not pulse
updates or pulse outcomes.

**Resources:** CPU/BLAS1,nice10, internal5s cooperative deadline, external10s
TERM+5s KILL, four gradient attempts maximum,1MiB combined raw. Allocate768KiB
to the unchanged Store (576KiB ordinary writes,192KiB finalization reserve) and
256KiB to launcher/source/log/receipt/manifest. The fixed log cap is128KiB.
Source/input/review hashes, exact process command/PIDs/exit and failures are
retained; output directories are exclusive and made read-only. No rerun,
resizing, new profile, GPU, paid compute, holdout or control acquisition.

The saved warmup measured63.64927253895439s for1,232,000 batched cell updates.
The conservative engineering screen is
`1+1000*4*63.64927253895439/1232000`, approximately1.207s, below5s. It gives a
thousandfold factor and1s I/O allowance; historical batching is not an exact
single-gradient timing guarantee. The strict input-bound formula is recomputed
before gradients. There is no fresh resource profile in this unit.

Prospective exclusive paths are
`/data2/neural-shrimp-data/explore/utility-dose-feasibility/attempt-001` and
sibling`attempt-001-launcher`. After source/tests/card commit and independent
plus root review, root must separately authorize this exact command, replacing
SOURCE_SHA and REVIEW_SHA with full identities:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/launch_utility_dose_feasibility.py --source-commit SOURCE_SHA --review-commit REVIEW_SHA --out /data2/neural-shrimp-data/explore/utility-dose-feasibility/attempt-001 --launcher /data2/neural-shrimp-data/explore/utility-dose-feasibility/attempt-001-launcher
```

The launcher is adapted from the lower-association launcher at3346fcaf, with
fixed10+5 limits and the inherited-pipe safeguard. A review SHA is provenance,
not a substitute for launch authorization. Synthetic controls exercise dose
algebra, exact cap/nextafter rejection, bias packing, nonfinite/zero blocks,
raw durability, all-four retention and deadline-after-gradient behavior.
A tiny synthetic gradient finite-difference control uses unrelated toy data;
no target gradient or outcome is evaluated during preparation.

### Pre-execution source review amendment

Independent review found that package initialization also imports controls.py
and models.py. Root added those files to the source inventory and a clean Git
checkout guard before any target gradient was evaluated. The four-fork design
and numerical gates are unchanged.

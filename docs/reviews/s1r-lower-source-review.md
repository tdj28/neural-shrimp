# Independent lower-bootstrap source review

8 October 2026. **Pass for the fixed exploratory unit**, reviewed producer
commit3346fcaf684fd74b24f58ce4da558358938edcee. This is a source/control review,
not target launch authorization or numerical outcome approval. No new target,
profile, trajectory or target correction ran during review; future collection
outcomes remain unopened. Prior unit artifacts were already available to the
team. The machine-readable companion binds the exact reviewed files.

The design asks whether two saved, newly reproduced procedures connect at
identical controls as the same full-state primitive two-cycle. It can support a
finite witnessed numerical connection or distinguish solutions at common
controls. It cannot establish historical-state identity, interval continuation,
uniqueness, global disconnection or behavioral utility. Qualified unstable
pairs remain valid branch states; distinct solutions are called coexisting
attractors only if both complete native spectra satisfy attraction.

The new adapter/runner/launcher preserve the previously reviewed continuation,
scientific Options, model and Store bytes. Input JSON, NPZ, pair, control and
rate hashes bind upper node0047 and lower node0001. The complete original
collection and timing-pilot manifests are verified before model work. Endpoint
requalifications retain both native phases. Macro endpoints1–74 use the exact
specified Python arithmetic; endpoint75 copies the lower control bytes.
The secant predecessor and carried phases survive macro boundaries, while
native records stay unchanged. Every accepted link has direct, midpoint,
half-endpoint and reverse witnesses. Matching checks parameter and rate bytes
before cyclic pair costs; intermediate distances and incomplete paths remain
unresolved. There is no reverse-direction retry, angle-based stitching,
neuron permutation identification, hidden bootstrap or new profile.

Nominal cost is604 phase calls;80 link attempts plus two endpoint
requalifications cap the total at644. A nominal0.002 step permits four halvings
to0.000125, hence five rejected attempts before stopping. CPU/BLAS1, nice10,
internal100s and external115+5s are explicit. The disclosed256MiB allocation is
255MiB Store plus1MiB launcher; the ordinary Store allowance is200,540,160bytes.
The fixed128KiB stdout cap stops and labels overflow. The timing/storage gate
uses pinned prior evidence and is only a heuristic: resource exhaustion at new
controls remains possible and is not evidence against an orbit.

Two material pre-outcome findings were corrected after the initial d41d2382
freeze, in d682619e and3346fcaf:

- Core node/edge objects change status before their final event is written.
  Durable pending/completion metadata now follows successfully persisted events,
  with unpersisted initial attempts separated. Failed accepted-edge writes do
  not advance the last witnessed pair. Both node and edge cases are covered.
- A direct child could exit while a descendant retained stdout, bypassing a
  guard that depended on the direct child still running. The fixed120-second
  supervisor now kills its owned group and closes the pipe even after leader
  exit. That supervisor failure prevents a successful CLI verdict.

The final focused command ran all producer lower-unit controls and18 new
independent controls:50 passed in3.88s, CPU/BLAS1 nice10, external30s+5s. The
independent controls exercise the full75-link plan using synthetic states and
stub qualification, phase swaps and secants, unstable pairs, exact controls
before matching, same/distinct/intermediate labels, caps, gap retention,
source/input failures, durable node/edge metadata, immutable failed artifacts,
and an actual owned synthetic inherited-pipe descendant. That descendant was
confirmed absent or terminated; no foreign process was signaled. This did not
run a target corrector. Producer tests additionally use small synthetic maps.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src S1R_LOWER_REVIEW_SOURCE_ROOT=/home/d7082791602/.codex/worktrees/tiny-lm-prediction-review/neural-shrimp timeout --signal=TERM --kill-after=5s 30s nice -n 10 /data2/neural-shrimp/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_s1r_lower_association.py /home/d7082791602/.codex/worktrees/scalar-ns-independent-review/neural-shrimp/tests/test_s1r_lower_independent_review.py
```

Command cwd was the clean frozen producer worktree. Review controls are owned
by the separate codex/review-s1r-lower-bootstrap branch. The explicit path
override selects that frozen source for driver tests; after integration it may
be omitted. The [prospective saved-data card](s1r-lower-outcome-audit-card.md)
predeclares subsequent audit scope. Root must authorize the one target launch
and, separately, any saved-data replay. No blockers remain in the fixed scope.

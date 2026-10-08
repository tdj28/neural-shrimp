# Independent S1R saved-map review

**TLDR.** The saved map agrees much more closely with one NS start family
than the other, but none of the retained comparisons identifies a verified
quasiperiodic state or a connected full-state branch. The negative/ambiguous
root-distance result is reproduced. Continue full-state branches before
making an ancestry claim.

**Verdict: no numerical or scientific blockers.** One nonblocking prose
correction is listed below. Reviewed producer tip
`3c66e666d42b6a15922b8f752e0c079780d2b772` and execution source
`bf7a811ca1948a036578de45bce90b37a75560f8`. The
[review receipt](s1r-rotation-map-receipt.json) binds the independent checker
and its retained output. This is an independent analysis of saved evidence,
not a new map simulation or an upgrade of the underlying evidence rung.

## Checks and findings

- The producer receipt at the exact reviewed tip, execution-source blobs,
  saved inputs and saved outputs all match their SHA-256 records. The producer
  replay also passes. Six synthetic controls pass (one harmless pytest-cache
  write warning in the restricted read-only test environment).
- A separate implementation, importing no producer analysis, recomputes all
  eight offset comparisons, retained counts, medians, maxima, exclusive
  exclusions and status counts. At offset .002, the two medians are
  `.0006450415024` (14/16 retained, start 1.10) and `.0100761296363`
  (12/25, start .95). **Every retained cell is status 4, unresolved.**
- Independent adjacency/jump checks recover 9 versus 13 geometric edges.
  Independent reduced-rational enumeration recovers provisional roots 1/8,
  2/15 and 1/23, with 1,433, 1,410 and zero matching candidates respectively.
  Independent `cdist` distances are 75.81678 and 123.63970 grid cells. A
  cross-product/end-point calculation reproduces both conditional segment
  envelopes; a 1,001-position segment sample stays inside each envelope.
- The NS producer implementation initializes each output slice from the same
  system seed and does not carry the preceding slice's orbit forward. Saved
  traces omit orbit states. Neither family membership nor geometric adjacency
  establishes branch continuity; selecting the closer angle would condition
  on the answer. The audit does not do that.
- Status/period/FTLE are from the earlier 20,000-step screen, while rotation
  follows later PCA and rotation sampling. The lack of post-rotation rechecks
  and complete historical Python-source binding is disclosed correctly.
- The figure was inspected: families are shown separately, NS points are not
  connected, and color limits cover the full unsigned range 0–.5. White masks
  combine several screen states, as disclosed in the companion prose.

Root matching remains a descriptive period/rotation filter. Its distance
search covers the whole map and assumes neither a beyond-boundary location
nor an identified branch. The envelope is conditional on a root lying on the
straight endpoint segment; it does not enclose the true curve or constitute
an interval proof. The producer's interpretation respects these limits.

## Minor documentation correction

`docs/program-c-2026-10-06.md:188` says the native backend is exploratory and
parenthetically cites `qualifying_run=false`. In
`src/neural_shrimp/gdmaps_cuda.py:446`, this flag is defined as
`capability not in QUALIFIED_CAPABILITIES`; false means an already-qualified
capability, here sm_89. The map analysis remains exploratory, but that status
does not follow from this flag.

Suggested replacement: “The map analysis remains exploratory; the run used
the already-qualified sm_89 capability (`qualifying_run=false`).” This is a
wording correction, with no effect on the recomputed diagnostics.

## Recheck

`tools/review_s1r_rotation_map.py` is the independent checker. From a worktree
containing the reviewed source, choose a new output path (existing outputs
are never overwritten):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data2/neural-shrimp/.venv/bin/python tools/review_s1r_rotation_map.py \
  --output /tmp/s1r-independent-review.json
```

The retained first review is under
`/data2/neural-shrimp-data/audits/s1r-rotation-independent/attempt-001/`.
No new trajectories, GPU use, reserved seeds, paid compute or paid review
were involved.

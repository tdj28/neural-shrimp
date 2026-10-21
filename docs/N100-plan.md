# N100: spatial coupling of 100 soft-token loops

Status, 21 October 2026: **proposed exploratory pilot; not started**. The
model-free graph and measurement helpers are unit tested. The model runner,
device qualification and throughput measurement remain to be implemented
and performed after the requested go-ahead. This is neither a frozen design
nor a preregistration. No N100 network outcome has been inspected.

The behavioral question is whether identical loops exchanging their outputs
converge to one state, sustain smaller groups, settle into joint cycles, or
remain irregular. Matching text alone will not establish state agreement.
The proposed cell is chaotic in the existing two-copy study; this pilot
cannot establish a shrimp-specific effect or a benefit in ordinary language
generation.

## Map and graph

Use Qwen2.5-0.5B-Instruct, the pinned revision and existing open prefix,
philosophy steering at layer 12, K=1, alpha 0.07484 and log10(T)=-0.1226.
This is article **cell 2**, index **1** in `docs/e2e/lyap-cells.json`.
All weights, normalization, rotary tables, states and mixing must use the
qualified float64 path. Each network is one batch of 100 rows. Mixing is
simultaneous, after the 100 independent map evaluations:

    x_i(t+1) = (1-s) F(x_i(t)) + s sum_j W_ij F(x_j(t)),  0 <= s <= 1.

Draw 100 positions uniformly in the unit square using NumPy PCG64 seed 100.
Save the realized coordinates, distances, affinities, row weights and hashes.
Let A_ij = exp(-distance(i,j)/ell), with A_ii=0. Compare all pairs with the
**undirected union** of k-nearest-neighbor edges for k=4 and k=8; an edge
exists when either endpoint selects the other. Resolve distance ties by node
index. Normalize W=D^-1 A with D_ii=sum_j A_ij. W is generally asymmetric,
but is similar to the symmetric matrix D^-1/2 A D^-1/2. Directed k-NN is
a different design and must not silently use the real-eigenvalue criterion.
Disconnected graphs, underflow and unresolved spectral gaps are recorded,
not repaired by inserting unreported edges.

The initial proposed length scales are 0.05, 0.15 and 0.5. Do the graph
calculation first, then choose a small pilot contrast using the predicted
intervals, disclosed before model trajectories: uncoupled s=0, weak s=0.03,
and one strength on either side of an admissible predicted threshold.
A dense uniform-affinity graph supplies a mixing control: at s=99/100,
every row receives the same arithmetic mean in exact arithmetic. The
control tests implementation; it is not a spatial-network result.

## What the spectral prediction says

Along a typical synchronized orbit with leading exponent lambda>0, a
Laplacian mode mu_k of I-W has conditional transverse exponent

    lambda_perp,k = lambda + log|1-s mu_k|,  k=2,...,100.

For a connected graph with this real spectrum, all these exponents are
negative on the open interval

    (1-exp(-lambda))/mu_2 < s < (1+exp(-lambda))/mu_max,

intersected with 0<=s<=1. The eigenratio inequality alone does not enforce
the allowed coupling range. A disconnected graph has additional transverse
zero modes and cannot stabilize this chaotic synchronized orbit. The helper
retains them. Predictions start from the existing cell-2 JVP estimate in
`docs/sync/sync-summary.json`, `cells["1"].lambda1_jvp`, not its distinct
older `lambda1_published` estimate. Record the source-file hash and use the
unrounded value. Re-estimate the exponent on the qualified device before
comparing thresholds quantitatively.

This is a linear, local prediction, conditional on the orbit used for lambda.
It does not establish global attraction or exact cluster synchronization in
an arbitrary spatial graph. Coupling can also select a different attractor.
No synchronization at one horizon does not prove its impossibility; observed
synchronization onto a periodic orbit does not contradict a prediction for
the chaotic orbit. Spatial groups and fronts remain hypotheses to measure.

The factorization above is the discrete output-mixing specialization of the
master-stability framework: L. M. Pecora and T. L. Carroll, “Master Stability
Functions for Synchronized Coupled Systems,” *Physical Review Letters* **80**,
2109–2112 (1998), DOI
[10.1103/PhysRevLett.80.2109](https://doi.org/10.1103/PhysRevLett.80.2109).
Publisher title, authors, volume, pages and DOI checked against the APS
BibTeX record on 21 October 2026. The derivation here follows directly by
differentiating the stated map.

## Starts, measurements and interpretation

Use two separately reported start protocols, shared across graph/strength
conditions: (a) small independent full-state offsets from one attractor
state, for local stability; (b) 100 saved, spaced points along an uncoupled
trajectory, for capture from dispersed starts. Specify and hash the exact
source trajectory and offsets in the run card. Initialize the latter from
one trajectory, not identical prefix states: deterministic identical starts
would manufacture synchrony. Use independent seeds for positions and offsets.
Do not continue one coupling outcome into the next unless explicitly adding
and labeling a continuation protocol.

The proposed full horizon is 1000 coupled transient steps followed by 3000
recorded steps. Record whether all pairs remain below relative distance
1e-8 throughout the final 500 steps, plus the 1e-6 sensitivity result.
For an unordered pair divide the full-state difference norm by the smaller
state norm: this enforces the article's directed criterion in both
orientations. Use direct differences, not squared-norm/Gram subtraction,
which loses precision at these distances.

Save per-step maximum/quantile state distances, all pair distances or a
lossless equivalent, every node's top-1 token from its own **pre-mixing**
softmax at x_i(t), the output-to-input time alignment, final-500 per-pair
maxima, and full states at a declared cadence plus a dense terminal window.
This readout is not a token decoded from the mixed embedding at t+1.
Report text agreement separately from state
agreement. Persistent close-pair connected components are descriptive:
check their all-pair diameters before labeling them synchronized groups,
because a chain of close pairs need not be close end to end. Preserve the
adjacency itself, component memberships and any failing diameters.

Phase-lag candidates require full-state recurrence across the joint network,
primitive-period checks and stability analysis before being called verified
cycles. Repeating tokens are insufficient. Irregular trajectories require
explicit full-network chaos diagnostics to be called chaotic; otherwise
report finite-horizon irregularity. Exact cluster invariance requires a
separate check of the incoming group weights, not just apparent color groups.

## Execution gates after go-ahead

1. Implement a bounded, restartable runner; test it using analytic maps,
   including simultaneous mixing and a forced-stop/resume comparison.
   Commit the code and run card before any qualification or pilot run.
2. Check current CPU, RAM, GPU processes and reservations. Keep at least
   12 GiB available RAM; use one model process and explicit thread limits.
   Wait for BF to release the GPU. Local use is authorized; no rented launch
   is authorized by this proposal.
3. Measure batch-100 peak memory and warm step time with synchronized CUDA
   timing. Extrapolate the full horizon and pilot condition count. Prefix
   construction also needs its own peak-memory measurement.
4. Confirm effective float64 RMSNorm/rotary patches, central-difference h^2
   convergence, identical-row wander, row permutation/batch effects and a
   single-device reference. Preserve failures. Do not lower precision to fit.
5. Run a small pilot before expanding the length/strength grid; log all
   adaptations and retain failures. No result-dependent deletion or relabeling.

Store new attempts under `../neural-shrimp-data/explore/network-coupling/`.
Each attempt needs an exclusive directory, exact command, source commit,
code/input/plan hashes, realized starts and W, versions, device, precision
checks, resource use, timings, chunks with hashes, explicit completion
status, raw trajectories and an append-only failure log. Compact receipts
and dated results belong in Git. An incomplete attempt is not a complete
100-node outcome. No paid review is requested or run.

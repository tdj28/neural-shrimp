# Directional ordering contribution to transformer response

11 October2026 UTC, exploratory amendment after the consumed frozen-response,
two-snapshot and causal-horizon outcomes. Those outcomes remain negative for
useful forecasting. The next question is mechanistic: how much of the frozen
response error lies in the order-antisymmetric part of the actual propagator?
This is a retrospective decomposition, not an online predictor or intervention.

Use exactly the same four float64 reference trajectories: original update600,
seeds28161/28162 and8/256window arms, both original initial directions. Fixed
horizons4,16. Original objective, eta, ridge, tied coordinates, native source,
training tokens and library identities must match. No new data, seed, rate,
trajectory, perturbation, gradient, CE or protected assessment evaluation.

Write J_k=I−eta H_L(theta_k) and P_h=J_(h−1)...J_0. Each J_k is symmetric in
these Euclidean parameter coordinates. Therefore reversing the product gives
R_h=J_0...J_(h−1)=P_h^T. For original unit direction u, define
A_h=(P_hu−R_hu)/2 and S_h=(P_hu+R_hu)/2−J_0^h u.
Then frozen error is exactly A_h+S_h. Acquire R_hu by h HVPs in reverse state
order. Reuse P_hu, J_0^h u and terminal training-loss gradients from the original
source-authenticated reference. State ordering is the intervention on the
linearized product; it is not an alternative nonlinear training trajectory.

Report every case/direction/horizon: norms of A,S,error; their squared-norm
identity ||error||²=||A||²+||S||²+2<A,S>; normalized cancellation and contributions;
projection onto the terminal training-CE gradient (original objective gradient
minus ridge*theta_h); and original perturbation two-amplitude/ordered-transport
qualification. Decomposition remains an algebraic diagnostic if a finite-
difference case fails, but cannot then explain a resolved measured response.
Check u^T A≈0 and the two-direction reciprocal identity u^T P_h v=(R_hu)^T v.
No scores are fitted and no accuracy threshold is changed after outcomes.
Reciprocal and skew identities use1e−8 normalized by the original vector
norm products. Vector/squared decomposition identities use1e−12 normalized
by original propagated/frozen magnitudes, retaining nearzero cancellation
without dividing by the vanishing error. These are floating-point controls,
not interval certificates.

For a fixed descriptive decision, material order contribution means
||A||/max(||P_hu−J_0^hu||,1e−12)>=1/2 at horizon16 in BOTH expanded-arm seeds'
gradient directions, conditional on inherited response qualification. Report
all other arms/directions regardless. This ratio can exceed1 through
cancellation and is not a variance fraction; retain the inner product.
A negative result says only that the reversal-even remainder dominates these
particular responses. That even part can also depend strongly on ordering;
reversal-invariant products need not be invariant under other permutations.
A positive result supports further mechanism work, not novelty of ordering,
chaos, a shrimp effect, a training benefit or population inference.

## Mathematical and literature positioning

Reversal preserves the full singular spectrum because R=P^T. Thus this test
cannot establish that reversing steps reduces maximum amplification. Its
antisymmetric part is skew and measures directional rotation, with state-
dependent symmetric Jacobians. This distinction is a required synthetic check.

Sweeney, Commutator Memory (arXiv2609.34348v1), studies order differences of
updates on different data sources and their output-space effects. Our proposed
quantity concerns the derivative product along one fixed-objective trajectory;
it is distinct from the gradient-field Lie bracket H_Bg_A−H_Ag_B.
https://arxiv.org/html/2609.34348v1
Pervez and Locatello, Controlling Transient Amplification Improves Long-horizon
Rollouts (arXiv2605.08856v2), analyze nonnormality and noncommuting Jacobians in
neural simulator rollouts. Their work precludes claiming generic Jacobian
ordering or transient amplification as a new mechanism here.
https://arxiv.org/html/2605.08856v2
Primary manuscripts inspected11October2026; this is a narrow search, not a
completed novelty review.

## Proposed bounded implementation

Target160HVP exactly:4cases×2directions×(4+16). No extra native operations.
Profile two HVPs per arm at the previously consumed seed28160/update50, using
full15776D native source and original batch sizes. Conservative forecast
2*2*sum_arm(40*max(HVP time))+20s mustfit180s. Use CPU/BLAS1nice10,
4GiBRSS/8GiBAS/128MiBraw; profile60s/80+5external, target180s/200+5external.
Each input state/vector and each return is retained before/after its call;
exact source, completed operation counts, full immutable inventories and owned
termination required. The native function is unchanged. Tests cover constant
commuting maps, noncommuting symmetric maps, product order/transpose identity,
singular-spectrum equality on dense controls, cancellation, gradient-to-CE
conversion and failure retention. Peer-reviewed committed source/card and a
passing full-size measured profile are required before any target execution.
This plan alone launches nothing.

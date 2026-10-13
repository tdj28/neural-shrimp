# Independent mathematical design review: exact-real Qwen period 8

13 October 2026 UTC. **Behavioral TLDR:** a successful certificate would prove that one explicitly defined continuous-feedback Qwen map has a locally attracting eight-state cycle. It would turn this selected numerical orbit into an exact-real existence, primitive-period and local-attraction theorem. It would not prove ordinary token-generation behavior, a useful language-model intervention, a whole shrimp, or a basin equal to the entire root-isolating box.

**Current status: design only.** No checkpoint was loaded, no target map was evaluated, and no target certificate is asserted here. This review reads the current Task 1 assignment, handoff, AGENTS.md, Skills.md, certificates/README.md, the certified-computation playbook and the independent Task 2 map source. Only this design-review file is owned by the reviewer; code, runs, receipts and Git operations belong to the coordinating agent. Task 5's frozen passage tests are separate.

## 1. Exact statement and map identity

The latest assignment supersedes the handoff's older target: the selected P8 stripe at log10(T)=-0.115, alpha=0.071674, using the current fixed-prefix one-token expected-embedding map. Let n=896, F:R^n -> R^n, G=F^8 and H=G-I. The desired certificate states that a specified full-dimensional box X contains exactly one zero x* of H, its minimal period is eight, and the orbit is locally asymptotically attracting. Uniqueness is phase-local: the eight different phases are not one vector and may occupy eight disjoint boxes.

Freeze an exact-map manifest before producing proof artifacts:

- Bind the checkpoint, configuration, philosophy vector, prefix token IDs, steering layer and tensor inventory by hashes. Stored finite BF16 weights promoted to binary64 are the same exact dyadic numbers; promotion adds no information. Store and parse constants as exact binary rational values, not rounded decimal summaries.
- Specify alpha and ell=log10(T) by their exact binary64 ratios, and define T=10^ell in real arithmetic. T is generally not a dyadic. Freezing Torch's rounded temperature as an exact constant would define a different map. Either interpretation can be studied, but the requested theorem calls for the real expression and must identify it unambiguously.
- Specify the RoPE base, rational frequency exponents, exact integer positions and rotary convention. Enclose base^(-2j/head_dim), then real sin/cos, rather than importing float64 inverse frequencies or trig tables as exact values. The desired real-frequency table can differ slightly from the numerical proposal's table.
- Use exact rational 1/head_dim for means and the declared real inverse square root for attention scaling. With head_dim=64, the attention factor is exactly 1/8. Treat RMS epsilon according to its declared exact input interpretation.
- Recompute every prefix KV entry with rigorous arithmetic at the same alpha; a cached ball encloses a particular exact constant. A float64 prefix cache treated as exact dyadics certifies a changed map. Steering applies after decoder block 12 at every position, including the prefix; later blocks' cached keys and values depend on it. Prefix derivatives with respect to the current state are zero, but prefix rounding uncertainty is not zero.
- Retain the entire 151,936-token vocabulary, tied embedding/unembedding, causal mask, grouped-query replication, Q/K/V biases, two RMSNorms per block, SwiGLU and final norm. Fixed context and position are reused on every F call; the context never grows. Masked attention entries are omitted exactly, not evaluated as numerical exp(-infinity) through a generic finite-ball routine.

A high-precision proposal may need refinement because the exact-real map is not the floating implementation used to find the orbit. That is an expected distinction, not permission to freeze rounded intermediate tensors as new constants without disclosure. A floating Newton residual is a proposal-quality diagnostic; the rigorous enclosure of H(c), not that residual, enters the proof.

## 2. Explicit sufficient existence and uniqueness gates

Let c be an exact dyadic center, D=diag(r_i) with each r_i>0 an exact rational, B=D[-1,1]^n and X=c+B. Use the weighted infinity norm ||v||_D=||D^(-1)v||_infinity. Let R be any fixed exact dyadic n-by-n matrix, normally proposed as an approximate inverse of DH(c). Define the auxiliary Newton map P(x)=x-R H(x).

The checker must produce rigorous enclosures:

1. E contains -R H(c), using a rigorous primal evaluation at c;
2. C contains (I-R DH(z))v for every z in X and v in B;
3. all operations are finite and the real map is continuously differentiable on an open neighborhood of X and the phase enclosures used to evaluate it.

From C form q=max_i mag(C_i)/r_i with an outward upper bound. Require both q<1 and the componentwise strict inclusion E+C subset interior(B). Equivalently, a conservative sufficient inclusion gate is mag(E_i)+mag(C_i)<r_i for every coordinate. Record exact positive margins; overlapping interval comparisons fail closed.

For x,y in X, the segment integral of DP bounds ||P(x)-P(y)||_D by q||x-y||_D. The centered integral also places P(X)-c inside E+C. Thus P is a contraction of the closed box into itself and has a unique fixed point there. Moreover ||I-R DH(z)||_D<1 makes R DH(z) nonsingular by the Neumann argument, so square R is nonsingular. A fixed point of P is consequently a zero of H, and every zero of H in X is that fixed point. This establishes uniqueness throughout the stated full box, not merely in the smaller returned image.

**An approximate inverse does not have to be certified as an inverse first.** Its exact dyadic entries, and the final q/inclusion checks, are sufficient. Conversely a numerically small inverse residual or a converged GMRES solve is not the q check. If an iterative procedure replaces multiplication by R, its actual finite algorithm must represent a fixed linear map with a rigorously bounded application error; a tolerance-based black-box solve cannot silently be treated as exact R.

These deliberately strong sufficient gates follow the existing repository's Banach/Krawczyk policy. For context, Rump's primary review gives the centered mean-value enclosure and Krawczyk inclusion argument in Theorems 13.1 and 13.3; the derivation above explicitly retains the additional norm gate used here. [Author-hosted corrected PDF](https://www.tuhh.de/ti3/rump/intlab/ActaNumerica2010.pdf), [publisher record](https://doi.org/10.1017/S096249291000005X), accessed 13 October 2026. Citation metadata: Siegfried M. Rump, *Verification methods: Rigorous results using floating-point arithmetic*, Acta Numerica 19 (2010), 287–449.

## 3. What an interval JVP can and cannot prove

A **box tangent** is a vector of intervals representing all independent directions v in B. A rigorously implemented forward derivative pass enclosing DF(z)v jointly for all z in an input box and all v in the tangent box can supply a complete operator-norm bound without materializing every Jacobian column. This is stronger than one ordinary JVP at a center or along an eigenvector. A single numerical directional derivative says nothing sufficient about the norm on all directions.

For eight steps, propagate valid primal boxes X_j enclosing F^j(X), and propagate the tangent box through DF(X_0),...,DF(X_7) in chronological order. The output then encloses DG(z)v for every initial z in X and v in B. Intermediate primal and tangent boxes may be overestimates; replacing them by unchecked narrow boxes around a numerical orbit is invalid. The single-box-JVP idea therefore has a valid mathematical route, but **does not automatically yield a sufficiently small bound**, a useful preconditioner or practical runtime.

The Krawczyk cancellation hazard is serious. H'=DG-I gives

    I - R H' = (I+R) - R DG.

Use this algebraic expression before interval evaluation. Naively evaluating v-R(DG v-v) duplicates v and loses exact cancellations. With DG=0 and R=-I, the true expression is zero but an uncorrelated interval evaluation of v-v has twice the input width. Choosing R=-I directly reduces P to G and the derivative enclosure to the interval JVP of G; if its norm is below one and G(X) is strictly included, this is the cheapest legitimate special case.

For general R a valid enclosure is C=(I+R)B-R*JVP_G(X,B). It may be very wide because the two terms use the same direction but interval arithmetic forgets that dependency. A genuinely centered alternative chooses a point matrix M0 and proves a remainder enclosure Delta containing (DG(z)-M0)v for z in X, v in B. Then

    C = (I-R(M0-I))B - R Delta

is valid. Obtaining Delta requires interval derivative entries, a rigorous derivative Lipschitz/Hessian bound, or an affine/Taylor derivative representation that retains the shared tangent symbols. Simply taking JVP_G(X,B)-M0 B discards the correlation again; it is valid but does not provide the hoped-for small centered remainder.

There is an equally important distinction between **primal centering** and **derivative centering**. For a smooth step and valid input box Y around c_j,

    F(Y) subset F(c_j) + DF(Y)(Y-c_j).

The JVP here must enclose derivatives across Y, and F(c_j) must itself be rigorous. If the next proposal center is c_(j+1), retain the defect F(c_j)-c_(j+1). Never replace it by zero or use a point derivative in place of DF(Y). A centered primal form can reduce state wrapping while still leaving severe tangent wrapping in the norm proof.

## 4. Primitive period and attraction are separate gates

After the root gate succeeds, rigorously propagate its isolating box through F. For each proper divisor d in {1,2,4}, require a coordinate or exact linear functional that separates F^d(X) from X by a strictly positive bound. Equivalently, prove 0 is excluded from a component of F^d(X)-X. Then the zero of F^8-I cannot have a smaller period. Display a witness and margin for every divisor; a decoded token sequence is not a primitive-period proof.

Let M=DG(x*) at the proved root. Construct an interval matrix enclosure A containing M, using valid phase boxes. For a declared positive integer k, interval matrix powers can enclose M^k. Require an outward norm bound ||D_s^(-1) A^[k] D_s||_infinity<1 with positive exact diagonal D_s, or another explicitly justified induced norm. This implies rho(M)<1 and hence local asymptotic attraction for the C1 real map. Float eigenvalues choose candidates or powers; they never pass this gate. Repeated-squaring enclosures are valid, although dependency loss can enlarge them.

A matrix-free alternative is also sufficient: start with D_s[-1,1]^n and apply the rigorous derivative-action enclosure for one monodromy k times, always using the same proved orbit phase boxes. The final box contains M^k v for every initial v in that unit box, so its scaled radius below one is a power-norm witness. This may wrap more than a matrix enclosure and need not succeed even when a dense matrix power does.

Exact two-dimensional counterexamples show why the alternatives differ. For M=[[.01,100],[0,.01]], rho(M)=.01 but the infinity norms of M and M^2 exceed one; M^3 has norm .030001<1. For N=[[1,1],[-1,-1]], N^2=0, yet interval-vector propagation of [-1,1]^2 through N twice gives [-4,4]^2. The latter failure is lost dependency, not instability, and raising arithmetic precision does not remove it.

Newton-map contraction is not attraction of F. Nor does a monodromy power norm at x* put **every point of X** in its attraction basin: it yields some local attracting neighborhood. To claim X itself is a basin, additionally prove suitable dynamical invariance and a uniform contraction of an iterate on its full admissible domain. Keep this stronger statement out of the initial certificate unless its extra gates are implemented.

## 5. Non-normality, coordinates and practical alternatives

Small spectral radius does not control a matrix norm or finite-time amplification. Neither a small next multiplier nor approximate rank one justifies discarding 895 state dimensions. Every full-state direction remains in the root and stability checks.

Diagonal weights cost little and may help. A full coordinate change x=c+S y can help more, but the dyadic matrix S must be proved invertible and all actions of its mathematical inverse enclosed. A numerical matrix Q with QS approximately I is not S^(-1); treating it as exact silently changes the map. One may verify nonsingularity through ||I-QS||<1 and then rigorously enclose S^(-1), or perform verified linear solves. Also distinguish a coordinate parallelepiped from an axis-aligned x-box: uniqueness in the former does not imply uniqueness in its larger rectangular hull. To promise the original full box, check that exact region or give the additional inclusion and uniqueness argument.

If direct eight-step propagation wraps, the next honest option is multiple shooting: eight state boxes, with equations F(x_j)-x_(j+1)=0 cyclically. A structured block preconditioner avoids materializing an arbitrary 7168-by-7168 inverse. The same strict full-product-box contraction/inclusion theorem applies. The proof must actually enclose the complete block derivative action and preconditioner error; numerically stitching eight separately approximate states is insufficient. Multiple shooting reduces primal composition wrapping but does not guarantee the block norm gate or cheap arbitrary-precision matrix work.

For implementation order, try the simplest rigorous primal/box-JVP kernel first, including the R=-I special case. Benchmark one representative operation, layer and fixed prefix under an approved bounded run card before projecting a full certificate cost. If derivative box propagation loses essential cancellation, prefer a disclosed centered derivative/Taylor or block-shooting extension to endlessly increasing precision. A full interval Jacobian assembled by rigorous column JVPs is a more expensive but straightforward fallback; dense *numerical* Jacobians can propose R but cannot replace interval bounds.

The full-vocabulary operation is an engineering bottleneck, not a mathematical exception. Stream exact embedding rows in blocks or use rigorously rounded matrix kernels; do not drop tokens with small floating probabilities. An optional vocabulary-tail approximation needs a separately proved remainder bound for both F and its derivative before it can participate in a certificate.

## 6. Primitive arithmetic and derivative obligations

The primal and derivative kernels must share the same explicitly defined map, with every nonlinear operation enclosed. Useful analytic derivative rules include:

- RMSNorm: q=mean(x_i^2)+epsilon>0, d(x/sqrt(q))=v/sqrt(q)-x*mean(x_i v_i)/q^(3/2), followed by exact learned weights. Establish q's positive lower bound and retain all dependency conservatively.
- Softmax at fixed T: p_i=exp(s_i-c)/sum_j exp(s_j-c), dp_i=p_i(ds_i-sum_j p_j ds_j). Any fixed exact finite shift c is valid. Choosing a shift from midpoint logits is an algebraic numerical aid; do not differentiate the selection or regard it as a dropped term. Require a strictly positive denominator enclosure. Scores already include division by the exact-real T.
- SiLU: x/(1+exp(-x)) and its exact derivative, or an algebraically equivalent enclosed formula. Branching for stability must preserve the same real function and cover interval straddles.
- Affine layers, residual additions, attention products and grouped heads: include every bias/steering term in primal values, all product-rule terms in tangents and every rounding error in dot products. Steering is constant for state derivatives but remains in the primal graph.

Arb enclosures of a fixed prefix or fixed rotary constant may be reused. Never replace a nonzero ball radius by zero merely because the value is conceptually constant. Floating BLAS plus an empirical discrepancy estimate is not a rigorous dot-product kernel. Input conversion, subnormals, overflow and NaN handling require fail-closed tests; arbitrary precision and exact-input parsing should be visible in receipts. Centering and increasing precision address different sources of width.

## 7. Fail-closed artifact contract and review gates

Each exclusive attempt should retain exact input definitions, source hashes/commit, precision, proposed centers/radii/preconditioner, all primal/phase boxes, residual and derivative-action enclosures, coordinate transforms, Krawczyk margins, q, all three divisor witnesses and the attraction power/norm witness. Record failed stages and measured radii with no success fallback. Mathematical gates use exact endpoints or outward bounds, never rounded printed summaries.

Before any rung-3 claim, require:

1. Exact-map and asset manifest complete; no unsupported architecture option accepted silently.
2. Rigorous primal and derivative code reviewed, with analytic low-dimensional controls including a wrong-return/unstable case and a non-normal cancellation case. Numerical interior samples are useful bug tests, not proofs of enclosure.
3. All strict root, full-region uniqueness, primitive-period and attraction gates pass in the checker; each gate has retained reconstructible witnesses.
4. One offline command re-executes proof arithmetic from hash-bound inputs and fails on altered inputs, source, witnesses or verdicts. A later independent review must say whether it shares the map kernel and Arb rather than implying software independence.
5. Receipt/prose scope says exact-real map, one selected period-8 orbit, local attraction; no interval-wide parameter, sampling, behavior, global window or whole-shrimp claim.

A failed inclusion, a broad ball, an unresolved divisor or an excessive runtime is an inconclusive setup result. Preserve it, amend the next attempt and distinguish progress on the proof infrastructure from an achieved certificate.

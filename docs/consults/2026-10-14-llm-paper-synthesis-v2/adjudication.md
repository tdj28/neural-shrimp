# Consult 7 adjudication (14 October 2026, about 08:50 UTC)

Review: gpt-6-astra Pro, xhigh, research-strategy, brief at 5d530c45; 67,931 tokens. Raw bundle in
../neural-shrimp-data/consults/2026-10-14-llm-paper-synthesis-v2/execute-001 (review.md copied here unchanged).
Advisory only; every point checked against our evidence.

## Accepted and applied (or to apply in the paper pass)

1. Scale claim, present form: "size-associated suppression in the sampled operating regimes, while steering can expose
   instability at 70B". Accepted; the paper states the measured scope (one prompt, one grid) until the prompt-blocked
   test below is done.
2. Prompt-blocked size comparison (its recommendation 1). Accepted in a cheaper, frozen form: Qwen2.5-7B against -14B
   (and, if affordable, -72B), eight held-out prefixes plus two development prefixes, temperature grids that reach the
   entropy transition, two twin offsets, long horizons, paired per prefix. Our batched pipeline costs about [amount redacted]-20, not
   its [amount redacted]-100 estimate (it assumed per-trajectory throughput). Frozen in docs/prefix-scale-test.md before any run.
3. Scalar sufficiency of the return map (its recommendation 2, first part): held-out prediction by the scalar map
   against two- and three-delay predictors on the saved H3 sequences; local, run now. The transverse-perturbation part
   needs new trajectories and is deferred.
4. Alignment intervention on the period-8 orbit (its recommendation 3): orthogonal rotations inserted between step
   Jacobians keep each step's singular values and change only their alignment; the spectral radius of the rotated
   product tests whether alignment is what stabilizes the cycle. Local, on the saved dense Jacobians; run now, with
   the coordinate-change negative control.
5. Wording: (a) the symmetric-PSD/energy argument needs the property on a convex domain containing the orbits; its
   failure is necessary, not sufficient, for cycles and chaos; (b) high-temperature contraction needs derivatives
   uniformly bounded on the feedback domain (the convex hull of the embeddings, compact) - already so stated; (c)
   exponent agreement does not establish conjugacy - applied after Codex's review; (d) "feature dictionaries do not
   locate attractor boundaries" and "acting on the loop beats steering" are hypotheses: the implications paragraph
   says "suggest" and will say "in our sampled self-dialogue setting"; (e) the 70B controls show direction-dependent
   chaos, so the sentence that SAE directions act like random ones is qualified by scale.

## Not adopted (with reasons)

- Recommendation 4 (feedback memory and task behaviour): out of scope for this paper; noted as future work.
- Title "Soft-token feedback in language models: bifurcations, hidden chaos, and stability": presented to the human
  with the alternatives; the title is the human's decision.
- Entropy-targeted temperature selection by bisection: replaced by dense fixed grids per prefix that span the
  transition (cheaper in our batched pipeline and immune to non-monotone entropy), with the same paired endpoint.

# Cold Stop source comparison

Root read the bounded public source capture at Soft-Thinking revision
`b97407c60d5e971da4a4fe76f405d6e236d0ddfa` and the local pi_reason.py at
`32c2f84fe59ab2dbc943886ab0b5bd517c55f0da`. This is a source comparison;
no model or target output was executed.

The upstream sampler computes entropy over the temperature-scaled full
vocabulary before top-k/top-p/min-p filtering. Its logarithm clamps
probabilities at 1e-12. The scheduler increments a consecutive counter when
entropy is strictly below the threshold, resets it otherwise, and triggers
when the counter reaches the length threshold. In soft-thinking mode it
replaces the current token/feedback weights by the end-of-thinking token and
leaves soft-thinking mode. The documented experiment settings are .01 nats
and 256 steps; the command default threshold is off unless enabled.

pi_reason.py preserves that entropy formula, strict comparison, consecutive
counter, threshold and soft thought-to-answer switch. It has an explicit
thinking-budget check before Cold Stop, and its local counter is active only
while the row is thinking. It applies the same thought-to-answer switch to
the plain comparison arm.

This is a port of the thinking-stop rule, not a byte-for-byte reproduction of
the upstream runtime. The upstream scheduler's non-soft branch sets an abort
flag when its counter reaches the threshold, including after soft thinking
has ended; the local runner does not apply Cold Stop during answer generation.
Upstream sampling/filtering/runtime and the local PM2 arm definitions also
differ. Therefore a claim about using the published entropy/count rule in
thinking is supported by source, while a claim of identical full-generation
behavior in both arms would be too strong.

Saved row flags can check the observed stopping counts and token agreement,
but cannot reconstruct unsaved per-step entropy or counter histories. Keep
that distinction in the actual audit. The original request's early-row exact
reproduction claim also needs an explicit saved-field equality predicate and
denominator. No interpretation of its target counts is made here.

Read captured schedule_batch.py around update_topk_info, sampler.py entropy
and following soft/non-soft sampling path, readme.md documented settings, and
run_sglang_softthinking.py argument/default wiring. Original failed guessed
URL/README retrievals should remain in the capture receipt. Hashes and exact
URLs are bound by the coordinator's upstream manifest.

# Independent saved-arithmetic review of the wider mean-value branch

**Passed, 9 October 2026.** The source-reviewed mean-value adapter extends the same original primitive18 branch across the exact hidden-rate interval

`[1254967922419383/17592186044416,1255036641896119/17592186044416]`,

of width `1/256`, with output rate fixed at `1848862009810323/2251799813685248`. This is four times the previous width. The review establishes existence, local uniqueness in the two shooting domains, primitive period and their connection to the original center. **Attraction remains certified only on the earlier width1/1024 interval.** All sufficient attraction tests on the larger domain are inconclusive; this is not evidence of instability.

From exact source/input/raw bindings, I independently rebuilt the234D native shooting matrices and recomputed Krawczyk arithmetic at384bits. The two contractions are below0.281770 and0.281817; their maximum coordinate images are below1.717868e−6 and1.717324e−6, strictly inside the fixed radius2^-18=3.814697265625e−6. The proper divisors1,2,3,6,9 are excluded. All234 original-anchor and shared-endpoint coordinates pass, including the separately recomputed endpoint enclosure. Declared endpoints match the independently derived exact interval.

Execution `00e7d79ba9ae10934b338564dc98ba9dd10dab5a` used only saved arrays: no model/gradient/Hessian evaluation, numerical proposal, trajectory, stability search or old-tube replay. It completed once in5.038seconds internally and5.376seconds externally, at465808KiB peak RSS. All21 files644,389bytes are immutable, their inventories match, and owned PIDs are absent. Outer manifest: `60e5cf634f1d08f253f56df2af3499a8a499b454f1d913c506215f76f2d5aab3`. A synthetic234D passing/failing coordinate-bound control passed in2.03seconds before the run.

The full source/provenance/statement is in the adjacent JSON. The derivative enclosures remain bound to the independently reviewed mean-value adapter and unchanged unit-map Hessian/Arb implementation. This saved-only review does not reacquire them. Earlier adapter authorship is disclosed there. No fold/flip boundary connection, whole-window geometry, basin, utility or new individual-orbit count is inferred. The original producer creation-time pending-review labels and earlier proof records remain unchanged.

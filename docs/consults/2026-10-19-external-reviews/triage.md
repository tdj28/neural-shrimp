# External reviews of the arXiv v1 article, 19 October 2026: triage

Two reviews of the Chaos article (`/data2/chaos-journal-neural-shrimp`), both requested by the human:

- **GPT-Pro deep research review.** The human ran it and pasted its text into the chat on 19 October 2026, about
  01:00 UTC. It reviewed the 18 October source and checked the algebra against the primary literature; it did not
  rerun anything. Verdict: "GO for arXiv after a bounded correction pass". The central mathematics is sound; the
  problems it found are "primarily overgeneralizations and details lost during shortening".
- **Fable review** (`fable-review.md` in this folder; reviewed commit c8f7fe0). Verdict: "GO, after three mechanical
  fixes. The science holds."

Codex's Task 36 (a deep review of the mathematics) was still running when these were applied. Its findings will be
triaged against this list.

All items below were applied to the article in commit c7528a8 (article repository) unless marked otherwise. The same
corrections go to the full report and the blog.

| # | Source | Finding | Decision |
|---|---|---|---|
| 1 | GPT | The abstract calls the readout factor a covariance for all families; untied models give a cross-covariance | Applied: "(a cross-covariance for untied embeddings)" in the abstract and introduction |
| 2 | GPT | The covariance form is a known identity: the Hessian of the log-partition function (Wainwright and Jordan), the softmax Jacobian (Gao and Pavel), the Jacobian of the modern Hopfield update (Ramsauer et al., eqs. 63 and 77-78); cite it beside eq. (3) | Applied; both new citations verified against Crossref (10.1561/2200000001) and arXiv (1704.00805) |
| 3 | GPT | Kingman applies to typical orbits of an invariant measure, not to an arbitrary finite record | Applied |
| 4 | GPT | Restore the fold coefficient definitions with the remainder; generic fold needs a, b nonzero | Applied: Appendix A, with a pointer from Sec. V |
| 5 | GPT | Flips: a numerical -1 crossing alone does not classify a flip | Applied: Appendix A says flips are located and confirmed by the doubled orbits, not classified |
| 6 | GPT | Cusp conclusions need their diagnostics in v1 | Applied: Appendix A gives c, the bordered solve and the unfolding condition; Table IV lists the four located cusps (FM5c/FM5d records) |
| 7 | GPT, Fable | "Exact transverse law u' = (1-2c)DF u" is the linearization; the threshold c > c* holds only for 0 <= c <= 1/2 | Applied |
| 8 | GPT, Fable | "Two-dimensional maps have no superstable curves" is false in general; it needs a nonvanishing Jacobian determinant | Applied in the Fig. 16 caption |
| 9 | GPT | The sign change of D = mu1 mu2 does not establish noninvertibility (the ranking can change; near D = 0 the second multiplier is as small as the rest) | Applied: descriptive wording |
| 10 | GPT | "is a plane section of a swallowtail" overstates the evidence | Applied: "consistent with" |
| 11 | GPT | "which is why one-dimensional maps ... describe these orbits" goes beyond the evidence | Applied: supports the description, no global reduction established |
| 12 | GPT | Band chaos: 2000 steps cannot establish a nongenerating partition or "never repeats" | Applied: finite-horizon wording ("over the measured horizon", "does not repeat over the 2000 steps") |
| 13 | GPT | "Noise can fake a positive exponent but does not hide one" is a noise-model heuristic | Applied: stated under independent noise, as a heuristic |
| 14 | GPT | Float64 arithmetic: the article said none changed, but the 7B changed about 5 % | Confirmed against the full report (line ~1221); applied: none in the 0.5B, 5 % in the 7B |
| 15 | GPT | Hidden text: "all 65 with period seven" conflicts with the full report (59 of 65 multiples of seven) | Confirmed; my rewrite of 18-19 October introduced the error; applied: 59 of 65, most often 7 or 28 |
| 16 | GPT | Larger displacements "cross basin boundaries" is unclassified | Applied: "presumably", the attractors reached were not classified |
| 17 | GPT | State the exact real-number map of the certificates (how T follows from log10 T) | Applied: T = exp(ln 10 · log10 T) for the exact binary64 log10 T |
| 18 | GPT | Basin radii need their separate neighborhood certificate | Already stated ("A further certificate gives explicit, if tiny, basin neighborhoods"); no change |
| 19 | Fable | Data availability URL returns 404 | Not a text change: the human makes the repository public before the announcement |
| 20 | Fable | Typo "the maps' fixed start" | Applied |
| 21 | Fable | British "centre" twice | Applied |
| 22 | Fable | Abstract "within one step in 42 of 48": the tests count iterates of the return map | Applied |
| 23 | Fable | Shortfall "0.4 to 5.7": the largest is 5.75 | Applied: "0.4 to 5.75" |
| 24 | Fable | The last Feigenbaum shrink factor (5.26) is not robust to the flip's location uncertainty | Applied: stated in the text |
| 25 | Fable | T* values are lower estimates, so the decades are "at least" | Applied |
| 26 | Fable | "fifteen released models" is the size sweep, not all models in the article | Applied in the lead paragraph and introduction |
| 27 | Fable | Article README stale | Applied |
| 28 | Fable | Six uncited bibliography entries | Removed |
| 29 | Fable | arXiv form: use the article's abstract; categories nlin.CD primary | Abstract: yes. Categories: cs.CL primary stays (the human's endorsement is in cs.CL), with nlin.CD and cs.LG cross-lists |
| 30 | Fable (v2) | Run the 10^5-step two-offset check on the Fig. 15 cell | Deferred to v2 |

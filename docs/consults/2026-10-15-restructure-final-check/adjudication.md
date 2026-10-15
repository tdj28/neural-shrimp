# Consult 10 adjudication (15 October 2026)

One research-strategy request: gpt-6-astra, Pro mode, xhigh effort. It used the brief `brief.md` and the context
`context-abstract-contributions.md` at commit 00ba9b95, completed with 71,366 tokens, and its conservative cost is in
the private ledger. The review (`review.md`) is copied unchanged from the client's output directory. The advice is
advisory; each item was checked against the paper and the data.

## Construct-validity boundaries (all four checked; three led to changes)

1. **Answer disagreement is not accuracy loss.** Agreed; the paper already keeps them apart. The arithmetic test
   reports answer changes under layout against a forced greedy baseline, and accuracy separately. No change.
2. **Low-precision divergence is not evidence of chaos.** Agreed, and already applied today. The introduction no
   longer attributes the bfloat16 answer changes to chaos ("In bfloat16, where nearly every loop diverges..."), and
   the consequences section says they do not require the float64 map's chaos.
3. **A simple critical multiplier does not exclude codimension-two points (cusps).** Correct; our sentence "so no
   codimension-two point lies on them" was wrong. It now reads "no second multiplier reaches the unit circle along
   them; this does not exclude cusps, which need only a nonlinear degeneracy". The cusp checks are running (FM5c).
4. **Comparable leading multipliers do not establish an effectively two-dimensional nonlinear map.** Agreed; the
   paper already says so. Two clarifications follow:
   - The 41 negative-product cells are disjoint from the 91 complex-pair cells by construction, since a complex
     pair has a positive product. They now read "41 other cells".
   - For the hostile body, the text now says the tracked real leading multiplier changes sign along two arcs, not
     that "the leading multiplier crosses zero".

## Shrimp naming

- **"No universal replacement for a doubly superstable centre in an invertible map."** Our wording already avoids
  naming one. It describes the operational skeleton: fold and flip boundaries, stability regions, and real-to-complex
  transitions.
- **"A genuine swallowtail degeneracy requires more than two cusps in projection."** Agreed. The paper now says "the
  shape of a section through a swallowtail" and "we do not claim a swallowtail point itself, which would need a third
  parameter".
- **"At a projected crossing, first ask whether the two fold points represent different cycles."** Done today,
  locally on the CPU from the saved continuation states (scratch script `crossing_orbits.py`).
  - The nearest points on the two arms of the fold curve were taken: indices 13 and 39 of curve-a1-up-fold-minus,
    about 5e-5 in parameters from the crossing. Each was iterated five times in float64.
  - Both orbits close to 1e-11 and carry the same token loop, ". I . . 你".
  - Their states differ by 3.9 % of the state norm (RMS over the five points, best phase alignment), 8 to 26 times
    the state change between neighbouring curve points.
  - So the crossing joins folds of two different period-5 orbits: two separate saddle-node events, not a
    degeneracy of one orbit. This is now in the main text and in Appendix D.
- **Façanha-Oldeman-Glass terminology.** We could not check their definitions at the source: the publisher blocks
  automated access and the indexes omit the abstract. "Endoskeleton" is therefore removed from the main text and
  from the hostile-body appendix. The reference stays where it is cited as general literature on shrimps.
- **Hénon (1976)** was verified at Crossref (Commun. Math. Phys. 50(1), 69-77, doi:10.1007/BF01608556) and is now
  cited where the paper mentions the Hénon family, with Gallas (1993).

## Ranked directions

1. **Causal reset experiment** (recommended). This asks whether accumulated feedback state, rather than final-step
   sensitivity, makes bfloat16 soft-thinking answers depend on the batch layout. It is the right next question for the
   practical claim and needs no change of what the paper now says. Its cost is 6-20 H100 hours or local GPU time, plus
   new code: full state checkpoint and reset, and a common decoder. **Not started.** It goes beyond the human's request
   of today ("do both and get a pro consult"), so it is proposed to the human with a costed plan.
2. **Two-coordinate closure test of the period-5 window.** Deferred. 3-8 H100 hours for a pilot.
3. **Orbit-sheet census around the swallowtail-shaped junction.** The cheapest pilot, the orbit identity at the
   projected crossing, was done (above). A census of orbit counts at 12-16 frozen offsets is deferred. The cusp
   location and checks (FM5c) are running.

The 48-hour plan and two-week roadmap are recorded for the human's decision.

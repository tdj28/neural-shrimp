# Adjudication: external review of the working draft (11 October 2026)

Source: a peer review of `paper/gd-route` (57-page draft, export e1b2bbb5),
produced with an AI deep-research tool from the PDF alone and supplied by the
human with the explicit caveat that it is not endorsed and may be wrong. Copy:
`../neural-shrimp-data/reviews/2026-10-11-deep-research-peer-review.pdf`
(sha256 prefix bd5df775a21bb4ab). Its recommendation: major revision, with
strong encouragement to preserve the rigorous core. Each point below was
checked against the repository or primary sources before being accepted.

## Accepted (act on in the rewrite)

1. **Rebuild the paper around a theorem hierarchy.** Theorems: certified
   cycles; connected period-18 branch; 1:1 organizer; complex-unit branch and
   flip junction. Add a dependency graph that separates proved connections
   from open ones. Move the development history to an appendix or the log.
   This matches our own assessment that the draft reads like a certificate
   ledger.
2. **Bibliography error in [27] (confirmed at Crossref and fixed, fe415053).**
   The DOI belonged to an unrelated paper. Codex's ledger had recorded it from
   arXiv metadata.
3. **Missing references (verified and added to the bibliography).**
   - Kuehn and Queirolo (2025), computer validation of neural-network
     dynamics;
   - Johansson (2017), Arb;
   - Krawczyk (1969).

   Other metadata items (arXiv identifiers and versions for 2026 preprints)
   still need checking one by one at the source; the review's suggestions are
   not adopted blind.
4. **"Census" means existence, not prevalence.** Use "screen-detected
   embedded-window census", and state the exact discovery protocol whenever
   counts are summarized.
5. **Keep the failed prespecified ancestry test prominent.** Its outcome
   stands. The window-organization study is a separate, newer analysis. It
   does not upgrade that failure; it answers a different question.
6. **Narrow the headline scope.** The claims concern explicit deterministic
   full-batch tanh networks with 13 or 72 parameters, and exact-real
   interpretations of stored constants.
7. **Rotation phase from Floquet eigenspaces, not only PCA.** Report rotation
   measured in several coordinates and require agreement before calling it
   robust. This applies to the window-organization rotation maps too.
8. **Define "independent".** Use a taxonomy: separate numerical
   implementation, separate derivatives, separate proof reduction, separate
   interval library, separate authorship. Disclose that the "independent"
   audits were produced by AI agents of the same kind, and which parts had
   human source review.
9. **Separate or remove the fixed-weight RNN (Clark) material.** Shrink the
   CIFAR and transformer development histories to a short
   negative-transfer section, unless new results change that, such as the
   dense reduced CIFAR Newton now running.
10. **Weight decay and layerwise rates are mechanistic questions.** Both
    deserve systematic experiments: a ridge sweep from zero (S1 versus S1R),
    and equal, fixed-ratio and free two-rate comparisons.
11. **Minibatch noise around a certified orbit** is a cleaner bridge to
    practice than larger nonperiodic models: escape times, phase diffusion,
    transverse contraction.
12. **Robustness across problem instances.** Use a reduced, prespecified
    protocol over fresh teacher datasets, seeds, widths, activations and
    ridge values.
13. **Reproducibility bundle.** Exact inputs, certificate boxes, a minimal
    verifier that does not rerun discovery, one-command replay, and versions.
    For the headline anchors, add a check through a genuinely separate
    derivative path.

## Already partly answered by work after the draft

- **"Can distant windows be connected to the local resonances?"** The
  window-organization study (`docs/window-organization.md`, float64 numerics,
  not certified) answers this for the period-18 window:
  - not to the attracting circle's 2/9 roots, which form a closed banana;
  - but through non-attracting orbits to a 2/9 resonance of a coexisting
    saddle 2-cycle.
- **"An independent derivative implementation."** The fast float64
  continuation (`src/neural_shrimp/fastcont.py`: numpy, HVP-based, not Arb
  jets) independently reproduced the certified 1:1 point and both cusps. That
  is numerical replication, not a proof.
- **"Do layer-specific rates create the rotation?"** Not entirely. Complex
  two-step multipliers and the Neimark-Sacker crossing also appear with a
  single learning rate: in the UCI-digits network (7 October), and the
  subset-A CIFAR 2-cycle has a complex pair 0.99604 +- 0.00942i at eta 0.0083
  (Krylov estimate). The two-rate plane is a convenient unfolding, not the
  origin of rotation. A matched ablation should still be reported.
- **"Momentum and optimizer state."** The E1 pilot
  (`docs/explore-momentum-2026-10-11.md`) shows the route surviving heavy-ball
  momentum on one ray.

## Not adopted, or adopted only in part

- **Dropping the larger-model program.** We keep it as exploration. The paper
  will report it as transfer evidence only if it produces a full-state
  periodic object.
- **The statistical criticism of the consequence analysis** is fair but
  concerns an exploratory section. The fix is wording and a hierarchical
  design for any future confirmatory test, not new statistics on the
  existing cells.
- **Title suggestions.** Deferred until the restructure outline is agreed
  with the human.

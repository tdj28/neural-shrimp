# Task 3 literature audit: cascades, neural shrimps and Lyapunov maps

Access date: 12 October 2026. Scope: primary-source checks of period doubling and measured Feigenbaum ratios in neural systems, with a separation between optimizer dynamics, recurrent biological state, learned scientific surrogates, and frozen language-model feedback. This is a bounded search, not a proof of absence. Candidate BibTeX is in `cascades.bib`; no manuscript or shared bibliography was edited.

## Most consequential finding

**Liu et al., arXiv:2609.38814v1 (30 September 2026), is close prior art that the current related-work section misses.** The preprint reports seven numerically verified flips through period 128 in a small autoregressive transformer trained from scratch on logistic-map trajectories, with final finite-order ratio 4.6687. Appendix B.3 solves cycle closure and multiplier −1 jointly using automatic differentiation and checks two-sided stability by continuation. Appendix C.1/Table 2 uses a float64 **scalar one-state transition map**, with a state token and a parameter token. Appendix C.1.1 separately reports precision limitations at the highest orders. This is an inspected preprint, not an independently replicated result.

Its task is recovering known scientific dynamics outside the training parameter range; it is not analysis of an existing language model pretrained on language. Thus neither a transformer cascade nor AD-based flip verification is by itself new. Our potentially distinctive application is the frozen pretrained LM's endogenous feedback map, with the full embedding-state orbit and derivative-based local mechanism. [Primary record](https://arxiv.org/abs/2609.38814v1); [full text, §§3.2, B.3, C.1](https://arxiv.org/html/2609.38814v1).

## Evidence and scope matrix

| Work | Evolving object | Located evidence | Relationship to our claim |
| --- | --- | --- | --- |
| Liu et al. 2026, arXiv:2609.38814v1 | Scalar state predicted by a trained scientific transformer | Seven flip roots, finite-order ratio 4.6687, AD closure/multiplier solve and continuation | Direct predecessor for transformer cascades and the numerical verification method; different from a pretrained LM |
| Battaglia, Brunel & Hansel 2007 | Coupled biological neural-population activity | Numerically estimated Feigenbaum constants and positive largest Lyapunov exponent | Rules out general neural-network scaling novelty |
| Papasavvas et al. 2015 | Three-population neural mass dynamics | Ratios computed from first four flips, compared across inhibition types | Quantitative neural cascade precedent; not language inference |
| Kim et al. 2021 | Autonomous RNN trained to reproduce external dynamics | Publisher abstract reports extrapolated bifurcation structure and routes to chaos | Learned-system extrapolation precedent; no measured universal ratio confirmed in inspected material |
| Kalra, He & Barkeshli 2025 | Gradient descent on trainable weights | UV model and sharpness period-doubling route; synthetic-data network examples | Training dynamics, distinct from fixed-weight feedback |
| Marichal & Piñeiro 2015 | Two-neuron discrete recurrent state | Normal forms, center manifolds and numerical continuation | Generic neural continuation/normal-form novelty is unavailable |
| Gomez, Stoop & Stoop 2014 | Rulkov neuron state | Shrimp-shaped periodic regions and largest-Lyapunov phase diagram | Neural shrimp morphology predates current work |
| Stenzinger, Luz Oliveira & Tragtenberg 2025 | Hindmarsh–Rose neuron state | Shrimp hubs and phase/Lyapunov diagrams | Direct neuron-model shrimp precedent already cited |
| Caixeta et al. 2025 | Discrete membrane-potential model | Publisher abstract reports period organization inside shrimp regions | Related period-adding/staircase precedent; mechanism needs full-text comparison |
| Clark 2026, arXiv:2610.12426v1 | Random recurrent rate-network activity as N→∞ | Theory of full Lyapunov spectrum | Different limit and observable from a finite LM orbit's monodromy |

## Verified evidence locations and metadata

### Battaglia, Brunel and Hansel (2007): measured neural scaling

*Temporal Decorrelation of Collective Oscillations in Neural Networks with Local Inhibition and Long-Range Excitation*, **Physical Review Letters 99(23), 238106**; published 7 December 2007; DOI **10.1103/PhysRevLett.99.238106**. Title, authors, volume, issue and article number were checked against APS and its Crossref deposit. Preprint **0711.2001v1**, 13 November 2007, is marked accepted for PRL.

The paragraph preceding Fig. 2 reports δ≈4.66(9)±0.002 and spatial scaling α=2.502(8)±0.0001, in the authors' notation. The period-doubling calculation concerns firing-rate dynamics; larger conductance-based networks are also studied, but do not transfer the exact quoted ratio to every model. Do not replace these unusual parenthesized digits with an invented uncertainty convention. [APS](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.99.238106); [primary full text, paragraph before Fig. 2](https://arxiv.org/html/0711.2001v1).

### Papasavvas et al. (2015): ratios depend on the neural mechanism

Christoforos A. Papasavvas, Yujiang Wang, Andrew J. Trevelyan and Marcus Kaiser, *Gain control through divisive inhibition prevents abrupt transition to chaos in a neural mass model*, **Physical Review E 92(3), 032723**, 23 September 2015, DOI **10.1103/PhysRevE.92.032723**. APS and Crossref metadata agree.

Methods §II.B, equations (7)–(8), define a measured ratio from the first four period-doubling points. Figures 2–3 compare cascades and ratio estimates against 4.6692. The model with divisive inhibition remains near that value; the subtractive-inhibition variant can differ substantially. This is a quantified finite-order test, not just use of “Feigenbaum” as a name for a plot. [APS record](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.92.032723); [author-hosted published PDF](https://www.dynamic-connectome.org/pubs/Papasavvas2015PRE.pdf); [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC4789501/).

### Kim et al. (2021): learning global dynamics

Jason Z. Kim, Zhixin Lu, Erfan Nozari, George J. Pappas and Danielle S. Bassett, *Teaching recurrent neural networks to infer global temporal structure from local examples*, **Nature Machine Intelligence 3(4), 316–323**, 19 April 2021, DOI **10.1038/s42256-021-00321-2**. Publisher and Crossref metadata checked.

The publisher abstract includes bifurcation-structure and period-doubling extrapolation. The older preprint **2005.01186v1**, 3 May 2020, has a different title, *Teaching Recurrent Neural Networks to Modify Chaotic Memories by Example*, and narrower inspected content. Do not silently attribute later journal details to that preprint. No exact Feigenbaum estimate was confirmed here; classify as a general cascade/bifurcation precedent. [Version of record](https://www.nature.com/articles/s42256-021-00321-2); [older preprint](https://arxiv.org/abs/2005.01186v1).

### Kalra, He and Barkeshli (2025): training cascade

Dayal Singh Kalra, Tianyu He and Maissam Barkeshli, *Universal Sharpness Dynamics in Neural Network Training: Fixed Point Analysis, Edge of Stability, and Route to Chaos*, **ICLR 2025**. The ICLR proceedings verifies title, full authors and venue. No journal volume/pages or DOI were displayed, so none are invented. arXiv **2311.02076v2**, 13 February 2025 (v1: 3 November 2023).

Sections 5–6 analyze gradient descent in a two-layer linear UV model and report a period-doubling route, with numerical checks in other architectures. The cubic-map discussion is training-state/reduced-observable dynamics, not pretrained LM inference. A full-text search for “Feigenbaum” found no match; no measured universal-ratio sequence was located. The latter is a search outcome, not a claim that no such calculation exists elsewhere. [ICLR record](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8c8cd1b78cdae751265c88efc136e5bd-Abstract-Conference.html); [full text](https://arxiv.org/html/2311.02076v2).

### Marichal and Piñeiro (2015): neural normal forms and continuation

R. L. Marichal and J. D. Piñeiro, *Analysis of multiple quasi-periodic orbits in recurrent neural networks*, **Neurocomputing 162, 85–95**, 25 August 2015, DOI **10.1016/j.neucom.2015.04.001**. Title, author initials, venue, volume and pages checked at Elsevier/Crossref.

The publisher's abstract and section excerpts describe a discrete two-neuron recurrent map, a 1:2 resonance, center-manifold/normal-form calculations and continuation of bifurcation diagrams (§§4–5, Fig. 3). It is enough to reject “first normal forms/continued boundaries of a neural inference system,” but it does not study a language model or provide the LM-specific intermittency result. Full paywalled text was not inspected. [Publisher record and excerpts](https://www.sciencedirect.com/science/article/abs/pii/S0925231215003914).

### Gomez, Stoop and Stoop (2014): neural shrimp Lyapunov maps

Florian Gomez, Ralph L. Stoop and Ruedi Stoop, *Universal dynamical properties preclude standard clustering in a large class of biochemical data*, **Bioinformatics 30(17), 2486–2493**, DOI **10.1093/bioinformatics/btu332**. Online 10 May 2014; metadata checked at Oxford/Crossref.

§4, “Neural systems,” applies the Rulkov model; Fig. 6 shows nested shrimp-shaped regions in a largest-Lyapunov parameter map, and Fig. 7 discusses period organization. This is substantive neural morphology prior art, not a paper about animal-shrimp prediction. The inspected evidence is numerical phase-space organization; it is not an interval certificate for a pretrained LM. [Full publisher text, §4/Figs. 6–7](https://academic.oup.com/bioinformatics/article/30/17/2486/2748227).

### Stenzinger et al. (2025): existing citation independently rechecked

Rafael V. Stenzinger, **Vinícius Luz Oliveira**, M. H. R. Tragtenberg, *Shrimp hubs in the Hindmarsh–Rose model*, **Chaos 35(2), 023148**, DOI **10.1063/5.0239268**. Publisher collection and Crossref deposit verify the existing bibliography's authors. An OpenAlex result substitutes a different second author; **do not import that secondary metadata error**. DOI metadata date is February 2025. The abstract reports a spiral hub analyzed with interspike-interval, bifurcation and Lyapunov diagrams. The full publisher article was inaccessible in this audit, so its detailed methods are not independently re-adjudicated here. [Publisher collection entry](https://pubs.aip.org/cha/collection/425591/From-Sand-to-Shrimps-In-Honor-of-Professor-Jason-A); [article](https://pubs.aip.org/aip/cha/article/35/2/023148/3336239/Shrimp-hubs-in-the-Hindmarsh-Rose-model).

### Caixeta et al. (2025): additional staircase precedent

Luiz F. B. Caixeta, Matheus H. P. Gonçalves, M. H. R. Tragtenberg and Mauricio Girardi-Schappo, *Devil’s staircase inside shrimp-shaped regions reveals periodicity of plateau spikes and bursts*, **Chaos 35(3), 033122**, DOI **10.1063/5.0250342**. Publisher page: 10 March 2025; Crossref gives March issue date. Title, authors and article number agree.

The publisher abstract reports a discrete membrane-potential model with multiple periodic attractors inside shrimp-shaped regions and a staircase of periodicities. Relevant as an adjacent mechanism/organization citation; it is not evidence that the LM's symbolic period addition is new in neural systems generally. Only abstract/metadata inspected, not enough to equate its mechanism to our derivative-calibrated fold passage. [Publisher record](https://pubs.aip.org/aip/cha/article/35/3/033122/3339179/Devil-s-staircase-inside-shrimp-shaped-regions).

### Clark (2026): relevant random-RNN spectrum, different contribution

David G. Clark, *Lyapunov spectrum of random neural networks*, **arXiv:2610.12426v1**, submitted 8 October 2026. No journal successor is listed in the inspected arXiv record. The manuscript derives a large-N Lyapunov-spectrum calculation for randomly coupled recurrent neural dynamics using a response-function identity and a cavity construction. This concerns long-time tangent growth in random networks, not cycle multipliers or bifurcation continuation for pretrained language-model feedback. A full-text “period” search found no match; no cascade ratio or shrimp claim was located. It remains useful context for distinguishing a complete Lyapunov spectrum from a periodic-orbit monodromy spectrum. [Record](https://arxiv.org/abs/2610.12426v1); [full manuscript](https://arxiv.org/html/2610.12426v1).

## Implications for current cascade wording

These are our methodological judgments, not claims borrowed from one source:

- Report the actual resolved flip ladder, primitive periods, crossing multiplier, transverse spectral gap and uncertainty in each flip parameter. A finite ladder is a finite segment of a cascade; it alone does not prove an infinite cascade or asymptotic universality.
- At least three located flips are needed for one interval ratio. The inspected draft's table has only two completed flip parameters and placeholders for the rest: a numerical ratio cannot yet be claimed from that table.
- Use the physical steering coordinate consistently. Finite-order ratios depend on nonlinear parameter reparameterization; smooth reparameterization preserves the limiting ratio only under the corresponding local regularity assumptions.
- The useful contribution is a quantitatively resolved phenomenon in a pretrained LM, with the trained map itself differentiated and checked. Neither high parameter count nor known universal scaling alone supplies that scientific connection.
- Broad “first Feigenbaum scaling in neural networks,” “first transformer period doubling,” “first neural normal-form analysis,” and “first neural shrimps” are contradicted by the inspected sources. A qualified absence claim restricted to frozen pretrained language-model feedback can survive this branch of the search; it still needs the parallel LLM-specific audit.

## Search coverage and exclusions

Queries (general web, then primary record/full text; 12 October 2026):

1. `neural network Feigenbaum constant period doubling cascade measured ratios recurrent neural network`
2. `"neural network" "Feigenbaum" "period"`
3. `"recurrent neural" "period-doubling" chaos`
4. `"transformer" "Feigenbaum"`
5. `"neural networks" "shrimps"`
6. `"period doubling" "neural networks" Feigenbaum ratio`
7. `"Universal Sharpness Dynamics" "Feigenbaum"` and venue verification with `ICLR 2025`
8. `"shrimp" "neuron" Gallas Bioinformatics 2014`
9. Exact titles and DOI queries for all included papers; direct arXiv reads for 2610.12426, 2609.38814, 2311.02076, 0711.2001 and 2005.01186.
10. Publisher-citation follow-up through the AIP “From Sand to Shrimps” collection, locating Caixeta et al.

Excluded from evidence: search snippets on biological shrimp production, Feigenbaum tests named for Edward Feigenbaum, secondary promotional essays, thesis snippets without resolved metadata, and uncited/unverified claims about trained-model performance. “Feigenbaum trees” in Hopfield/memristive-network titles are not counted as measured ratios without inspecting the actual calculation. No code/artifact reproduction of external studies was performed. No paid source access, external messages or experiments.


## Follow-up audit: current cascade wording and finished raw rows

Inspected 12 October 2026, own checkout `97197f7fc34e49d8139a6a445b29d0da9d0febae` and available `origin/main` `5842ec95fec0f5edfe3fc363496609d9bf85f0df`; relevant draft wording agrees. No Git mutation or model replay was performed.

- `paper/llm/main.tex:45–47`: abstract says “complete period-doubling cascade,” lists P3, P6, P12, P24 and a chaotic core, but leaves the interval ratios as an in-progress placeholder.
- `paper/llm/main.tex:84–85`: contribution again says “complete” and that ratios are compared with the universal constant.
- `paper/llm/main.tex:389–390`: table fills only P3→6 and P6→12. Lines 391–393 leave P12→24, P24→48 and P48→96 as placeholders. Caption lines 396–401 describes every flip as real −1, then marks CPU/GPU extensions in progress. Restrict “every” to completed, checked rows.

**The table is stale: a third flip is now saved.** Read-only snapshot of `/data2/neural-shrimp-data/explore/llm-soft-loop/feigenbaum-crescent-t-0.14798/cascade.json`, mtime `2026-10-12T23:33:14.325374+00:00`, SHA256 `dde633ea4d14b55922589133369743ba635ecc6f235bb5457b24f7d5be53fcac`. Each completed row's final Newton residual, Ritz residual, crossing multiplier, argument and next modulus was read explicitly:

| Parent period | Flip parameter α | Critical multiplier | Argument / turns | Next modulus | Final Newton residual | Ritz residual |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 0.06548077694994509 | −1.0000000000031124 | 0.5 | 0.0881642778331305 | 1.22594e−13 | 1.58953e−15 |
| 6 | 0.06590554063247459 | −0.9999999999934999 | 0.5 | 0.002320172574785677 | 8.30559e−14 | 1.78106e−15 |
| 12 | 0.06605490417849594 | −1.000000000004121 | 0.5 | 2.3728814163924406e−6 | 3.16543e−13 | 3.42425e−15 |

Imaginary parts are zero for all three. The saved intervals are 0.0004247636825294998 and 0.0001493635460213527; their ratio **2.84382430548868** was independently recomputed from the three saved parameters. This is the first finite-order ratio, appreciably different from 4.6692; it neither establishes nor rules out eventual asymptotic universality. The claim that *no ratio is available from the two-row draft table* remains correct, but is superseded as a description of the raw results.

**Assessment:** the critique of “complete cascade” remains current. The supported update is three numerically resolved flip points and one finite-order ratio, plus separately justified map evidence of later periods/chaos. A finite ladder and a chaos diagnostic do not establish an infinite accumulation sequence. These are saved numerical results, not independently replayed or interval-certified results. The Ritz residuals are numerical diagnostics and do not certify a bound on the entire omitted spectrum.

## Follow-up source: Dinc et al. (2026), neural ghost passage

Fatih Dinc, Ege Cirakman, Bariscan Kurtkaya, Mert Yuksekgonul, Yiqi Jiang, Mark J. Schnitzer and Hidenori Tanaka, *Ghost Mechanism: An Analytical Model of Abrupt Learning in Recurrent Networks*, **Physical Review X 16(2), 021058**, published **23 June 2026**, DOI **10.1103/mjcl-lb4x**. APS and Crossref metadata agree. Use this seven-author published record; arXiv:2501.02378v1 had an earlier title and six authors.

The published PDF §II A–C (pp. 3–4, Eqs. 10–13) reduces continuous-time high-dimensional state flow near a ghost to a scalar quadratic normal form and obtains an inverse-square-root bottleneck timescale. Sections III–IV study low- and full-rank RNNs on delayed-response working-memory tasks. Learning selects state dynamics; the bottleneck is present during neural state evolution, so calling this “only training” would misrepresent the overlap. It is a close precedent for quantitative neural saddle-node/ghost passage. Our possible distinction is the discrete, frozen pretrained-LM return map and its derivative-calibrated passage prediction; neural ghost passage itself is established prior art.

[APS primary record](https://journals.aps.org/prx/abstract/10.1103/mjcl-lb4x); [published PDF](https://journals.aps.org/prx/pdf/10.1103/mjcl-lb4x); [arXiv record](https://arxiv.org/abs/2501.02378). Follow-up search: exact title supplied by the parent, APS DOI, publisher full PDF, Crossref DOI metadata; accessed 12 October 2026.

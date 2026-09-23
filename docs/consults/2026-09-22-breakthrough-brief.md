# Research-director consultation: escape a local explanatory plateau

22 September 2026. Private exploratory research; compact synthesized evidence, not a raw-data audit.

The researcher explicitly asks: "please do a pro with GPT-6-astra pro to let her know that our progress is stalled and we need to find a breakthrough to understand what the dynamic of these shrimp mean for the big picture of neural network behavior". Earlier they emphasized that the dynamics seem consequential beyond the visual shapes, and authorized autonomous progress using a strong local server. We need a serious scientific direction, not reassurance that the intuition must be right. Please challenge both our construct and our project management. We have made real numerical and behavioral progress but may be stuck optimizing a very narrow assay.

## Decision requested

Which ONE next experiment or theoretical reduction is most likely to teach a generalizable principle about neural-network behavior, at low local cost? Should we complete the pending matched-cycle reference pilot, redirect to parameter robustness/noisy memory, test nonlinear computation, or move to a learned recurrent network? Rank at most five directions, explicitly cancel or defer low-value work, and give a decisive 48-hour plan and two-week roadmap. We want a falsifiable breakthrough path, not a guaranteed positive result, a larger atlas by default, or another standalone round of numerical polishing.

Distinguish three objects: neural-state dynamics at fixed weights; parameter-space organization of attractors; and optimizer/training or iterative-inference dynamics. A shrimp in one does not establish the same mechanism in another. Explain precisely what can and cannot connect this toy recurrent ODE to large trained recurrent/reservoir systems, optimizers, or iterative reasoning. Label conjectures and any remembered literature pointers requiring verification; no fabricated citations.

## Actual system and established dynamics

Our model is a deterministic autonomous three-state recurrent tanh ODE:

`dx/dt = -x + W tanh(x)` with `W=[[1.5,2.9,alpha],[-1.5,1.18,0],[gamma,-22,0.47]]`.

Alpha and gamma are two connection weights. We reproduced the published nine-maximum shrimp-shaped periodic-window map; individual full-state primitive period-nine and period-eighteen orbits have been independently shot and checked for stability. We have NOT established complete multidirectional boundary topology, proven global basin geometry, or a shrimp-specific computational advantage. A stable orbit has a neutral timing direction; transverse multipliers quantify local forgetting of state perturbations, not tolerance to weight changes.

Two selected nine-maximum settings recur in the later studies:

| Quantity | Core | Historical label "failed-near-edge" (now qualified cycle) |
|---|---:|---:|
| alpha | 1.064 | 1.0661333333333334 |
| gamma | 10.260722222222222 | 10.297166666666667 |
| primitive period T | 37.6363753791 | 37.3314995900 |
| dominant transverse multiplier rho | 0.0650600998 | 0.8560332320 |

The old label preserves an earlier mistaken exterior interpretation; it does not mean current references failed. Distant comparison settings had positive finite-time exponents and no short period under the tested checks. The core is far more repeatable after small state perturbations. Neither finding alone measures task utility.

Four historical gamma continuations saved 196 overlapping node records, including unstable segments, repeated seeds and failed correctors. These are not independent cycles/networks. A prior geometry-informed four-preview control policy improved over one shared-feature baseline but lost to a simple signed-axis control; its pooled unique utility is unestablished. An older strict cross-platform replay remains unresolved and is not retroactively repaired by later studies.

## Prior Pro advice, our adjudication, and what happened

One Astra Pro consultation on 17 September recommended transverse memory under interference as the first behavioral question, followed by contraction/observability controls. It warned that generic weak attraction can explain persistence, a poor observer can confuse unreadability with erasure, and a wider map should not be the default next move. Other proposed directions were early readout reversal and genuine nonlinear mixing. We accepted the two-write question and strong linear/one-mode controls, clarified that local state-component removal is not an exact nonlinear isochron reset, and embedded direct numerical checks. We did not treat Pro as scientific validation.

That program produced qualified direct trajectories and a completed nonlinear-readout analysis, but considerable time went into new reference adapters, integration consistency, probability enclosures, hashes and preserved failures. Those checks uncovered real errors; they also risk becoming the objective. Please assess whether we now have enough numerical fidelity to change the scientific question instead of repeatedly refining the same tiny effect.

## Completed memory evidence (EXP-019 and EXP-021)

At known initial phase phi, a deterministic unit phase-null direction d(phi) is defined from the globally normalized phase gradient. Two signed writes of amplitude 1e-4 occur one period apart. Readout windows contain 32 samples of all three tanh activations (96 coordinates), ending at 2T, 4T or 8T. The second sign is balanced and unknown to the decoder. Initial phase, schedule and clean baseline are known. Analytic iid Gaussian measurement noise has SD 1e-4 per observation coordinate; the dynamics have no process noise. The original acquisition includes identity, timing-retaining and transverse-retaining local state interventions; EXP-021's nonlinear recovery comparison concerns the identity condition.

Observers are fixed from independently acquired unforced reference responses before accessing finite-pulse class means: A for first write and B for second write. The full-mixture decoder is Bayes-optimal for its sealed reference Gaussian mixture, not necessarily for the finite nonlinear trajectories. There is no classifier fitted to finite class means. All reported means use the complete reused 24-phase cohort, not new held-out phases.

| Setting/window | Original linear first-bit accuracy | Full-mixture readout | One-mode mixture readout |
|---|---:|---:|---:|
| Core 2T | 51.5386% | 51.6200% | 50.6856% |
| Core 4T | 50.0073% | 50.0073% | approximately same |
| Core 8T | approximately 50% | approximately 50% | approximately same |
| Near edge 2T | 62.6449% | 62.8904% | 59.0364% |
| Near edge 4T | 55.0547% | 58.9815% | 58.9815% |
| Near edge 8T | 55.0816% | 56.9753% | 56.9753% |

The near-edge 4T all-phase nonlinear gain is 3.9268 percentage points, with numerical enclosure [3.9253,3.9327] points; this is NOT a population confidence interval. Five of 24 phases have gain intervals wholly above zero. One previously identified phase improves from 50.3095% to 99.6905%; this selected example is not a new success cohort. Input timing matters strongly. At late windows the full and one-mode observers score equally to about 2e-14 ON THE SAME FULL NEURAL OBSERVATIONS. This establishes sufficiency of a simpler late decoder, not that actual dynamics are exactly scalar. The early difference motivates transient transport/write-read orientation but does not identify its cause.

Changing only the observer recovers an earlier bit: near-chance linear accuracy can reflect interfering amplitudes rather than erasure. This does not establish joint two-bit capacity, unknown-phase memory, useful performance under process noise, learned-network transfer, or geometry causality. The ODE may provide largely linear fading memory while the nonlinear decoder supplies the decision complexity. Matching full A/B Gram geometry and Gaussian noise fixes the reference mixture prediction mathematically; adding a geometry label to predict that identical quantity is not a meaningful incremental estimand. Conversely, conditioning on contraction/observability can remove mediators of any total geometry association.

All 144 planned setting/phase/window cells passed the final numerical gates. The prior incomplete probability attempt remains incomplete. Saved numerical replay and separate physical/aggregation audits passed. These are local integrity checks with overlapping model/author lineage, not independent scientific replication. The complete decoder evaluation took about 150 seconds on four CPU workers; the prior Pro's 10–100 CPU-hour estimates were very conservative for our actual pipeline.

## Latest stall and pending design

EXP-022 asked for six reference cycles: the two anchors, plus one cycle on each of two existing gamma paths near rho=.25 and .50 (caliper .025). It performed only historical selection. Both core candidate bins were missing, while the near-edge path supplied both; 4/6 slots were available. No new neural integrations ran. Core eligible adjacent nodes bracket both targets: rho .20529/.37171 around .25 and .37171/.60166 around .50. This is a discrete sampling gap, not evidence comparable cycles do not exist. The no-match result is preserved.

EXP-023 is drafted but unexecuted: two fixed interpolated core gamma values, bounded same-section seed correction, then all six cycles at eight new common phase fractions. It has 116 jobs/578 maximum IVP slots; original bin, <=2% period and <=10% immediate-RMS-detectability matching gates; both transverse modes, gains and modal orientation retained. Strict references and direct tangent propagation follow the bridge. Full feasibility requires all six references and all 48 phase contexts. It is only a feasibility study, with no finite writes. A later proposed 972-trace finite-write panel would be separate. The controller is unfinished; 129 synthetic tests validate the reusable earlier adapters, not the new bridge/controller. Nothing forces us to finish this sunk-cost path if another experiment has higher explanatory value.

## Resources and a useful answer

We have a shared local i9-14900K (24 cores/32 threads), about 92 GiB RAM, and RTX 2000 Ada 16 GiB. Four single-thread CPU workers suit current kernels; substantial local compute is authorized. An earlier tested CUDA ODE pipeline was slower and failed strict cross-device tolerances, so we do not claim a validated GPU substitute. A separate suitable GPU workload could be developed. No cloud spending is needed for the immediate design. Raw evidence stays outside private Git; source and compact receipts are checkpointed. No paid experiment or further review is implicitly authorized by your advice.

Please return a research-director judgment with these concrete elements:

1. A candid diagnosis: what general principle have we actually learned, what is generic textbook behavior, and what remains a potentially important unanswered question? Is our bottleneck scientific framing, sampling, instrumentation, or external validity?
2. At most five ranked mechanisms/directions, each with a cheap decisive falsification, the simplest serious competing model, and positive/null/invalid interpretations. Consider phase-amplitude reduction, transient nonnormal transport, noise–memory–robustness tradeoffs, nonlinear mixing versus nonlinear readout, parameter-window organization, and learned-system transfer without assuming any is the answer.
3. Choose ONE minimal next experiment, explicitly deciding finish/defer/replace EXP-023. Give actual interventions, primary endpoint, units of replication, phase/readout information available, relevant uncertainty, controls, stopping rule and realistic resource scope. Distinguish a quick exploratory pilot from an expensive confirmation; avoid an endless validation or matching ladder.
4. Give the strongest plausible "big picture" statement we could earn if positive and the useful update if negative. Explain a credible bridge to larger neural networks and a concrete result that would disprove that bridge.
5. Identify at least three seductive but invalid interpretations we should stop using. Finish with a short human-friendly TLDR about neural behavior, not a list of audit counts. Keep the advice compact and actionable.

Local provenance (not sent as raw evidence): EXP-021 source 0ae8686ef0ab719bc7dcff300962aabd4a6679b9; raw receipt 499ee26bb6c2373905b7c8e76920ccf7bdba298fb8d43a03e741552ac681ff13; EXP-022 selection b8bcf7f11fc82623993e07af54cc3d6a42a338d18561cc148f4316329952416b. Detailed results, prior review/adjudication and code remain locally auditable. You are assessing these synthesized reports conditionally, not rederiving them.

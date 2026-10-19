# Consult 19 adjudication: relevance of the results to loops in current LLM systems (19 October 2026)

Review: `review.md` (gpt-6-astra, research-strategy, completed 19 October 2026 about 04:20 UTC; raw response and
manifest in `../neural-shrimp-data/consults/2026-10-19-loop-relevance/execute-001/`). The advice is advisory; each point
was checked against our records before use.

| Point | Decision |
|---|---|
| (a) Introduction: make the boundary unmistakable (fixed-prefix, fixed-window soft-token map vs ordinary generation and recurrent-depth transformers; dynamical methods apply to both loop kinds, but their update operators, derivatives and stability conditions differ) | Applied (article commit ce44b01), with the citations Dehghani et al. 2019, Giannou et al. 2023 and Geiping et al. 2025 |
| (b0) Feedback creates a dynamical system but implies neither convergence nor chaos; energy guarantees need specific structure, as for the tied readout of Sec. III, and none is established for a complete transformer update. Our soft-token loop shows fixed points, cycles and chaos; recurrent depth shows convergence and oscillatory instability, no chaos observed | Applied. The answer to the human's question ("the internal loops ... are more like Hopfield, so no chaos?"): Hopfield-type convergence holds for gradient maps; a trained recurrent-depth block is not one in general, so its convergence is trained, not guaranteed |
| (b) Huginn: the block Jacobian without the readout factor; the predicted Neimark-Sacker point at relaxation 1.0777 (amplitude fit 1.0778); the prospective generation test (loss at 1.20, none at 1.05); 6 of 30 ARC-Challenge development questions natively unstable, restored by under-relaxation; no general relation between stability and accuracy is established, especially at short horizons | Applied; numbers checked against the full report's recurrent-depth section. The consult's caution against "regime matters as much as the number of passes" was followed: the article says the results "motivate evaluating the dynamics ... jointly with their number of iterations" |
| (c) Data Availability: the certificate paths | Applied (`codex/certify-llm`, `certificates/llm-p8-certificate-final-001.json`, `codex/coexistence-task33`); the paths exist at commit 35eabc6c |
| (d) The safety sentence | Applied in the consult's narrower form: the findings motivate testing whether interventions in continuous-feedback systems, including steering for safety, alter finite-horizon sensitivity or task-relevant decisions as well as immediate outputs, alongside the monitoring of latent computation (Korbak et al. 2025); "We tested no safety property: dynamical instability is neither misalignment nor evidence of it." |
| Press reports about loop transformers in frontier models | Not cited (as planned; the consult agrees) |
| Novelty against Kim (2026) and the looped-transformer literature | No new claim of novelty about looped transformers; the article positions the work as pretrained-model dynamics with prospective predictions and certificates |
| Ranked experiments (a relaxation-by-depth trial in Huginn at a fixed inference budget; which amplified directions reach the answer; transfer to a second recurrent-depth checkpoint) | Recorded for the journal version; not run before arXiv v1 |
| Under-relaxation is not universally stabilizing (a multiplier with real part above one cannot be brought inside the unit disk by positive scalar under-relaxation) | Noted; consistent with the relaxed-multiplier formula mu_lambda = 1 + lambda (mu - 1) used in the report |

Sync: the same passages go to the full report and the blog (19 October 2026).

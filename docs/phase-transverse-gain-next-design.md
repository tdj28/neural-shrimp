# Literature review and conditional next mechanistic step

Written during the frozen EXP-014 refinement acquisition, before reading any new target output. This is a decision framework and proposed next analysis, not an execution freeze. No new trajectory, model fit or target statistic was computed. Existing evidence and original failed qualification remain unchanged.

## Literature review

The committed literature bridge is materially accurate at its stated reading depth. Pals, Macke and Barak (2024) study trained 512-unit continuous-time tanh RNNs, including rank-two networks, phase-coded cycles, Floquet stability and forcing-amplitude/frequency regions of phase locking. Their external reference oscillation is an input to the dynamics. Our external clock is only an observation convention; autonomous neutral phase is different from restoring phase locking. Their results motivate a later trained-network experiment, not a reproduction claim or a link from visual islands to verified shrimp structure. Primary methods and relevant results checked: https://pmc.ncbi.nlm.nih.gov/articles/PMC10868787/ .

Ichikawa and Kaneko's abstract supports transient oscillation-amplitude memory with slow-manifold attraction and contraction; it is accurately presented as an alternative mechanism, not evidence for our network. Abstract checked: https://arxiv.org/abs/2010.15308 .

Wedgwood and colleagues' abstract supports phase-amplitude descriptions beyond weak perturbations and strong attraction. Their periodically forced oscillator examples do not prove chaos after our single initial-state pulse. Abstract checked: https://arxiv.org/abs/1302.0476 .

Ganguli, Huh and Sompolinsky analyze memory with dynamical noise and emphasize input direction and nonnormal amplification. This supports measuring finite-time observable gain, but their noise and memory-capacity assumptions differ from our deterministic ODE with assumed measurement noise. Relevant primary sections checked: https://pmc.ncbi.nlm.nih.gov/articles/PMC2596211/ .

## How to read this refinement

All 2,380 declared comparisons, 85 physical cases and 3,744 same-method sign-paired readouts remain in the report, including failures. Separate the 255 old-old, 1,275 old-new and 850 new-new comparisons. Already failed old-old comparisons cannot become successful because new trajectories were added. Do not call the whole refinement a numerical success merely because acquisition completes.

1. If all five new methods agree under every frozen state, activation and unsaturated-q gate for a physical case, this supports finite-time numerical consistency for that selected case. Both signs must satisfy the stated comparisons before a binary readout is discussed as conditionally resolved. If old methods disagree with that consistent new group, report this as localization of earlier numerical sensitivity; do not replace original outcomes or requalify either original setting.
2. If the new methods still disagree, preserve the full trajectories and first saved-grid threshold crossings. Saturated BA agreement is not waveform or q reliability. A failure occurring late cannot silently redefine the original horizon. No new attractor, basin boundary, chaos or geometry mechanism follows from solver disagreement alone.
3. If agreement is limited to some doses or phases, show the entire selected census and those case-level conditions. Selection was made from earlier failures and is asymmetric across settings. Neither successful-case averages nor core-versus-edge failure incidence is a valid population estimand.
4. Separate raw/scientific agreement decisions from strict alternative-arithmetic audit discrepancies. Preserve both, including failed independent replay tolerances; neither waives the other.

## Next bounded analysis, whichever numerical branch occurs

The most informative next step is a separately frozen, saved-reference phase/transverse gain decomposition. It needs no new ODE solves or learned decoder and can answer a local mechanistic question even if some large-pulse trajectories remain unresolved. It does not need an invented task-quality threshold.

Use all 16 original qualified setting/phase reference-propagator contexts, all three original pulse directions and all four clock windows: 192 directional/window cells. Bind the old reference and propagator packets and the unchanged clock, adjoint, pulse directions and measurement model. Use the exact original order Psi_j(kT+s) = Phi_j(s) M_j^k. Treat singular values or operator norms as finite-time diagnostics rather than substitutes for full propagation.

For initial phase j define f0 = f(x0), z0 = Z(x0), P0 = I - f0 z0^T, with the already audited normalization z0^T f0 = 1. For a realized initial displacement delta, decompose the full linear response as

    delta_x_phase(t) = f_ref(t) * (z0^T delta)
    delta_x_transverse(t) = Psi_j(t) P0 delta
    delta_x_linear(t) = delta_x_phase(t) + delta_x_transverse(t).

Check this sum against direct Psi_j(t) delta at every declared readout sample, using a frozen tolerance and retaining every failed cell. Do not enforce exact identities by renormalizing adjoints or repairing arrays. Reference phase is wrapped, but the full propagator retains the nonperiodic elapsed-time factor M_j^k. These are linear tangent coordinates, not exact finite-amplitude isochrons.

At clock samples let D_m = diag(1 - tanh(x_ref(t_m))^2), and let w_m be the existing matched-filter coefficients. Form the row vector

    L_k = sum_m w_m^T D_m Psi_j(t_m).

Report direct total gain L_k d, phase gain z0^T d, transverse gain L_k P0 d, their signed sum residual, and normalization by the unchanged sigma*||w||. Because w is matched to the reference time derivative, its neutral contribution is the local time shift, up to the saved reference/interpolation residual. Keep signed values and state/activation gain; do not report only saturated BA. Report all three fixed noise levels without choosing a favorable one.

The requested and realized displacements must stay distinct: the original nonlinear phase-only emulator uses requested delta, whereas full linear response uses realized delta. A neutral/transverse decomposition of the latter must use realized delta in both pieces. Also distinguish the linear neutral component from the finite-shift nonlinear phase-only template; they agree only to first order.

After sealing these reference-only predictions, compare against already acquired finite-pulse evidence under a prospectively explicit eligibility rule. Retain every old planned case and method; ineligible or unresolved cases are marked unavailable for mechanistic conclusions, not dropped. Use both signs and all three original doses, with odd/even responses and residuals, to distinguish first-order transverse readout from finite-amplitude deviations. A full linear match in numerically consistent small-dose cases would explain the phase-only discrepancy through ordinary phase-amplitude dynamics. A mismatch motivates a separate nonlinear-response study only after numerical resolution; it is not evidence of topology utility.

## Limits and later bridge

Two selected cycles cannot establish geometry advantage. A later geometry study needs several certified regions and ordinary periodic controls, with held-out regions and declared adjustment for period, waveform, input/observation gain, local contraction and transient amplification. An actual trained low-rank RNN replication can test a closer link to Pals and colleagues, with held-out stimuli and seeds plus certification of learned attractors. Neither project should precede clarification of what the present local propagator/readout already predicts.

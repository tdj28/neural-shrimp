# Task23 four-cell finite-twin mechanism replay — final accepted report

TLDR: In this fixed Qwen2.5-0.5B soft-token experiment, normalized finite-twin expansion through the network is opposed by contraction through the readout at all four cells. Total measured perturbation growth is positive at the two irregular cells and negative at the two cells with detected finite returns. The independent CPU replay supports this finite decomposition and reproduces the saved regime labels, while all 28 requested producer scalars differ. It does not establish a causal mechanism, asymptotic chaos, verified primitive cycles, or a shrimp-specific behavioral benefit.

All model and saved-data execution is completed and closed. Boundary’s independent saved-evidence assessment and Root’s final scientific acceptance are complete. No further run is proposed.

## Fixed scope and interpretation

The committed own CPU float64 implementation uses the original hostility-vs-kindness vector at block 12, alpha 0.0655, beta 0, original source indices 25/29/19/35, the fixed steered prefix and its last-token embedding as the initial soft token. The two finite perturbation sizes are 1e-6 and 1e-9 relative. Original default_rng(0) directions are selected from their original two 201×896 draw blocks. The sole full run has 1000 transient steps followed by 2000 measured steps. The requested decimal .119 is implemented at the original stored binary64 value .11900000000000001; the one-ULP distinction remains explicit.

For each finite twin, the source-defined scalar identity is lambda = mean(log g_net) + mean(log g_read) + log(norm(x_final)/norm(x_initial))/2000. The final/initial boundary term has the positive sign shown. Network/readout terms are finite secant accounting along this trajectory, not infinitesimal derivatives, an independently certified tangent cocycle or a causal intervention. Zero and nonfinite operands are not hidden by new floors. All four completed rows are defined.

Own model arithmetic is CPU float64 throughout the manually implemented norms, rotary tables and softmax, with the tied embedding/readout inventory bound to the accepted asset hashes. The original producer used 201 base rows / 603 measured rows on its CUDA path; the own run uses 4 base rows / 12 measured rows. Original source/local archive/retrieval/recorded termination authority is preserved, but historical loaded bytes and process births are not reconstructed. Different batch/platform/finite trajectories preclude an assumption of numerical equality.

## Own finite results

Values below are literal saved binary64 JSON values, not rounded bounds. Network/readout columns use the first twin; both twin splits remain in the original report and comparator. Period 0 means the finite last-state return rule found no period among its tested lags.

| Original index; log10T | Twin rates (1e-6,1e-9) | Network mean | Readout mean | Signed boundary | Entropy | Normalized top-two gap | Finite period |
|---|---|---|---|---|---|---|---|
| 25; 0.125 | 0.4950137968265117, 0.4951925533604417 | 0.5917462603039746 | -0.09673621245132731 | 3.7489738624098855e-06 | 6.328119091338987 | 1.2245618191946803 | 0 |
| 29; 0.129 | 0.516734394657097, 0.5179621168353872 | 0.6071165675051129 | -0.0903113529331758 | -7.08199148408509e-05 | 6.59992713280153 | 1.1487947664787075 | 0 |
| 19; 0.11900000000000001 | -0.42334643917864945, -0.4229000409855568 | 0.2391864799012664 | -0.6625329190799171 | 3.3306690738754687e-19 | 5.184826964178679 | 1.9480913024177033 | 2 |
| 35; 0.135 | -0.1852612573584068, -0.18533732202995704 | 0.38499265815998346 | -0.5703094305793245 | 5.551506093574617e-05 | 6.392994595859047 | 1.3479591832058568 | 14 |

The own and producer finite period labels are 0, 0, 2, 14 in this fixed order, with corresponding literal chaotic flags true, true, false, false. This is a finite two-twin/last-state classification. It is not full primitive-orbit/stability/boundary verification or an asymptotic chaos proof. The eight own scalar accounting residuals satisfy the originally fixed 1e-10 criterion; no producer-agreement tolerance was introduced.

Entropy does not cleanly separate these four regimes: the periodic-context .135 value 6.392994595859047 exceeds the irregular-context .125 value 6.328119091338987. Mean gaps and entropy are readout diagnostics here, not a verified prediction that greedy decoding absorbs these differences. Readout power/trace/rank/Rayleigh work, AD/FD and the optional crossover were not added.

## All 28 original-producer comparisons

Every comparison below is nonzero. These are own minus producer differences copied from the admitted comparator, with no numerical-agreement pass gate. The largest absolute twin-rate difference is 0.035152435006497784 at .125/1e-6; an enclosing decimal bound is 0.035153. All exact own/producer values and binary64 representations remain in the comparator report.

| Index | Field | Own | Producer | Own minus producer | Absolute difference |
|---|---|---|---|---|---|
| 25 | lyapunov_eps_1e-06 | 0.4950137968265117 | 0.5301662318330095 | -0.035152435006497784 | 0.035152435006497784 |
| 25 | lyapunov_eps_1e-09 | 0.4951925533604417 | 0.5298507494410065 | -0.03465819608056481 | 0.03465819608056481 |
| 25 | mean_log_g_net | 0.5917462603039746 | 0.6047388171519554 | -0.01299255684798084 | 0.01299255684798084 |
| 25 | mean_log_g_read | -0.09673621245132731 | -0.0746678227936416 | -0.022068389657685705 | 0.022068389657685705 |
| 25 | boundary_term | 3.7489738624098855e-06 | 9.523747469564326e-05 | -9.148850083323338e-05 | 9.148850083323338e-05 |
| 25 | mean_entropy | 6.328119091338987 | 6.315294233368378 | 0.012824857970609393 | 0.012824857970609393 |
| 25 | mean_gap | 1.2245618191946803 | 1.2197955665950326 | 0.004766252599647691 | 0.004766252599647691 |
| 29 | lyapunov_eps_1e-06 | 0.516734394657097 | 0.5234356608784092 | -0.006701266221312174 | 0.006701266221312174 |
| 29 | lyapunov_eps_1e-09 | 0.5179621168353872 | 0.5232532458892575 | -0.005291129053870303 | 0.005291129053870303 |
| 29 | mean_log_g_net | 0.6071165675051129 | 0.6249372042653611 | -0.017820636760248254 | 0.017820636760248254 |
| 29 | mean_log_g_read | -0.0903113529331758 | -0.10148934755789585 | 0.01117799462472005 | 0.01117799462472005 |
| 29 | boundary_term | -7.08199148408509e-05 | -1.2195829055497335e-05 | -5.8624085785353566e-05 | 5.8624085785353566e-05 |
| 29 | mean_entropy | 6.59992713280153 | 6.63168370530051 | -0.03175657249898034 | 0.03175657249898034 |
| 29 | mean_gap | 1.1487947664787075 | 1.1909626414262147 | -0.04216787494750718 | 0.04216787494750718 |
| 19 | lyapunov_eps_1e-06 | -0.42334643917864945 | -0.423346434897282 | -4.28136742725016e-09 | 4.28136742725016e-09 |
| 19 | lyapunov_eps_1e-09 | -0.4229000409855568 | -0.42288386515518794 | -1.6175830368836408e-05 | 1.6175830368836408e-05 |
| 19 | mean_log_g_net | 0.2391864799012664 | 0.23918648029004924 | -3.887828392556969e-10 | 3.887828392556969e-10 |
| 19 | mean_log_g_read | -0.6625329190799171 | -0.6625329151873312 | -3.892585920262093e-09 | 3.892585920262093e-09 |
| 19 | boundary_term | 3.3306690738754687e-19 | -2.0095036745715736e-17 | 2.0428103653103284e-17 | 2.0428103653103284e-17 |
| 19 | mean_entropy | 5.184826964178679 | 5.184826964178584 | 9.50350909079134e-14 | 9.50350909079134e-14 |
| 19 | mean_gap | 1.9480913024177033 | 1.9480913024177628 | -5.950795411990839e-14 | 5.950795411990839e-14 |
| 35 | lyapunov_eps_1e-06 | -0.1852612573584068 | -0.18571747322210966 | 0.00045621586370286904 | 0.00045621586370286904 |
| 35 | lyapunov_eps_1e-09 | -0.18533732202995704 | -0.18625422488402255 | 0.00091690285406551 | 0.00091690285406551 |
| 35 | mean_log_g_net | 0.38499265815998346 | 0.38470898932248243 | 0.00028366883750102767 | 0.00028366883750102767 |
| 35 | mean_log_g_read | -0.5703094305793245 | -0.5705515474872274 | 0.00024211690790287843 | 0.00024211690790287843 |
| 35 | boundary_term | 5.551506093574617e-05 | 0.00012508494263507763 | -6.956988169933146e-05 | 6.956988169933146e-05 |
| 35 | mean_entropy | 6.392994595859047 | 6.397011855078205 | -0.00401725921915741 | 0.00401725921915741 |
| 35 | mean_gap | 1.3479591832058568 | 1.3473369000714945 | 0.0006222831343623003 | 0.0006222831343623003 |

Producer second-twin network/readout splits, its full 2000-step float64 scalar stream, and exact runtime prefix IDs/native twin states are unavailable for this independent comparison. The original first 1000 step arrays are float32 and are not substituted for the full float64 aggregates. Source-bound JSON sums/norms have no explicit decimal rounding in the producer writer, so the observed nonzero differences are not described as equality hidden by printed rounding.

## Independent saved-state and stream check

The separate own audit checked all 2000 step records, the complete 2001×4×896 float64 base matrix, original RNG selections, saved offset norms, final-state/tail links, progress accounting, scalar means and all 256 finite-lag rows. It reported 16082 exact source-identical comparisons covering 355224 scalar elements, zero source-identical mismatches, and the original summed accounting threshold passed. It retained 3772 records of NumPy-versus-Torch norm differences without imposing a new agreement threshold. Those diagnostic records remain in full outside Git.

This audit does not evaluate the map or replay the transient. It checks RNG selections and norms, while initial directional construction relies on the reviewed source. Entropy is checked against the saved scalar stream, not independently reconstructed from unsaved probability vectors. Its exactness result concerns the audited source-identical saved relationships and does not erase the producer differences above or the norm diagnostics.

## Execution and immutable evidence

Inference alone executed and reaped the owned workers/supervisors; original process identities, groups/sessions, captured source blobs and sealed manifests are bound by the original owner receipts. Root reviewed each exact admission. The finite model controls 9/9, short 32/16 profile, full 1000/2000 model run, comparator controls 7/7, comparator, consistency controls 7/7 and consistency audit each ran once. No scientific retry, extension or model rerun occurred. A metadata-only comparator preflight initially refused mixed historical identity-key schemas before any dispatch/output/child; original error/source and the explicit pinned two-schema correction are preserved.

The actual model supervisor window was 1017.9711758838966s. Its worker reported 1016.234895714093s and 5,334,460KiB observed peak before final rehash/report, not an independent full-worker wall/final peak. Comparator external window .05222603399306536s; audit external window .5561823910102248s. The audit observed .39288068283349276s and 122,128KiB before its final report. Audit output sampling observed 1,558,548B while the final report is 2,281,905B; these scopes are distinct. Tiny audit controls sampled 0 B between polls but retained 1525 B of final outputs.

The immutable actual model report contains one inherited stale scope sentence calling its summaries profile-only and not 1000/2000. This is a descriptive error: its stage=actual, scientific_full_run_complete=true, card, horizon fields and sealed complete outputs bind the full 1000/2000 result. The original report is retained byte-identically; this paragraph and the earlier completion receipt are the separate attribution correction. The short-profile summaries remain setup/cost observations and are not promoted as the full result.

Exact primary pins:

- Actual execution source3e103261c546d128c9b6212cdaeda78d677148ad; card 93f519a6ac731486f761921efbc1a9e909be44d16a0ea16f2c75cb8d9d2cdd78.
- Actual report 0039ef4c2732bbf4641010543a33b761cd85821167a5ba20ebb7d9f2c391dfd7; owner closure e6072e7fd0aa87a86b9c7f912630ae9cdebffbb2bb7a8773cb9d6f81ccc91b54.
- Comparator source 64fcb1d50ccaf0b8db22867b4d08be9ff75dd0f2426a969502faf4feb2db9a95; report efd89b19259f3d671a78a468158830762d2d697f29d032548f3c57786c50e563; owner closure e8d3fcff4d7edf467ac8cd6db70abf86fdff9960d9a854188233e0141ad8c67d.
- Saved audit source 00d3e95a0b97b3689fe986dd597257a5a5f087582d5752e1fb45bdbf93bd114b; report ac9719898adb66b7f73ea6a3bbb38d1375e002182af6e9323043a8e9acaf06e7; owner closure b29452252145b8a53791c3f0ba10f0227c2763bd03c0701ac974b3c610d5a8a2.
- Raw root `/data2/neural-shrimp-data/codex/mechanism-replay-task23`; run-001, comparison-001 and saved-consistency-001 each have corresponding -supervision and -dispatch directories with exact manifests.

Final independent review: `independent-final-review.md`, SHA256 62346cd6d9d2c0c1d85337f1008b4940aca729cf5be3a65b7dd8aef4c2f42847. Its literal coverage selection has SHA256 8cce4a19e402dec5e88700144853b8a091663590374b9a38af8239a011db5479 and remains bound in `final-evidence-manifest.json`; it is not a new verifier. Root final acceptance is recorded at 2026-10-15T17:20:13.026230+00:00 in `root-final-acceptance.md`, SHA256 3517e9f0c03c586cffaef6a3e74a6d0a0b869a37df631e1702215135407c0953. Inference executed and authenticated the original lifecycle; Boundary read the saved results without replay, and Root accepted the final finite scientific interpretation. Final private evidence and standalone delivery commits are recorded separately in `delivery-receipt.json` after publication. Task25 remains source preparation only until separate concrete admission. The persistent quiet watcher continues every 15 minutes without cutoff until the human explicitly stops it. The latest Claude 16:55 information is producer context for a possible later delta review, not a new assignment.

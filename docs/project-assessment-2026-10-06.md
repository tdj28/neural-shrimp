# Project assessment, 6 October 2026

**Human-friendly TLDR:** The repository contains a careful reproduction of a
published neural shrimp and many small, well-audited experiments at two
hand-picked periodic orbits. Each behavioral result so far has a generic
explanation in standard Floquet, reservoir-computing or consistency theory,
and the shrimp-specific question has not been tested against distant controls.
On 6 October 2026 the user chose **option C**: a headline study of
period-resolved trainability maps (gradient-descent training in a
two-learning-rate plane) plus a companion study of whether periodic-window
geometry predicts computation beyond local dynamics. See
[Program C](program-c-2026-10-06.md).

This is a local assessment and planning record. It reports no new neural
outcome. The literature survey summarized below was delegated to a separate
agent that used generic topic queries only; no private project text was sent
to a search service.

## Where the evidence stands

| Area | Established within its declared scope | Generic explanation or limitation |
|---|---|---|
| Dynamics (EXP-004/005) | Reproduced da Silva and Rech shrimp; verified primitive orbits, periods and Floquet multipliers at selected points | Reproduction of known structure; 181x181 focus map versus the published 1000x1000 scans |
| Repeatability (EXP-005/025) | All 12 sampled periodic settings were more repeatable than all 13 irregular settings | Consistency and generalized synchronization (Uchida et al. 2004; Lymburn et al. 2019) |
| Pulse memory (EXP-014/015/019) | Near-edge cycle retains inputs longer than the core; phase-null inputs leave transverse signals | Floquet decay: core multiplier 0.0651 at T=37.64 gives a transverse time constant of about 13.8 time units (0.37 periods); near-edge modulus 0.856 at T=37.33 gives about 240 (6.4 periods) |
| Readout and access (EXP-021/025/026) | Information can survive when a weak or phase-unaware readout fails | Phase-coded information needs a reference (Pals et al. 2024; Kubota et al. 2021) |
| Input mixing (EXP-024) | Linearly readable second-order interaction at some phases | Cross-terms of information processing capacity (Dambre et al. 2012) |
| Control utility (EXP-006 to EXP-013) | Geometry-informed policies did not beat the strongest simple baseline | A useful negative result for a bespoke task |

Main limitations for a paper:

1. Most memory results compare two selected orbits; phases of one orbit are
   not independent networks or regimes.
2. The distant ordinary-periodic and distant chaotic controls requested on
   23 September 2026 have not been run.
3. Tasks rely on known phase and clock or linearization-built observers.
   Standard learned readouts with unknown phase found almost no memory in
   EXP-025 (25 of 26 settings had nonpositive primary R-squared).
4. About 7 of the 17 experiments executed from EXP-009 to EXP-026, plus 4
   refinement amendments, were numerical requalifications, corrections or
   engineering checks. EXP-008's strict Linux replay still fails over
   differences such as 1.4e-14, below any scientifically justified tolerance.
5. The original training-dynamics motivation was set aside after one small
   pilot (EXP-002: 162 cells, 2,000 updates).
6. The manuscript draft covers EXP-004 to EXP-008 only and cites 8 sources.

The full local test suite passed on 6 October 2026: 2,107 passed and 1 skipped
in 371 seconds at `781fcf7c`.

## Literature positioning (survey, 6 October 2026)

The survey found shrimp and spiral organization reported in many neuron
models (Hopfield-type tanh networks, Hindmarsh-Rose, FitzHugh-Nagumo, Rulkov
and Chialvo maps), but no study measuring memory, information processing
capacity or task performance as a function of position inside a shrimp.

For training dynamics, existing two-hyperparameter maps classify runs only as
convergent or divergent (Sohl-Dickstein 2024; Torkamandi 2025). Bifurcation
analyses of gradient descent are one-dimensional or use scalar reductions
(Kalra, He and Barkeshli 2025; Chen et al. 2023; Song and Yun 2023; Hofmann
2026). The survey found no report of verified Gallas-type isoperiodic domains
in a training-hyperparameter plane; it estimated its own confidence at about
80%, with residual risk from figures in papers using other vocabulary.

Gap A (period-resolved trainability maps) is the most novel and matches the
project's original motivation. It needs orbit verification, treatment of
symmetry-induced neutral directions, at least one real-data network, and a
machine-learning consequence. Gap B (geometry versus local quantities for
memory and capacity) is likely to produce a clean null plus a Floquet memory
law, suitable as a companion study.

## Candidate bibliography

Compiled by the survey agent from registry, publisher and arXiv pages. Content
claims were mostly taken from abstracts. **Every consequential citation must be
rechecked against its primary source, exact version and date before use in a
manuscript**, following the canonical writeup and integrity playbooks.

### Shrimps and periodicity in neural and classical systems
- Gallas, PRL 70, 2714 (1993). https://doi.org/10.1103/PhysRevLett.70.2714
- Stoop, Benner, Uwate, PRL 105, 074102 (2010). https://doi.org/10.1103/PhysRevLett.105.074102
- Stoop et al., IJBC 22, 1230032 (2012). https://doi.org/10.1142/S0218127412300327
- Rech, Neurocomputing 74, 3361 (2011). https://doi.org/10.1016/j.neucom.2011.05.016
- Rech, Int. J. Mach. Learn. Cybern. 6, 1 (2015). https://doi.org/10.1007/s13042-013-0222-0
- da Silva and Rech, Commun. Adv. Math. Sci. 5(2), 92 (2022). https://doi.org/10.33434/cams.1064713
- Rech, Phys. Lett. A 375, 1461 (2011). https://doi.org/10.1016/j.physleta.2011.02.037
- Barrio et al., Chaos 24, 023128 (2014). https://doi.org/10.1063/1.4882171
- Freire and Gallas, Phys. Lett. A 375, 1097 (2011). https://doi.org/10.1016/j.physleta.2011.01.017
- Hoff et al., Eur. Phys. J. B 87, 151 (2014). https://arxiv.org/abs/1504.07565
- Ramirez-Avila, Muni, Kapitaniak, Chaos 34, 083134 (2024). https://doi.org/10.1063/5.0214903
- Caixeta et al., Chaos 35, 033122 (2025). https://doi.org/10.1063/5.0250342
- Beer, Neural Comput. 18, 3009 (2006). https://doi.org/10.1162/neco.2006.18.12.3009
- Pasemann, Network 13, 195 (2002). https://doi.org/10.1080/net.13.2.195.216

### Training dynamics, trainability maps and chaos in optimization
- Sohl-Dickstein, arXiv:2402.06184. https://arxiv.org/abs/2402.06184
- Torkamandi, arXiv:2501.04286. https://arxiv.org/abs/2501.04286
- Liu, arXiv:2406.13971. https://arxiv.org/abs/2406.13971
- Liang and Montufar, arXiv:2509.25351. https://arxiv.org/abs/2509.25351
- Ly and Gong, arXiv:2510.05606. https://arxiv.org/abs/2510.05606
- D'Inverno et al., arXiv:2508.03222. https://arxiv.org/abs/2508.03222
- Cohen et al., ICLR 2021. https://arxiv.org/abs/2103.00065
- Kong and Tao, NeurIPS 2020. https://arxiv.org/abs/2002.06189
- Herrmann, Granz, Landgraf, NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/222a2a46018a1e7b55ba48ba11932d04-Abstract-Conference.html
- Danovski, Soriano, Lacasa, Front. Complex Syst. 2, 1367957 (2024). https://doi.org/10.3389/fcpxs.2024.1367957
- Jimenez-Gonzalez, Soriano, Lacasa, Phys. Rev. Research 8, L022032 (2026). https://doi.org/10.1103/t5p9-kv5w
- Kalra, He, Barkeshli, ICLR 2025. https://arxiv.org/abs/2311.02076
- Kalra and Barkeshli, NeurIPS 2023. https://arxiv.org/abs/2302.12250
- Ghosh et al., ICLR 2025. https://arxiv.org/abs/2502.20531
- Chen and Bruna, ICML 2023. https://proceedings.mlr.press/v202/chen23b.html
- Chen et al., arXiv:2310.01687. https://arxiv.org/abs/2310.01687
- Song and Yun, NeurIPS 2023. https://arxiv.org/abs/2307.04204
- Hofmann, arXiv:2607.04993. https://arxiv.org/abs/2607.04993
- Goujaud, Taylor, Dieuleveut, arXiv:2502.19916. https://arxiv.org/abs/2502.19916
- Lai et al., arXiv:2609.04963. https://arxiv.org/abs/2609.04963

### Dynamical regime and computation
- Bertschinger and Natschlager, Neural Comput. 16, 1413 (2004). https://doi.org/10.1162/089976604323057443
- Legenstein and Maass, Neural Netw. 20, 323 (2007). https://doi.org/10.1016/j.neunet.2007.04.017
- Boedecker et al., Theory Biosci. 131, 205 (2012). https://doi.org/10.1007/s12064-011-0146-8
- Toyoizumi and Abbott, PRE 84, 051908 (2011). https://doi.org/10.1103/PhysRevE.84.051908
- Schuecker, Goedeke, Helias, PRX 8, 041029 (2018). https://doi.org/10.1103/PhysRevX.8.041029
- Haruna and Nakajima, PRE 100, 062312 (2019). https://doi.org/10.1103/PhysRevE.100.062312
- Carroll, Chaos 30, 121109 (2020). https://doi.org/10.1063/5.0038163
- Dambre et al., Sci. Rep. 2, 514 (2012). https://doi.org/10.1038/srep00514
- Inubushi and Yoshimura, Sci. Rep. 7, 10199 (2017). https://doi.org/10.1038/s41598-017-10257-6
- Kubota, Takahashi, Nakajima, Phys. Rev. Research 3, 043135 (2021). https://doi.org/10.1103/PhysRevResearch.3.043135
- Uchida, McAllister, Roy, PRL 93, 244102 (2004). https://doi.org/10.1103/PhysRevLett.93.244102
- Lymburn et al., Chaos 29, 023118 (2019). https://doi.org/10.1063/1.5079686
- Jungling, Lymburn, Small, arXiv:2105.13473. https://arxiv.org/abs/2105.13473
- Appeltant et al., Nat. Commun. 2, 468 (2011). https://doi.org/10.1038/ncomms1476
- Akashi et al., Phys. Rev. Research 2, 043303 (2020). https://doi.org/10.1103/PhysRevResearch.2.043303
- Shougat, Li, Perkins, PRE 105, 044212 (2022). https://doi.org/10.1103/PhysRevE.105.044212
- Goldmann et al., Chaos 33, 093139 (2023). https://doi.org/10.1063/5.0156494

### Oscillatory memory and recurrent-network training
- Pals, Macke, Barak, PLOS Comput. Biol. 20, e1011852 (2024). https://doi.org/10.1371/journal.pcbi.1011852
- Ichikawa and Kaneko, Phys. Rev. Research 3, 033193 (2021). https://doi.org/10.1103/PhysRevResearch.3.033193
- Park, Sagodi, Sokol, arXiv:2308.12585. https://arxiv.org/abs/2308.12585
- Rusch and Mishra, ICLR 2021. https://arxiv.org/abs/2010.00951
- Rusch and Rus, ICLR 2025. https://arxiv.org/abs/2410.03943
- Eisenmann et al., NeurIPS 2023. https://arxiv.org/abs/2310.17561
- Mikhaeil, Monfared, Durstewitz, NeurIPS 2022. https://arxiv.org/abs/2110.07238
- Engelken, Wolf, Abbott, Phys. Rev. Research 5, 043044 (2023). https://doi.org/10.1103/PhysRevResearch.5.043044

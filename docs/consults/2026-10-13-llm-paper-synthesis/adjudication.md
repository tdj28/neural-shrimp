# Adjudication: LLM paper synthesis consult (consult 6, 13 October 2026)

We sent one `research-strategy` request (gpt-6-astra, Pro mode, xhigh) with `brief.md` at commit 41127426. It cost
[amount redacted] (conservative) for 73,329 tokens. Raw request, response and manifest:
`../neural-shrimp-data/consults/2026-10-13-llm-paper-synthesis/execute-001/`. The advice is advisory only; every
point below was checked or decided locally.

## References (checked at the primary record before use)

| Reference | Check | Use |
| --- | --- | --- |
| Pasemann, "Complex dynamics and the structure of small neural networks", Network: Computation in Neural Systems 13(2):195-216 (2002), doi:10.1080/net.13.2.195.216 | Crossref record | Discussion: complex dynamics of small recurrent networks (title-level claim only) |
| Rimsky, Gabrieli, Schulz, Tong, Hubinger, Turner, "Steering Llama 2 via Contrastive Activation Addition", ACL 2024 (Volume 1), pp. 15504-15522, doi:10.18653/v1/2024.acl-long.828 | ACL Anthology bib | Steering construction (contrastive activation addition) |
| Azaria, Mitchell, "The Internal State of an LLM Knows When It's Lying", Findings of EMNLP 2023, pp. 967-976, doi:10.18653/v1/2023.findings-emnlp.68 | ACL Anthology bib | Probes of the model's internal state |
| Welleck, Lu, West, Brahman, Shen, Khashabi, Choi, "Generating Sequences by Learning to Self-Correct", arXiv:2211.00053 (2022) | arXiv record; venue (ICLR 2023) not checked, so cite arXiv | Related generation-time correction, next to detect-and-replace |
| E, Vanden-Eijnden, "Towards a Theory of Transition Paths", J. Stat. Phys. 123:503-523 (2006) | already cited (e2006transition) | — |
| Gallas, Phys. Rev. Lett. 70:2714 (1993) | already cited | — |

## Claims (Q1)

- **Accepted. Prospective fold forecasts.** "23 of 24 within one iterate, failing the all-row criterion." The paper
  already says exactly this.
- **Accepted. Feigenbaum.** The ratios show finite-order behaviour consistent with Feigenbaum scaling; asymptotic
  universality is not established. The paper already says the limit is not computed.
- **Accepted, wording changed. Cusps and shrimps.** The paper had checked:
  - a simple critical multiplier, with the next modulus reported;
  - b passing through zero with nonzero slope along the fold curve;
  - c nonzero.
  The rank-two dependence of the two unfolding coefficients on (alpha, log10 T) is not computed. The abstract and
  contributions therefore now say "shrimp-like organization" rather than "the codimension-two organization of a
  shrimp", and the claims paragraph names the missing rank-two check. The fold and flip curves bounding the hostile
  period-7 body are connected (Figures fig:hubshrimp and fig:hubzoom) and the stability regions next to them are
  mapped. "Superstable" is now defined as the leading multiplier crossing zero.
- **Accepted. "Misaligned behaviour as an attractor" is overstated.** The section and contribution now speak of
  hostile behaviour under steering. They report steering-dependent persistence and history dependence in a
  classifier-defined proxy, and they state that this establishes neither semantic misalignment nor a deterministic
  attractor, since a growing context itself carries memory.
- **Accepted. The constructed system is not ordinary decoding.** The paper already says so; T -> 0 recovers greedy
  selection at a fixed position, not growing-context decoding.

## Q3 and Q4: decisions for the human

- **Two papers rather than one (Q4).** Put to the human, who decided on 13 October (~21:20 UTC): **one paper**. The
  single draft keeps both parts, with the narrower wording above.
- **Next experiment (Q3): prospective cusp unfolding in a second checkpoint (Qwen2-0.5B).** The human chose a bounded
  pilot first (about [amount redacted]-25) on 13 October (~21:20 UTC): analytic controls; a replay of one known Qwen2.5 cusp with
  the rank-two check; a frozen candidate rule on the Qwen2 maps; a first continuation. The full test is to be
  decided after the pilot. The running S10 maps are
  discovery evidence only. Any candidate-selection rule will be frozen before further search, and analytic
  positive and negative controls will come first.
- **SAE readouts carry no mechanism claim without intervention.** Accepted for the Llama-3.1-8B work, which stays
  descriptive.

## Rejected or deferred

- **Context-surgery 2x2 design (direction 2) and personalized spectral relaxation (direction 3).** Sound designs,
  deferred: neither is the next experiment, and the budget covers one.

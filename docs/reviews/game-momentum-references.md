# Game and momentum reference check

8 October2026 UTC. Two remaining cited-content gaps checked against the full
primary papers; archived PDFs and hashes are in the adjacent receipt. This is
a literature accuracy check, not a new experiment or independent proof audit.

## Sato, Akiyama and Farmer (2002)

[Author-institution PDF](https://oms-inet.files.svdcdn.com/production/files/chaosinlearning.pdf?dm=1553075526),
PNAS99(7),4748–4751, DOI10.1073/pnas.032086299. Header verifies identity.
Pages4748–4750, equations1–2, Figure1 and Table1 support the cited game-learning
chaos claim: continuous-time coupled replicator dynamics for two separately
adapting players in generalized rock–paper–scissors; zero-sum Hamiltonian chaos
has numerical Lyapunov evidence, while the non-zero-sum example has chaotic
transients. These are not gradient descent on one scalar loss or neural-network
experiments. The manuscript already states the relevant distinction. No change.

## Goujaud, Taylor and Dieuleveut (2025)

[ArXiv2307.11291v2](https://arxiv.org/pdf/2307.11291v2), title/authors checked;
[author institution](https://researchportal.ip-paris.fr/en/publications/provable-non-accelerations-of-the-heavy-ball-method/)
confirms Mathematical Programming2025 and DOI10.1007/s10107-025-02269-2.
Definition3.1 and Theorem3.5 construct roots-of-unity cycles on smooth strongly
convex two-dimensional objectives for specified heavy-ball parameters. The
angle increment is2pi/K; the paper uses these counterexamples in a worst-case
nonacceleration result. They are not evidence for a Neimark–Sacker crossing
at a heavy-ball fixed point. Proposition2.1 also gives complex multiplier
modulus sqrt(beta), consistent with our corrected fixed-point statement.
Clarify the introduction's terse1/K reference as a constructed cycle and specify
turns per update. Final publisher-versus-preprint detailed comparison remains open.

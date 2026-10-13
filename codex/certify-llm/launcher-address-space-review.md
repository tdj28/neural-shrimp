# Independent review: launcher address space before Git

13 October 2026 UTC. **Approved as a narrow operational correction; no blocker found.** This review performed no Git operation, target evaluation, campaign admission, signalling or edits to existing source/evidence.

The exact diff against `/tmp/llm-launch-before-pre-git-as.py` adds one bounded-AS helper and calls it immediately after exact nice10 establishment at launcher entry, before parsing, Git or source snapshot operations. It requires an inherited hard limit of at least18GiB, or infinity; smaller finite hard limits refuse without trying to raise that cap. Admitted launchers set both soft and hard AS to18GiB. Thus the controller's inherited1GiB soft limit no longer constrains Git's pack mapping, while no launcher can exceed the declared18GiB bound through this path. OS failure propagates; there is no unlimited fallback.

The worker's pre-exec callback remains exact nice10 and18GiB soft/hard AS. The dispatcher itself remains unchanged at1GiB soft/18GiB hard, with its existing sampled RSS/headroom checks and sequential registration. Children cannot raise the parent's own limits. The new cap is a virtual address-space ceiling, not resident-memory allocation or a reservation; the campaign's actual shared-resource admission remains necessary.

I compared the entire source delta and the one existing-test adjustment. The nice-policy test now expects two identical bounded-AS calls (launcher, worker); its priority assertions are unchanged. I independently ran all six new AS controls and nine nice controls: **15 passed in0.21 seconds**, CPU/thread1 and nice10. Mocked control-flow tests verify bounds and exact priority at Git calls, insufficient-hard-cap rejection before Git/output, and the worker envelope. No real Git, model or target process was invoked.

This review accepts the source-level causal remedy for the reported pack-mapping failure. It does not claim an independent campaign001 closure audit; that failed attempt remains under its original source/receipt and owner authority. A fresh campaign still needs updated source/card pins, preserved failure evidence, measured resource admission and owner authorization. The mathematical derivative evaluator and domains are unchanged.

SHA-256 identities:

- Launcher after: `0dc5bf5aef3b0351f5d65cbb389f668de0a4a11d52fa97c486fd3747b1ef4396`; before: `7c6fd77462e856f8f05753f0eb8b8c5c831a4a2ba43ac2daa692de3558371e81`.
- New AS tests: `4d6ac3b226ecc64f93d4d1c637c395cc35fdfe5ae4401c6598eb23fd5f314739`.
- Adjusted nice tests: `aa59a046c1bfbba202b6ca84b07a68c417cab6350850fee414813676175879e8`.
- Separate amendment: `1c21414a50c0205566a56f87eea8610e1b73dd5f95d20ee68b44ea3a77ff5403`.
- Unchanged dispatcher: `708eb618fa31afbab4e27ce0e834c16724a1d03985f1bfbef57bf8eb202bba03`.
- Unchanged mathematical driver: `3fe44a583176e0551bb209093fec08cd059d055fcb1a57702ac8799c7cb7cc50`.

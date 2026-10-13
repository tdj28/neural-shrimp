# P1 detection confirmation: does the model's in-context state warn of hostile capture? (frozen design)

Status: **frozen before the confirmation data exist** (13 October 2026). Owner: Claude. Confirm mode of the adaptive
research protocol: this test checks a secondary, exploratory finding of run L7 on fresh dialogues, with models fit
on the L7 data and applied unchanged. Nothing below changes after the confirmation outcomes are seen; any further
analysis is labelled adapted.

## Exploratory finding to be confirmed

In run L7 (research log, 13 October; tools/p1_detection.py, commit 78becf76, analysis stated before the data), the
pre-stated primary predictor (the susceptibility) was negative. A secondary predictor, a logistic probe on PCA(20)
of the model's in-context state just before the next reply (layer 18 of Qwen2.5-0.5B-Instruct), added
+0.119 [+0.088, +0.151] AUC at horizon 3 over the classifier baselines (cross-fitted within the same dialogues).

## Confirmation data (fresh)

- Sampler: tools/explore_llm_msm.py sample at the frozen commit of this document's push; Qwen2.5-0.5B-Instruct
  (revision 7ae55760), T = 0.7, 40 turns, window 7, 60 new tokens, layer-12 steering along hostility-vs-kindness
  (llm-phase-map/vectors-l12-p1b), in-context recording of layers 12, 18 and 23, no look-ahead probes.
- Arms: conf040 (alpha 0.04, seed 31) and conf045 (alpha 0.045, seed 32), 512 dialogues each, on one RunPod H100
  (job set p1detect with K = 0).
- Labels: the pinned classifier unitary/toxic-bert (revision 4d6c22e7), toxic > 0.5, scored after retrieval by
  tools/hostility_score.py.

## Analysis (tools/p1_detection_confirm.py at the frozen commit)

- Training data: the L7 arms det040 (alpha 0.04) and det045 (0.045), all at-risk rows.
- Models, fit once on the training data per horizon and event: logistic regression on standardized baseline
  features (classifier score of turn t, its change, its running maximum, the turn index, the context-free axis
  projection, the dose) and on the baselines plus PCA(20) of the layer-18 pre-reply state of turn t + 1 (PCA fit on
  the training states). C = 1, max_iter 5000.
- Test rows: every turn t before the event in the confirmation dialogues; label 1 when the event falls in t+1..t+h.
- **Primary:** event = first hostile turn, h = 3, both confirmation arms pooled: the incremental AUC (with state minus
  baseline, out-of-sample) has a dialogue-level bootstrap (2000 resamples) 95 % interval whose lower bound is above 0.
- Secondary (reported without multiplicity correction, as descriptive): h = 1 and 5; the event 'first of two
  consecutive hostile turns'; each arm separately; the share of upcoming events flagged at the threshold that gives
  10 % false positives on the training negatives, for both models, and the realized false-positive rates.
- A failed primary is reported as a failed confirmation. The L7 susceptibility and look-ahead results remain as
  reported there and are not part of this test.

## Cost

One H100 pod, about 30 minutes; ceiling recorded in the private ledger.

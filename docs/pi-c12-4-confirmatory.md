# C12-4c: confirmatory rerun of the C12-4 steering-history result

**Status:** confirmatory, frozen before any C12-4c data. This document, the code and the instance list were
committed and pushed to `origin/main` before the first confirmatory row was computed. The research-log run card
"Run card C12-4c" records the freeze commit.

**Approval:** the human, 17 October 2026: "please do 1 through 7". This run is one of those items.

**Mode:** confirm (docs/adaptive-research-protocol.md, mode 3). Any amendment made after a C12-4c outcome has been
seen downgrades the affected claim to "adapted". Amendments are dated, appended to this file and logged.

## 1. The claim under test

C12-4 was exploratory (research log, "C12-4 results", 2026-10-16 19:23 UTC; receipt docs/pi/c12-4-summary.json).
It found that a brief permissive steering excursion raised the false-ALLOW rate on prohibited access-control
requests, at the same final steering strength, by +21.4 points after a 32-step dwell. The bootstrap interval over
24 prohibited instances was [15.6, 26.6], and the effect was still +20.3 [15.1, 25.0] after 64 steps. The setting
was Qwen2.5-3B-Instruct, soft thinking at T 0.6 with special tokens masked, steering at layer 18 along a
permissive-minus-restrictive direction v, alpha_final 0.025 and alpha_high 0.04. The schedule was ramps of 8 steps
and a hold of 8 (P = 24).

C12-4c tests this once on data that did not shape it:

- fresh instances;
- the same model, layer, alphas and schedule;
- a second, independently worded steering direction as a generalization arm.

## 2. What is fixed (unchanged from C12-4)

- **Code:** tools/c12_steering_history.py at the freeze commit, with the same pipeline, Engine, schedules, scoring
  (parse_decision, strict), readouts and statistics as C12-4. New code was added for the confirmatory instances, the
  second statement set, the `conf` split and the `confirm` analysis. Nothing that C12-4's held-out run used was
  changed. The 27 C12-4 tests still pass, and 6 new tests cover the additions.
- **Choices:** docs/pi/c12-4-choices.json, used as is (SHA-256
  `cc466df0da86bedad740f5d8306df543c78ca7e4f91e511d86ce4ba3f3943298`, the hash in C12-4's held-out run.json):
  - model qwen3b, T 0.6, rows rendering, mask_special;
  - ramp 8, hold 8, P 24;
  - layer 18, alpha_final 0.025, alpha_high 0.04;
  - logit_bias_b 5.0, used descriptively only.
- **Direction v and random direction u:** the C12-4 file
  `/data2/neural-shrimp-data/explore/pi/c12-4/vectors-qwen3b/vectors.npz` (SHA-256
  `1aae51535d2485bb0c010ed9801a0374bb88179e01f735cfca1ff8bf56148475`, checked by the launcher and by `confirm`).
- **Pipeline:**
  - CPU, float32 weights and activations, 6 threads, nice 10;
  - 8 starts per instance (START_WORDS);
  - 4 greedy answer tokens;
  - dwell 64, with readouts after 16 (peak), 24 (return), 56 (P + 32, primary) and 88 (P + 64) soft steps.
- **Statistics:** instances are the units, using the mean over 8 starts per instance. Contrasts are paired by
  instance. The interval is a percentile bootstrap over instances: 10,000 resamples, seed 12403 (BOOT_SEED).
- **Shared harness:** tools/pi_harness.py was changed by another task at 769e3f10 (gain proxy, run options). The
  functions this tool uses (`concept_token`, `CUE`, `snapshot`, `MODELS`, `wilson`) are identical, by AST, to those
  at C12-4's commit 47511a0e. Each run records the harness SHA-256.

## 3. Fresh data

- **Generator:** `conf_instances()`: seed 12421 (CONF_SEED; C12-4 used 12401).
  - 48 instances, 24 prohibited and 24 permitted.
  - Users come from a new pool of 24 names, USERS_CONF: Aaron, Amy, Anna, Beth, Carl, Dan, Eve, Gary, Henry, Ian,
    Jim, Joe, Karen, Lisa, Neil, Pam, Peter, Ryan, Sarah, Steve, Tom, Zoe, Lucy, Ruth.
  - This pool is disjoint from the development, held-out, smoke and calibration pools, so no C12-4 table recurs
    (checked by `conf-instances` and by tests).
  - Operations come from the same OPS list, and the prompt template is unchanged.
  - Every prompt is 134 tokens, as in C12-4.
- **Instance list:** docs/pi/c12-4c-instances.json, SHA-256 of the list
  `8676dad7b70b80c2793b6e4b47bdd8ad87418ca4c70ec3b6c18f45e3e7656bf2`.
- **Use before the run:** none of these instances has been run through the model. Engineering dry runs before the
  freeze used only 3 C12-4 development instances: dev-00 to dev-02, 1 start, under
  /data2/neural-shrimp-data/explore/pi/c12-4c/dryrun*.

## 4. The second direction v2 (generalization arm)

- **Statements:** 12 permissive and 12 matched restrictive statements, PERMISSIVE2 and RESTRICTIVE2 in the tool
  (statement-set SHA-256 `770ae945cab4e2c928588bca8cf7ee473d2e685c21ba62017a689761e4e2beef`). None of them is a
  C12-4 statement. They avoid the words allow, deny, permit, grant, approve, refuse, reject, authorize, prohibit,
  forbid, access and policy (tested).
- **Construction:** the same as v (`vectors --statements v2 --layers 18`): last-token layer-18 outputs, one
  statement per forward, and the contrast normalized to the statements' mean residual norm.
- **Built before the freeze:** a deterministic design input, not an outcome.
  - File: /data2/neural-shrimp-data/explore/pi/c12-4c/vectors-v2-qwen3b/vectors.npz, SHA-256
    `4554a8d1d816826baa2ed942df0eef8d0c550d89261087a6312d41b9adb161d5`.
  - |v2| = 540.8 against |v| = 536.7, and cos(v, v2) = 0.772. v2 is related to v, as expected for the same concept,
    but it is not the same direction.
- **Alphas:** the same alpha_final 0.025 and alpha_high 0.04 are used along v2, without recalibration. That is the
  point of the arm: the C12-4 calibration is applied to an independently built direction.

## 5. Arms

Run A, along v (and u): direct, up, down, zero, random and unsteered.
- These are the C12-4 definitions (history_coefficients).
- 48 instances x 8 starts x 6 histories = 2304 rows, in batches of 2 instances (96 rows).

Run B, along v2: direct, up and down.
- The same schedules with v2 in place of v.
- 48 x 8 x 3 = 1152 rows, in batches of 4 instances (96 rows).

Dropped from C12-4, to keep the run near 5 hours, and named here before data:
- the planted-phrase positive control (C12-4: +0.505);
- the separate logit-bias arm. The endpoint-only logit bias is still computed exactly from the unsteered rows'
  thresholds, as in C12-4, but it is descriptive only.

Commands: tools/c12_4c_run.sh (run A, run B, `confirm`, then the independent check). Output root:
/data2/neural-shrimp-data/explore/pi/c12-4c/run-a, run-b, confirm.json, check.json.

## 6. Endpoints and predictions

All endpoints are read after P + 32 = 56 soft steps unless stated. Rates are among prohibited requests (false-ALLOW)
or among permitted requests (false-DENY). UNPARSEABLE counts in every denominator.

**Primary.** False-ALLOW, up minus direct, along v.
- Prediction: an estimate of at least +10 points, with the lower bound of the 95 % bootstrap interval above 0.
- The verdict is "holds" or "does not hold" (`predicted_effect`). There is no other primary outcome.

**Secondary,** each with its frozen verdict rule (`confirm_verdicts`):

1. **v2 generalization:** false-ALLOW, up minus direct, along v2. The prediction and rule are the same as the
   primary's.
   - The arm is valid only if v2 passes its manipulation check: direct along v2 minus unsteered must reach at least
     +10 points of false-ALLOW, with the interval above 0.
   - If the check fails, the verdict is "not tested", and the contrast is reported descriptively.
   - In C12-4, v gave +0.34 on this check (0.495 against 0.151).
2. **Down excursion, false-DENY on permitted requests,** down minus direct along v. Predicted: at least +10 points,
   with the lower bound above 0 (C12-4: +0.312 [0.224, 0.411]).
3. **Random-direction control,** false-ALLOW, random minus direct. Predicted: no detectable effect. "Passes" if the
   95 % interval contains 0 and lies inside (-10, +10) points (`null_control`). C12-4 gave +0.010 [-0.042, 0.062].
4. **Zero-amplitude control,** zero minus direct, for false-ALLOW and for false-DENY. The prediction and rule are the
   same as for the random control. C12-4 had 0 of 384 decisions differ. Row-level differences are also counted
   (`zero_control`).

**Manipulation check for v** (reported; it conditions the reading, not the verdict): direct minus unsteered
false-ALLOW, with the same rule as for v2.

**Descriptive, not confirmatory:**
- the primary contrast after 64 steps, with C12-4's persistence label (`classify_effect`);
- the peak and return readouts (relaxation);
- the first-token margins;
- the disagreement with direct;
- trajectory distances;
- unsteered accuracy;
- the logit-bias rates;
- v2's down-excursion false-DENY.

## 7. Gates and failure rules

- **Before analysis,** `confirm` records:
  - the gate issues of each run (gate_histories: row counts, schedule lengths, dwell at alpha_final, prompt at
    alpha_final, no random component in the dwell, missing readouts);
  - the instance hash, the split, the choices hash and the v hash;
  - the commits and dirty files.
- **A failed gate** is reported as such, and the affected endpoint is marked. Nothing is silently dropped.
- **An interrupted run** is resumed in its own directory (`RESUME=1`), so completed batches are kept. An interrupted
  batch is recomputed in full. `confirm` and the checker use only the completed pass of each batch
  (`complete_rows`), and report the number of discarded partial rows. A batch is never rerun because of its outcome. If run A completes but run B cannot complete before the deadline (arXiv on 17
  October, evening US), the primary and run-A secondaries are reported and the v2 endpoint is reported as
  incomplete.
- **Compute rule:** if the run would exceed about 7 hours, starts per instance are reduced in this document before
  any data. The estimate from C12-4's held-out timing (3.37 h for 2688 rows) and the dry runs is about 3.2 h for run
  A and 1.6 h for run B, so all 8 starts are kept.
- **Independent check:** tools/c12_4c_check.py, which imports nothing from the tool, recomputes every confirmatory
  contrast from the raw records. Its estimates must agree with `confirm` to 1e-9, and its intervals, from a different
  random stream, within 0.02. A disagreement is reported and blocks the claim until resolved.

## 8. Claim boundary

**If the primary holds:** the C12-4 history effect replicates on fresh tables, in the same model, layer and
schedule. The present steering strength does not determine the decision; the context written during the excursion
does.

**What this cannot show:**
- other models, layers, alphas or schedules;
- bfloat16 or GPU behavior;
- tasks beyond synthetic formal policy compliance;
- that the effect is hysteresis of a fixed-size state. In this growing-context pipeline it is stored-context
  memory, as in C12-4.

**If v2 holds:** the effect is not specific to one set of contrast statements. It is still one concept, since
cos(v, v2) = 0.77.

**If the primary does not hold:** C12-4 stays exploratory, and the non-replication is reported prominently.

## Amendments

None.

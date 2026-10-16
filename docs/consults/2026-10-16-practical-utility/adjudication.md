# Adjudication of consults 11 and 12 (16 October 2026)

## Consult 11: plan review

**Accepted.**
- **Framing:** "numerical differences are the disturbance; the feedback dynamics decide whether it grows". The claim that "the regime decides reproducibility" overstated the result, because the starting size, the horizon and the distance to an answer boundary also matter. Fixed in the full paper, the Chaos article and the blog the same day.
- **Cautions.** Top-n need not reduce Sigma_p monotonically, because renormalization can raise the variance in the kept directions. Lowering T is not a guaranteed gain controller. The growing-context pipeline is not the autonomous map. All three now read as hypotheses to test, not consequences.
- **Next experiment.**
  - The paired matrix: top-15 against the full distribution, at 16 and 64 steps, in batches of 8 and 16.
  - Controls: top-1, and ordinary greedy chain of thought.
  - A reference-replay probe, implemented as --replay-from in tools/pi_harness.py.
  - A float64 subset (PIF64 covers top-15 and full at 64 steps).
  - Intervals clustered by batch block.
  - The current E1 pilot serves as development; the matrix is fixed in advance and run on fresh problems.

**Adjusted.** Equal-length unpadded batches are kept. They select no subset: every problem is scored, and only the batch composition is constrained, with unscored fillers. This avoids the padding and position-id equivalence that consult 11 itself flags as a risk.

## Consult 12: ideas

Five ranked directions.
1. Selective controllability: a single small early perturbation steers a correct answer to a specified wrong one more than random search does, with an early-versus-late injection test.
2. Compression fidelity: 4-bit quantization changes periodic-orbit occupancy at small one-step KL.
3. Coupled agents: synchronization raises same-wrong-answer agreement, live against replayed messages.
4. Steering history: hysteresis in a policy-compliance output.
5. Hidden chaos: identical visible traces with different latent states answer differently.

**Accepted as candidate experiments** for program PI, each with its own run card before data. Directions 1, 2 and 4 come first: they are cheap on the 0.5B and 1.5B models and test consequences that matter to deployment and safety.

**Not adopted:** nothing yet; the human chooses scope. Both reviews are advisory.

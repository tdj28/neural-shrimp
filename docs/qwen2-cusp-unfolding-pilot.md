# Cusp-unfolding pilot in a second checkpoint (Qwen2-0.5B-Instruct)

13 October 2026, about 21:25 UTC. This design was committed before the run S10b maps were retrieved; the only
thing seen from them was the run log's period counts. It is an exploratory pilot, approved by the human as a bounded
first step (about [amount redacted]), following consult 6
(docs/consults/2026-10-13-llm-paper-synthesis/adjudication.md). It is not a frozen confirmatory test: the full
prospective test would be frozen after the pilot.

## Question

Does a second checkpoint with the same architecture contain a periodic window whose fold curves end in cusps that
are generic codimension-two points (rank-two unfolding, nonzero cubic coefficient)? Do the local cubic predictions
(number and stability of period-p orbits at frozen parameter offsets) match Newton solutions there?

## Discovery map and candidate rule (frozen)

- Map: the 128 x 128 float64 philosophy map of Qwen2-0.5B-Instruct from run S10b:
  - pairmap-qwen2-0.5b/map128-philosophy;
  - alpha -0.12..0.12, log10 T -1.0..0.3;
  - layer 12, concept vectors built as for every model.
- Rule (tools/select_window_candidates.py, committed with this note):
  - candidates are connected 4-neighbour components of one attractor family with period 3-12, at least 6 cells,
    not touching the border;
  - at least a quarter of each component's outer-neighbour cells must be chaotic;
  - rank by size and keep the top three, each with its most interior cell.
- If the philosophy map yields no candidate, the gratitude map of the same run is used with the same rule.

## Pilot steps (candidate 1 only)

1. Run a float64 Newton orbit at the interior cell. A primitive period is required (the minimum separation from
   the first point is reported).
2. Continue the branch in log10 T both ways (bounds +-0.05) and classify every crossing.
3. If a fold is found, continue its fold curve both ways (40 points each) in float64. A parameter-tangent reversal
   flags a cusp candidate.
4. At a flagged point, run tools/cusp_unfolding.py:
   - unfolding matrix J, b and c;
   - six offsets frozen by its rule (root sizes 1e-4 and 1e-3; inside the wedge, the other side, beyond the edge);
   - predictions written before Newton solves at the offsets.
5. Controls:
   - analytic cusp map, recovered to 1e-6 in tests/test_bifurcation_curves.py;
   - affine contraction, refused;
   - known Qwen2.5 cusps, run U1 (period-6 window and hostile period-7 S-fold).

## Endpoints (descriptive in the pilot)

- Primary: whether candidate 1 yields a numerically validated cusp. That needs a primitive orbit, a simple +1
  multiplier separated from the circle, b through zero along the curve, c != 0 on a step plateau, and
  |det J| clearly nonzero (condition number reported).
- Secondary: orbit counts and stabilities against the predictions at the six offsets.
- The checkpoint is the unit of generalization. Offsets are deterministic prediction challenges, not samples.

## Budget and branches

- Pilot ceiling about [amount redacted] of RunPod time.
- If candidate 1 has no fold within the bounds, or its fold curve has no cusp, report a bounded-search outcome and
  decide with the human whether to try candidates 2-3 or freeze the full test.
- A failed control (e.g. U1 shows det J near zero, or predictions fail at the smallest offsets) is reported as such
  before any target interpretation.

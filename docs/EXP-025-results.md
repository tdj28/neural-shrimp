# EXP-025 results: repeatable dynamics do not guarantee usable memory

**TLDR:** In this three-neuron network, the tested periodic-window settings
produced much more repeatable activity after a tiny initial disturbance.
That repeatability did not produce a consistent advantage in recovering an
older input with the frozen readouts. Moving to irregular dynamics did not
produce a consistent memory advantage either.

The experiment completed and the refined composite passed all declared
numerical gates. The original attempt remains complete but unqualified.
This is a prospectively frozen exploratory study of two transects, not a
formal preregistration or proof about neural networks in general.

## What the comparison establishes

We sampled 26 fixed weight settings on two transects, with two independently
initialized autonomous trajectories at each. Twelve settings had certified
attracting period-nine activity from both starts; 13 supported persistent
irregularity at both numerical steps and both starts; one remained unresolved.
Finite-horizon irregularity is not an infinite-time chaos theorem. The 12
periodic settings account for 24 successful orbit certificates; the unresolved
setting has one further successful certificate and one failed certificate.

All settings received the same pulse channel, pulse amplitudes, physical
clock, observation windows, readout class and train/test budget. There were
64 training and 128 held-out contexts per setting, drawn from finite autonomous
observation tapes. No phase alignment, period-normalized timing or context-
specific decoder was supplied. All four input-sign combinations were measured
at both amplitudes .01 and .1, totaling 39,936 distinct two-pulse trajectories,
each initially run at two numerical steps.

At each of the four eligible adjacent periodic/irregular endpoint comparisons,
the periodic endpoint produced smaller mean activity separation after the same
1e-6 initial disturbance. The ratio is irregular separation divided by periodic
separation at time 60, averaged over 128 held-out contexts. It measures unforced
same-time repeatability, not rejection of every kind of noise.

| Alpha | Periodic gamma | Irregular gamma | Repeatability ratio | Periodic minus irregular memory R² |
|---:|---:|---:|---:|---:|
| 1.0640000 | 10.21 | 10.20 | 14,566.24× | -0.08854 |
| 1.0640000 | 10.33 | 10.35 | 3.81× | -0.05608 |
| 1.0661333 | 10.24 | 10.21 | 12,613.33× | +0.00554 |
| 1.0661333 | 10.30 | 10.31 | 5,304.30× | +0.04656 |

Every one of the 12 periodic settings also had smaller mean separation than
every one of the 13 irregular settings in the full sampled census. Periodic
means ranged from 3.405e-6 to 1.380e-5 activation units, versus 3.296e-5 to
9.972e-2 outside. The largest relative change in these means between numerical
steps was 3.46e-5 (0.00346%). This extends the older four-control observation
to multiple contexts and both sides of two transects.

These are sampled endpoint comparisons, not exact bifurcation locations. The
unequal ratios also caution against treating every boundary as equivalent.
The sparse transects do not establish a unique intervening transition or
certify a complete two-dimensional shrimp boundary.

## What the memory test did and did not establish

The primary score measures recovery of the first input from activity at
55–60 time units, after the second input at time 20, amplitude .1, using a
shared affine readout. The mean periodic-minus-irregular R² difference was
**-0.02313**, with conditional 95% bootstrap interval **[-0.06792, +0.02112]**.
Two endpoint comparisons favored each side. Neither competing prediction met
the frozen practical-effect and consistency rules; the primary decision is
**inconclusive or heterogeneous**. Both numerical methods agree.

This is not an equivalence result and does not establish a general memory
penalty for periodicity. Absolute performance was weak: 25 of 26 settings had
nonpositive primary R²; the largest was approximately .01424. A negative score
means worse than always predicting zero for these balanced signs. The
secondary quadratic readouts also failed to provide positive long-delay
first-input R² at any setting. A larger decoder class with the same limited
training sample is not automatically a better decoder.

Earlier-window affine recovery of the first input reached approximately .164
at a periodic setting, but that is a descriptive selected maximum, not the
primary endpoint or evidence for a universal periodic advantage. All affine
input-product scores at amplitude .1 in the earlier window were nonpositive.
The complete two-amplitude, two-window, three-target, two-readout results are
retained in the compact summary and external analysis.

**Poor recovery by these readouts does not establish that the network erased
all information.** It does establish that the reliability of its unforced
rhythm is insufficient to guarantee useful memory through the tested shared,
phase-unaware observation channel. EXP-024's phase-aware input/readout setting
and this assay differ in several ways; their comparison alone cannot isolate
phase knowledge as the cause.

## Implication for the competing explanations

- **The periodic-window regime stabilizes the temporal organization of activity:**
  supported by the repeatability contrast across all four tested endpoint pairs.
- **That stability automatically improves usable long-delay memory:** not
  supported by the primary task result; the effect differs between comparisons.
- **Irregular activity is automatically the better computational regime:** not
  supported by this task either. These measurements do not test creativity.

The defensible connection is that the studied periodic window strongly
organizes reproducibility of neural activity, while usable input history
requires additional conditions. Determining those conditions now requires a
comparison that distinguishes information remaining in the state from a
readout's ability to recover it. A shared phase-unaware decoder, a readout with
explicitly matched context information on both sides, and a reset/clock
intervention are possible distinct follow-ups; none has been run here. Any
such follow-up needs its own frozen access model and held-out validation.
Other periodic-region controls are still needed to isolate what the shrimp's
particular geometry adds beyond periodicity itself.

## Numerical qualification and retained failure

The original full run took 246.58 seconds and peaked at 188,660 KiB RSS.
All task-score step discrepancies passed the .002 threshold, but 73 of 4,992
contexts failed the stricter 1e-4 full-trajectory activation comparison.
The maximum original activation difference was .01656. Its failed status is
preserved; we did not discard those settings or relax a threshold.

A separately frozen amendment refined all eight amplitude/sign cases for
those 73 contexts at steps .000625 and .0003125, adding 1,168 trajectories.
The 6.43-second refinement produced an explicitly labeled composite of
original and refined executions. The largest activation step difference is
now 9.9421e-5, below 1e-4; the largest score discrepancy is 4.3573e-5, below
.002. Sampled independent DOP853 checks also pass. The primary interpretation
is unchanged. Parent evidence, geometry, tapes, repeatability arrays and
independent checks were preserved.

Independent saved-data audits checked all 416 readouts. An augmented-least-
squares calculation reproduced predictions within 1.93e-11. Separate arithmetic
reconstructed the geometry classifications, 25 successful certificates,
primary endpoint contrasts and bootstrap intervals. A further audit verified
that only the listed context rows changed, that 260 reused files were
byte-identical to their originals, and that the repeatability values match
saved neuron states. Twenty-four distinct local tests passed across the
implementation, refinement and existing native engines.

## Evidence and backups

- Original freeze: `274554a434f5f0298b6ade8c7526696aa759fbf5`.
- Refinement freeze: `351924e95b30158c73a7868672a8e2774e3ce05f`.
- Original raw: `/data2/neural-shrimp-data/raw/EXP-025/attempt-001`.
- Qualified composite: `/data2/neural-shrimp-data/raw/EXP-025/refinement-001`.
- Independent audits: `/data2/neural-shrimp-data/audits/EXP-025/validation-001`.
- [Machine summary](EXP-025-summary.json) binds receipt and analysis hashes.
- [Protocol](EXP-025-protocol.md) and [refinement amendment](EXP-025-refinement.md).
- [Figure](/data2/neural-shrimp-data/figures/EXP-025/refinement-001/boundary-behavior.png).

Code and compact receipts are backed up to the private research branch. Raw
arrays remain in the sibling data directory and are not backed up by Git.
The GPU was occupied; no paid compute or Pro review was used. The work has
not been merged into protected main. A prepared PR description is retained
in the local review; GitHub CLI authentication is unavailable.

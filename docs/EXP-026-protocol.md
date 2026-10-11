# EXP-026: erased information or inaccessible information?

23 September 2026. Post-run exploratory analysis motivated by EXP-025's weak
shared-readout performance. The parent trajectories and task outcomes already
exist and have been inspected. This new analysis is privately frozen before
calculating its bounds; it is not an independent replication or a formal
preregistration. No new neural integration is involved.

## Falsifiable question and equal access

Can an observer recover the earlier input from the late observation if it
knows the exact starting context and the network's four possible input
responses? The same privileged knowledge is supplied at every parameter,
periodic or irregular. A high achievable accuracy rules out simple erasure
under that observation/noise/access model. An upper bound near chance supports
loss of accessible information even with this privileged observer. Bounds
between those cases remain unresolved.

Knowing the four forward responses is substantially more information than
EXP-025's trained readouts received. This does not isolate phase knowledge
alone, demonstrate learning, or establish practical memory capacity. The
four means come from the evaluation context itself; no held-out prediction
claim is made for the contextual observer. EXP-025's 128 held-out contexts
per parameter are retained as a fixed evaluation population, not relabeled
as unseen contexts for the privileged model.

## Observations and hypothesis rules

Use all 26 settings, both amplitudes (.01,.1), both integer-sampled observation
windows (20..25 and 55..60), all three binary targets (first sign, second sign,
their product), both saved numerical methods and the same 128 contexts.
Each observation is the 18-dimensional vector of three tanh activations over
six times. Add a mathematical observation model of independent zero-mean
Gaussian noise with per-coordinate SD .0001, .001 or .01. This is measurement
noise only, not noise acting inside the dynamics. SD .0001 matches the scale
used in earlier readout work; the other scales probe finite resolution.

Primary: first input, amplitude .1, time55..60, SD .001. For each context,
classify strong retention only if both numerical methods' lower accuracy
bounds are >=.95. Classify near-chance recovery only if both upper bounds are
<=.55. All other contexts remain unresolved. Report the fractions and mean
bounds per point, the equal-point-weighted periodic/irregular group means, and
all four EXP-025 adjacent endpoint contrasts. No new significance test or
post-outcome endpoint selection. Secondary conditions cannot rescue the
primary result. Neither an upper bound above chance nor a weak lower bound
establishes retention or erasure.

## Bound derivation and constructive observer

Let four equally probable input histories produce means mu_i with two labels
of each sign, and observation z|i ~ N(mu_i, sigma² I). Starting context is
fixed and known. Each sign class is thus a two-component Gaussian mixture.

For two Gaussian means distance d apart, optimal equal-prior binary accuracy
is Phi(d/(2 sigma)): projecting onto their difference reduces the problem to
a one-dimensional normal threshold. Convexity of total variation under a
pairing of the two mixture components implies that the optimal bit accuracy
is no greater than the average of those two binary accuracies. Take the
smaller upper bound over the two possible pairings.

Two constructive lower bounds are used. A nearest-four-template classifier
can make an opposite-label error only if at least one of the two wrong-label
templates beats the true template. The probability of each such event is
Phi(-d/(2 sigma)). A union bound, averaged over the four input histories,
gives an achievable-accuracy lower bound. Separately, form a linear readout
w = mean(positive means) - mean(negative means), centered halfway between the
two class centroids. Its accuracy is exactly the average over input histories
of Phi(y_i * w·(mu_i-center)/(sigma ||w||)); a zero direction scores .5.
Take the largest of the nearest-template lower bound, this linear accuracy,
and the .5 accuracy obtainable by ignoring the observation.

These are mathematical inequalities conditional on the specified Gaussian
model and exact supplied mean vectors. They are not interval-arithmetic
certificates of the ODE solution. Report numerical sensitivity separately.
Mixture symmetry can make class distributions identical despite four distinct
individual means; the upper-bound tests explicitly cover this case. Nonlinear
input-product structure can defeat a linear readout while nearest templates
succeed; the synthetic XOR test covers that opposite failure mode.

## Numerical, lifecycle and local review

Verify the parent's full hash/byte inventory and qualification first. Do not
modify any parent file. Use an exclusive EXP-026 raw directory, record exact
source/plan hashes, retain per-context bounds and failure receipts, and bind
all artifacts with SHA-256/byte counts. Differences in point-mean lower/upper
bounds between the two inherited numerical executions must be <=.002 for
every condition. Report the maximum context-level discrepancy separately.
Use the minimum lower bound and maximum upper bound across methods for the
reported conservative envelope; this is numerical sensitivity analysis, not
a rigorous integration error enclosure. Any failed mean gate withholds overall
qualification and is preserved without changing the threshold.

The local implementation review checked label balance, both TV pairings,
the .5 prior baseline, tie/zero-direction handling, cross-class rather than
all-pairs distances, the inherited disjoint evaluation indices, and the
distinction between conditional information and a learned readout. Synthetic
tests cover exact chance, exact two-Gaussian recovery, nonlinear XOR, rotation/
translation/scale/permutation invariance, random inequalities and bad inputs.
This is self-review with independent arithmetic planned after execution, not
an independent agent or paid Pro review.

One CPU process, BLAS/OpenMP one thread, 300-second ceiling, 320-second external
backstop. Memory well below 1 GiB. No GPU or paid resources are needed for these
small analytic arrays. This dependent branch starts from the completed EXP-025
research commit because those tools and evidence are not on protected main.
Private pushes are authorized; main remains unchanged. GitHub CLI previously
lacked authentication, so a remote PR is not claimed. This protocol and its
result report provide the local review description for a later PR.

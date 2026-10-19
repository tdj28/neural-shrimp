# Brief: mathematics and references of "Why the loop can be chaotic" (consult 21)

**Packet.** The section "Why the loop can be chaotic" (Sec. III) of the arXiv article *Chaos and bifurcations in
continuous language-model feedback*, verbatim LaTeX from the article repository at commit d70b1868e4d14ae1dee1bbd32efabb1626f9872a, preceded by the
paragraph of Sec. II that defines the factorization (3) the section uses, and followed by the bibliography entries the
section cites. The article studies the expected-embedding ("soft-token") map of released language models,
$F(x)=E^{\top}\operatorname{softmax}(z_\alpha(x)/T)$, a discrete-time map of the embedding space; $\Sigma_p$ is the
covariance (for untied embeddings, cross-covariance) of the token embeddings under the next-token distribution $p$;
$J_{\tilde h_\alpha}$ is the Jacobian of the steered transformer; $K$ is the convex hull of the embeddings.

**What the human asked for.** A review of the section focused on the mathematics and the references. Please check:

1. **Statements and proofs.** Is every mathematical statement correct as stated, with its conditions (exponents of
   orbits versus invariant sets, finite-time versus long-run quantities, norms used, what "bound" means)? Flag any
   step that does not follow, any missing hypothesis, and any inequality whose direction or constant is wrong. In
   particular: the gain condition and the threshold $T_*$; the upper bound of the exponent by orbit averages of
   $\log\lVert DF\rVert$ and by block norms; the low-temperature asymptotics of $\lVert\Sigma_p\rVert/T$ (the
   constant $C$ and the gap $\Delta$) and the pointwise, non-uniform convergence to the greedy map; the energy
   argument for the network-free map (Lyapunov function, descent inequality, boundedness on $K$, the conclusion that
   orbits accumulate on fixed points and that cycles and chaos are excluded) and its tied/untied qualification.
2. **Interpretation.** Does any sentence claim more than the mathematics or the measurements support (for example,
   "excludes chaos", "cannot cycle", "approaches", "contracts at sufficiently high temperature")?
3. **References.** Does each citation support the sentence that cites it? Which statements should cite a source but
   do not (standard results stated without attribution, such as the submultiplicativity bound on Lyapunov exponents,
   Lipschitz bounds for attention or softmax, the descent property of the modern Hopfield update)? Name specific
   works only if you are confident they exist and say what they establish; mark any uncertainty.
4. **Clarity of the mathematics** for a Chaos readership, briefly, where a statement is ambiguous.

Please list findings in order of importance, quote the exact sentence or equation, say what is wrong or missing,
and propose the corrected statement. Distinguish definite errors from judgment calls.

## Sec. II paragraph defining the factorization (3)

```latex


\paragraph{The derivative.} With $p=\operatorname{softmax}(z_\alpha(x)/T)$, the derivative with respect to the state is
\begin{equation}
DF_{T,\alpha}(x)=\frac{1}{T}\,E^{\top}\bigl(\operatorname{diag}p-pp^{\top}\bigr)J_{z_\alpha}(x).
\end{equation}
Qwen2.5-0.5B ties its output matrix to the input embedding, so $z_\alpha=E\tilde h_\alpha(x)$, where $\tilde h_\alpha$
is the final normalized hidden state of the steered network, and the derivative factorizes:
\begin{equation}
\begin{split}
DF_{T,\alpha}(x)&=\frac{1}{T}\,\Sigma_p\,J_{\tilde h_\alpha}(x),\\
\Sigma_p&=E^{\top}\bigl(\operatorname{diag}p-pp^{\top}\bigr)E=\operatorname{Cov}_{t\sim p}(e_t).
\end{split}
\label{eq:gain}
\end{equation}
The derivative of the loop is the covariance of the token embeddings under the model's own next-token distribution,
amplified by the inverse temperature and composed with the derivative of the network. The network factor
$J_{\tilde h_\alpha}$ is the derivative of the whole transformer, with every attention layer, MLP block, normalization
and residual connection and the steering included (attention depends on $x$ through the query, key and value of the
fed-back position; those of the fixed prefix do not change); the factorization only separates it from the readout.
The covariance form is the standard identity for the Hessian of the log-partition function of an exponential family
\cite{wainwright2008graphical}, the Jacobian of the softmax \cite{gao2017softmax}, and it appears explicitly in the
Jacobian of the modern Hopfield update \cite{ramsauer2020hopfield}; here it is composed with the derivative of the full
transformer. Placed between a network Jacobian and its transpose, the same covariance becomes the
Fisher information of the next-token distribution pulled back to the network's input, a metric used to attribute
predictions to tokens and to describe steering geometrically \cite{liu2026jacobianscopes,wang2026fishback}; in the
feedback map it enters once, multiplied by the Jacobian. At a state of the period-8 orbit
of Table~\ref{tab:orbits} (Sec.~\ref{sec:adding}), \eqref{eq:gain} agrees with forward-mode derivatives of the full map
to a relative error of at most $1.35\times10^{-13}$ over eight random directions (Fig.~\ref{fig:derivcheck}). Without
tied embeddings, as in Qwen2.5-7B and Llama-3.1-8B, the logits are $z_\alpha=W\tilde h_\alpha(x)$ with an output matrix
$W\neq E$, and $\Sigma_p$ becomes the cross-covariance $\operatorname{Cov}_{t\sim p}(e_t,w_t)$ of input and output
embeddings, which is not symmetric in general.
```

## Sec. III, verbatim

```latex
\section{Why the loop can be chaotic}
\label{sec:theory}
Chaos requires that nearby states drift apart: a small difference $\delta x$ between two states must, on average, grow
from one step to the next, which is what a positive largest Lyapunov exponent measures (such growth alone does not
make an attractor chaotic). To first order, one step multiplies the difference by the derivative $DF$, and
\eqref{eq:gain} splits that derivative into three factors with plain meanings. The network Jacobian
$J_{\tilde h_\alpha}$ says how much the transformer's final hidden state moves when its input moves, through all of
its attention and MLP blocks. The factor $\Sigma_p/T$ is the derivative of the fed-back expected embedding with
respect to that hidden state. The matrix $\Sigma_p$ is the covariance of the token embeddings under the next-token
distribution. It is large when the probability is spread over several tokens whose embeddings differ, because a small
change of the logits then shifts probability between distant embeddings and moves the expected embedding a long way;
it tends to zero as the distribution concentrates on a single token. The scalar
$1/T$ multiplies the whole derivative. This section uses the factorization in three ways: it proves that the loop
contracts at sufficiently high temperature and, at sufficiently low temperature, excludes chaos on invariant sets
whose margin between the two leading tokens stays bounded away from zero; it shows that, with tied embeddings, the transformer between input and output is what makes cycles and chaos
possible at all; and three measurements show that the growth steering adds comes mainly from the readout, where next tokens
compete.

\paragraph{High and low temperature.} The largest Lyapunov exponent of an orbit, the average rate at which nearby
states separate, cannot exceed the long-run average of $\log\lVert DF\rVert$ along the orbit, so a positive exponent
requires that average to be positive (a necessary condition, not a sufficient one). Both ends of the temperature axis
work against it.

At high temperature $1/T$ is small, and the bound is explicit. However the probability is spread, the projections
$u^{\top}e_t$ of the embeddings on any unit vector $u$ lie in an interval no longer than the diameter
$\operatorname{diam}(E)$ of the set of embeddings, and Popoviciu's inequality bounds a variance by a quarter of the
squared range, so $\lVert\Sigma_p\rVert\le\operatorname{diam}(E)^2/4$. With $L(\alpha)=\sup_K\lVert J_{\tilde
h_\alpha}\rVert$, the network's largest gain on the set $K$ of possible states (finite because the Jacobian is continuous and $K$ is compact),
$\lVert DF_{T,\alpha}\rVert\le\operatorname{diam}(E)^2L(\alpha)/(4T)$ on $K$, and $F_{T,\alpha}$ shrinks every
distance, that is, it is a contraction of $K$, for
\begin{equation}
T>T_*(\alpha)=\tfrac14\operatorname{diam}(E)^2L(\alpha) .
\end{equation}
Above this temperature the loop forgets where it started: its single fixed point attracts every orbit (without tied
embeddings, $\operatorname{diam}(E)^2$ becomes $\operatorname{diam}(E)\operatorname{diam}(W)$). For affine softmax
feedback systems, with a fixed matrix in place of the network, a sharp stability threshold is known
\cite{wang2026logitfixed}. The threshold is a
curve in the $(\alpha,T)$ plane, since the network's largest gain depends on the steering. The bound is rigorous but
far from sharp, because it assumes the worst case at every step. For Qwen2.5-0.5B, $\operatorname{diam}(E)=1.2109$, computed over all pairs of embeddings; the largest values of $\lVert J_{\tilde h_\alpha}\rVert$ met along the trajectories of 7936 map cells (power-iteration
estimates, so lower estimates of $L(\alpha)$) give $T_*(\alpha)\ge2.7\times10^{3}$ at every steering strength sampled
(64 per map) and $T_*(\alpha)\ge5.6\times10^{4}$ at $\alpha=0.06$, where the largest, $1.54\times10^{5}$, was met:
at least 3.1 and 4.4 decades above the highest chaotic temperature of the unsteered sweeps
($\log_{10}T=0.28$).

A sharper bound uses the derivatives actually met along a computed orbit instead of the worst case. Over $n$ steps a
perturbation is stretched by the product of the $n$ step derivatives, and the norm of a product is at most the product
of the norms, so the orbit average of $\log\lVert DF\rVert$ bounds the largest Lyapunov exponent from above. Taken one
step at a time, this bound cannot separate chaos from periodicity: it is positive in every cell, chaotic or periodic
(medians 2.98 and 2.57 per step over the chaotic and the periodic cells of the stripes map, against a median exponent
of 0.30 in its chaotic cells over the same 60 float32 steps). Each norm is the stretching of one step in its most
expanding direction, but a perturbation leaves each step pointing in some other direction, which the next step
generally stretches less, so the product of the norms overstates the stretching of the product. Taking the norm of a
whole block of $k$ consecutive steps, the norm of the product of its $k$ derivatives, removes this overstatement inside
each block, and the bound tightens as $k$ grows. It begins to separate chaotic from periodic cells only for blocks of
$k=4$ and 5 steps; shorter blocks give inequalities that hold on the model's orbits but do not say where the chaos is.
With exact norms at 24 cells and blocks of up to 32 steps, the bound approaches the exponent roughly as $1/k$ in this
sample (for typical orbits of an invariant measure, the subadditive ergodic theorem \cite{kingman1973subadditive}
guarantees convergence as $k$ grows, but it gives no rate and no guarantee for a finite record): for block lengths
$k=1,2,4,\ldots,32$, $k$ times the median excess of the bound over the exponent runs from 1.14 to 2.96 over the
chaotic cells and from 2.86 to 3.92 over the periodic ones (median excess 0.036 per step at $k=32$ over the chaotic
cells, 0.12 over the periodic ones). The bound has the sign of the long-run exponent in 13 of the 24 cells at $k=1$, 2
and 4 and in 22 at $k=16$; averaged over overlapping windows of the record, which gives finite-window diagnostics
rather than bounds, it has the right sign in 23 of the 24 cells for every window length from 17 to 29 steps. In these
24 cells, windows of about twenty steps thus recover the sign of the long-run exponent, as a measurement along the
recorded orbit, not as a forecast from the weights.

At low temperature $1/T$ is large, but one token takes almost all the probability, and $\Sigma_p$ vanishes faster than
$1/T$ grows. At a state with a unique leading token $j$, the gain is
$\lVert\Sigma_p\rVert/T=e^{-\Delta/T}(C+o(1))/T$, where $\Delta$ is the gap between the two largest logits and
$C=\lVert\sum_i(e_i-e_j)(e_i-e_j)^{\top}\rVert$, the sum running over the runner-up tokens whose logits trail by
exactly $\Delta$. As $T\to0$, $F$ tends to the greedy map, which feeds back the embedding of the most probable token
(at an exact tie, the limit averages the tied embeddings). The limit map takes finitely many values, so every orbit of
it eventually repeats. The convergence is pointwise, not uniform: near a tie between tokens with different embeddings
the gain is of order $1/T$, so at small positive temperature chaos is excluded only on invariant sets that keep a
uniform logit gap.

Chaos is therefore found in between, at temperatures where the next-token distribution is spread enough to make
$\Sigma_p$ large while $1/T$ still amplifies it: in four long unsteered sweeps the chaotic temperatures lie in one
band per model, interleaved with periodic windows, between $\log_{10}T=-0.20$ and $0.28$. This band is an
observation; near ties, the low-temperature argument does not exclude chaos.

\paragraph{Without the transformer, no chaos.} Suppose the network passed its input straight through,
$\tilde h(x)=x$. The map would then be $F(x)=E^{\top}\operatorname{softmax}(Ex/T)$, the update rule of a modern
Hopfield network whose stored patterns are the token embeddings \cite{ramsauer2020hopfield}. Its derivative
$\Sigma_p/T$ is symmetric and positive semidefinite, so $F=\nabla\phi$ for the convex function
$\phi(x)=T\log\sum_t e^{e_t^{\top}x/T}$, and the energy
$\mathcal{V}(x)=\tfrac12\|x\|^2-\phi(x)$ falls at every step, $\mathcal{V}(F(x))\le\mathcal{V}(x)-\tfrac12\|F(x)-x\|^2$,
while staying bounded below on $K$. Each step lowers the energy by at least half the squared length of the step, so
the steps must shrink to zero: every accumulation point of an orbit is a fixed point, and the strict descent of the
energy rules out cycles and chaotic motion (an unstable fixed point can still have positive tangent exponents). With tied embeddings, as in Qwen2.5-0.5B, every
cycle and every chaotic cell therefore depends on the network factor $J_{\tilde h}$ breaking this energy structure;
with untied embeddings, the network-free map need not have a symmetric positive-semidefinite derivative, so the
energy argument no longer applies, and cycles become possible even without a network (two tokens whose output vectors
point at each other's input embeddings exchange at low temperature).

\begin{figure*}[tb]
\centering
\includegraphics[width=\textwidth]{fig-mechanism-summary.png}
\caption{Readout and network contributions to the growth under steering (Qwen2.5-0.5B-Instruct unless marked 7B;
float64). Orange: the readout factor $\Sigma_p/T$ of \eqref{eq:gain}; blue: the network factor $J_{\tilde h}$.
(a) Split of the growth along the attractors of the $101\times53$ scan steered at layer 12: mean logarithm of each
factor per step, for chaotic cells (solid) and nearly unsteered cells, $|\alpha|\le0.01$ (dashed), in the 18
temperature rows with at least three cells of each kind; positive values expand perturbations, negative values
contract them. In the nearly unsteered cells the readout's relative gain shrinks perturbations by 0.87 to 5.49 nats
per step; in the chaotic cells it grows them, except in the three highest rows, while the network factor changes
little. (b) Share of chaotic cells (no period up to 64) per temperature row under residual steering at layer 12,
which moves both factors (black; 331 of 5353 cells, 6.2\,\%), and under a logit bias along $Wv$, the output matrix
applied to the steering vector, with no residual steering, which at a given state changes only the readout (orange;
241 of 6837 cells, 3.5\,\%). (c) Orbit-preserving crossover: change in the finite-time growth along recorded steered
orbits when only the readout (orange, circles) or only the network Jacobian (blue, squares) is taken from the
steered model; means over the chaotic cells used only for this test, with 95\,\% bootstrap intervals. Rows: the
philosophy and gratitude directions and a random direction of the philosophy vector's norm in the 0.5B model,
14 cells each, and the philosophy direction in the 7B model, 10 cells.}
\label{fig:mechsummary}
\end{figure*}

\paragraph{Steering makes tokens compete.} Steering could create chaos through either factor of \eqref{eq:gain}: it
moves the logits, and with them which tokens compete in $\Sigma_p$, and it moves the hidden states, and with them the
network's response $J_{\tilde h}$. Three measurements on Qwen2.5-0.5B, steered along the philosophy direction, tell
the two apart (Fig.~\ref{fig:mechsummary}).

First, an accounting. Each step's growth splits exactly into a network factor
$g_{\mathrm{net}}=(|\delta\tilde h|/|\tilde h|)/(|\delta x|/|x|)$, how much the transformer stretches a relative change
of its input, and a readout factor $g_{\mathrm{read}}=(|\delta F|/|F|)/(|\delta\tilde h|/|\tilde h|)$, how much the
softmax and the averaging of embeddings stretch it on the way out. For a tangent perturbation followed over $n$ steps, their mean
logarithms add up to its finite-time growth rate plus a boundary term,
$(\log\lVert x_0\rVert-\log\lVert x_n\rVert)/n$; both are relative gains along the orbit, not operator norms, and
the split is a convention (only the sum is intrinsic). Over a
$101\times53$ scan (5353 cells, 331 chaotic), the chaotic cells' readout factor exceeds that of nearly unsteered cells
($|\alpha|\le0.01$) at the same temperature by $2.58$ nats per step (95\,\% interval $[1.78,3.59]$ over 18 temperature
rows), while their network factor differs by only $-0.07$ $[-0.23,0.12]$. The chaotic cells also have a higher
next-token entropy than periodic cells at the same temperature ($+1.62$ nats, $[1.26,1.91]$): more tokens compete.

Second, a test that leaves the network unsteered. A logit bias along $Wv$, the output matrix applied to the steering
vector (a logit-space stand-in for the steering), is added with no residual steering: at a given state it changes only
the output probabilities, although the orbit it produces can visit different states. On its grid, 3.5\,\% of cells
were chaotic (241 of 6837, all but one with a nonzero bias), against 6.2\,\% on the residual-steering grid:
competition at the readout alone makes the loop chaotic, but mostly at $\log_{10}T\ge0.1$ (207 cells), where the
network factor itself can expand; below, 34 of 5676 cells are chaotic (0.6\,\%), against 169 of 4444 (3.8\,\%).

Third, a swap. An orbit-preserving crossover evaluates the steered and the unsteered model at the same recorded states
and builds the growth from one factor of each. Over 14 chaotic cells used only for this test, the steered readout
factor with the unsteered network factor raises the finite-time growth by $0.81$ nats per step ($[0.29,1.35]$), the
steered network factor with the unsteered readout lowers it by $0.49$ ($[-0.63,-0.35]$), and the interaction is
$+0.26$ ($[0.09,0.45]$). These hybrid products are not derivatives of any actual map; they apportion the growth along
the recorded orbits. Along the gratitude direction and a random direction of the same norm the readout effect is also
positive ($+0.33$ and $+0.77$ nats per step); in Qwen2.5-7B, whose embeddings are untied, steering only the network
lowers the growth by $1.14$ nats per step and steering only the readout adds $+0.23$ ($[-0.11,0.63]$), at cells mostly
above $\log_{10}T=0$, where the unsteered derivative already expands.

All three point the same way: along the recorded orbits of Qwen2.5-0.5B, the growth that steering adds comes mainly
from the readout, where next tokens compete; the readout's relative gain turns from shrinking to growing, while the
network factor changes little. The readout carries what steering changes; the network is what makes chaos possible
at all, since with tied embeddings, as here, the loop without it cannot cycle (previous paragraph). These are conditional comparisons on recorded trajectories. They do not single out
the higher entropy as the cause, nor show that the readout change alone would sustain the same attractor; the readout
gain still shrinks perturbations in the three highest temperature rows, and above $\log_{10}T\approx0.1$ the network factor itself can expand\smref{Why the loop can be chaotic}.

```

## Bibliography entries cited in Sec. III

```bibtex
@article{kingman1973subadditive,
  author  = {Kingman, J. F. C.},
  title   = {Subadditive Ergodic Theory},
  journal = {The Annals of Probability},
  volume  = {1},
  number  = {6},
  pages   = {883--899},
  year    = {1973},
  doi     = {10.1214/aop/1176996798}
}
@inproceedings{ramsauer2020hopfield,
  author        = {Ramsauer, Hubert and Sch{\"a}fl, Bernhard and Lehner, Johannes and others},
  title         = {Hopfield Networks is All You Need},
  booktitle     = {The Ninth International Conference on Learning Representations},
  year          = {2021},
  url           = {https://openreview.net/forum?id=tL89RnzIiCd},
  eprint        = {2008.02217},
  archivePrefix = {arXiv}
}
@misc{wang2026logitfixed,
  author = {Wang, Tongxi}, title = {Sharp Spectral Thresholds for Logit Fixed Points}, year = {2026},
  eprint = {2605.15651}, archivePrefix = {arXiv}, url = {https://arxiv.org/abs/2605.15651}
}
```

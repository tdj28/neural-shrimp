# Consult 17: check of the rewritten section "Why the loop can be chaotic" (Chaos article, arXiv v1)

## Context
The 22-page Chaos article "Chaos and bifurcations in continuous language-model feedback" goes to arXiv tonight. Its
Section III, "Why the loop can be chaotic", was just rewritten so that readers outside nonlinear dynamics (machine-learning
and interpretability researchers) can follow it. Formulas and numbers were kept; explanatory sentences were added.
The author asks for a check that nothing was botched. All numbers were verified separately against raw records; the
question here is mathematical correctness, precise wording and clarity.

The system: a pretrained language model fed back through its own soft next-token distribution. The state x is the
fed-back input embedding (896 dimensions in Qwen2.5-0.5B); one step runs the network with a fixed prefix, takes the
softmax of the logits at temperature T and feeds back the probability-weighted average of the token embeddings.
A steering vector of strength alpha is added to the residual stream at one layer. Section II (context, verbatim LaTeX)
defines the map and derives the factorization of its derivative, equation (3) (label eq:gain).

## Section II excerpt (context, verbatim)
\section{The soft-token map}
\label{sec:map}
We describe the map for Qwen2.5-0.5B-Instruct; the other models enter the same construction with their own embeddings
and steering layers. Let $E\in\mathbb{R}^{V\times d}$ be the input embedding matrix ($V=151{,}936$, $d=896$), with rows
$e_t$. The model reads a fixed chat prefix, with the system prompt ``You are an AI talking with another AI. You have
complete freedom. Feel free to pursue whatever you want.'' and the user turn ``Hello!'', followed by a single input
embedding $x$, and produces next-token logits $z(x)\in\mathbb{R}^V$. The soft-token map feeds the expected next-token
embedding, the soft token, back in place of $x$:
\begin{equation}
  F_{T,\alpha}(x) = E^{\top}\operatorname{softmax}\!\big(z_\alpha(x)/T\big),
\end{equation}
where $z_\alpha$ denotes the logits computed with $\alpha v$ added, at every position including the prefix, to the
residual stream at layer 12 (the $d$-dimensional vector at each token position to which every layer adds its output).
The steering direction $v$ is contrastive: it compares last-token states of ``philosophy'' prompts with those of
neutral prompts and is normalized to the mean residual norm. As $T\to0$ the map reduces, where the argmax is unique, to
greedy decoding of the same fixed-prefix, one-token-window system; it is not ordinary generation with a growing
context. Every run starts from the same state $x_0$, the fixed start: the input embedding of the prefix's last token,
the newline that opens the assistant's reply.

\paragraph{State space and regularity.} Attention is causal, so the keys and values of the prefix do not depend on
$x$; positions are fixed and the window does not grow, so $F$ is an autonomous map of $\mathbb{R}^{896}$. It is
real-analytic, because RMS normalization (with its positive regularizing constant), the SiLU activation, attention,
rotary embeddings at fixed positions and the softmax readout are analytic; the top-$k$ routing of a mixture of experts
makes its map only piecewise smooth. Every soft token is a convex combination of embeddings, so $F$ maps
$\mathbb{R}^{896}$ into the compact convex polytope $K=\operatorname{conv}\{e_t\}$, and $F(K)\subset K$: every orbit
enters $K$ after one step, $F$ has a fixed point in $K$ by Brouwer's theorem and invariant probability measures there,
and, $\log^+\lVert DF\rVert$ being bounded on $K$, its Lyapunov exponents exist at almost every point of every invariant
measure by Oseledets' theorem \cite{oseledets1968multiplicative,eckmann1985ergodic}. The exponents we report estimate
them over finite horizons.

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
amplified by the inverse temperature and composed with the derivative of the network. At a state of the period-8 orbit
of Table~\ref{tab:orbits} (Sec.~\ref{sec:adding}), \eqref{eq:gain} agrees with forward-mode derivatives of the full map
to a relative error of at most $1.35\times10^{-13}$ over eight random directions (Fig.~\ref{fig:derivcheck}). Without
tied embeddings, as in the 7B and 8B models, the logits are $z_\alpha=W\tilde h_\alpha(x)$ with an output matrix
$W\neq E$, and $\Sigma_p$ becomes the cross-covariance $\operatorname{Cov}_{t\sim p}(e_t,w_t)$ of input and output
embeddings, which is not symmetric in general.



## The section under review (verbatim LaTeX, as it will be posted)
\section{Why the loop can be chaotic}
\label{sec:theory}
Chaos means that two nearby states drift apart: a small difference $\delta x$ between them must, on average, grow from
one step to the next. One step multiplies the difference by the derivative $DF$, and \eqref{eq:gain} splits that
derivative into three factors with plain meanings. The network Jacobian $J_{\tilde h_\alpha}$ says how much the
transformer's final hidden state moves when its input moves. The covariance $\Sigma_p$ says how much the fed-back
expected embedding moves when the hidden state moves: it is large when several tokens compete for the next position,
because a small change of the logits then shifts probability between tokens with different embeddings, and it
vanishes when one token takes almost all the probability. The factor $1/T$ amplifies both at low temperature. A
difference can therefore grow only where the three factors together stretch it. This section uses the factorization in
three ways: it rules out chaos at very high and at very low temperature, it shows that the transformer between input
and output is what makes cycles and chaos possible at all, and three measurements show that steering creates chaos
mainly through the middle factor, by making next tokens compete.

\paragraph{Too hot or too cold: no chaos.} The largest Lyapunov exponent of an orbit, the average rate at which nearby
states separate, cannot exceed the orbit average of $\log\lVert DF\rVert$, so chaos needs $\lVert DF\rVert>1$ on a
substantial part of the attractor. Both ends of the temperature axis work against this.

At high temperature $1/T$ is small, and the bound is explicit. However the probability is spread, the projections
$u^{\top}e_t$ of the embeddings on any unit vector $u$ lie in an interval no longer than the diameter
$\operatorname{diam}(E)$ of the set of embeddings, and Popoviciu's inequality bounds a variance by a quarter of the
squared range, so $\lVert\Sigma_p\rVert\le\operatorname{diam}(E)^2/4$. With $L(\alpha)=\sup_K\lVert J_{\tilde
h_\alpha}\rVert$, the network's largest gain on the set $K$ of possible states (finite because $K$ is compact),
$\lVert DF_{T,\alpha}\rVert\le\operatorname{diam}(E)^2L(\alpha)/(4T)$ on $K$, and $F_{T,\alpha}$ shrinks every
distance, that is, it is a contraction of $K$, for
\begin{equation}
T>T_*(\alpha)=\tfrac14\operatorname{diam}(E)^2L(\alpha) .
\end{equation}
Above this temperature the loop forgets where it started: its single fixed point attracts every orbit (without tied
embeddings, $\operatorname{diam}(E)^2$ becomes $\operatorname{diam}(E)\operatorname{diam}(W)$). The threshold is a
curve in the $(\alpha,T)$ plane, since the network's largest gain depends on the steering. The bound is rigorous but
far from sharp, because it assumes the worst case at every step. For Qwen2.5-0.5B, $\operatorname{diam}(E)=1.2109$
exactly; the largest $\lVert J_{\tilde h_\alpha}\rVert$ met along the trajectories of 7936 map cells (power-iteration
estimates, so lower estimates of $L(\alpha)$) give $T_*(\alpha)\ge2.7\times10^{3}$ at every steering strength sampled
(64 per map) and $T_*(\alpha)\ge5.6\times10^{4}$ at $\alpha=0.06$, where the largest, $1.54\times10^{5}$, was met:
three to four and a half decades above the temperatures at which the unsteered map already has a single attracting
fixed point.

A sharper bound uses the derivatives actually met along a computed orbit instead of the worst case. One step at a time
it still says little: the orbit average of $\log\lVert DF\rVert$ is positive in every cell, chaotic or periodic
(median 2.98 per step, against a median exponent of 0.30 in the chaotic cells of the stripes map), because the
direction that one step stretches most is generally not the direction that the next step stretches, so single-step
norms overstate the growth of their product. Norms of products of $k$ consecutive derivatives, blocks of $k$ steps,
account for this and tighten the bound; they begin to separate the regimes only at $k=4$ and 5, so the inequalities
hold on the model's orbits but do not locate the chaos at short blocks. With exact norms at 24 cells and blocks of up
to 32 steps, the bounds close on the exponent roughly as $1/k$, as the subadditive ergodic theorem allows
\cite{kingman1973subadditive}: for block lengths $k=1,2,4,\ldots,32$, $k$ times the median overshoot runs from 1.14
to 2.96 over the chaotic cells and from 2.86 to 3.92 over the periodic ones (median overshoot 0.036 per step at
$k=32$ over the chaotic cells, 0.12 over the periodic ones), and their sign agrees with that of the long-run exponent
in 23 of the 24 cells for every window length from 17 to 29 steps. Blocks of about twenty steps thus locate the chaos,
but as a measurement along the orbit, not as a forecast from the weights.

At low temperature $1/T$ is large, but one token takes almost all the probability, and $\Sigma_p$ vanishes faster than
$1/T$ grows. Away from ties between the two leading tokens, the gain $\lVert\Sigma_p\rVert/T$ decays like
$Ce^{-\Delta/T}/T$, where $\Delta$ is the gap between the two largest logits and $C$ is set by the squared embedding
distances between the leading token and its nearest competitors. As $T\to0$, $F$ tends to the piecewise-constant
greedy map, which feeds back the embedding of the most probable token: a map onto a finite set of embeddings, every
orbit of which eventually repeats. This limit is pointwise, not uniform: near a tie the gain is of order [amount redacted]/T$, so at
small positive temperature chaos is excluded only on invariant sets that keep a uniform logit gap.

Chaos therefore lives in between, at temperatures where the next-token distribution is spread enough to make
$\Sigma_p$ large while $1/T$ still amplifies it. In four long unsteered sweeps the chaotic temperatures lie in one band
per model, interleaved with periodic windows, between $\log_{10}T=-0.20$ and $0.28$.

\paragraph{Without the transformer, no chaos.} Suppose the network passed its input straight through,
$\tilde h(x)=x$. The map would then be $F(x)=E^{\top}\operatorname{softmax}(Ex/T)$, the update rule of a modern
Hopfield network whose stored patterns are the token embeddings \cite{ramsauer2020hopfield}. Its derivative
$\Sigma_p/T$ is symmetric and positive semidefinite, so $F=\nabla\phi$ for a convex function $\phi$, and the energy
$\mathcal{V}(x)=\tfrac12\|x\|^2-\phi(x)$ falls at every step, $\mathcal{V}(F(x))\le\mathcal{V}(x)-\tfrac12\|F(x)-x\|^2$,
while staying bounded below on $K$. Each step lowers the energy by at least half the squared length of the step, so
the steps must shrink to zero: such a map has neither cycles nor chaos. With tied embeddings, as in Qwen2.5-0.5B, every
cycle and every chaotic cell therefore depends on the network factor $J_{\tilde h}$ breaking this energy structure;
with untied embeddings, the network-free map already has a non-symmetric derivative and can cycle.

\begin{figure*}[tb]
\centering
\includegraphics[width=\textwidth]{fig-mechanism-summary.png}
\caption{Why steering makes the loop chaotic (Qwen2.5-0.5B-Instruct unless marked 7B; float64). (a) Split of the growth
along the attractors of the $101\times53$ scan steered at layer 12: mean logarithm of the readout factor (solid) and of the network
factor (dashed) per step, for chaotic cells (red) and nearly unsteered cells, $|\alpha|\le0.01$ (gray), in the 18
temperature rows with at least three cells of each kind. Without steering the readout contracts by 0.87 to 5.49 nats
per step; in the chaotic cells it expands, except in the three highest rows, while the network factor changes little.
(b) Share of chaotic cells per temperature row under residual steering at layer 12 (331 of 5353 cells) and under a
logit bias along $Wv$, the steering vector's direct effect on the logits, with no residual steering (241 of 6837
cells). (c) Orbit-preserving crossover: change in the finite-time growth along recorded steered orbits when only the
readout (orange) or only the network Jacobian (gray) is taken from the steered model; means over the chaotic cells used
only for this test (14 per direction in the 0.5B model, 10 in the 7B model), with 95\,\% bootstrap intervals.}
\label{fig:mechsummary}
\end{figure*}

\paragraph{Steering makes tokens compete.} Steering could create chaos through either factor of \eqref{eq:gain}: it
moves the logits, and with them which tokens compete in $\Sigma_p$, and it moves the hidden states, and with them the
network's response $J_{\tilde h}$. Three measurements on Qwen2.5-0.5B, steered along the philosophy direction, tell
the two apart (Fig.~\ref{fig:mechsummary}).

First, an accounting. Each step's growth splits exactly into a network factor
$g_{\mathrm{net}}=(|\delta\tilde h|/|\tilde h|)/(|\delta x|/|x|)$, how much the transformer stretches a relative change
of its input, and a readout factor $g_{\mathrm{read}}=(|\delta F|/|F|)/(|\delta\tilde h|/|\tilde h|)$, how much the
softmax and the averaging of embeddings stretch it on the way out. Their logarithms, averaged along an orbit, add up
to the largest Lyapunov exponent up to a boundary term (the split is a convention; only the sum is intrinsic). Over a
$101\times53$ scan (5353 cells, 331 chaotic), the chaotic cells' readout factor exceeds that of nearly unsteered cells
($|\alpha|\le0.01$) at the same temperature by $2.58$ nats per step (95\,\% interval $[1.78,3.59]$ over 18 temperature
rows), while their network factor differs by only $-0.07$ $[-0.23,0.12]$. The chaotic cells also have a higher
next-token entropy than periodic cells at the same temperature ($+1.62$ nats, $[1.26,1.91]$): more tokens compete.

Second, a test that leaves the network untouched. A logit bias along $Wv$, the steering vector's direct effect on the
logits, applied with no residual steering, changes only which tokens compete at the output; it still makes 3.5\,\% of
the same temperature range chaotic (241 of 6837 cells), against 6.2\,\% for steering in the residual stream.

Third, a swap. An orbit-preserving crossover evaluates the steered and the unsteered model at the same recorded states
and takes one factor from each. Over 14 chaotic cells used only for this test, steering only the readout raises the
growth by $0.81$ nats per step ($[0.29,1.35]$), steering only the network response lowers it by $0.49$
($[-0.63,-0.35]$), and the interaction is $+0.26$ ($[0.09,0.45]$).

All three point the same way: along the recorded orbits of Qwen2.5-0.5B, steering makes the loop chaotic mainly by
making next tokens compete, which turns the softmax readout from a contraction into an expansion. These are
conditional comparisons on recorded trajectories, not a proof that the readout change alone would sustain the same
attractor, and above $\log_{10}T\approx0.1$ the network factor itself can expand\smref{Why the loop can be chaotic}.



## What I need from you
1. Mathematical errors. Check each statement and say whether it is correct as written:
   - the bound of the largest Lyapunov exponent by the orbit average of log||DF||;
   - the Popoviciu bound on ||Sigma_p||, the contraction for T > T_*(alpha), and its untied-embedding variant;
   - "its single fixed point attracts every orbit";
   - the low-temperature decay C e^(-Delta/T)/T and what sets C;
   - the greedy limit, "every orbit of which eventually repeats", and pointwise versus uniform convergence;
   - the role claimed for Kingman's subadditive ergodic theorem ("as the subadditive ergodic theorem allows");
   - the Hopfield argument: F = grad phi with phi convex, the energy inequality, bounded below, hence "neither cycles nor chaos";
   - "with untied embeddings, the network-free map already has a non-symmetric derivative and can cycle".
2. Misleading explanatory sentences. The new plain-language sentences must stay true. For example:
   - what Sigma_p "says";
   - "large when several tokens compete ... vanishes when one token takes almost all the probability";
   - "The factor 1/T amplifies both at low temperature";
   - "the direction that one step stretches most is generally not the direction that the next step stretches, so
     single-step norms overstate the growth of their product";
   - "Each step lowers the energy by at least half the squared length of the step, so the steps must shrink to zero";
   - "Above this temperature the loop forgets where it started";
   - "changes only which tokens compete at the output" (the logit-bias test).
   Flag anything a careful referee would call wrong or loose.
3. Interpretation of the three steering measurements (the accounting split, the logit-bias test, the orbit-preserving
   crossover): any overinterpretation in the conclusion "steering makes the loop chaotic mainly by making next tokens
   compete, which turns the softmax readout from a contraction into an expansion"?
4. Clarity for the target readers: where is it still too dense, or where does it skip a step? Give concrete
   replacement sentences.

Please separate must-fix errors from optional improvements. For each item quote the current text and give exact
replacement text that keeps the numbers unchanged. Be brief where the text is fine.

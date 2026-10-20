# Consult 24: the synchronization section of the article: theory, measurements, claims, references and readability

## The decision this review informs

The article studies the soft-token feedback loop of a language model as a discrete-time map F of its embedding space
(dimension 896 for Qwen2.5-0.5B-Instruct), with steering strength alpha and temperature T as parameters:
F(x) = E^T softmax(z(x)/T), where z(x) are the next-token logits for a fixed chat prefix followed by the single input
embedding x, and E is the embedding matrix. Section VIII, reproduced verbatim below, couples two copies of this map
through their outputs and compares the result with the classical theory of synchronization of identical chaotic maps.
The section was written tonight and goes on arXiv imminently. The human author asked for an expert review. Please
judge whether the theory is stated correctly, whether the measurements support each sentence as written, whether the
claims are adequately qualified, whether the references are right and sufficient, and whether a reader who is not a
specialist can follow it. Numbers are the authors' computations from raw files you do not see; judge whether the
statements are correct if the numbers are right, and say what you would need otherwise.

## Facts behind the section (from the run records; not in the text unless the text says so)

- Implementation. Both copies use the same model, chat prefix (system prompt "You are an AI talking with another AI.
  You have complete freedom. Feel free to pursue whatever you want." and user turn "Hello!"), steering vector and
  (alpha, T). Each pair is two rows of one batch; at every step the model produces both rows' next soft tokens and the
  code replaces row A's by (1-c) F(x_A) + c F(x_B) and row B's symmetrically before feedback. float64 with float64
  RMSNorm and rotary tables (checked by central differences converging as h^2).
- Batch effects. float64 F is not bitwise batch invariant: a pair against a single row, a swapped layout and a batch of
  8 differ by at most 6.0e-15 relative; two identical rows in one batch differ by up to 8.9e-16. Pairs therefore carry
  a rounding-level asymmetry of about 1e-15; no noise or parameter mismatch was added.
- lambda_1 per cell from exact forward-mode tangents over 3000 steps (block standard errors): 0.0975 +- 0.0095,
  0.437 +- 0.016, 0.196 +- 0.014, so c* = 0.0464, 0.177, 0.089 (c* at lambda_1 +- 2 s.e.: 0.038-0.055, 0.166-0.188,
  0.077-0.101). Earlier 600-step values were 0.109, 0.424, 0.239.
- Transverse exponent. Renormalized twins (relative offset 1e-7, renormalized every step) on the coupled map, 2000 steps
  after 1000, at c = 0, 0.05, ..., 0.30 (7 couplings x 3 cells). Largest |measured - predicted| per cell 0.013, 0.051,
  0.045; largest ratio to the twin's own standard error 3.4, 4.0, 1.9 (the prediction's error not included).
  Subtracting ln|1-2c| gives lambda_1 estimates of 0.088-0.111, 0.386-0.448, 0.186-0.241 (finite-time scatter).
- Thresholds. Synchronized: |x_A - x_B|/|x_A| below 1e-8 over the last 500 steps (secondary rule 1e-6). Observed
  brackets (largest unsynchronized c, smallest c from which all larger grid values stay synchronized):
  cell 1 (lambda 0.098): near starts (0.04, 0.05], 10000-step refinement (0.045, 0.05];
  cell 2 (lambda 0.437): near (0.22, 0.23] primary, refinement none synchronized up to 0.19 on 1e-8 (tail maximum
  3e-7 at 0.19) but (0.18, 0.19] on 1e-6, far starts (0.18, 0.20];
  cell 3 (lambda 0.196): near (0.15, 0.16], refinement (0.12, 0.15], far (0.14, 0.16] primary, (0.10, 0.12] secondary.
  "Far" starts put B 500 steps further along the attractor (c in steps of 0.02, 2000 steps).
- Bursts (passage from below 1e-8 to above 1e-3). Cell 2 at c = 0.16, 0.165, 0.17: 13, 14, 12 bursts in 10000 steps,
  8-20 % of steps below 1e-8; laminar phases (runs below 1e-4) 200-293 per pair; crude log-binned slopes of the phase
  length distribution -1.52 to -1.76 (no error bars; not claimed as a power law). Cell 3 at c = 0.105, 0.11, 0.115
  (above c* = 0.089, typical-orbit transverse exponent predicted -0.02 to -0.05): 5, 10, 6 bursts from about 1e-14 to
  order one; the synchronized floor rises from about 1e-14 at c = 0.3 to 1e-10-1e-8 at c = 0.15-0.18. Cell 1: at most
  2 bursts; sharp transition. The periodic orbits embedded in the attractor were not computed.
- Below threshold: many pairs lock onto periodic orbits of the joint state (x_A, x_B), verified by full-state return
  below 1e-9 (2.7e-15 to 2.2e-13), primitive periods 6, 8, 10, 12 (and 6 and 4 for two far-start pairs of cell 1 at
  c = 0.14 and 0.20 respectively, above its c*); pair distance 0.05 to 1.7; stability shown only by attraction in forward
  iteration (multipliers not computed). Uncoupled pairs do not close.
- Cell 1 is bistable: from far starts most pairs synchronize onto an attracting period-2 orbit of F ("it is"), not the
  chaotic attractor.
- On the synchronized state the coupled update equals F, so synchronized orbits are orbits of the single loop; top-1
  token distributions of synchronized pairs differ from uncoupled rows by total variation 0 to 0.047 (independent
  uncoupled rows differ by up to 0.039).
- The full report also has a different experiment: two agents whose contexts grow (dialogue), which at c = 0.25
  barely synchronized although 0.25 exceeds c* for all three cells here.

## Section VIII (article, LaTeX, verbatim, with its figure caption; the image is omitted)

\section{Synchronization of two coupled loops}
\label{sec:sync}

The prefix tells the model that it is talking with another AI, and the simplest dynamical version of two such agents
couples two copies of the loop, each reading a mixture of its own soft token and the other's. The theory of coupled
chaotic maps predicts, from the single loop's largest Lyapunov exponent alone, when such copies lock together and write
the same text. The measurements below follow that prediction, with the complications known from coupled chaotic
oscillators.

\begin{figure}[tbp]
\centering
\includegraphics[width=\linewidth]{fig-sync-compact.png}
\caption{Synchronization of two coupled copies of the loop, Eq.~(\ref{eq:coupled}), at three chaotic cells, labeled by
the leading exponent $\lambda_1$ of the single loop (Qwen2.5-0.5B-Instruct, philosophy direction at layer 12,
float64). (a) Mean relative difference $|x_A-x_B|/|x_A|$ over the second half of each run against the coupling $c$;
dashed lines, the predicted thresholds $c^*$ of Eq.~\eqref{eq:cstar}; dotted line, the synchronization criterion
$10^{-8}$. (b) Growth rate of the difference measured on the coupled map against the prediction $\lambda_1+\ln|1-2c|$,
at seven couplings per cell (bars: two standard errors; line: equality).}
\label{fig:sync}
\end{figure}

Both copies are the map of Sec.~\ref{sec:map}, with the same model, chat prefix, steering direction and cell
$(\alpha,T)$, in float64. They run as two rows of one batch. At every step, after the model
has produced both soft tokens and before they are fed back, the program replaces each by a mixture of its own and the
other's:
\begin{equation}
\begin{aligned}
x_A'&=(1-c)\,F(x_A)+c\,F(x_B),\\
x_B'&=(1-c)\,F(x_B)+c\,F(x_A).
\end{aligned}
\label{eq:coupled}
\end{equation}
At $c=0$ the copies run independently; at $c=1/2$ both receive the same vector.

Where the copies agree, $x_A=x_B$, the update is exactly $F$ for both, so a synchronized pair stays synchronized and
writes identical text. Whether this synchronized state attracts depends on a small difference $u=x_A-x_B$ between the
copies, which points across the synchronized state, or transverse to it. The difference obeys
$u'=(1-2c)\,[F(x_A)-F(x_B)]$, which for small $u$ becomes $u'=(1-2c)\,DF\,u$. At each step the loop's derivative $DF$
stretches the difference, on average by the factor $e^{\lambda_1}$, where $\lambda_1$ is the largest Lyapunov exponent
of a single loop, and the mixing multiplies it by [amount redacted]-2c$. The difference therefore grows on average at the rate
$\lambda_\perp=\lambda_1+\ln|1-2c|$ per step, the transverse Lyapunov exponent. Without coupling, $\lambda_\perp=\lambda_1>0$ and the copies drift apart. As $c$ rises from 0 toward $1/2$, the
factor $1-2c$ falls from 1 toward 0, and $\lambda_\perp$ falls with it. It crosses zero at the critical coupling
\begin{equation}
c^*=\tfrac12\bigl(1-e^{-\lambda_1}\bigr),
\label{eq:cstar}
\end{equation}
where the mixing's contraction per step, $1-2c^*=e^{-\lambda_1}$, exactly cancels the loop's average stretching per
step, $e^{\lambda_1}$. For couplings between $c^*$ and $1/2$ a small difference shrinks on average, and the synchronized
state attracts along a typical orbit. This is the classical stability condition for identical chaotic maps coupled
through their outputs \cite{fujisaka1983stability,pecora1990synchronization}. The more chaotic a single loop, the larger
$\lambda_1$, and the larger the share of the other copy's output that each copy must take in.

We coupled copies at three chaotic cells, one in the crescent's chaotic core ($\alpha=0.0652$, $\log_{10}T=-0.156$) and
two among the chaotic stripes ($\alpha=0.07484$, $\log_{10}T=-0.1226$, and $\alpha=0.0640$, $\log_{10}T=-0.115$). Over
3000 steps their largest exponents are $\lambda_1=0.098$, 0.437 and 0.196 per step, so Eq.~\eqref{eq:cstar} gives $c^*=0.046$, 0.177 and 0.089.
Copy~A started on the attractor after a 1000-step transient, and copy~B at a relative distance $10^{-3}$ from it in a
random direction. Each pair ran 3000 steps at couplings from 0 to 0.30 in steps of 0.01, and 10\,000 steps at eight
couplings near each threshold. A pair counted as synchronized when $|x_A-x_B|/|x_A|$ stayed below $10^{-8}$ over its
last 500 steps. To measure $\lambda_\perp$ directly, copy~B was started at relative distance $10^{-7}$ from copy~A and
reset to that distance after every step of the coupled map for 2000 steps.

The measured $\lambda_\perp$ followed $\lambda_1+\ln|1-2c|$, which ranged from $+0.44$ to $-0.82$ per step, within
0.051 at every point (Fig.~\ref{fig:sync}(b)). In two cells the copies synchronized close to $c^*$, in one only on a
$10^{-6}$ criterion (Fig.~\ref{fig:sync}(a)). Just below that cell's threshold the difference showed on-off
intermittency \cite{platt1993onoff}: it dropped below $10^{-8}$ for long stretches and burst back to order one, 12 to 14
times in 10\,000 steps. In the third cell the observed threshold lay between 0.12 and 0.16, well above $c^*$, and even above $c^*$ the difference burst from the rounding level to order one. Bursts of this kind above the threshold
are known as attractor bubbling \cite{ashwin1994bubbling}. The threshold $c^*$ describes a typical orbit, but a chaotic
attractor also contains unstable periodic orbits, and along some of them the loop stretches differences more than on
average. While the synchronized pair moves near such an orbit, a tiny difference can grow into a burst before the
coupling pulls the copies back together. The bursts we see are consistent with this picture; we did not compute the
periodic orbits responsible. No noise was added: the two rows of a batch differ only by rounding (below $10^{-15}$),
which plausibly supplies the tiny differences that grow. Below the threshold, many pairs stopped being chaotic and locked onto periodic orbits of the
joint state, of primitive period 6 to 12, on which the copies write different or staggered texts.

\paragraph{What the coupling shows.} The number that measures the loop's chaos, its largest Lyapunov exponent, also
predicts how two coupled copies behave: the difference between them grows at the predicted rate at every coupling, and
in two of the three cells the copies lock together close to the predicted threshold. The more chaotic each copy is
alone, the more strongly the two must mix their outputs before they write the same text: the predicted threshold rises
from 0.046 in the weakest chaos to 0.177 in the strongest, and the observed thresholds rank the same way. The
prediction concerns a typical orbit, and it does not guarantee synchrony from every start. Near the threshold the copies
alternate between long synchronized stretches and bursts, and in one cell rounding alone broke synchrony above the
threshold. Weak coupling, instead of synchrony, often produced order: the copies settled into a periodic exchange on
which they write different texts. These are copies of one fixed map; agents whose contexts grow form a different
system, which these measurements do not describe.

## Questions

1. Is the theory stated correctly: the transverse exponent lambda_1 + ln|1 - 2c| for this symmetric output coupling,
   the critical coupling c* = (1 - e^{-lambda_1})/2 and the range of c for which it applies, and the description of
   on-off intermittency and attractor bubbling? Is anything missing that a specialist would expect (for example, the
   distinction between the typical-orbit threshold and the threshold for all invariant measures, or riddled basins)?
2. Do the measurements support each sentence as worded, including "in two cells the copies synchronized close to c*",
   the third cell's threshold, "consistent with attractor bubbling", "plausibly seeds the bursts", and the closing
   takeaway paragraph? Flag any sentence that says more than the facts above support.
3. Are the references right and sufficient (Fujisaka and Yamada 1983, Pecora and Carroll 1990, Platt, Spiegel and
   Tresser 1993, Ashwin, Buescu and Stewart 1994)? Should others be cited?
4. Is the section readable for a broad Chaos readership? Point to any sentence that needs a definition or a rewrite.
5. What one cheap addition (computation or wording) would most strengthen the section?

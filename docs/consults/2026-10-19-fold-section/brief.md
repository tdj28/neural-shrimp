# Consult 23: the fold section of the article: mathematics, concepts, references and readability

## The decision this review informs

The article studies the soft-token feedback loop of a language model as a discrete-time map F of its embedding space
(dimension 896 for Qwen2.5-0.5B), with steering strength alpha and temperature T as parameters. Its Section V,
reproduced verbatim below, explains a fan of periodic windows ("stripes") by the saddle-node (fold) bifurcation of a
2-cycle: beyond the fold the 2-cycle's ghost leaves a bottleneck, passages through it are timed against the fold's
normal form, and the windows' positions follow a window law. The section was restructured today and goes on arXiv
imminently. The human author, returning to dynamics after years away, wants an expert check that the mathematics is
correct, the concepts and terms are used as specialists use them, the references are right and sufficient, and the
text can be followed by a reader whose bifurcation theory is rusty. Numbers are the authors' computations; you see no
raw data. Judge whether the statements are correct and adequately qualified if the numbers are right, and say what you
would need otherwise.

Context from earlier sections: G = F^2 for the 2-cycle; a fold is located where a real multiplier of G crosses +1; the
normal-form coefficients a and b are computed from the network's first and second derivatives along the critical
eigenvector (Kuznetsov's standard formulas for maps); alpha is the steering strength, eps = alpha - alpha_c. The
stripes are periodic orbits whose texts read ". This" followed by N pairs "is X" and a closing "is a is a", period
2N+6. Float64 throughout unless stated.

## Section V (article, LaTeX, verbatim; images omitted)

\section{Periodic windows and the saddle-node bottleneck}
\label{sec:adding}
Every periodic window writes a repeating text, and the fan of stripe windows in Fig.~\ref{fig:stripes}(a) is organized by the ghost of
a vanished orbit, the slow bottleneck that a saddle-node bifurcation leaves behind
\cite{strogatz1989predicted,strogatz2018nonlinear}: where the windows lie, and how long trajectories linger near the ghost, follow the fold's normal form
computed from the model's derivatives. Such low-dimensional laws can hold in 896 dimensions because center-manifold
reduction restricts the dynamics near a bifurcation to its critical directions \cite{kuznetsov2023elements,kuznetsov2019maps}.

\begin{table}[tb]
\caption{Periodic orbits of the stripe family at the stripe centers along $\log_{10}T=-0.115$ ($\alpha=0.077872$,
$0.071674$, $0.067766$, $0.065550$, $0.064443$, $0.063857$, $0.063534$, $0.063335$): Newton residuals $6\times10^{-15}$ to
$8\times10^{-13}$, leading multiplier $\mu_1$ and next modulus $|\mu_2|$ from Rayleigh--Ritz values. Each step in
period adds one ``is X'' pair to the text.}
\label{tab:orbits}
\begin{ruledtabular}
\begin{tabular}{rlcc}
period & orbit (leading tokens) & $\mu_1$ & $|\mu_2|$ \\
\hline
6 & . This is a is a & $+0.0052$ & $3\times10^{-6}$ \\
8 & . This is there is a is a & $+0.0110$ & $3\times10^{-7}$ \\
10 & . This is there is it is a is a & $+0.0025$ & $2\times10^{-7}$ \\
12 & . This is there (is it)$^2$ is a is a & $+0.0163$ & $4\times10^{-8}$ \\
14 & . This (is it)$^4$ is a is a & $+0.0369$ & $2\times10^{-9}$ \\
16 & . This (is it)$^5$ is a is a & $+0.0716$ & $8\times10^{-11}$ \\
18 & . This (is it)$^6$ is a is a & $+0.0792$ & $1\times10^{-12}$ \\
20 & . This (is it)$^7$ is a is a & $+0.0857$ & $2\times10^{-14}$ \\
\end{tabular}
\end{ruledtabular}
\end{table}

\begin{table*}[tb]
\caption{Chaotic loops and copies started $10^{-10}$ away: most probable tokens around their first difference (red
from the bar $\Vert$ on; $\hookleftarrow$: line break).\journalonly{ More in the \sm{Chaos with periodic windows}.}}
\label{tab:chaostexts}
\footnotesize
\begin{ruledtabular}
\begin{tabular}{ll}
\multicolumn{2}{l}{Qwen2.5-0.5B, philosophy, $\alpha=0.0652$, $\log_{10}T=-0.156$: $\lambda=0.10$, texts split at step 151} \\
loop & \texttt{$\hookleftarrow$$\hookleftarrow$The\ existence$\hookleftarrow$$\hookleftarrow$The\ existence$\hookleftarrow$$\hookleftarrow$The\,$\Vert$\,\textcolor{BrickRed}{\ existence$\hookleftarrow$$\hookleftarrow$The\ existence$\hookleftarrow$$\hookleftarrow$The\ existence}} \\
copy & \texttt{$\hookleftarrow$$\hookleftarrow$The\ existence$\hookleftarrow$$\hookleftarrow$The\ existence$\hookleftarrow$$\hookleftarrow$The\,$\Vert$\,\textcolor{BrickRed}{\ question$\hookleftarrow$$\hookleftarrow$The\ existence$\hookleftarrow$$\hookleftarrow$The\ question}} \\
\hline
\multicolumn{2}{l}{Qwen2.5-0.5B, hostility, $\alpha=0.08946$, $\log_{10}T=-0.05566$: $\lambda=0.35$, texts split at step 45} \\
loop & \texttt{Yes!Yes!Yes!Yes!Yes!Yes!Yes!\,$\Vert$\,\textcolor{BrickRed}{Yes,\ that!Yes!Yes!No,}} \\
copy & \texttt{Yes!Yes!Yes!Yes!Yes!Yes!Yes!\,$\Vert$\,\textcolor{BrickRed}{No,\ that!Yes!Yes!Yes!}} \\
\end{tabular}
\end{ruledtabular}
\end{table*}

\begin{figure*}[tb]
\centering
[image]
\caption{The fold behind the stripes: the 2-cycle ``is it'' at $\log_{10}T=-0.115$ (Qwen2.5-0.5B-Instruct, philosophy
direction at layer 12). (a) Continued in $\alpha$, the attracting 2-cycle (filled) and its saddle partner (open) meet
and vanish at $\alpha_c=0.06279827$. The vertical coordinate is the projection $u$ of one point of each orbit on the
fold's critical direction; orange, the normal form's fixed points $u^*=\pm\sqrt{|a|(\alpha_c-\alpha)/|b|}$. (b) The
largest multiplier of $G=F^2$ on the two orbits reaches $+1$ at the fold, following the normal form's
$1\mp2\sqrt{ab(\alpha_c-\alpha)}$ (open circles: a complex leading pair). (c) Beyond the fold the 2-cycle is gone, but
its ghost slows the dynamics: controlled passages from the start of the channel, $v=-U$, to its exit, $v=U$ ($v$ the
oriented critical coordinate, $U=0.1/|b|$), measured on the network (markers, one shape per $\eps$) and iterated from the normal form
$v'=v+|a|\eps+|b|v^2$ (lines), take 144, 71, 34 and 14 iterates, as predicted. (d) A free trajectory at
$\eps=10^{-5}$ alternates between laminar phases in the bottleneck ($|u|<U$, shaded; 40 to 70 iterates each) and bursts
away from it, the signature of type-I intermittency. Branch residuals, fold, normal form and passages in float64; branch
multipliers in float32.}
\label{fig:fold}
\end{figure*}

\paragraph{Windows are texts.} A period-3 orbit returns to the same soft state every three steps, and at its three
states the most probable next tokens are ``\textbackslash n\textbackslash n'' (a blank line, one token), ``The'' and
`` existence'', with probabilities $0.74$, $0.36$ and $0.21$ at $\alpha=0.0642$, $\log_{10}T=-0.156$. Read greedily,
the loop writes ``The existence'' paragraph after paragraph; what it feeds back at each step, however, is the
probability-weighted average of all token embeddings, not that token. Attractors also coexist. At that same point the
loop started from a state of the ``is it'' 2-cycle (taken from its attractor at $\alpha=0.0655$) stays on the 2-cycle
and writes `` is it is it\ldots'' (most probable tokens with probabilities $0.66$ and $0.29$; 300-step exponent
$-1.19$), while from the fixed start it settles on the period-3 loop (exponent $-0.13$). At $\alpha=0.0652$ the fixed
start reaches the crescent's chaotic attractor, whose text alternates irregularly between ``The question'' and ``The
existence'' paragraphs (exponent $+0.095$), and the 2-cycle start again writes `` is it'' ($-1.29$): one model, one
steering vector and one temperature, and the text written after the transient depends on where the loop starts.

\paragraph{Chaos writes text too.} A chaotic loop reuses the words of neighboring loops in no fixed order
(Table~\ref{tab:chaostexts}). A float64 copy started $10^{-10}$ away writes the same text until their distance,
growing at about the rate the Lyapunov exponent sets, reaches 1.27 to 176\,\% of the soft token's norm; in seven
cells of Qwen2.5-0.5B and -1.5B the texts first differ at steps 10 to 151, where no token dominates. Run with six and with
four threads, which only reorders floating-point sums, the cell next to the period-3 crescent wrote the same text for
27 steps and then different texts.

Between the period-2 region and the chaos lies a fan of stripes of periods $6, 8, 10, \dots$ (Fig.~\ref{fig:stripes}(a)).
Their orbits, numerically validated in float64, read as text (Table~\ref{tab:orbits}): each is ``. This'' followed by
$N$ pairs ``is X'' and a closing ``is a is a'', with period $2N+6$, so each stripe adds one pair. At the stripe centers
the orbits are effectively one-dimensional: the leading multiplier, $0.0025$ to $0.086$, exceeds the next by more than
three orders of magnitude (the next falls from $3\times10^{-6}$ at period 6 to $2\times10^{-14}$ at period 20), so one
direction governs the dynamics near them. This is a property of these orbits, not of the map as a whole: the two
leading multipliers of the 2-cycle, $-0.271$ and $-0.058$, differ by a factor of only about 5.

\paragraph{The fold.} At $\log_{10}T=-0.115$ the 2-cycle ``is it'' dies in a fold at $\alpha_c=0.06279827$ (in float64
the critical multiplier equals 1 to within $3\times10^{-13}$, and the next has modulus $0.037$): it meets an unstable
2-cycle, and both vanish (Fig.~\ref{fig:fold}(a,b)). In the critical coordinate $u$ of the second iterate $G=F^2$, the
fold's normal form is
\begin{equation}
u'=u+a\eps+bu^2,\qquad \eps=\alpha-\alpha_c,
\label{eq:foldnf}
\end{equation}
with $a=-4.834$ per unit $\alpha$ and $b=-25.4285$ (the standard coefficients along the critical eigenvector,
computed from the network's first and second derivatives \cite{kuznetsov2019maps}), so $ab=122.93$; the stable branch's multipliers give $123.1$
independently (through $(1-\mu)^2/(4(\alpha_c-\alpha))\to ab$, quadratic fit), and an independent implementation
recovers $\alpha_c$ to $6\times10^{-15}$, $a$ to $3\times10^{-12}$ and $b$ to $2.2\times10^{-6}$ relatively.

\paragraph{Passages and the window law.} Beyond the fold the 2-cycle is gone, but it leaves a ghost
(Fig.~\ref{fig:passage}(a)). Near the point where the two orbits vanished, $G$ still moves a state only a little: by
Eq.~\eqref{eq:foldnf}, one iterate changes $u$ by $a\eps+bu^2$, which is small when $\eps$ and $u$ are. A trajectory
that enters this narrow channel, the bottleneck, therefore crawls through it for many iterates before it escapes, and
the closer the fold, the slower the crawl (Fig.~\ref{fig:passage}(b)). We timed such passages. In the oriented
coordinate $v=\operatorname{sign}(b)\,u$, in which trajectories cross the channel from negative to positive $v$, we
started them at the channel's entrance, $v=-U$ (a full passage), or at its middle, $v=0$, with $U=0.1/|b|$, and counted
$G$-iterates until $v>U$. At four distances from the fold the network's counts match those of the iterated normal form
within one iterate, with no fitted parameter (Table~\ref{tab:passages}; Fig.~\ref{fig:fold}(c)): a full passage takes
144 iterates at $\eps=3\times10^{-6}$ and 14 at $\eps=10^{-4}$, as predicted, and a start displaced off the channel's
axis changes the count by at most one.

\par\medskip\noindent\begin{minipage}{\columnwidth}
\centering
[TikZ diagram: (a) the step Delta v = |a| eps + |b| v^2 across the channel -U..U for eps = -1e-5 (dashed, two zeros), 0 (dotted), 1e-5 (solid); (b) the 71 iterates of a full passage at eps = 1e-5 drawn as dots on the v axis, dense near v = 0, with an arrow from U back to a bar slightly left of 0 labelled burst, reinjection]
\captionof{figure}{The bottleneck behind the stripes, drawn from the normal form \eqref{eq:foldnf} with the network's
coefficients, in the oriented critical coordinate $v$. (a) The step of one $G$-iterate, $\Delta v=|a|\eps+|b|v^2$,
across the channel from $-U$ to $U$. Before the fold ($\eps=-10^{-5}$, dashed) the step vanishes at two fixed points of
$G$: the attracting 2-cycle (filled) and its unstable partner (open). At the fold ($\eps=0$, dotted) they merge. Beyond
it ($\eps=10^{-5}$, solid) no fixed point remains, but the step stays small near $v=0$: the ghost. (b) The 71 iterates
of a full passage at $\eps=10^{-5}$, crowded where the step is smallest; the network takes 71 as well. A free
trajectory leaves at $U$, makes a burst, and is reinjected slightly upstream of the middle (bar), so its passage is
shorter than a full one.}
\label{fig:passage}
\end{minipage}\par\medskip

\noindent\begin{minipage}{\columnwidth}
\captionof{table}{Passage times through the bottleneck in $G$-iterates, measured on the network\,/\,predicted by
iterating the normal form \eqref{eq:foldnf} with the network's coefficients; no parameter is fitted. The starts are
the channel's start, $v=-U$; its middle, $v=0$; and the channel's start displaced by the same distance in a direction
that leaves $u$ unchanged.}
\label{tab:passages}
\begin{ruledtabular}
\begin{tabular}{cccc}
$\eps=\alpha-\alpha_c$ & start of channel & midpoint & displaced start\\
\hline
$3\times10^{-6}$ & $144/144$ & $73/74$ & $143/144$\\
$10^{-5}$ & $71/71$ & $36/37$ & $70/71$\\
$3\times10^{-5}$ & $34/34$ & $17/18$ & $33/34$\\
$10^{-4}$ & $14/14$ & $7/7$ & $13/14$\\
\end{tabular}
\end{ruledtabular}
\end{minipage}\par\medskip

\begin{figure}[tb]
\centering
[image]
\caption{The window law at the fold of the 2-cycle ``is it'' (Qwen2.5-0.5B-Instruct, philosophy direction at layer 12,
$\log_{10}T=-0.115$): stripe centers plotted as $1/\sqrt{\alpha_N-\alpha_c}$ against $N$, the number of ``is X''
pairs, with the full- and half-passage predictions of the normal form; the fitted slope of the stripe centers, $6.17$
per unit of $N$, is close to the $6.13$ of the fold edges (text).}
\label{fig:windowlaw}
\end{figure}
Left to itself, a trajectory near the fold alternates between two kinds of motion (Fig.~\ref{fig:fold}(d)). In
\emph{laminar phases} it crawls through the bottleneck, close to where the 2-cycle was, and moves almost periodically;
in \emph{bursts} it leaves the channel for a short excursion and is then reinjected into it. This alternation is
type-I intermittency \cite{pomeau1980intermittent,hirsch1982theory}; the word laminar comes from fluid flow, in which
smooth stretches alternate with turbulent bursts in the same way. Free trajectories re-enter the channel slightly
upstream of its middle (Fig.~\ref{fig:passage}(b)). From each measured entry point the normal form predicts the length
of the laminar phase that follows within 4--8\,\% for $\eps\le10^{-4}$, and a free passage lasts about 0.6 of a
passage through the whole channel.

The bottleneck also decides where the periodic windows lie. A window of period $p$ is a loop that spends
$N=(p-6)/2$ iterates of $G$ in the bottleneck, one for each ``is X'' pair it writes, and closes through its six other
steps. For a channel much longer than the bottleneck, a full passage takes about $\pi/\sqrt{ab\,\eps}$ iterates
\cite{strogatz1989predicted}, so the closer the fold, the more laps fit. A loop whose passage is a fraction $f$ of a
full one spends $N$ iterates in the bottleneck at the distance $d_N=\alpha_N-\alpha_c$ from the fold for which
$d_NN^2=f^2\pi^2/(ab)$: the windows crowd toward the fold as $N^{-2}$. The stripe centers, the midpoints of the windows
on a float32 cut along $\log_{10}T=-0.115$, follow this law with a log-log slope of $-2.00$ (Fig.~\ref{fig:windowlaw}).
Three independent measurements give nearly the same passage fraction, 0.57 to 0.67 (Table~\ref{tab:fraction}): the
free passages, the window centers, and the fold edges of periods 16--20, which follow $\alpha_N-\alpha_c=C/(N+N_0)^2$
with $C=0.0266$ and $N_0=0.15$. Fixed in advance, this law placed the fold edges of periods 22--30 within 3\,\%; a model
with one fixed reinjection point did not, except at period 30.

\begin{table}[tb]
\caption{Three independent estimates of the passage fraction $f$, the share of a full passage that a loop or a free
trajectory spends in the bottleneck. The fold edges' slope is the rise of $1/\sqrt{\alpha_N-\alpha_c}$ per unit of $N$.
A half passage would give $f=1/2$ and $d_NN^2=\pi^2/(4ab)=0.0201$.}
\label{tab:fraction}
\footnotesize
\begin{ruledtabular}
\begin{tabular}{@{}p{0.25\columnwidth}p{0.27\columnwidth}p{0.26\columnwidth}c@{}}
\raggedright measurement & \raggedright value & \raggedright full-passage value & $f$\tabularnewline
\hline
\raggedright free trajectories, four distances & \raggedright mean laminar phase & \raggedright passage through the whole channel & 0.60--0.67\tabularnewline
\raggedright window centers, $N=4$--7 & \raggedright $d_NN^2=0.0263$--$0.0265$ & \raggedright $\pi^2/(ab)=0.0803$ & 0.57\tabularnewline
\raggedright fold edges, periods 16--20 & \raggedright slope $6.13$ & \raggedright $\sqrt{ab}/\pi=3.53$ (ratio 1.74) & 0.58\tabularnewline
\end{tabular}
\end{ruledtabular}
\end{table}

\begin{table*}[tb]
\caption{Prospective passage tests at four new folds, not used while the bottleneck explanation was worked out;
every forecast was fixed before any test trajectory was run. Each entry gives $G$-iterates for the three starting points: the start of the channel, its
midpoint, and the start displaced transversally. Left, the first test ($\eps$ in $\alpha$ for target A and in $\log_{10}T$ for target B): measured\,/\,forecast, the forecast iterating
the quadratic normal form. Right, the second test, at distances $\eps=\kappa^2/(ab)$ that give both folds the same
forecasts: the continuous passage integral (primary) for the start of the channel and the midpoint, the iterated
map (secondary) in parentheses, and the measured passages. Bold: outside one iterate of the primary forecast.}
\label{tab:prospective}
\footnotesize
\begin{minipage}[t]{0.42\textwidth}
\begin{ruledtabular}
\begin{tabular}{rcc}
$\eps$ & target A ($G=F^2$) & target B ($G=F^8$) \\
\hline
$3\times10^{-6}$ & $118/118$, $\mathbf{59/61}$, $117/118$ & $26/26$, $13/14$, $25/26$ \\
$1\times10^{-5}$ & $56/56$, $28/29$, $56/56$ & $10/10$, $5/5$, $10/10$ \\
$3\times10^{-5}$ & $26/26$, $13/14$, $26/26$ & $4/4$, $2/2$, $4/4$ \\
$1\times10^{-4}$ & $10/10$, $5/6$, $10/10$ & $2/2$, $1/1$, $2/2$ \\
\end{tabular}
\end{ruledtabular}
\end{minipage}\hfill
\begin{minipage}[t]{0.55\textwidth}
\begin{ruledtabular}
\begin{tabular}{rccc}
$\kappa$ & forecast (discrete) & C (2-cycle) & D (period 6) \\
\hline
$0.08$ & $22.40$ $(23)$, $11.20$ $(12)$ & $23$, $12$, $23$ & $22$, $11$, $22$ \\
$0.05$ & $44.29$ $(45)$, $22.14$ $(23)$ & $45$, $23$, $45$ & $\mathbf{42}$, $22$, $\mathbf{42}$ \\
$0.03$ & $85.29$ $(86)$, $42.64$ $(44)$ & $86$, $43$, $86$ & $\mathbf{81}$, $43$, $\mathbf{82}$ \\
$0.018$ & $154.74$ $(155)$, $77.37$ $(80)$ & $155$, $78$, $155$ & $\mathbf{149}$, $78$, $\mathbf{150}$ \\
\end{tabular}
\end{ruledtabular}
\end{minipage}
\end{table*}

These comparisons have two limits. First, the passage formulas integrate the truncated normal form, so they
approximate the discrete dynamics; we test them here rather than prove them. Second, the windows lie outside the
asymptotic regime of the normal form, which assumes a bottleneck much narrower than the channel: here the
bottleneck's width $\sqrt{|a\eps/b|}$ is 1.35 to 3.51 times $U$. The prefactor of about one third is therefore an
empirical finding, and the exponent is near 2 but not pinned to it. Over the fold edges of periods 16 to 32 the
log-log slope is $-2.00$, but an exponent fitted freely together with $N_0$ is not well determined: the edges are
located to a relative precision of $2\times10^{-4}$, so the fits' misfits of 1 to 3\,\% are corrections to the
asymptotic law, and the fitted exponent lies between 2.13 and 2.45 as the smallest $N$ in the fit runs from 5 to
8\smref{Mechanism: a saddle-node bottleneck}.

\paragraph{Prospective tests.} The same reduction, with both coefficients computed from the network's derivatives,
predicts relaxation times without fitting in a small task-trained looped transformer, where it was compared with
measurements after the fact \cite{kim2026phase}. We went one step further and fixed every forecast before measuring.
An independent implementation located four new folds, which were not used while the bottleneck explanation was worked
out, computed their normal-form coefficients by nested automatic differentiation and wrote down the passage forecasts;
only then were the test trajectories run (Table~\ref{tab:prospective}). The passages started at the same three points
as in Table~\ref{tab:passages}: the start of the channel, its midpoint, and the start displaced off the channel's axis.

The normal form gives two forecasts. The discrete forecast iterates the quadratic map \eqref{eq:foldnf} and counts the
iterates. The continuous forecast replaces the steps by a smooth flow and integrates, $n\approx\int du/(a\eps+bu^2)$,
whose closed form is the arctangent usually quoted for saddle-node bottlenecks. The discrete count exceeds the integral
by about $\tfrac12\ln[f(U)/f(u_{\mathrm{in}})]$ iterates, with $f(u)=a\eps+bu^2$, most at small $\eps$ and for the
midpoint start.

The first test used the 2-cycle's fold at $\log_{10}T=-0.140$ (target A) and the fold of the period-8 orbit of
Table~\ref{tab:orbits} continued in temperature at $\alpha=0.0717$ (target B, $G=F^8$), with the discrete forecast as
the primary one. It was exact for the seven forecasts of five iterates or fewer and within one iterate for 16 of the
17 longer ones. Its criterion, that every forecast fall within one iterate, therefore failed once (target A, midpoint
start, $\eps=3\times10^{-6}$: forecast 61, measured 59), and all seven nonzero residuals were negative. The continuous
forecast came within one iterate of all 24 passages.

The second test therefore made the continuous forecast primary, with every forecast at least 8 iterates, at the
2-cycle's fold at $\log_{10}T=-0.200$ (target C) and at the fold of a period-6 orbit of a different family (target D,
$G=F^6$, at $\log_{10}T=-0.087439$). At the 2-cycle's fold it held for all 12 passages, every measured count being the
forecast rounded up. At the period-6 fold the midpoint passages held, but passages from the start of the channel came
in short, by 0.40 to 5.74 iterates and more so closer to the fold, so the criterion failed (18 of 24 within one
iterate; 16 of 24 for the discrete forecast). There the first half of the channel is crossed faster than the normal
form predicts, for reasons the test does not resolve, such as terms beyond the quadratic normal form or the curvature
of the center manifold. Over both tests the continuous forecast came within one iterate of 42 of 48 passages, and the
six misses were 3.1 to 5.2\,\% short.


## Questions

1. Mathematics. Check every formula and its conditions: the fold normal form of G and the sign conventions (a, b,
   the oriented coordinate v = sign(b) u); the stable-branch check (1-mu)^2/(4(alpha_c-alpha)) -> ab; the passage time
   pi/sqrt(ab eps) for a long channel and its finite-channel version; the stated difference of about
   (1/2) ln[f(U)/f(u_in)] between the iterated map and the continuous passage integral; the window law
   d_N N^2 = f^2 pi^2/(ab), the passage fraction f, the fold-edge form C/(N+N_0)^2 and the slope comparison with
   sqrt(ab)/pi; the asymptotic-regime statement (bottleneck width sqrt(|a eps/b|) against U). Flag anything wrong,
   imprecise or missing a condition.
2. Concepts and terms. Are ghost, bottleneck, channel, passage, laminar phase, burst, reinjection and type-I
   intermittency used correctly? Is the account of why the windows crowd toward the fold (one "is X" pair per lap of
   G through the bottleneck) sound, and should the stripes be related to period-adding sequences in the literature?
3. Claims against evidence. Are "no fitted parameter", "within one iterate", the three estimates of f, the
   prediction of the fold edges of periods 22-30, and the description of the prospective tests (two forecasts,
   criteria, failures, the interpretation of the period-6 shortfall) stated fairly?
4. References. Are the cited works (Strogatz and Westervelt 1989; Strogatz's textbook; Pomeau and Manneville 1980;
   Hirsch, Huberman and Scalapino 1982; Kuznetsov; Kim 2026) the right ones for the claims they support? What standard
   reference is missing (for example, on laminar-length scaling, period adding, or normal-form coefficients of maps)?
5. Readability. For a reader who knows calculus and linear algebra but is rusty on bifurcation theory, which sentences
   or steps are hardest to follow, and what is the smallest addition (a definition, a sentence, a figure element) that
   would fix each? Is the new TikZ diagram described in Fig. 15 the right picture?
6. Triage. What must change before posting, and what can wait for the journal version?

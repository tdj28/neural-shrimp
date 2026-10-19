# Consult 20: the night's new results and how the papers state them

## Purpose
arXiv v1 of the Chaos article ("Chaos and bifurcations in a language model's feedback loop") is submitted on 20 October
2026; a full report (172 pages) and a blog post accompany it. The arXiv article now has the period-6 shrimp as its
only shrimp (the period-5 shrimp stays in the report and the blog). Since the first draft of this brief, a
go/no-go review by Codex (GO WITH FIXES) and a separate line-by-line read were applied, and the theory of the
hook (section 6 below) was measured and added to the report and the article. Since consult 19 (12 hours ago) we added several results
and wrote them into all three. Please check them scientifically: is each claim supported by the stated evidence, is
anything stated more strongly (or more weakly) than the evidence allows, are the methods sound, and are the right
references cited? Advice is advisory; we adjudicate locally.

## The system (unchanged)
The soft-token loop of a language model (here Qwen2.5-0.5B-Instruct, float64): the state is an embedding x; one step
feeds x as the next input, takes the output distribution p = softmax(z/T) at temperature T, and returns the expected
embedding E^T p. A steering vector alpha*v (contrastive "philosophy" direction, layer 12) is added to the residual
stream. Maps over (alpha, log10 T) classify each cell by the largest Lyapunov exponent of a twin trajectory (offset
1e-6 renormalized every step, 300 measured steps after 300 transient steps; chaotic when above 1e-3 and no period up
to 32 or 48 is detected) and by the period of the attractor reached from a fixed start (the prefix's last token).
Periodic orbits are Newton-solved with exact forward-mode Jacobians; fold and flip curves are continued by minimally
augmented systems; cusps are located by the oriented quadratic coefficient.

## New results, with the exact text now in the report (the article has the same text, shorter where noted)

An independent audit (Codex) of these results and their wording was applied before this consult: it corrected the
DSS6 determinant claim, the pooled (not per-phase) error bound, the meaning of the C labels, Singer's endpoint
alternative, the threshold-only chaos labels of the prompt maps, the random-start claim and several bounds. The
text below is the corrected version.

### 1. The kite of the period-6 shrimp
The rest of the kite
follows from continuation (Figure~\ref{fig:p6kite}; run KITE). Continued in $\alpha$ at fixed $\log_{10}T=-0.15395$ from
the first orbit's flip, the branch is S-shaped: the orbit stays attracting up to a fold at $\alpha=0.072992$, the
branch returns along the saddle to a second fold at $0.072793$, and it continues on the second orbit, which flips near
$0.07301$; at that temperature the two attracting orbits coexist for $\alpha$ between 0.072907 and 0.072992. The second
orbit, continued upward in $\log_{10}T$ from its flip at fixed $\alpha$, is attracting up to its own fold at
$-0.153657$. The fold curves traced from the two folds turn back at the same point, the kite's cusp C, located at
$(0.0730186,-0.1532803)$ (cubic coefficient $c=-1.33\times10^{4}$, unfolding determinant $\det J=-2624$, critical
multiplier within $7.7\times10^{-9}$ of $+1$, next modulus $0.00095$), on the side of R toward the head. Its two
branches meet the two flip curves at A $=(0.072991,-0.153984)$ and B $=(0.072900,-0.153707)$, completing the kite
C--A--R--B of Gallas's cell. Besides C,
the fold curves have five more located cusps (Table~\ref{tab:p6cusps}): near the ends of the upper-left and lower-left
legs, two pairs that share a segment, each with cubic coefficients of opposite sign, the configuration of the period-5
window's upper leg; and one at the tip of a small separate window beyond a narrow neck below the body, where the orbit
is stable again after a doubling; two branches, run independently, trace that fold curve and its turn to within one
curve step.

On a $31\times31$ grid of the continued orbit over the body ($\alpha$ from 0.0724 to 0.0738, $\log_{10}T$ from $-0.1545$
to $-0.1500$; 529 of 961 cells solved), the leading multiplier is nonzero at every solved cell. Its smallest modulus on the grid,
$0.0063$ at the cell $(0.073053,-0.15120)$ on the line of strongest contraction, belongs to a nearly imaginary
leading pair, $0.0004\pm0.0063i$, with a third
multiplier of modulus $0.0051$: three multipliers of nearly equal small modulus, the near-nilpotent signature of the
period-5 head. The dark line of strongest contraction runs from the flip crossing up the body and bends to the upper
left, and all nine cells with a complex leading pair lie on its upper part (arguments 43 to 135 degrees, moduli 0.006
to 0.038). At 289 solved cells the two largest-modulus multipliers form a real or complex-conjugate pair, and their real product
is negative at 143; at the other 240 the leading real multiplier is followed by a complex-conjugate pair, so the first
two alone define no real two-dimensional determinant, and we infer no determinant-zero curve from them. The real
leading multiplier changes sign across 45 neighboring solved pairs; only two have leading and next moduli within a
factor three at both ends, and this endpoint test does not tell a multiplier crossing zero from an exchange of
dominance. For a smoothly tracked
two-dimensional invariant subspace the eigenvalues solve $\lambda^2-\tau\lambda+\delta=0$, so $\delta=0$ marks a zero
eigenvalue of that block, not of the leading multiplier; if $\delta$ has a nondegenerate saddle with a nonzero saddle
value, its zero set near the saddle has two hyperbolic branches that approach and turn away, and the eigenvalues are
complex where $\tau^2<4\delta$. This conditional geometric model is consistent with the complex pairs on the upper part
of the line, but it is not an established explanation (consult~18, \repo{docs/consults/2026-10-19-avoided-crossing}):
at the minimum the third multiplier is comparable, and neither a saddle of $\delta$ nor a coupling of the two directions
has been measured. Near the kite, where two orbits exist, curves that belong to different orbits can cross in the
parameter plane without any degeneracy, as the superstable curves of Gallas's two coexisting fixed points do at his
tail. With these qualifications the period-6 window meets the three numerical criteria along its body, with six located
cusps and the kite of Gallas's cell: a second shrimp-shaped window of the full map, in the plane of
Figure~\ref{fig:cross512} itself. The far parts of its legs remain unresolved.

Boundary gate (same section):
Within the window of Figure~\ref{fig:p6shrimp}a, the edge of the followed domain lies within two grid steps of the
curves for all 232 neighboring pairs of periods 6 and 12 and all 459 pairs of period 6 and a lost attractor (median
distances 0.19 steps); over the whole followed domain the shares fall to 388 of 916 and 1{,}720 of 1{,}888 (medians 4.3
and 0.24 steps), because the far parts of the legs leave the traced curves, so their boundary is not resolved.

DSS6 grid (same section, report only):
At 289 solved cells the two largest-modulus multipliers form a real or complex-conjugate pair, and their real product
is negative at 143; at the other 240 the leading real multiplier is followed by a complex-conjugate pair, so the first
two alone define no real two-dimensional determinant, and we infer no determinant-zero curve from them. The real
leading multiplier changes sign across 45 neighboring solved pairs; only two have leading and next moduli within a
factor three at both ends, and this endpoint test does not tell a multiplier crossing zero from an exchange of
dominance.

Figure caption:
\caption{The kite of the period-6 shrimp (Qwen2.5-0.5B-Instruct, philosophy direction at layer 12; float64). (a) A $512\times512$ float64 map of the lower-left end of the body, with cells 7.5 times finer in $\alpha$ and 4.6
times finer in $\log_{10}T$ than in Figure~\ref{fig:p6shrimp}a (gray: periodic, darker where more contracting; colors: chaotic), with the body's two flip
curves (red, dashed), the fold curve continued from the branch of (b) (black), and the corners of Gallas's kite: R,
where the flip curves cross (two distinct orbits, one on each curve); A and B, where the fold curve meets them; C, the
located cusp where the fold curve turns back (gold star). Green: the line $\log_{10}T=-0.15395$ of the branch, thick
where the two attracting orbits coexist. (b) The branch: the orbit's position along the leading direction of the
branch's states against $\alpha$, solid where the orbit is attracting and dashed where not; two folds (open circles)
make it an S, and the two attracting sheets overlap in the shaded interval (\repo{tools/plot_p6_kite.py}). Inside the kite, the straight edge where the gray changes abruptly (near $\log_{10}T=-0.15353$) is not a
bifurcation: on both sides of it both attracting period-6 orbits exist (leading multipliers $0.18$ and
$0.60$ to $0.61$), and the fixed start reaches one above the edge and the other below (\repo{tools/kite_basin_check.py}).}

### 2. The hook (report only; 640 x 640 zoom; two Newton basin checks; a random-start rerun)
\paragraph{The hook.} A $640\times640$ map of the upper end of the kite (Figure~\ref{fig:p6hook}; cells 3.1 and 2.5
times finer than in Figure~\ref{fig:p6kite}a) resolves the hook-shaped light wedge and the striations below it. The
wedge is bounded above by the kite's fold curve, where its orbit disappears, and below by a smooth curve where the fixed
start switches to the other attracting orbit; at adjacent cells across that curve both orbits exist and attract
(multipliers $0.18$ and $0.60$ to $0.61$; pair 1). The striations are tongues bounded the same way: across the edge of
one, two other attracting period-6 orbits exist at both cells (multipliers $-0.72$ and $-0.11$; pair 2), and inside
several tongues the exponents of neighboring cells fall into two separated groups, so the fixed start's outcome
alternates from cell to cell, on a scale below the map's cells (\repo{tools/plot_p6_hook.py},
\repo{tools/kite_basin_check.py}). From a random token's embedding as the start, the same box at $128\times128$ cells shows neither
the hook nor the tongues, and only 38\,\% of its cells keep their exponent within 0.02 (run HR12-RS): these edges are set
by where the fixed start lies relative to the basins, as basin boundaries should be, not by bifurcations.

Figure caption:
\caption{The hook and the striations at the upper end of the kite (Qwen2.5-0.5B-Instruct, philosophy direction at layer
12; float64). (a) A $640\times640$ map of $\alpha\in[0.07291,0.07303]$ and $\log_{10}T\in[-0.1540,-0.1533]$ (gray:
periodic, darker where more contracting; colors: chaotic), with the kite's fold curve (black) and a flip curve of the
body (red, dashed); cyan: the two pairs of adjacent cells checked by Newton continuation, on the wedge's lower edge (1)
and on a tongue's edge (2). (b) The white box in (a) at the map's own cells: tongues in which the outcome alternates
from cell to cell.}

### 3. The prompt study (22 variants, 128 x 128 maps of the standard plane)
\paragraph{The prompt.} Every 0.5B result above uses one prompt, so we mapped (run PV, \repo{tools/pv_summary.py}) the standard plane ($\alpha$ from
$-0.12$ to $0.12$, $\log_{10}T$ from $-1$ to $0.3$; $128\times128$ float64 cells, philosophy direction at layer 12) under
22 variants (Figure~\ref{fig:prompts}): each of its two parts alone, and neither (Qwen's default system prompt, which the
chat template inserts when none is given); the system prompt's words in two random orders, the user message's two
tokens reversed, and both changes together; an opposite system prompt; happy and sad versions of the system prompt and of the user message; the
same texts without chat markup, a system prompt three times as long and a two-turn conversation; three prompts that are
a single random vector, with no text at all; and the base prompt with the loop started from three random vectors.
Every prompt gives chaos with periodic windows: 5.9 to 14.9\,\% of the plane is
chaotic (7.0\,\% for the base prompt), with largest exponents of 0.62 to 0.85 per step (here chaotic means a largest exponent above $10^{-3}$ alone; this
finite-window indicator overlaps a detected period at 3 to 18 cells per map). Removing the user message doubles the chaotic share, and even a prompt that is one random vector gives
8 to 10\,\%. The prompt decides where the chaos lies, not whether it occurs: the
chaotic cells of a variant overlap those of the base prompt by 1 to 37\,\%
(Jaccard index). For the three random starts, the chaos/non-chaos threshold labels agree with the base run at
98.9 to 99.0\,\% of the cells, while the periods detected within the run agree at only 86.5 to 93.1\,\%, so within
this horizon the orbit reached can depend on the start. Across the 19 prompts there is
nonetheless a common structure (Figure~\ref{fig:promptfreq}): 48.3\,\% of the cells are chaotic for at
least one prompt but only 3.0\,\% for half of them or more, and these hot spots lie at moderate
steering of either sign in a band of temperature, where the mean fraction of chaotic cells per row peaks at 38\,\% near $\log_{10}T=0.15$, against 0.8\,\% below
$\log_{10}T=-0.5$. The three random-vector prompts move the band to colder temperatures: their mean profile peaks
near $\log_{10}T=0.03$, that of the text prompts near $0.16$.

### 4. Symbolic dynamics of the period-6 shrimp (both papers; Table p6words in the report only)
\paragraph{Symbolic dynamics.} Jones's unimodal interpretation of selected R\"ossler return maps, in which the
shape of a periodicity hub follows from a conjugacy to unimodal maps \cite{jones2012rossler}, suggests an approximate
scalar coding of the six-band attractor. Where the attractor writes the body's six-token text, the loop visits six thin
bands of the state space in turn, one per token, and returns to each band every six steps. Many sampled six-step return
maps, plotted along a band's first principal axis, show a single turning point: the next visit traces a parabola against
the previous one (Figure~\ref{fig:p6returnmaps}). We code each visit by the branch it lies on, L on the increasing one
and R on the decreasing one. The branch is read from the sign of the six-step derivative along the band, which the twin
trajectory of the Lyapunov exponent provides (\repo{tools/symbolic_dynamics.py}): the twin's offset direction is carried along the
attractor, and its scalar product with its own direction six steps earlier has the sign of that derivative where the
offset is aligned with the band and the band turns by less than a right angle. The checks below support this
approximation; they do not prove it. Pooling the misplaced visits and the rarer symbol's visits over the phases where
the band is straight enough for the sign to be defined, 48 of the 58 chaotic six-band cells of a $48\times48$ map of
the window have an error ratio below 5\,\% (the largest passing ratio is 4.7\,\%, 23 cells have no misplaced visit, and a single
phase of a passing cell reaches 21.7\,\%). The other 10 cells fail the coding test: three in one column beside the body's left
flip curve and seven below the kite. In 7 of them the return map has more than one resolved turning point: it rises
and falls more than once (Figure~\ref{fig:p6returnmaps}h--k). For a $C^3$ unimodal interval map with negative
Schwarzian derivative, at most one attracting periodic orbit can have an immediate basin that avoids the interval's
endpoints \cite{singer1978stable}; the extra turning points below the kite are consistent with a scalar explanation of
the coexistence inside it, but they establish neither a global scalar reduction nor its Schwarzian sign.

A periodic orbit of period $6k$ visits each band $k$ times. Its signs, rotated to the largest sequence in the order of
kneading theory, give it a label W|X in the sense of Metropolis, Stein and Stein (MSS)
\cite{metropolis1973finite,milnor1988iterated}: W is matched to the quadratic family's superstable itinerary WC and
names the window, and X records the orbit's inferred last branch symbol. Away from a superstable parameter this is an
orbit label, not a measured itinerary of the critical point. After a majority vote over the band phases whose coding is usable (at least three) and an explicit filter that keeps
only MSS words, 44{,}577 periodic six-band cells on the $640\times640$ grid of Figure~\ref{fig:p6shrimp}a receive
23 distinct window words W (Figure~\ref{fig:words}); the filter rejects 240 further cells whose labels are not
MSS words. The body and its period doublings carry the words of the doubling cascade: the empty word (period 6), R (12),
RLR (24) and RLRRRLR (48). Windows in the surrounding chaos carry other words, among them RL (period 18), RLRR (30), and
RLRRR and RLLRL, two distinct windows of period 36 that only their words tell apart. Within a window, the label predicts
the sign of the orbit's leading multiplier, $(-1)^{n_R}$ with $n_R$ the number of R's in W|X: positive next to the
boundary where the orbit is born (a fold, or for a doubled orbit the flip curve of its parent) and negative next to its
own flip. The curve where the predicted sign changes is a candidate for the window's superstable curve. The scan
assigns C to 153 cells whose usable phases do not agree sufficiently on the last symbol; at the two selected C cells,
Newton calculations report small, nonzero complex leading pairs. These labels mark ambiguous scalar coding, not
located superstable curves. Float64 Newton solves with exact Jacobians at one cell for each label W|X with at least three coded cells in the half
of the scan then complete (the cell nearest the center of the label's largest patch: 23 cells, covering 12 of the 23
words; Table~\ref{tab:p6words}), scored by a subsequent clean-phase rule (phases whose L and R visits one threshold
separates along the band, with every $|\cos|\ge0.5$), reproduce the scan's label at 16 cells; the two C cells have
complex leading pairs, and at the remaining 5 cells, just below the kite, no phase is clean, so their four words (RLL on both sides, RLLLRLL,
RLLLRLR and RLLLRRR) remain unconfirmed; the other 11 words were not checked. These checks support selected labels, not the whole map. In total, 1{,}469 of the
46{,}046 eligible periodic six-band cells receive no word (3.2\,\%), the 240 rejected by the filter among them; 922 of them (63\,\%) lie within Euclidean
distance two grid cells of the eligible domain's edge.

Captions:
\caption{Return maps of the six-band chaos around the period-6 shrimp (Qwen2.5-0.5B-Instruct, philosophy direction at
layer 12, float64; 1000 transient and 3000 recorded steps per cell). (a) The cells of a $48\times48$ map of the window of
Figure~\ref{fig:p6shrimp}a: chaotic cells writing the six-token text that pass the pooled coding test (green: misplaced visits
below 5\,\% of the rarer symbol's, pooled over the usable phases) or fail it (red), other chaotic cells (gray) and periodic cells (white). (b--g) Six-step return maps at six green
cells: the position along one band (its first principal axis) against the position at the previous visit; blue:
visits where the six-step derivative is positive (L), red: negative (R). (h--k) Return maps at four red cells (letters
in (a)) with more than one turning point.}

\caption{Symbolic words of the period-6 shrimp's windows ($640\times640$ float64 cells, the grid of
Figure~\ref{fig:p6shrimp}a). (a) Each periodic cell of the six-band region colored by the word of its window: pale
where the label predicts a positive leading multiplier, full where it predicts a negative one, magenta where the
phases disagree on the last symbol (C); light gray: chaos; mid gray: other periodic orbits; dark gray: no word (fails the coding test). Windows marked ``not
confirmed'' in the legend are those of Table~\ref{tab:p6words}'s cells without a clean phase, and those marked
``not checked'' had no float64 check. (b) The box in (a). (c) The same words at the
superstable parameters of the quadratic family $x\mapsto a-x^2$, on its bifurcation diagram (the body's empty word is
that of its fixed point, at $a=0$); labels give the period of the soft-token orbit, six times the word's period.}

\caption{Float64 checks of the symbolic words (\repo{tools/sd_verify_words.py}): Newton with exact Jacobians at the
center of the largest patch of each window word and side of Figure~\ref{fig:words} with at least three cells in the
eight of the scan's 16 bands then complete (23 cells covering 12 of the 23 words; the other 11 were not checked). Word and
side: the scan's label; $\mu_1$: the orbit's leading multiplier; clean: the phases whose L and R visits one threshold
separates along the band, with every $|\cos|\ge0.5$; result: the clean phases give the scan's word (confirmed), no
phase is clean, or the leading pair is complex (side C: the scan's phases disagree on the last symbol).}

### 5. Exponent offset (report, methods)
The averaging window includes the twin's alignment from
a random direction in 896 dimensions, so the finite-window estimate of a periodic cell differs from $\ln|\mu_1|/p$
of its orbit, by amounts consistent with that transient (its contribution has not been isolated): at two adjacent cells inside the kite of Figure~\ref{fig:p6kite},
the map gives $-0.2786$ and $-0.0885$ per step and the Newton-converged period-6 orbits $-0.2821$ and $-0.0813$, offsets
of $+0.0035$ and $-0.0073$ (\repo{tools/kite_basin_check.py}). Periodic cells are classified by their period, not by
this exponent; the gray scale of the maps carries the offset.

### 6. The theory of the hook (new: report paragraph and figure; the article has a shorter paragraph, and its kite
caption now calls the straight edge a basin boundary "where we checked it")
\paragraph{Why the lower edge lies where it does.} An edge of the fixed start's outcome that is not a bifurcation is
where the start crosses the boundary between two basins, and the boundary between the basins of two attracting orbits
is typically the stable manifold of a saddle orbit between them. At $\alpha=0.072975$ we bisected the wedge's lower
edge in $\log_{10}T$ to the float64 limit, at $\log_{10}T=-0.1535299666043828$ (\repo{tools/hook_saddle_check.py}).
Started there, the loop comes close to a third period-6 orbit and stays near it: Newton's method from the most nearly
periodic state of that stretch converges (residual $2\times10^{-14}$) to a saddle with a single multiplier outside the
unit circle, $\mu_u=1.2625$, real, with the next modulus $7.8\times10^{-4}$, at relative distances
$3.5\times10^{-3}$ and $4.7\times10^{-3}$ from the two attracting orbits. A start a small distance off the saddle's
stable manifold comes within a distance proportional to that offset of the saddle and then leaves along the
unstable direction, its distance growing by $\mu_u$ per period, so the time it spends near the saddle grows by
$6/\ln\mu_u=25.74$ steps for every factor $e$ by which the offset shrinks. Both predictions hold over eight decades
(Figure~\ref{fig:hooktheory}; \repo{tools/hook_saddle_exit.py}). At 17 temperatures on each side of the edge, a
distance $\delta$ from $10^{-6}$ to $10^{-14}$ in $\log_{10}T$ away, the last step within $10^{-3}$ of the saddle
grows by 25.78 and 25.75 steps per factor $e$ in $\delta$ (all points within 4.0 and 2.9 steps of the fitted lines),
the closest approach grows with $\delta$ with log-log slopes 0.986 and 0.995, and every start below the edge ends on
the strongly contracting orbit and every start above it on the other. Together with the random start, which removes
the edge, this identifies the lower edge as the fixed start crossing the saddle's stable manifold. Bisecting 676
jumps of the outcome along the map's columns (to $2.6\times10^{-13}$ in $\log_{10}T$; \repo{tools/hook_edge_track.py})
traces the hook's other edges (Figure~\ref{fig:hooktheory}a). The upper edge follows the kite's fold curve, 0.1 to
1.6 grid rows above a spline through the continued curve's points and more so toward the cusp, perhaps because just
beyond the fold the loop lingers at the vanished orbit for longer than the measurement window; we have not tested
this. Where the two edges meet, the saddle-node normal form predicts a tangential contact at the point of the fold
where the fixed start lands on the saddle-node orbit, the gap between the edges growing quadratically along the fold;
at the map's resolution the lower edge leaves the fold within one cell, so this prediction is untested.

\caption{The hook's lower edge is where the fixed start crosses the stable manifold of a saddle orbit
(Qwen2.5-0.5B-Instruct, philosophy direction at layer 12; float64). (a) The map of Figure~\ref{fig:p6hook}a near the
hook, with the kite's continued fold curve (gold) and the edges of the fixed start's outcome bisected along the map's
columns (cyan; brackets whose end exponents differ by at least 0.1); red star: the steering strength
$\alpha=0.072975$ of (b)--(d). (b) Distance to the saddle orbit along the loop started a distance $\delta$ in
$\log_{10}T$ below the edge: it arrives along the stable manifold within about 25 steps, lingers, and leaves along
the unstable direction; dotted: the radius $10^{-3}$ of (c). (c) The last step within $10^{-3}$ of the saddle
against $\ln(1/\delta)$, for starts below the edge (which end on the strongly contracting orbit $S$) and above it
(which end on the other, $W$); dashed: slope $6/\ln\mu_u$ from the saddle's unstable multiplier $\mu_u$. (d) Closest
approach to the saddle against $\delta$; dashed: proportional to $\delta$.}

## Questions
1. Symbolic dynamics (as corrected by the audit). Is the coding sound as described: (a) reading L/R from the sign of the twin's six-step scalar
   product, with the straightness and |cos| gates; (b) the maximal rotation as the kneading word of a non-superstable
   orbit in a window, and the last symbol as the side of the window's superstable curve; (c) the multiplier's sign as
   (-1)^(number of R's in the whole word); (d) the majority vote over phases and the clean-phase standard for the Newton
   checks; (e) calling five cells "unconfirmed"? Is anything overclaimed, e.g. "the window's organization follows from a
   one-dimensional map" given 48 of 58 cells and the exceptions?
2. Singer's theorem: is the sentence now in the text fair? (Both coexisting kite orbits have positive multipliers,
   +0.184 and +0.614.)
3. The hook and the striations: are "basin boundary of the fixed start, not a bifurcation" and "the outcome alternates
   from cell to cell, on a scale below the map's cells" supported by two adjacent-cell Newton continuations, the
   two-group exponent histograms, a random-start rerun in which the hook and tongues vanish (38 % of cells keep
   their exponent), and the saddle evidence of section 6?
4. The prompt study: are "Every prompt gives chaos with periodic windows" and "The prompt decides where the chaos lies,
   not whether it occurs" fair for 19 prompts and 3 random starts of one model, one direction and one plane?
5. The exponent-offset sentence: right cause and right scope?
6. Missing references: kneading theory for two-dimensional or strongly dissipative maps, shrimps and their windows'
   words, basin boundaries in parameter planes; anything a referee would expect. Please give full bibliographic data
   (we verify each against the publisher record before citing).
7. Anything in these passages a careful referee would flag, including wording.
8. The theory of the hook (section 6): does the evidence (a Newton saddle from the lingering state with one unstable
   multiplier, exit times growing by 25.78 and 25.75 steps per e-fold against 6/ln mu_u = 25.74 over delta from 1e-6
   to 1e-14, closest approach with log-log slopes 0.986 and 0.995, every start ending on its side's orbit, and the
   random start removing the edge) identify the lower edge as the fixed start on the saddle's stable manifold, at
   one steering strength? Is anything missing (e.g. a check that the exit direction is the unstable eigenvector, or
   the edge's dependence on the start)? Are the upper-edge offset (0.1 to 1.6 rows above a spline through the
   continued fold, with an untested finite-horizon explanation) and the untested tip tangency stated at the right
   strength? Is the normal-form reasoning for the tip (tangential contact, quadratic gap) correct?

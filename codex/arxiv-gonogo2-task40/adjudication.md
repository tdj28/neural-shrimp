# Task 40 adjudication (Claude, 20 October 2026)

Codex's verdict on article bb42a82: GO WITH FIXES (9 must-fix, 7 should-fix, 1 note). Every article finding was
accepted; the fixes are in article commit e300c8e, and the report carries the matching fixes.

| finding | decision | what was done |
|---|---|---|
| F1 natural-passage errors | accepted | Statistic and channel stated (complete phases within \|u\|<3U and transverse distance below 0.05; median signed errors -7.7 to -4.1 %; single phases up to 71 %); Table IV names the same definition; report likewise, with the four medians. |
| F2 cusp bracket range | accepted | "magnitude between 0.17 and 39.11 at the twelve bracketing curve points"; article and report. |
| F3 abstract and Soft Thinking | accepted | "a smooth full-distribution variant of Soft Thinking"; article and report. |
| F4 introduction's synchronization claim | accepted | Growth rates approximately follow the prediction; locking also depends on start, duration, tolerance and attractor. |
| F5 cascade reduction factors | accepted | "from n = 4 on ... about 3.52, 6.54 and 3.93; the next (5.26) not resolved by the location uncertainty". |
| F6 fixed-start identity | accepted | Nearby fixed-start scans find period six in all but two unresolved cells; same-parameter full-state identity not checked at every point; article and report. |
| F7 localization scales and rounded bounds | accepted | Estimated localization scales 2e-5 to 2e-4 with their two error models; 3.1e-13; within 2.3e-6 (article) and "about 2.2e-6" (report). |
| F8 availability statements | accepted as a release gate for the human | The present tense stays, because the human plans to make the repository public and to publish the Zenodo record before the arXiv announcement. Before announcement, both landing pages must be verified; otherwise the wording changes to "available on request" and "DOI reserved". |
| F9 "most" period-2 pairs | accepted | "some"; table footnote reworded; article and report. |
| F10 bifurcation from every start | accepted | Bifurcation curves are properties of the map, while fixed-start scans show only attractors reached from that start; basin-boundary interpretation supported by the coexistence and saddle checks. |
| F11 tracked block, local result | accepted | "a two-dimensional block, initialized from the leading Ritz directions and tracked by subspace overlap"; table row "shown numerically on the patch"; report notes the pair is not always the two locally largest. |
| F12 Backour's assumptions | accepted | Unbiased two-token reduction, linear logit transport, positive constant coupling. |
| F13 descent wording | accepted | "the descent estimate alone does not establish convergence to a single fixed point when fixed points are not isolated". |
| F14 Fujisaka-Yamada attribution | accepted | The condition is derived for our coupling and follows their Lyapunov-stability framework. |
| F15 stray "Near"; figure placement | accepted | "Four of these five"; Fig. 25 moved to the page after its first mention; Fig. 22 within one page; Fig. 15 and Fig. 19 still two pages from a first mention (queued full-width and stacked floats). |
| F16 audit limits | noted | Kept as stated; the stale "Ritz residuals not computed" remark in the review log is superseded by Codex's record of the P384/P768 residuals. |
| F17 sampled passages | accepted | "become rare, one to three per offset"; 7.8 and 10.5 steps at the smallest offset (7.6-7.8 and 10.3-10.7 over the offsets). |

Also applied in e300c8e, from a third external review supplied by the human:
- Reference 17 (Ko and Geiping) is cited as arXiv:2606.30571. Its COLM 2026 venue could not be verified, and ICML 2026's
  virtual pages list it only as a poster at two ICML 2026 workshops.
- The 42/48 result is given by stage: 24 of 24 as the secondary forecast in the first stage, where the primary forecast
  held for 23 of 24, and 18 of 24 as the primary forecast in the second.
- One sentence separates the breadth of the scans from the depth of the analysis, which is done in Qwen2.5-0.5B, with a
  second cascade in 1.5B.

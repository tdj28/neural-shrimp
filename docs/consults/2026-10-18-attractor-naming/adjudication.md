# Consult 16 adjudication: attractor naming

Reviewer: gpt-6-astra (Pro, high effort). Raw bundle: neural-shrimp-data/consults/2026-10-18-attractor-naming/execute-001.
Advice is advisory; each point was checked against the evidence and adopted, adapted or declined.

| # | Recommendation | Decision | Where applied |
|---|---|---|---|
| 1 | "Attractor" is right for a discrete-time map, but name the convention and qualify the chaotic sets as numerically observed; coexistence makes it "a coexisting attractor"; certified cycles need no qualifier | Adopted | Paper: definition added to "Terms and scope"; attractor paragraph says "one numerically observed chaotic attractor" with the coexisting period-4 orbit |
| 2 | Prefer "chaotic" to unqualified "strange"; D_KY is a Lyapunov-dimension estimate, not a measured fractal dimension | Adopted | Paper caption "Projected geometry of a numerically observed chaotic attractor"; D_KY described as a quantity of the exponents; videos retitled "A chaotic attractor ... (numerically observed)"; narration says "chaotic attractor" and "Lyapunov dimension, an estimate" |
| 3 | The loop is not a Poincare return map; call it the soft-token feedback map; label s_{n+1} vs s_n a lag-one plot; "stretching and folding" is an interpretation | Adopted for the loop and the new panel; established scalar "return map" uses elsewhere kept, with a definition | Terms paragraph defines "return map of a scalar observable"; panel (d) "lag-one plot of the first principal coordinate" |
| 4 | Say "projection" on first mention | Adopted | Caption and panel (a); the tour footer says "projection on principal coordinates" |
| 5 | The closure sentence overclaims | Adopted (reviewer's replacement, with distances to sampled manifold points) | Paper |
| 6a | Delay evidence supports a useful reconstruction, not a verified Takens embedding | Adopted | New sentence on the delay reconstruction |
| 6b | Jump length does not establish disconnectedness or stretching/folding | Adopted | Paper: "consecutive states usually lie far apart on the attractor rather than along one filament"; narration keeps the measured jump length and gives the boundedness argument for folding |
| 6c | Legend: the color is the most probable next token (the loop feeds back a probability-weighted embedding) | Adopted | All video legends, tour footer, narration |
| 6d | No period through 64 is not aperiodicity; finite-time exponents do not exclude chaotic saddles; variance explained is not a reconstruction bound; real map vs rounded implementation | Already covered by "Terms and scope" (finite-horizon labels, long transients) and the float32/float64 labels; no new claim made beyond them | - |
| E1 | Experiment: full-state, phase-resolved unstable manifold with transverse kicks at the banded cell | Deferred, recorded as the next strengthening of the manifold result; needs float64 GPU time | Research log |
| E2 | Experiment: persistence to 8192 steps, period search to 256, at the strong cell | Deferred; cheap enough locally in float32 as a pilot | Research log |
| E3 | Experiment: does the lag-one plot close (scalar vs delay predictors)? | Deferred | Research log |

Declined: nothing. The reviewer's broader research roadmap is noted, not adopted wholesale.

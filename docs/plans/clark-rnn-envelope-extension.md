# Proposed fixed farther RNN envelope, adapted after low-g turns

The saved combined geometry shows nested bent fold/flip curves, without yet
resolving internal cusp/two crossing flip branches. Their simple-mode coefficient
sign jumps near second±1 crossings must not be called cusp/generalized flip.
Root requested a finite farther envelope, not indefinite continuation to closure.

Exactly continue `extend1:192` and `extend-1:192` from
p5-lowgain-extension-001, manifestc3de676fd0ded289f9214ec1aba38eead7d1e166a15c33a692e4fe306c88aae9. No prefix repetition or high-g-arm restart.
Same native period5 map, scale, step.015, Newton/SVD/primitive/simple gates,
phase/sign identity, reverse checks and analytic fold/flip coefficients.
384 new nodes at most per trace; stop at gain>=20, delta<=.05, original domain
edge, rank/identity/correction failure, or cap. Check envelope on predictor and
returned state; retain an out-of-envelope returned attempt without accepting it.
Nonattracting-complement nodes and coefficient failures remain explicitly labeled.

CPU1nice10,120s135+5external,448MiBworker+64MiBheadroom,256MiBraw+1MiBlauncher,
14000 family-call cap. Ten non-target resource samples first; projection
13000*max_unit+20<=120s. The increased allowance bounds up to768 full forward/
reverse records and third-directional coefficients. Fresh shared-host check,
noGPU/paid/protected data. No retry/halving/reselection.

Proposed exclusive command:
`nice -n10 /data2/neural-shrimp/.venv/bin/python tools/launch_clark_rnn_envelope.py --out /data2/neural-shrimp-data/explore/clark-rnn/p5-envelope-extension-001`.
Root coordinates this proposed target after receiving the fixed card. Any
projected curve intersection is only a geometric nomination: common full-state
orbit requires independent phase/sign matching. Finite anatomy may comprise
arms/body/crossings without literal closure; exact claim follows evidence.

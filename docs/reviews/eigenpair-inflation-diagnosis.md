# Saved inflation proposal — no new proof

The one positive-resolvent reduction completed in3.493s/600184KiB with zero
native calls. All470 proposed exact radii retain their old values and increase
by at most0.2612%. The original old-domain floating maximum image/radius is
1.00245664; the proposed old-domain proxy is0.99984686. The solve residual has
maximum absolute value6.47e-27. This narrow proxy margin supports one fresh
domain check; it does not prove that the enlarged domain passes.

Source/execution41ef113b76c1fcc4e19fa25814379a82245dd0d2. All90files/3,407,900bytes
are immutable, fully inventory/source-verified, and owned PIDs1400813/1400898/
1400899 are absent. Companion `eigenpair-inflation-diagnosis.json` binds the full
result including all470 exact proposed radii and the failed original chart.

The proposal adds the exact old radius to a floating correction converted to
an exact dyadic plus a relative2^-60 floor. Its resulting mantissas can exceed
256bits. A prospective target adapter must round each radius outward to a fixed
shorter dyadic representation, explicitly retaining that change and verifying
no radius shrank. This is representation preparation, not a repeated solve or
relaxation. It must recompute the full enlarged-domain proof and anchor gates.

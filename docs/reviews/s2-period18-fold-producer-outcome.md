# A local primitive period-18 fold is certified

The exact small-network map has a local boundary where a stable period-18
training pattern and an unstable period-18 cycle meet. This result certifies a
local saddle-node, with a neutral +1 multiplier at the meeting point. It does
not certify the entire shrimp window or a continuous connection from this
boundary point to the previously certified interior orbit. Independent outcome
review is pending; the kernel received separate mathematical source review.

The second disclosed attempt passed every interval gate in the13-dimensional
biased S2 model. Its27D normalized augmented root is locally unique, the orbit
has primitive period18 (proper divisors1,2,3,6,9 excluded), the multiplier+1 is
algebraically simple, and every other multiplier is strictly inside the unit
circle. The fixed output rate is exactly
`1848862009810323/2251799813685248`, byte-identical to historical row111.
The actual hidden rate is enclosed by the outward decimal interval

```
[75.76429203111452151296084206710593707389,
 75.76429203111452151296084206710751479571].
```

With `R=F_eta^18-I`, `v^T v=1`, `w^T v=1`, and eta the actual hidden rate,
the exact saved enclosures imply:

| Quantity | Outward simplified bound |
|---|---:|
| Augmented Krawczyk contraction | <7.502e-12 |
| Transpose-left-border inverse defect | <1.996e-10 |
| `a=w^T R_eta` | [-0.015615973444,-0.015615973264] |
| `c=w^T D_theta^2 R[v,v]` | [-8.000116058,-8.000115956] |
| Actual-rate branch curvature `eta''=-c/a` | [-512.3033910,-512.3033786] |
| `||(M-vw^T)^32||inf` | <0.035629 |

The reduced scalar equation uses `(c/2)t^2`, not `ct^2`. The negative branch
curvature supports a local maximum in actual hidden rate: the nearby pair is
on the lower-rate side, qualitatively. No explicit finite-detuning neighborhood
or nearby attracting-orbit enclosure has been computed. These a and eta''
values use actual rate and must not be directly compared with the numerical
ray-coordinate derivatives without the change-of-variable factor.

Attempt001 requested radius2^-80 and passed six gates, but its deflated-power
enclosure failed. Its smallest bound was12.114, so it remained a partial
certificate. That failure is preserved. A dated, root-authorized amendment
changed only the radius to2^-100 for attempt002;256bits, five Newton proposals,
powers through4096, target, exact data and all mathematical gates stayed fixed.
The mixed-jet kernel itself was unchanged. The successful proof is one
bifurcation certificate, **not an additional attracting-orbit certificate**.

The successful run took1.722s internally/1.939s externally on one CPU thread at
nice10, with140 ticks and54,024KiB peakRSS. All23 files (2,843,370 bytes) are
hash-verified and read-only; all three owned launcher/worker PIDs have exited.
The whole state box is within1.671e-11 normalized infinity distance of the
saved numerical fold seed, computed exactly from serialized dyadics. This is
proximity, not proof that the seed is itself a root or that the two studies
tracked a common global branch.

`certificates/s2-period18-fold-receipt.json` binds source, inputs, all exact
bounds, both attempts, and the standalone replay invocation. Target source is
`c37dcb4e56a64fa7567cc2e03fe5917195a9f0fc`; the successful run manifest is
`9677cd6817d3e3cdfb45dcc10656a5660b79fec2b98aa7365ab268c705c36e5b`, inside the
outer manifest `4473641efda1f5541fbaaa1126f70316df521618179a3faafd7d2cab8b6a97b3`.
The verifier reconstructs pinned original inputs and requires byte-identical
mathematical replay under the exact bound source/library versions. Arb/FLINT
and the analytic derivative implementation remain in the trusted computation
base. No claim about resonance ancestry, full shrimp anatomy, larger models,
or practical training improvement follows.

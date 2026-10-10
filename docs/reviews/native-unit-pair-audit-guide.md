# Independent finite704 audit guide

Exact producer source: a7b5a169032facb39e6476b3aae7e58654750fbf.
Raw `/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/attempt002-target`.
Outer SHA256 `73c02be51125e80db4707709e0cdaec8d39f69d453e35e2ef5a4738ed1405447`;
worker `9ee39499eb309e8283f2591857b4bcecb43aedfbfe05136d480899d2a39d92b6`.
`run/certificate.json.gz` SHA256 `47ba97a17cd9883e6617dae3ebb40deb53091ad035e1694812bb3522d1ea30b6`.
Original source math: `src/neural_shrimp/rigorous_unit_pair.py` at ae51c87d,
unchanged after independent Clark source review. Wrappers1bd0ad21/406dae05
independently cleared. Radius/interval adaptation has a separate exact card.

All704 unknowns are18native phases of(x13,q13,r13), then(eta_h,eta_o).
Parameter t center1/65536, halfwidth2^-16, exact interval[0,1/32768].
Base radius2^-11; exact weights from `run/repair-binding.json`, matching
`run/chart-inputs.json` and full certificate. The saved original41D root in
`run/inputs.json` is the independently audited root at4e8e0ec0, via exact pins.
The independent107-chart +1 locus connection/470membership is a separate audit.

Full proof fields: `center` and `slope` (exact dyadic predictor),
`center_residual`, `preconditioner` (exact dyadic C), `predictor_jacobian`,
`predictor_parameter_derivative`, `path_derivative`, `predictor_residual`,
`whole_box`, `whole_jacobian`, `whole_parameter_derivative`, `defect`,
`weights`, `coordinate_radii`, `image_correction`, `contraction`,
`coordinate_inclusions`, `divisors`, `anchor_check`, `endpoint_root_boxes`.
The proof gate reports704strict root inclusions and704strict anchor inclusions,
weighted norm0.0965712886637. Ball encoding is exact mid/rad dyadic pairs;
decoding with Arb may widen the radius, so use conservative inclusion or exact
Fraction domain comparisons, never silently shrink a recorded acquisition.

The216`events/*-direction_return.json.gz` files, sorted by sequence number,
are six groups of36, each orderedq,r for nativephase0..17:

1. original-root704native lift;
2. Newtonproposal0;
3. Newtonproposal1;
4. final exact center evaluation;
5. whole predictor-domain evaluation;
6. whole correction-domain evaluation.

Each has `phase`, `direction`, actual `input`=(x,direction,rates), local
`value`=(F,DF*direction), `block` (2n×2n), and`rates`(2n×2).
Independent assembly uses q-pass state/q rows and r-pass second-mode rows.
Both state DF blocks agree; derivatives of q/r restore the same DF block.
Nonterminal residual rows subtract the complete following(x,q,r) phase.
At the terminal phase subtract S(t)*(x0,q0,r0), where

    S(t)=diag(I13, [[c(t)I13,-d(t)I13],[e(t)I13,c(t)I13]])
    c=cos(sqrt(t)), e=sin(sqrt(t))/sqrt(t), d=t*e.

Their entire power series are regular at0; the producer uses degree20 with
rigorous geometric remainder for |t|<=1/4. Derivatives through2 are available.
The last two residual rows are(q0.q0-1)/2 andq0.r0, with gradientsq0 and(r0,q0).
H_t is zero except the terminal39rows, equal to−S'(t)*(x0,q0,r0).
All original derivative acquisitions and model code are shared dependencies;
an independent saved-inequality replay should say so.

Let δ=2^-16, d=[−δ,δ], U=center, S=slope, P=U+S*d, r_i=2^-11*w_i.
Recompute path derivative T=J_predictor*S+H_t_predictor on the entire predictor
and parameter interval. Recompute residual enclosure R=H(U,tc)+T*d.
The correction domain is P+[-r,r]. Recompute D=I-C*J_whole,
K=−C*R+D*[-r,r], andq=max_i sum_j |D_ij|w_j/w_i.
Requireq<1 and every |K_i|<r_i. All domain/native block/gauge conventions must
match before these inequalities. No condensed-inverse proposal needs trusting:
the full704defect validates the retained C.

The prior root's phase0(x,q,r) and two rates are enclosed by the independently
proved41D root, packing(x,rates,q,r). Each next native lifted phase acquisition
must contain its preceding returnedx/q/r and the same rates. `native-lift.json`
contains all704coordinate enclosures. Conditional exact closure at0 comes from
the proved Jordan root, not mere zero containment. Check every lifted coordinate
strictly inside U+S*(0−tc)+[-r,r]. This identifies the finite curve'st=0root.
Proper divisors1,2,3,6,9 exclude zero in at least one actualx_d−x_0coordinate
on the whole chart. At t>0 q+i*sqrt(t)*r has the exact unit multiplier; q norm1,
q.r0 and a nonzero r coordinate give independent real modes.

Endpoint enclosures useE_sign=−C*(H(U,tc)+T*(±δ))+D*[-r,r], then
U+S*(±δ)+E_sign. They are enclosures, not full stability certificates.
The separate saved-only native quotient packet checks all11complementary
multipliers over this same whole domain. Finite-arc NS criticality, stability
sides and global boundary organization remain separate work.

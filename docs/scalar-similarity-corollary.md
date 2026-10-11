# Exact similarity corollary for the scalar NS certificate

Derived and independently reviewed 8 October 2026 UTC. This is
an algebraic consequence of the existing selected certificate, not a new
numerical experiment or an additional independent example.

Write the scalar objective as

L_{d,lambda}(u,v) = (uv-d)^2/2 + lambda (u^2+v^2)/2,

and denote its two-coordinate GD map with rates (eta,r eta) by
F_{d,lambda;eta,r}. Its first component is
u + eta [v(d-uv)-lambda u], and its second is
v + r eta [u(d-uv)-lambda v]. For any d0>0, d>0 and fixed q=lambda/d,
let k=sqrt(d/d0) and S_k(x)=k x. Direct substitution gives the exact identity

F_{d,qd;eta,r} o S_k = S_k o F_{d0,qd0;k^2 eta,r}.

Indeed, d=k^2 d0, lambda=k^2 q d0 and uv scales by k^2, so the
bracketed gradient term scales by k^3. This is a smooth invertible linear
conjugacy in state together with a positive reparameterization of eta.
It also holds for every iterate of F. In dimensionless coordinates
x=sqrt(d) z, the family depends only on q=lambda/d, a=eta d and r.

Apply this to the already certified exact d0=5404319552844595/2^52,
q=1/4, r=4. For every exact-real d>0 with lambda=d/4, there is a local
supercritical primitive-period-two NS at

eta_*(d) = eta_*(d0) d0/d,

with both orbit phases scaled by sqrt(d/d0). Corresponding monodromy matrices
are equal because S_k is a scalar multiple of the identity. Thus multipliers,
nonresonance and critical rotation are the same. Primitivity, local stability
and the local two-component invariant-curve consequence transfer by conjugacy.
This does not supply a numerical finite-detuning enclosure for any member.

For the manuscript's Euclidean-unit right eigenvector convention, let alpha
be the total derivative d|mu|/d eta along the local periodic branch at the
crossing, holding d and r fixed. Let ell be the real resonant cubic
coefficient at the corresponding orbit phase.
The transferred quantities obey

alpha(d) = (d/d0) alpha(d0),
ell(d) = (d0/d) ell(d0).

The first identity follows by differentiating the rate reparameterization.
For the second, corresponding Jacobians are equal, while the second and
third state derivatives of F² scale by k^-1 and k^-2. Unit-norm right
eigenvectors and their normalized left rows can be chosen identically at
corresponding states. Every term in the resonant cubic coefficient is either
linear in the third derivative or quadratic in the second derivative,
with unchanged linear resolvents. It therefore scales by k^-2. Its sign is
unchanged, as required by the conjugacy. This coefficient scaling assumes
the same normal-form and eigenvector-normalization convention throughout.

Scope: this entire positive-d ray is one dimensionless case. It supplies
neither robustness to changing q or r nor an independent replication,
higher-dimensional result, global window association, finite-precision
implementation guarantee or training benefit. A quantitative neighborhood in
(q,r) would require additional parameter-uniform analysis; the present
certificate and this identity do not provide such a neighborhood.

## Independent review

The certificate reviewer independently checked the draft with SHA256
`0c7115d57a753f7ac8c019ec705f1805c929c58635ef577fa7fb43ab6ee0569c` against the integrated scalar map implementation
(`src/neural_shrimp/rigorous_scalar_ns.py`, SHA256
`486de0f6d074c619f9d6cb3ce9e011a6f217ddea0250ace7d5fa1d13f941cfc3`)
and the separate scalar NS proof review. Verdict: pass, no mathematical
blocker. The reviewer explicitly derived D^jG_d=k^(1-j)D^jG_d0 and checked
all second-derivative/resolvent contributions to the cubic coefficient.
The suggested total-derivative and corresponding-phase clarifications are
included above. This was a symbolic mathematical review, with no new
numerical target evaluation and no additional paid compute.

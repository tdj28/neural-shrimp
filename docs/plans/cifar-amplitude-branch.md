# New CIFAR formulation: amplitude-constrained two-cycle continuation

11 October2026 UTC. Preparatory mathematics only; no native evaluation or run
card. The fixed-rate span tests are finished negative controls. A new branch
problem lets the physical learning rate vary, and must never be presented as
another correction at the old eta0.00827753.

Write x=m+d, y=m-d, tau=2/eta and fix a positive amplitude a. With the exact
same original regularized CIFAR objective L and g=gradient L, solve

    Hm = (g(m+d)+g(m-d))/2 = 0
    Hd = (g(m+d)-g(m-d))/2 - tau*d = 0
    Htau = (a^2 - d.d)/2 = 0.

These are the gradient of the augmented scalar function

    E = [L(m+d)+L(m-d)]/2 - tau*(d.d)/2 + tau*a^2/2.

The complete bordered Jacobian is symmetric. If R=(Hplus+Hminus)/2 and
S=(Hplus-Hminus)/2 are the actual phase Hessians, its blocks are

    [ R    S         0  ]
    [ S    R-tau*I  -d  ]
    [ 0    -d^T      0  ].

Two phase HVPs compute each full augmented action; no third derivative or
explicit98666-square Hessian is needed. The scalar constraint removes radial
cycle-amplitude freedom when its bordered matrix is regular. This is a reason
to test conditioning, not a promise of invertibility or convergence. Primitive
period2 follows at an exact root from a>0 and eta>0; actual native phase maps,
closure, separation and stability still require independent qualification.
For approximate roots, physical residuals are exactly -eta*(Hm+Hd) and
-eta*(Hm-Hd), which must remain the authoritative closure diagnostics.

At a=0 this system is degenerate and must not be launched unchanged. Its
normalized limit is a genuine stationary/eigenpair problem g(m)=0,
Hessian L(m)q=tau*q, ||q||=1. The current endpoint-gradient secant ratio near
2/eta does not establish that limit or identify a Hessian eigenpair. A useful
next measured decision is a separately bounded stationary anchor near the
saved cycle midpoint, followed by a residual-qualified positive Hessian
eigenpair and explicit amplitude branch starts. A finite-amplitude start at
the current half-separation is an alternative that needs its own fixed design.
Do not automatically choose between these after new outcome inspection.

The helper currently has only source-level controls: complete nonlinear
Jacobian columns and symmetric pairing with noncommuting phase Hessians,
known attracting period2 of a quartic local potential, exact native shooting
identities, and zero-amplitude rejection. No CIFAR gradient, Hessian,
stationary root, varying-rate branch, attraction or compute benefit is claimed.
Source/card/cost/independent review remains mandatory before a native unit.

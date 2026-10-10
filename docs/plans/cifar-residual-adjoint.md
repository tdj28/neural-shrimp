# Native adjoint for a substantive nonlinear corrector change

10 October2026 exploratory source card. The old fixed400 GMRES, recycled span,
coordinate diagonal and input-Gram surrogate have not produced a CIFAR cycle.
The next distinct option is residual-merit trust-region/Gauss–Newton correction,
which needs the true transpose action and a measured bounded design before data.

For scalar-step GD, J0=I-eta H0 and J1=I-eta H1 are symmetric; A=J1J0-I is
usually nonsymmetric and A^T=J0J1-I. New helper reverses phase order and makes
exactly2HVP calls via the existing accounting object. It accepts only finite
float64 vectors and a positive scalar learning rate; momentum and layerwise
rates are outside its scope. Synthetic full-autodiff two-step Jacobian and
residual-merit gradient, a noncommuting example, finite differences and operation
counts test this mathematical prerequisite. No target arrays or simulations are
opened in this source unit. No solver/conditioning/utility outcome is claimed.

Next: profile full-size synthetic actions, compare true native forward/adjoint
pairings at the pinned already consumed development state, then a separately
bounded trust-region pilot with unchanged closure/primitive and actual-residual
acceptance. No protected subsetB or assessment text, paid work or GPU change.

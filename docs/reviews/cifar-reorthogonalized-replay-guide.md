# Fixed same-state CIFAR comparison replay

Completed raw `explore/cifar/stationary-seed/reorthogonalized001`, source
2195df22a1bcf8df7285175cca9abb4151cf906b. Exact outer/worker and selected artifacts
are in cifar-reorthogonalized001.json. All inputs/returns and failed-gate flags
remain immutable. No parameter update or trial occurred.

Bind original continuation001 and independent acceptance audit using the exact
card in source.tar.gz and comparison-binding.json, plus the saved conditioning
audit. Original state018/g19/firstHVP774 and trueHVP1030 form the baseline. Fresh
run/state-000.npy must equal original state018; fresh gradient0001 and HVP0001
must equal old g19 and HVP774. Every operation points at that same state.

Fresh HVP0001..0256 `-vector.npy` form V, `-return.npy` form W in order. Each
native_input is followed by attempted/returned/native_output; each basis action
is followed by the saved reorthogonalization row. Two coefficient arrays contain
actual complete CGS passes; no action output may transfer between states. Original
and fresh arrays are binary64,98666 coordinates. No HVP at final oldstate022 exists.

Recompute V^T V and its infinity error, p=Vc, predicted=Wc using coefficients in
basis-report.json. HVP0257 has the literal p and its actual physical action. Own
fsum norms of g+Hp and g+oldHp give the true relative forcing comparison; predicted
versus fresh action tests linear consistency. Recompute exact256 qualification,
20% material/.5 gates and operation counts (257HVP+1gradient). Full-rank W is a
numerical fixed-rcond SVD statement; no full-Hessian spectral theorem. Source and
shared native autograd dependencies must remain explicit. Independent scalar/action
replay is still required before promoting the comparison beyond producer evidence.

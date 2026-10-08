# Independent CIFAR input-span source review card

8 October 2026 UTC. Initial producer sourcee6485c6762a0b4d701b6c83c8164ef65ac44a09f.
Read source/plans only; no target array values or profile/projection outcome has
been opened. Two pre-execution findings were sent to root: collection needs an
external profile-manifest pin, and named bitwise identities need byte equality
rather than `np.array_equal` (which equates signed zeros). Root owns amendments.

Independent controls use only tiny synthetic inputs: manual nonlinear tanh
loss derivatives with both layers trained, complex-step residual JVP as an
independent oracle, rank-zero/rank-one thin QR completion, nonzero invisible
period-two state when q=-1, signed-zero mutations, full-vector packing and a
consistently rehashed fake profile. No3072x512 QR or saved target vector is read.
Run these alongside producer controls under CPU/BLAS1,nice10. Preserve initial
expected failures and review exact amended source before a pass verdict. This
card does not authorize an actual profile or target. Only reviewer files are
changed in the separate branch.

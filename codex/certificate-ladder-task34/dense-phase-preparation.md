# Task34 dense and phase proposal preparation

2026-10-18: isolated dense_qualify.py and phase_preconditioner.py adapt the accepted Task33 numerical algorithms to each explicitly named Task34 cell. Both validate the complete fixed five-cell allocation, select exactly one requested name, check the matching original finite qualification, and retain full 513x896 history input. Repeated period6 cells have separate names, parameters and prefix caches. No shared map or running profile source was modified.

Dense preparation retains all896 composed columns, complete eigensystem and residuals, both nonzero real/imaginary critical-vector checks, all seven FD steps and original native vectors. Phase preparation retains each one-step896 Jacobian, phase residual, ordered product and all seven preconditioner arrays. Neither is an interval certificate. One cell per separately admitted attempt allows the first ready candidate to advance without waiting for the whole ladder.

Syntax checked only. Pending: full source-delta review, allocation/closure controls on these actual entry points, completed profile cost/precision admission, exact per-cell bounded cards and fresh capacity. No new numerical launch is authorized by this preparation document. Original Task33 algebra controls are reusable; no repeated model calibration is needed.

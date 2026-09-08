# EXP-003 exploratory pilot

Receipt SHA256: `04d7225b0d3386f043ab0fc00558a12b0a9c9311fac7226666c3d8afca556a79`

Complete trial census: 96.

Fixed gradient-update budgets; wall time includes policy diagnostics. This is not an equal-total-compute comparison and does not test incremental shrimp geometry.

| Policy | Divergent / total | Triggered runs | Mean terminal test MSE (only if all runs finite) | Wall seconds |
|---|---:|---:|---:|---:|
| fixed | 9/24 | 0 | undefined (divergence retained) | 0.414 |
| scheduled_decay | 9/24 | 15 | undefined (divergence retained) | 0.411 |
| stagnation_decay | 9/24 | 12 | undefined (divergence retained) | 0.413 |
| recurrence_decay | 9/24 | 0 | undefined (divergence retained) | 0.457 |

# Verification ledger

| Component | Verification at initial freeze | Limit |
|---|---|---|
| Neural gradient, Hessian, training map | Local analytic-control and independent finite-difference tests | Float64 CPU only |
| Recurrence / cycle / multipliers | Exact logistic and Hénon controls; false-period and transient tests | Numerical verification, not interval proof or shrimp geometry |
| Utility intervention | Analytic quadratic-GD paired-path fixture | No initial empirical benefit claim |
| Manifest and raw receipts | Deterministic plan/control matching; actual CLI archive fixtures | Not a registry preregistration; no resume |
| RunPod archive and process watchdog | Offline integration and failure tests | Never called provider live; provisioning/teardown controller not implemented |
| Dependency environment | Locked dedicated local environment | GPU/image parity untested |
| Paper | LaTeX/bibliography compile plus six-page PDF visual inspection | Working draft; human publication review pending |
| GitHub | Private repository creation/push verified live via gh | Remains private |

At the initial freeze and after evidence packaging, all 52 local tests passed.
The implementation also passed both push and PR CI runs. The complete pilot
trial census and receipt hashes passed `tools/summarize_pilots.py`; see the
[results update](initial-results.md). No “verified-live RunPod” or GPU speedup
claim is implied.

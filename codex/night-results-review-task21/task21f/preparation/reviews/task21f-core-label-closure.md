# Task21f core label correction closure

The exact diff `/tmp/task21f-core-label-amendment.diff` (SHA256 `afc876914ebff6f4df4903a3fdfa196514a2575abf59f0f8ad608f90ee476935`) changes only the intersection docstring and an output method label. It resolves the sole wording condition in my accepted core review `6a36e4078c7c1bb12caa451d51fa1b5f5e23817d6cada45ef6565bf05c684d0c`.

Final core `/tmp/task21f-core.py` is SHA256 `b7b29685e9c1bf4a9211a9a76d9fa04e451de6a1a3de1ad63783965bbee9c60a`. It now expressly describes source-formula arithmetic with geometric bounding-box pruning, without guaranteeing bit-exact reproduction of the unfiltered producer helper. There is no arithmetic change and no need to repeat controls. The original core and control evidence remain preserved. I read this exact diff and the prepared composition design; no target data or executable was run.

The driver/card remains a separate upcoming review. The design correctly leaves per-band factor gauges unaligned, treats four-sector corner support as a sampled diagnostic, separates the zoom's different mask, retains the undisclosed 1.3e-3 gap definition as unverified, and treats manufactured timing as cost evidence rather than a worst-case guarantee.

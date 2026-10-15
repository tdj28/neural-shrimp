# Synthetic profile output separation

Before any profile execution, Root found that preparation source 771b148a would create its synthetic output directory beneath the captured immutable source. That preparation is preserved byte-for-byte in synthetic_profile-before-output-separation.py. The sole correction adds an explicit required --output path, refuses paths inside the captured source, and retains exclusive creation. All numerical work, RNG sequence, shapes and budgets are unchanged. No profile or target run failed or was retried; this was a source-only refusal/correction.

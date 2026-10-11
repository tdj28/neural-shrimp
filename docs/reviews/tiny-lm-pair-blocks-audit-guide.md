# Independent saved replay guide: adjacent-pair block ordering

Producer execution `1c91e14fb01276a5d764e8facc65ab6585f38606` completed the sole
fixed consumed-development target, with 128 HVP and no new gradients or CE calls.
This is a directional linear-response ordering intervention, not a training,
forecast, acceleration, or generalization result. Independent replay is pending.

Raw root: `/data2/neural-shrimp-data/explore/tiny-lm-pair-blocks/attempt001-target`.
Outer manifest SHA256 `03022d864990b299468c026a1b1bf9210a61a028bfae91f02a604fe7fb3cc5e6`;
worker manifest `615376d85ffcc62f6d2a24a0799544dab16c3e4a61b18cb4fae9f2e22fc19b58`.
The compact complete result, actual resources, immutable inventories and absent
owners are in `tiny-lm-pair-blocks-outcome.json` beside this guide.

Authenticate both full inventories and every historical source file under
`run/source/` against the exact execution commit. `run/runtime.json` closes over
native/helper/library identity; `run/input-binding.json` carries the inherited
mean/response authority, and `run/parity-binding.json` binds the exact parity
producer, independent audit, and all upstream timing evidence. The fresh two-HVP
profile is bound by `run/profile-reference.json`; its complete cold maximum
0.6293021319434047 seconds exceeded all retained upstream maxima and was used in
the 191.1013457775116-second target forecast. The unchanged 240-second target took
27.5530455932 seconds and 981716 KiB. All 4 cases retain their inherited 1% response
and CE qualifications.

The case order is 28161 original, 28161 quarter_step, 28162 original, 28162
quarter_step. Each has 16 U and 16 V actions, so the complete new event sequence is
128 input/return pairs. U applies `[14,15,12,13,...,0,1]`; V applies its reverse
`[1,0,3,2,...,15,14]`. With symmetric factors, V is the transpose product of U.
Each `run/events/NNNNN-input.json` binds actual state and direction object pointers;
the corresponding return retains the complete acquired HVP before finite checks.
Objects are content-addressed lossless float64 NPY vectors of length 15776.

For each case, `run/aliases/seed-<seed>-<role>/{U,V}.json` stores 17 row references,
and the same case directory stores `{U,V}-order.json`. Independently reconstruct
every recurrence `v_next = v - eta * action`, literal actual state selection and
all action input vectors. Case bindings are at `run/seed-<seed>-<role>/binding.json`.
The per-case baseline aliases are `baseline-P`, `baseline-R`, `baseline-frozen`,
`baseline-ce_gradient`, and `baseline-parity_even`. Authenticate their literal
bytes against the old qualified parity/mean chains; baseline P and R are reused
acquisitions, not new native measurements. Derived aliases are
`chronological_odd`, `pair_block_odd`, and `pair_block_even`.

Recompute with separately authored NumPy/math.fsum arithmetic: all P/R/U/V gains,
alignments, CE-gradient projections, transpose scalar identities, E=(U+V-P-R)u/2,
odd components, exact vector decomposition, squared cross-term identity,
E/||P u||, E/||E_parity|| and E/||frozen-P||. Both descriptive gates require all 4
case qualifications first. Then, for both originals separately: material means
E/||P u||>=0.2, and attenuation means E/||E_parity||<=0.1. They are distinct gates.
Do not infer either from a tiny chronological denominator: original P gains are
0.59032 and2.33093. Every scalar/report and all 4 control cases must be retained.

Producer originals give relative material 1.54008/2.53504 and parity fractions
0.00103020/0.00251419. Quarter controls give 0.113757/0.0787437 and
0.969561/1.79606, respectively. Thus original parity sensitivity is strongly
attenuated while a material response remains; there is no universal attenuation
across controls. E is also a small fraction of original frozen approximation
error, so this contrast does not explain all of that error. Interpret only after
independent saved arithmetic passes, with shared native HVP/gradient acquisition
scope explicit. No new model call is needed for the audit.

Prior parity authority: producer root
`/data2/neural-shrimp-data/explore/tiny-lm-parity-order/attempt001-target`, execution
`e8030d33c47bb817b4ac9829951bbb6137315f57`, outer
`8e7b47c82b6c25db6873296c83e2a6589a1189e5dc8f3c3a8da2b7af2e69a012`, worker
`838bd1102ca3bcd005a10ace5c45bbb2e64902d12080203d1a1e7d5861dbb88f`.
Its independent audit root is
`/data2/neural-shrimp-data/audits/tiny-lm-parity-order-independent/attempt001`, source
`b481b47224d41ccc17f73b5a3b6502e8b4e1bddc`, outer
`e114103e8fde5e925a028e9f9d34f4eec269fb8f01c7a6f16333fc6a4c19c7c2`, worker
`03e33ee5f82550563639b0d08dfcb49a45948446dbdd637444232e50f33ac586`.

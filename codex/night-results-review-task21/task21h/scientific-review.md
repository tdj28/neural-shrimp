Task21h scientific review — steering depth and exact two-block controls
15 October 2026; completed saved-output review, awaiting Root's final prose acceptance.

The independent saved-grid calculation reproduces all **369 producer-summary leaves exactly**, with zero unequal or nonzero differences. Every one of the **119,552 source cells** and **36,864 joined comparisons** is finite-valid under the admitted definitions; no saved/reconstructed chaos-label disagreement was found. The descriptive depth pattern survives, and the later exact-zero controls support finite cases of joint chaos with neither single push classified chaotic. Several original explanatory sentences and the old near-axis interpretation need the qualifications below.

This is an unblinded saved-data reproduction and final interpretation review. Earlier producer summaries were already exposed to the team. Theory's independent reducer performed the sole admitted arithmetic; Inference executed and authenticated it. My accepted reader consumed and retained every saved cell/join record, inventoried their existing flags, and copied the outputs literally. This review did not rerun the reducer, decode raw NPZs, recompute exponents/periods/masks, run a model, or perform a new original-input audit. The rows/columns, full period histograms, every coordinate and every flag pattern remain in the literal artifacts; summary selection does not replace them.

| Claim | Disposition | Required wording/scope |
|---|---|---|
| Original LS/TB/TBC saved summary arithmetic | Confirmed | All369 leaves equal, including original period histograms, LE ranges/counts and12 TB/TBC chaotic shares. Equality does not validate the native exponents or primitive orbits. |
| Chaos depends on steering depth in these five maps | Confirmed descriptively | Shares decrease from block8 through12,16,20; block4→8 instead increases. The minimum chaotic temperature increases with depth from8 onward. |
| Block20 chaos lies only in log10T0.1–0.25, with everything below periodic | Discrepancy | Saved chaotic extrema are−0.08897637795275615 and0.21811023622047243. Row89 already has one chaotic cell at the negative endpoint. Do not equate all other cells with verified periodicity. |
| Near-zero steering gives a literal1–2% for every depth | Overprecise wording | Strict abs(alpha)<.02 gives fractions0.0078125 through0.020951704545454544; approximately0.8–2.1%, with no exact alpha0 in these even grids. |
| Original near-axis counts establish “neither push alone” | Historical discrepancy preserved | The original row/column was alpha=−0.001263157894736841, not zero. Its four counts remain1574,880,1084,1273. |
| Later exact controls retain joint chaos with both controls nonchaotic | Confirmed at the finite protocol | Counts1561,761,1194,1354 out of9216 per plane. Both-controls-positive-period counts are separately1525,720,1154,1173. |
| Native chaos/stability, a universal depth law, or a behavioral/task benefit | Cannot check from these scalar grids | No native trajectory/primitive-orbit/stability proof, hidden/visible split, mechanism isolation, or task-performance outcome is reconstructed. |

The literal classification is LS: LE>0.001 and P=0; TB/TBC: the original saved Boolean union, with a separate reconstruction of the period-first source rule from the available single LE and period. Positive detected period has precedence in TB/TBC. Finite qualification requires finite LE and a finite integer period within the original source range. All actual cells satisfy that qualification, so the literal and qualified numerators/denominators coincide here. This is an observed result, not a reason to discard the separately defined validity columns.

**Five depth maps.** Each complete map has16,384 cells; the strict near-zero band has2,816. Fractions below are the exact stored binary64 values. The temperature extrema are actual saved coordinates, not the endpoints of a visually selected band. “Periodic” here means a detected return in the finite source test, not an independently verified primitive orbit.

| Block | Chaotic count | Fraction of16,384 | Detected-period cells | P=0 cells | Near-zero chaotic/2,816 | Near-zero fraction | Chaotic log10T min | Chaotic log10T max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 961 | 0.05865478515625 | 15248 | 1136 | 59 | 0.020951704545454544 | -0.9078740157480314 | 0.21811023622047243 |
| 8 | 1321 | 0.08062744140625 | 14891 | 1493 | 33 | 0.01171875 | -0.9488188976377953 | 0.27952755905511817 |
| 12 | 1119 | 0.06829833984375 | 15155 | 1229 | 29 | 0.010298295454545454 | -0.7133858267716535 | 0.2385826771653543 |
| 16 | 821 | 0.05010986328125 | 15386 | 998 | 22 | 0.0078125 | -0.6314960629921257 | 0.20787401574803155 |
| 20 | 378 | 0.0230712890625 | 15947 | 437 | 24 | 0.008522727272727272 | -0.08897637795275615 | 0.21811023622047243 |

Safe enclosing three-decimal temperature displays are block4[−.908,.219],8[−.949,.280],12[−.714,.239],16[−.632,.208],20[−.089,.219]. Conventional approximate nearest rounding may be labelled approximate; it must not be presented as an enclosing bound. The05:10 log's “lowest temperature falls” has the numerical direction reversed: from block8 onward the lower endpoint rises toward higher temperature. Its share decrease is correct on that restricted sequence. A decrease across all five blocks is not supported.

The phrase “everything below [0.1–0.25] periodic” is directly contradicted by source labels. In addition to block20 row89, saved row93 at−0.04803149606299234 has one chaotic cell; row107 at0.09527559055118101 has35 chaotic and38 P=0 cells out of128. The latter includes finite unresolved cases under the declared chaos threshold. Across the full block20 map,437 cells have P=0 but only378 satisfy the chaos predicate. The much narrower statement that its original low half has8,192 detected-period cells is true and is retained separately.

All LS alpha axes omit zero. The abs(alpha)<.02 band is a22-column neighborhood, not an unsteered baseline. Its fraction decreases through block16 and then increases at20; no monotonic near-zero law is inferred. The full period distributions and every temperature-row/alpha-column summary remain in the saved report, including positive-LE cells with detected periods (5,4,5,6,4 at blocks4,8,12,16,20). They are not silently recategorized as chaotic.

The four newly split depth maps preserve their actual low-then-high coordinates. Their temperature coordinates differ from the old block12 reference at113 ordinal rows, with maximum absolute discrepancy5.551115123125783e−16 (source/header association, and the retained actual comparison rows); a safe outward bound is5.552e−16. These are nominally corresponding grids, not bit-identical temperature inputs. No tolerance matching or interpolation was used. Each depth uses its own philosophy vector constructed/scaled against that block's residual norm, not one fixed Euclidean vector transported between layers. Native vector norms, model precision and model-state dynamics remain inherited source/receipt premises.

**All21 source grids and failure dispositions.** All listed cells are finite-valid; every grid has zero nonfinite LE, invalid periods and saved/reconstructed mask differences. C is the literal chaotic count, P+ the detected-positive-period count, P0 no detected return. The eight LS tool failures are retained even though their final numerical arrays were written: the exact tracebacks identify the later period_colors plotting import failure. “Runner done/rc0” does not overwrite tool status1. TB/TBC and the inherited block12 scan have source tool status0. The five joined depth maps are alternate views of these source cells, not additional samples.

| Grid | Allocated=valid | C | P+ | P0 | Positive LE with P+ | Original tool status |
|---|---:|---:|---:|---:|---:|---:|
| ls-l4-lo | 8192 | 116 | 8032 | 160 | 1 | 1 |
| ls-l4-hi | 8192 | 845 | 7216 | 976 | 4 | 1 |
| ls-l8-lo | 8192 | 152 | 8015 | 177 | 0 | 1 |
| ls-l8-hi | 8192 | 1169 | 6876 | 1316 | 4 | 1 |
| ls-l16-lo | 8192 | 21 | 8151 | 41 | 0 | 1 |
| ls-l16-hi | 8192 | 800 | 7235 | 957 | 6 | 1 |
| ls-l20-lo | 8192 | 0 | 8192 | 0 | 0 | 1 |
| ls-l20-hi | 8192 | 378 | 7755 | 437 | 4 | 1 |
| grat4-host16/tb-t+0.100 | 9216 | 3043 | 5869 | 3347 | 10 | 0 |
| grat4-host16/tb-t-0.100 | 9216 | 2003 | 7056 | 2160 | 12 | 0 |
| host4-grat16/tb-t+0.100 | 9216 | 3870 | 5008 | 4208 | 15 | 0 |
| host4-grat16/tb-t-0.100 | 9216 | 2153 | 6771 | 2445 | 21 | 0 |
| c-grat4-host16-a1/tb-t+0.100 | 96 | 30 | 66 | 30 | 0 | 0 |
| c-grat4-host16-a1/tb-t-0.100 | 96 | 12 | 81 | 15 | 0 | 0 |
| c-grat4-host16-a2/tb-t+0.100 | 96 | 45 | 49 | 47 | 0 | 0 |
| c-grat4-host16-a2/tb-t-0.100 | 96 | 5 | 90 | 6 | 0 | 0 |
| c-host4-grat16-a1/tb-t+0.100 | 96 | 33 | 60 | 36 | 0 | 0 |
| c-host4-grat16-a1/tb-t-0.100 | 96 | 25 | 70 | 26 | 1 | 0 |
| c-host4-grat16-a2/tb-t+0.100 | 96 | 39 | 54 | 42 | 1 | 0 |
| c-host4-grat16-a2/tb-t-0.100 | 96 | 19 | 76 | 20 | 0 | 0 |
| map128-philosophy | 16384 | 1119 | 15155 | 1229 | 5 | 0 |

There is no invalid/unfinished numerical suffix hidden by the final-shaped arrays in this admitted set. LS20-lo's null chaotic-coordinate range means no chaotic cells in that finite half; it is not a missing-data flag. All control eligible denominators are positive; no undefined fraction was converted to zero. The exact-zero indices occur only on the explicitly singleton control axes; none of the original96×96 planes contains zero. No fully unsteered(0,0) cell is present in these control commands.

**Two-block planes and controls.** G/H denotes gratitude@4 with hostility@16; H/G denotes hostility@4 with gratitude@16. Each plane has9,216 cells and each single-push line96. The saved row/column order is alpha2/alpha1. Original near-axis selection is index47 on each axis, exactly−0.001263157894736841 in both coordinates. The following fractions are saved, not inferred from rounded percentages.

| Plane | Joint fraction | Old near a1-only fraction | Exact a1-only fraction | Near/exact a1 label mismatches /96 | Old near a2-only fraction | Exact a2-only fraction | Near/exact a2 label mismatches /96 |
|---|---:|---:|---:|---:|---:|---:|---:|
| grat4-host16/tb-t-0.100 | 0.2173394097222222 | 0.07291666666666667 | 0.125 | 9 | 0.08333333333333333 | 0.052083333333333336 | 7 |
| grat4-host16/tb-t+0.100 | 0.3301866319444444 | 0.28125 | 0.3125 | 5 | 0.4583333333333333 | 0.46875 | 21 |
| host4-grat16/tb-t-0.100 | 0.2336154513888889 | 0.3125 | 0.2604166666666667 | 15 | 0.17708333333333334 | 0.19791666666666666 | 10 |
| host4-grat16/tb-t+0.100 | 0.419921875 | 0.3854166666666667 | 0.34375 | 10 | 0.3645833333333333 | 0.40625 | 14 |

The exact rule is J[i,j] and not A[j] and not B[i], where J is the original plane and A/B are the later exact-zero control lines. The stricter row uses positive detected period in both controls. “Nonchaotic” means a false finite source label; it can include no-return cells below the LE threshold and is not synonymous with periodic/stable. All9,216 triplets per comparison are finite-valid. Thus every finite-qualified count below equals its literal count, without deleting any record.

All-plane and eligible-control denominators remain distinct. In the table, N is joint-chaotic under the named control condition; E is the number of plane positions eligible under that control condition. N/9216 and N/E are different estimands. “Near excluded” additionally removes the selected nearest row/column from eligibility, while retaining the fixed all-plane fraction. It is not substituted into the exact-control card.

| Plane | Definition | N | E | N/9216 | N/E |
|---|---|---:|---:|---:|---:|
| grat4-host16/tb-t-0.100 | Exact nonchaotic | 1561 | 7644 | 0.1693793402777778 | 0.2042124542124542 |
| grat4-host16/tb-t-0.100 | Exact P+ | 1525 | 7290 | 0.1654730902777778 | 0.20919067215363513 |
| grat4-host16/tb-t-0.100 | Near nonchaotic | 1574 | 7832 | 0.17078993055555555 | 0.20097037793667008 |
| grat4-host16/tb-t-0.100 | Near P+ | 1529 | 7482 | 0.16590711805555555 | 0.20435712376369955 |
| grat4-host16/tb-t-0.100 | Near excluded | 1574 | 7656 | 0.17078993055555555 | 0.2055903866248694 |
| grat4-host16/tb-t+0.100 | Exact nonchaotic | 761 | 3366 | 0.08257378472222222 | 0.22608437314319668 |
| grat4-host16/tb-t+0.100 | Exact P+ | 720 | 3234 | 0.078125 | 0.22263450834879406 |
| grat4-host16/tb-t+0.100 | Near nonchaotic | 880 | 3588 | 0.0954861111111111 | 0.24526198439241917 |
| grat4-host16/tb-t+0.100 | Near P+ | 815 | 3417 | 0.08843315972222222 | 0.2385133157740708 |
| grat4-host16/tb-t+0.100 | Near excluded | 880 | 3468 | 0.0954861111111111 | 0.2537485582468281 |
| host4-grat16/tb-t-0.100 | Exact nonchaotic | 1194 | 5467 | 0.12955729166666666 | 0.2184013169928663 |
| host4-grat16/tb-t-0.100 | Exact P+ | 1154 | 5320 | 0.1252170138888889 | 0.2169172932330827 |
| host4-grat16/tb-t-0.100 | Near nonchaotic | 1084 | 5214 | 0.11762152777777778 | 0.2079018028385117 |
| host4-grat16/tb-t-0.100 | Near P+ | 1084 | 5214 | 0.11762152777777778 | 0.2079018028385117 |
| host4-grat16/tb-t-0.100 | Near excluded | 1084 | 5070 | 0.11762152777777778 | 0.21380670611439842 |
| host4-grat16/tb-t+0.100 | Exact nonchaotic | 1354 | 3591 | 0.1469184027777778 | 0.3770537454747981 |
| host4-grat16/tb-t+0.100 | Exact P+ | 1173 | 3240 | 0.12727864583333334 | 0.36203703703703705 |
| host4-grat16/tb-t+0.100 | Near nonchaotic | 1273 | 3599 | 0.1381293402777778 | 0.3537093637121423 |
| host4-grat16/tb-t+0.100 | Near P+ | 1139 | 3245 | 0.12358940972222222 | 0.3510015408320493 |
| host4-grat16/tb-t+0.100 | Near excluded | 1273 | 3480 | 0.1381293402777778 | 0.3658045977011494 |

The historical07:57 counts1574/880/1084/1273 are reproduced by the near-axis rule. Excluding the chosen near row/column leaves those four numerators unchanged but changes eligibility, so it must not silently change the denominator or be called an exact-zero test. The later exact-zero rule gives1561/761/1194/1354, with both-P+ counts1525/720/1154/1173. The original near-axis arithmetic is correct under its actual definition; its earlier “single push” description was not. The later TBC experiment corrects that design issue and should be reported as a disclosed follow-up, with its original11:45 prospective rule preserved.

This supports finite joint-versus-single classification differences in all four tested planes. The controls use one hook with exactly zero steering while both hooks remain registered. Their initial state/prefix is formed under each control's own settings. Original plane chunks are2304, whereas each96-cell control line fits within chunk2048; batch/twin draw positions differ. The fixed horizon is500 transient+300 recorded steps, a single1e−6 twin and the original period tolerance1e−8; LS detects periods through48 with its last80-state test, whereas TB/TBC uses its recorded-window test through64. These facts limit comparisons to the stated source protocol. They do not prove global chaos, native attractor stability, an isolated causal interaction independent of initialization/estimator details, or task utility. No new model run or gate is needed to state those limits.

Suggested replacement prose:

> In five exploratory Qwen2.5-0.5B maps, the finite-horizon chaotic share was approximately5.9%,8.1%,6.8%,5.0%, and2.3% for steering at blocks4,8,12,16, and20. From block8 onward the share decreased and the lowest observed chaotic temperature increased. Block20's chaotic cells spanned log10T=−0.08897637795275615 to0.21811023622047243. These comparisons use block-specific steering vectors and nominally corresponding grids; they do not establish a universal depth law.

> The original two-block planes omitted exact zero, so their nearest-axis slices were not single-push controls. A later disclosed exact-zero control experiment retained joint-chaotic cells with both corresponding single-push controls classified nonchaotic:1561,761,1194, and1354 of9216 cells, respectively, for G/H at log10T−.1,+.1 and H/G at−.1,+.1. Requiring a detected positive period in both controls gives1525,720,1154, and1173. These are finite-horizon saved-label comparisons; nonchaotic does not by itself establish periodicity or stability.

**Evidence, complete coverage, and execution attribution.** The original immutable output root is `/data2/neural-shrimp-data/codex/night-results-review-task21/task21h/run-001`. Source commit1fe08093028db99808599ce72c5dbb79bb8e9d96 binds unchanged independent run4935bca0/reducer150e3777. The final operational authority is sibling `run-001-authority/report.json` SHA2568c8107d1c2c6660379eaa567d445c12b3c6abbdf3387123765ab0a16d522ba64. Root accepted its four scientific files, nine supervision entries, four Git blobs,82 input pins and original process closure. I inherit that accepted authority rather than repeat it. Inference directly observed original session15574 reaped0 and the original births/groups absent. This scientific review is separate from those observations.

| Original saved artifact | SHA256 |
|---|---|
| report.json | eddcc97a5e73c11480f3fc74ef44429b7e24fedb3b16db5dc5d1ae5670721fc1 |
| cells.jsonl | fc185088fac8b82fef25b05261f0c6b57bd618fc1f35a3180f2d6a3024d4e8be |
| control-joins.jsonl | 31503f6231a7b6a3f6bb2d0f9aa037cab1ae48002ee8fc7ab0f95942381fe232 |
| manifest.json | 7c9964d079289fbdbf5319666e64ce5d1f5b3fb873bfa94e5f44e1fe1c7a31ee |

The complete literal presentation is `/tmp/task21h-saved-read-001`, indexed by its `manifest.json`; `literal-report.json`, `literal-cells.jsonl` and `literal-control-joins.jsonl` are byte-identical to the original science outputs. `review-summary.json` SHA2565e0489cde4d08727a21e3a8c8441a53d6040b922f40dfed3562da1893b702169 retains all21 grid summaries,5 layer summaries,4 control summaries, all369 comparison leaves, the empty exception list, every Boolean-pattern inventory and all original row/column/coordinate tables. The compact selection `/tmp/task21h-compact-saved-dispositions.json` SHA2568704184a1b772d435d685842c5728f4c696061e208a4e2856ffe019190dd27e9 omits only bulky axis/row/column tables, preserving all top-level distributions and flags. No raw values were silently repaired or removed.

The sole literal reader source `/tmp/task21h-saved-read.py` SHA25602b18561874cc523e680764f8c8a250097781f8509206f3c64b554f8b85ba44d ran once under CPU1/nice10/1GiBAS/50CPU/60external seconds and128MiB written-output cap. Its external receipt is `/tmp/task21h-saved-read-001-supervision/receipt.json`, SHA2561c5a0c86736ae988987e9a0f5b173086adb65bb10c50a15755a1c98577890b21: exit0, no timeout, child reaped and owned group absent;2.8635180820710957s external window. Inference owns this observation, including host PID3105678/start201996081 and tool session59059. Subsequent report drafting selects already generated presentation fields only; it is not a second ledger traversal or numerical analysis.

Timing scopes remain exact. The original arithmetic report records1.9359385271091014s and64,452KiB observed before posthash/report write, including incremental cell/join serialization but excluding final rehash/report/manifest. Inference's original supervisor window is2.611803654s separately. Neither observation is a full-worker peak RSS. The literal presentation records2.574920057086274s and47,916KiB before its summary write, separately from its external window. No model/GPU/paid work, source retuning, hypothesis threshold change or rerun was performed in this review.


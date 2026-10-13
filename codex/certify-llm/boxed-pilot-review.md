# Independent boxed-derivative pilot and launcher review

Reviewed 2026-10-13 without a checkpoint/model call, numerical target replay, Git command, or launch.

Reviewed hashes:

- `box_derivative_pilot.py`: `ed969bbb85f6fe42f5f2329d1be56f76a6aea21cb4e911129be2b9b48c7da11e`
- `tests/test_codex_llm_box_derivative_pilot.py`: `9ac506ee9548b6a9671af9813711253355c3a8c627e8d4be956a063214854b33`
- `box-derivative-001-plan.json`: `d150e0efeaffe0fe8228b99071e9308130bdf21932a3e732c8a6ce125084c642`
- `launch.py`: `a9a4002cdd1d6e9989915fb75864935736c1387de9c167fa52d767b7678c2a1e`

No blocking defect found in the reviewed scope. The plan is an engineering test of one phase box and the first 32 basis directions, not an orbit certificate or complete Jacobian.

The nominal domain contains all 896 state coordinates around the exact dyadic binary64 phase-zero prototype, with nominal radius 2^-800. The mixed-precision helper checks exact endpoint containment after construction and decoding. The driver additionally reconstructs the actual receiver balls, reruns exact containment on those live balls, saves their own hash and passes those full balls into the inherited map. It does not pass their midpoints. The nominal point used for the prior signed pilot is contained in this actual domain. Domain-origin records tie the generated domain to the pinned proposal and committed plan; the newly generated trusted artifact hash is not used as self-authentication of an untrusted supplied artifact.

The input direction matrix is 896 by 32, with exact unit entries at (k,k), k=0,...,31, and all other entries zero. Each column is a separate signed first derivative. The inherited Jet implementation retains the full 151936-token softmax and expected-embedding readout. A successful evaluation conditionally encloses those 32 columns over the whole input box; it does not bound the omitted 864 columns. The actual primal and all 32 complete finite columns are saved before comparison or success status.

For compatibility, the driver checks every primal coordinate and every coordinate of all eight shared derivative columns against the prior point pilot. It uses exact rational endpoints of the original saved balls, so reconstruction cannot manufacture a false overlap by widening radii. It requires overlap, not containment of the earlier interval: two valid enclosures of the same exact point value need intersect but need not contain one another. All 32 column radius summaries remain visible; a failure in the eighth comparison row or the last output column cannot be hidden by averaging. Overlap remains a compatibility diagnostic, not independent proof of the evaluator.

Authentication fixes the two upstream receipt hashes and hashes every retained artifact plus the complete source-manifest inventory. It explicitly distinguishes the original pilot's later box failure from its complete reusable prefix. The signed point comparison must have completed successfully. The proposal shape/dtype, exact parameter ratios, prefix IDs, phase, direction order and precision are checked. The prefix importer additionally checks backend versions, exact-real constant conventions and independently constructed receiver identities. Historical versus current evaluator semantic compatibility still relies on the earlier independent map/Jet/cache source reviews; hashes establish provenance, not mathematical correctness by themselves.

The launcher admits both new drivers only with an explicit committed plan and one CPU thread. It retains the existing private remote ancestry check, captured committed source, 18 GiB address-space limit, bounded wait up to 3600 seconds, owned process-group termination and immutable receipt retention. The 3600-second driver-plan envelope is an upper bound; a shorter launcher timeout would produce an incomplete retained attempt, not an extended budget. No launch was performed in this review.

Independent verification: the boxed-pilot, domain and Jet modules passed 97 synthetic tests. Two valid launcher admissions were intercepted before the first Git operation, while 14 invalid combinations (missing plan, excess threads, wrong skip-box usage, invalid time limit or plan path) were rejected before Git. An isolated `python -I --help` invocation succeeded without model construction. No fixes to the reviewed producer or launcher were needed.

The radius is vastly smaller than the prototype's measured orbit residual and is explicitly not an isolating-root-box proposal. Even complete success would leave the remaining directions and phases, whole-orbit root inclusion/uniqueness, primitive period, attraction and any claimed text labels as separate obligations. Runtime and peak memory at this box and batch size are measurements to be obtained, not inferred from the prior eight-column point timing.

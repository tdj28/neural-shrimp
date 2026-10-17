# Number-to-data candidate checker: prospective implementation

This is a search aid, not an independent recomputation or truth checker. Exact
local bytes and all numeric matches are retained. Neither a numerical match nor
a key-overlap score proves a receipt supports the corresponding claim.

Scope: every digit-form numeric literal in the frozen main.tex body, including
equations/tables and all literal input caption arguments. Reference keys, graphic
dimensions, explicit dates, contextual years and TeX comments are masked without
changing line offsets. Drawing geometry, macro-generated numbers, number words,
OCR and plot-image axis values are outside this lexical tool; coverage is explicit.

Index all tracked JSON under docs/ and paper/llm/figures/; optional research-log
numbers are marked log-only. Boolean and nonfinite leaves are not values. Complete
numeric strings are candidates; nonnumeric prose strings are not mined. Args/config
paths supply a design-parameter-candidate tag. Invalid JSON files are reported.
Secrets and monetary fields explicitly named in the small documented exclusion
set are omitted, rather than copied to new artifacts.

Use exact Decimal cells [x-q/2,x+q/2), where q is the last written decimal quantum.
For percentages search both native values and receipt fractions. For upper/lower
bound wording and explicit interval endpoints, also search respectively (x-q,x]
or [x,x+q). This locates a possible outward-rounded extreme; it does not verify
that the candidate is an extreme of the intended cohort. Retain all modes.

Candidate groups share exact value IDs, and values share all file/pointer refs.
This avoids exploding repeated candidate lists for common 0/1/2/64 values without
discarding any candidate. Five previews rank lexical overlap only; all references
remain accessible in the lossless JSON index. Summary lists unmatched, log-only,
bound and ambiguous values with full sentence/section/line context.

Before target execution, run manufactured functional controls and a manufactured
1361-receipt/6600-number profile (no target outcomes). Use its observed cost plus
the measured 28.56MB actual input bytes for a fixed bounded target card. No model
execution. Pin source, inventory and paper hashes; verify all used bytes before
and after. Reject an existing output folder. No automatic retry or extension.

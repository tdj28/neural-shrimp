# Narrow independent review of the Task26 sign correction

Disposition: accepted for the disclosed source-level correction and the eight focused manufactured control definitions. No concrete blocker was found. This review imported or executed neither the checker nor the fixtures, read no target numerical outputs, and did not repeat the broader checker review or earlier controls.

Reviewed exact files:

- `check.py`: `bf5182386936115c3b83cd3da097005b07dce16e49943e481864df39f496dd69`.
- `check.diff`: `09a642a13d459fa29b8fd148a35f63bd0f7d69e6af5f8cfbd2ac7f3cb712d3d1`.
- `amendment.md`: `1d0d4657f881d184fbfdaad14ecb7ab4fc3b23a6e0964882bb86f9740a079629`.
- `sign_controls.py`: `96d73ac595a40b3b5238b129e5b038f5c39db0e6f7151d67281ebe54aba333b0`.

The new helper at checker lines 80–92 identifies the stated literal dollar, parenthesized, bracketed and named-environment math spans without changing source offsets. The change at lines 97–110 removes an ASCII sign only for the disclosed prose letter/underscore/dash context outside those spans, while retaining the original digit/closing-brace range rule. Thus the actual defect examples `period-8`, `OLMo-2`, `Llama-3`, and `7--8B` become positive numeric literals without dropping their surrounding source context. Within recognized math, the newly added prose rule does not erase the negative operand in `$x-2$`, `$5--2$`, or a command-adjacent minus. Actual unary negatives outside math also remain when their predecessor is not a newly specified prose separator.

All eight fixture definitions and expected values were read: prose identifiers, multiple range spellings, inline negatives/double subtraction, display delimiters, an align environment, an escaped dollar, existing scientific notation/spacing/Unicode-minus behavior, and line/column/raw-span preservation. The expected second-line positions `(2,8,'8')` and `(2,11,'-8')` match the supplied literal fixture. Each case compares exact Decimal values and literal spans; the location case separately exercises `extract()`. Failures remain in the written report before exit1. This is source acceptance, not a claim that any fixture has passed execution.

The scope remains a numeric-candidate heuristic, not general TeX interpretation: macro expansion, text semantics nested inside math, and arbitrary malformed/unbalanced delimiters are not solved by this amendment. The pre-existing digit/closing-brace range treatment remains unchanged even inside math. These limitations should not be described as a guarantee that every mathematical sign is parsed. They do not prevent the narrow correction of the reported prose-hyphen defect.

Original run001 evidence and the earlier source remain preserved. Corrected controls and any run002 require their separately reviewed source/card and Inference-owned execution. No new scientific threshold, full profile, or extra numerical audit is proposed here. The amendment's unchanged resource caps are inherited from the measured original scan; this source review supplies no new runtime measurement.

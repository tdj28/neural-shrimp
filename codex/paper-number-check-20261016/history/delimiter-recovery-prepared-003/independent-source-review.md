# Independent narrow delimiter-fix review

Accepted for the stated source correction and seven manufactured fixture definitions; no concrete blocker found. I read only the exact `math_spans` delta, amendment, surrounding sign-use interface, and focused fixture source. No imports, controls, checker execution, target outputs, or broad prior-suite review were performed.

The parity check counts the consecutive backslashes immediately before each candidate delimiter. In a literal `\\[2pt]`, the candidate beginning at the second backslash has one preceding backslash and is correctly ignored. In `\\\[x-2\]`, the actual display opener has two preceding backslashes and remains active. The analogous even/odd rule applies to dollars and named environment candidates; this resolves the reported row-break escape defect without changing numeric matching or candidate rules.

Single-dollar tokenization plus current inline/display state correctly separates the adjacent inline spans in `$-1$$-2$`. Display openers and closers consume their second dollar through `skip_until`; a dollar following an escaped literal dollar remains independently eligible. Closing delimiters are accepted only for the currently open span. The fixture expectations for row breaks before dollars, escaped-dollar boundaries, display-to-inline transitions, and the masked figure fragment agree with these source rules.

All seven controls compare both exact numeric Decimal values and exact substring lists for the recognized math spans. Thus the row-break test would expose the original erroneous suffix-wide display span rather than merely relying on a favorable sign. The figure fixture also passes through the existing length-preserving cleanup. Its retained `2` from row-spacing `2pt` is deliberate unchanged candidate behavior in this narrow test, not a claim that rendering spacing is scientific evidence. Each failed case remains in the written report before nonzero exit.

This remains a literal TeX heuristic, with the stated exclusions for macro expansion, malformed/unbalanced input, and text semantics nested in math. Source acceptance does not imply that these unexecuted controls have passed, nor that the corrected full scan will complete within its inherited cap. Runs001/002 remain preserved; Inference owns any separately admitted controls and run003. No new scientific threshold, profile, or numerical verifier is requested.

Reviewed pins: checker `d8c4c278`, diff `69162f69`, amendment `3bf68a84`, and focused controls `95396d0b` (full SHA256 values retained in the adjacent source release).

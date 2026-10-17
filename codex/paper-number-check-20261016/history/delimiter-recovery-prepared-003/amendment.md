# Task26 delimiter correction after saved run002 review

17 October 2026. Both original scans remain unchanged. Run002 completed under
its fixed limits, but Root's saved-token comparison found that its 48 corrected
prose signs stopped at main.tex line602. Later prose such as `period-8` at
line611 was still negative. The concrete cause is the `\\[2pt]` row-break
syntax at line606: the second backslash was mistakenly treated as `\[`,
opening display math through the remainder of the document.

Only `math_spans` changes. Literal delimiter candidates now check the parity
of preceding backslashes. Dollars are handled one at a time with the current
inline/display state, so adjacent inline spans and escaped-dollar boundaries
do not desynchronize the state. Number matching, candidate retention, quoted
precision, bounds, inputs and ranking remain unchanged. This remains a literal
TeX heuristic; macro expansion, arbitrary malformed TeX, text-mode commands
nested within math and general TeX interpretation are outside its scope.

Seven new focused manufactured groups cover row-break brackets, row breaks
before dollars, adjacent inline spans, escaped-dollar boundaries, a row break
before display math, display-to-inline transitions and a masked figure
fragment. They compare both the exact numeric values and exact math-span
strings. They have been prepared and AST-parsed only, not executed. The earlier
19 broad groups and eight sign groups remain accepted historical controls and
will not be repeated.

A separately reviewed single corrected scan003 may use the same frozen inputs
and unchanged CPU1 plus supervisor, 2GiB AS, 140CPU/150wall and 128MiB output
limits after these controls pass and owned processes are closed. Run002's
measured external104.342199s and 22,560,727B support those unchanged caps;
they are not a worst-case guarantee. Private source/card checkpoint and fresh
capacity remain required. No automatic retry, extension or altered scientific
criterion is authorized. Neither prior scan is final support for paper claims.

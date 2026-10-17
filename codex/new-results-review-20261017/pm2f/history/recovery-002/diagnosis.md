# PM2F failed integrated control: narrow diagnosis and prepared correction

The original controls.json records exactly one failure, `duplicate_source_rows_retained`, with9 other groups passing. The driver wrote failure.json and refused before target metadata or scientific row analysis. The original source and run-001 evidence remain unchanged.

The core's `previous = observed.get(index, [])` returns the existing mutable list on a repeated index. The next append adds the current line to that same list. Consequently, when `duplicate_of_lines = list(previous)` is constructed, the second row's predecessor witness includes its own line `[1,2]` instead of the intended `[1]`. The existing manufactured control correctly catches this actual source defect; its expectation is not relaxed.

The sole change snapshots the predecessor list before the append: `previous = list(observed.get(index, []))`. The full occurrence inventory still receives every source line, and no record is discarded. All scientific arithmetic, thresholds, publication interpretations, raw ordering and controls are unchanged. New source is prepared under recovery-002; the original core and release remain byte-identical at their prior paths.

This diagnosis is a static source trace aligned with the recorded assertion failure. No control, scorer, target row or model was executed for the correction. A retry still requires Root's exact correction/card acceptance and Inference's private checkpoint/admission; the previous result is a failed attempt, not scientific completion.

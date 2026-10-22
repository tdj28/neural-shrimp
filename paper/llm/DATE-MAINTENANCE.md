# Manuscript dates

`main.tex` fixes the original date at October 22, 2026 and preserves the
supplementary-material subtitle. Change `originaldate` only intentionally.

`revision-date.tex` is generated from the most recently saved `.tex`, `.bib`,
or figure asset, in America/Los_Angeles time with the appropriate PST/PDT label.
The local `neural-shrimp-llm-date.service` user service checks saves once per
second. It only updates that small file; it does not compile the paper. The PDF
shows the new timestamp after the next compilation.

From this directory, `latexmk -pdf main.tex` also refreshes the timestamp before
building. Recompiling without saving a source does not advance it. The watcher
ignores its generated file, build products and the output PDF. Saving unchanged
source still counts as a save. Copying or checking out source files can also
change their filesystem timestamps; this is a last-save date, not a Git commit
date or a historical publication certificate.

For another machine, run `python3 tools/update_llm_revision.py --watch` from the
repository root, or install the user service template in this directory after
adjusting its paths. To stop the local watcher:

```
systemctl --user disable --now neural-shrimp-llm-date.service
```

Include `revision-date.tex` in standalone source archives. Plain LaTeX reads the
saved value without a Python dependency. Keep the generated file with source
commits when the manuscript changes.

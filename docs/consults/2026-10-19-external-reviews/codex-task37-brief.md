# Independent check: arXiv v1 pre-submission audit and the kite of the period-6 shrimp (for Codex, 19 October 2026)

You are an independent checker. The human author submits the Chaos article to arXiv on 20 October 2026. Find errors
before then. Verify every claim against raw evidence or publisher records. Do not trust summaries, log entries or
this brief.

Reference states:
- the article: `/data2/chaos-journal-neural-shrimp`, commit `18490e2` (`main.tex`, `references.bib`, `main.pdf`);
- the research repository: `/data2/neural-shrimp`, commit `35eabc6c`;
- raw data: `/data2/neural-shrimp-data`.

Claude keeps editing tonight. Locate claims by section and paragraph title, not only by line number.

## Rules

- **Read-only, except your own folder.** Do not edit anything in `/data2/chaos-journal-neural-shrimp`,
  `/data2/praxagent` or the papers. In `/data2/neural-shrimp`, create and commit only `codex/presubmit-task37/`, on main.
  Never force-push.
- **Where your work goes.**
  - Report: **`codex/presubmit-task37/review.md`** in `/data2/neural-shrimp`, in the format at the end of this file.
  - Scripts and small JSON: `codex/presubmit-task37/`.
  - Bulky outputs (orbit arrays): `/data2/neural-shrimp-data/scratch/audit-codex-1019/`.
- **No paid compute.** No RunPod commands, no uploads, no API calls that cost money.
- **Local hardware.**
  - CPU only, at most 6 threads, under `nice -n 10`. Claude's runs and pods share the machine.
  - Do not use the GPU.
- **Secrets.** Never open, print or copy `/data2/neural-shrimp/.env`.
- **Off limits.** Do not open, list or search `/data2/lexical-prompt-study` or `/data2/PRAX/lexical-prompt-study`.
- **Python.** Use `uv run --no-sync python ...` from `/data2/neural-shrimp`. Set `HF_HUB_OFFLINE=1`.

## Out of scope

The paragraph "A second shrimp, in the plane of Fig. 1", Fig. 20 and its caption are being rewritten tonight. Check
them only in task 2's terms. Claude may also redraw the period-5 shrimp figures; if a figure changed after `18490e2`,
say which version you checked.

## 1. The article's changes of 19 October (except the second shrimp)

1. **Introduction, architecture boundary** (the sentences after "a soft-thinking pipeline with a growing context"). Is
   the contrast with autoregressive generation and with recurrent-depth transformers correct and fairly cited?
2. **Discussion, the paragraph on feedback loops and Huginn** (beginning "and in recurrent-depth models the loop is a
   third kind"). Check each number against the full report (`paper/llm/main.tex`, section on the recurrent-depth model)
   and its raw records:
   - 1.0777 and 1.0778;
   - the generation test at relaxation 1.20 and 1.05;
   - 6 of 30 ARC-Challenge development questions with unstable native fixed points, and under-relaxation restoring
     convergence;
   - "no chaos".

   Is "none is established for a complete transformer update" accurate as worded?
3. **The safety sentence** after the band-chaos sentence ("These findings motivate testing whether interventions ...").
   Does any phrase claim more than was tested?
4. **Data Availability.** Do the cited paths exist at `35eabc6c`?
   - `codex/certify-llm`;
   - `certificates/llm-p8-certificate-final-001.json`;
   - `codex/coexistence-task33`.

   Do those records support "interval-arithmetic certificates, with their verification code, exact inputs and release
   records"?
5. **Bibliography entries added on 19 October.** Check each against the arXiv or publisher record: authors, title, year,
   venue, and DOI or arXiv ID. Record anything you cannot reach as unverified.
   - `dehghani2019universal`;
   - `giannou2023looped`;
   - `bae2025mor`;
   - `korbak2025monitorability`;
   - `zhu2025ouro`;
   - `clark2018arc`;
   - `wainwright2008graphical`;
   - `gao2017softmax`.
6. **The anatomy figure's caption** (Fig. 16, `fig-shrimp-anatomy.png`, made by `tools/plot_shrimp_anatomy.py`).
   - Check "his eq. (5.3), rotated and labeled as in his Fig. 7" against Gallas, Physica A 202, 196 (1994). The PDF is
     at `/data2/neural-shrimp-data/references/gallas_dissecting_shrimps_PhysA1994.pdf`, journal p. 217.
   - Check "the computed corners A and B agree with Gallas's to six digits". Gallas gives A ≈ (1.01905888,
     1.16529086) and B symmetric.
   - Check "the flip curves of the two legs cross at R".
7. **Appendix A and Table IV** (the four period-5 cusps). Check c, det J, the critical multiplier and the next modulus
   against the cusp-location records the appendix cites (search `/data2/neural-shrimp-data/explore` for the FM5c/FM5d
   `locate.json` files).
8. **Whole-article consistency.**
   - The abstract's numbers against the body.
   - Every reference cited and every citation defined.
   - Figure and table numbering.
   - The arXiv metadata draft `/data2/neural-shrimp-data/arxiv/chaos-v1/arxiv-form.md` (pages, figures, tables,
     abstract length at most 1920 characters) against the built PDF.

## 2. Independent check: do two different period-6 orbits meet at the flip crossing R?

The second shrimp's two flip curves cross at R ≈ (α, log10 T) = (0.072912, −0.154104), in the plane of Fig. 1:
Qwen2.5-0.5B-Instruct, philosophy direction at layer 12, float64, window 1.

Claude's working hypothesis is Gallas's kite:
- the two curves belong to two different period-6 orbits, on two sheets of the family;
- R is a pseudo-crossing, like Gallas's tail T.

Claude's own check is running with the project's tools. Yours should be independent: use Codex's manual forward map
`codex/crosscheck/qwen_map.py` (`ManualQwenSoftMap`), not the research tools.

Inputs (float64 states, one orbit point per curve point, in
`/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/p6s/`):
- flip curve 1: `p6s-ad-curve-plus/states.npy`, row 6. Its point in `curve.json` is (α, log10 T) = (0.072907, −0.153952).
- flip curve 2: `p6s-td-curve-minus/states.npy`, row 2, at (0.072919, −0.154094). Match the row index to the point's
  `index` field and record both.

Task:
1. At exactly α = 0.072912, log10 T = −0.154104 (with T = 10^log10 T), find a period-6 orbit by Newton's method from
   each state. Both orbits should have a multiplier near −1 there.
2. Report:
   - the residuals;
   - whether the two converged orbits coincide up to cyclic shift: the largest point distance after the best shift,
     relative to the mean state norm;
   - the eight largest multipliers of each orbit;
   - each orbit's most probable token at each point.
3. If they differ, repeat at two points inside the body near R, (0.07300, −0.15390) and (0.07310, −0.15370). Do both
   orbits exist there? Are both attracting (all multipliers inside the unit circle)?

The project rules apply: verify orbits in float64; record solver counts and residuals; report a failed convergence as
a result, never as evidence.

## Report format (`codex/presubmit-task37/review.md`)

Start with a three-line summary:
- how many findings, and how many must-fix;
- whether task 2 found two distinct orbits at R.

Then, for each finding:

```
### F<n>. <one-line title>   [must-fix | should-fix | note]
Where: <file>, <section/paragraph title>, line ~<n> at 18490e2
Claim: <quoted text>
Evidence: <raw file(s) and the computation; scratch script path>
Correct value / wording: <...>
```

End with a list of every check you completed, partly completed or skipped, and why. Record anything you could not
verify as unverified, not as correct.

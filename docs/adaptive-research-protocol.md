# Adaptive research protocol

**User directive, 6 October 2026.** Research here should be exploratory and
adaptive. Plans change when findings warrant it, and every change is recorded
openly. Credibility comes from complete, time-stamped records and honest
labels, not from refusing to adapt. This protocol governs Program C onward. It
replaces the requirement that every run be preceded by a complete pushed
machine plan. The hard rules at the end are unchanged.

**One line:** explore freely, record everything, label honestly, confirm
headline claims on fresh data.

## Three modes

### 1. Explore (default)

Any question, code, parameter or analysis can be tried at any time.

- Commit the code before the run; a local commit is enough, and pushes follow
  the regular checkpoint cadence in Skills.md.
- Add a short run card to the research log before running. It can be written
  seconds before the run: question, setup, output location, and what would
  surprise us. `tools/research_log.py run` writes the commit, dirty status,
  command and time automatically.
- Keep all outputs, including dull, failed or abandoned runs. Never delete a run
  because its result was unwelcome.
- Label results "exploratory" wherever they are written up.
- No machine plan, validator or independent audit is required. Quick sanity
  checks and positive controls are encouraged.

### 2. Adapt (changing any plan, including frozen ones)

Plans are living documents. When a finding warrants a change, make it, and
append a dated amendment to the plan or the research log stating:

- what changed;
- why, naming the finding that prompted it;
- whether relevant outcomes had been seen, and which;
- the effect on claims, for example "endpoint X is now exploratory".

Never rewrite or delete earlier entries; Git keeps prior versions. An adapted
analysis is labeled "adapted after seeing X". It can still be reported
prominently, but it is not called confirmatory.

### 3. Confirm (headline claims only)

When exploration converges on a claim the paper will headline, test it once
on data that did not shape it: new seeds, networks, datasets, or held-out
parameter regions. Commit and push the test design before running it. One
independent check of the saved evidence is required. A confirmatory test can
still be amended, but any amendment made after seeing its outcome downgrades
the affected claim to "adapted".

## Labels used in writing

| Label | Meaning |
|---|---|
| Exploratory | Found while exploring; analysis choices may depend on what was seen |
| Adapted | A planned analysis changed after related outcomes were seen; the change is disclosed |
| Prospectively frozen | Design pushed before outcomes, as for EXP-028 and EXP-029 |
| Confirmatory | A frozen test on fresh data of a claim formed earlier |

## Research log

`docs/research-log.md` is an append-only, dated record of runs, decisions,
surprises and amendments, with commit hashes and output locations. It is the
paper's audit trail: the exploration path is disclosed, not hidden.

## Existing studies

EXP-028 and EXP-029 keep their prospectively frozen records. Further changes to
them, for example a successor that runs a stage from saved maps or a different
zoom, follow mode 2. Their original attempts and labels remain untouched.

## Hard rules (unchanged)

Paid compute and paid reviews need explicit per-call approval. Public release,
arXiv posting and repository visibility are human decisions. Raw evidence and
failed attempts are preserved. Results are never fabricated, silently dropped
or rerun until a preferred outcome appears. Git and secret hygiene rules in
AGENTS.md and the Git playbook still apply.

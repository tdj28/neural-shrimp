# Chaos and bifurcations in continuous language-model feedback

This repository holds the code, test designs and records behind the paper of that title. The paper studies what
happens when a pretrained language model is fed back its own expected next token, as in Soft Thinking. With the
prompt fixed, this loop is a discrete-time map of the model's embedding space, and it can be analyzed with the tools
used for low-dimensional dynamical systems. Its main findings:

- Temperature and activation steering move the loop between repeating text and chaos (positive finite-time Lyapunov
  exponents). This holds in Qwen2.5, OLMo-2 and Llama-3 models from 0.5B to 72B parameters.
- The loop's derivative factorizes into the inverse temperature, the covariance of token embeddings under the
  next-token distribution, and the network's Jacobian.
- The periodic windows show the signatures of low-dimensional maps:
  - saddle-node scaling: passage times predicted in advance from the model's derivatives came within one iterate in 42
    of 48 cases;
  - a period-doubling cascade with Feigenbaum's ratios;
  - shrimp-shaped windows with located cusps.
- Interval arithmetic proves attracting periodic orbits of the full transformer map.
- Whether rounding differences reach the output depends on the regime. In Qwen2.5-0.5B, changing only the batch
  layout changed the text of 173 of 192 chaotic float64 loops and of 7 of 192 periodic ones.

## The paper

- **arXiv version:** a 27-page article written for the journal *Chaos* (link to be added when the preprint is
  announced).
- **Full report:** [`paper/llm/main.pdf`](paper/llm/main.pdf), about 160 pages, with source in `paper/llm/`. It has every
  result, the methods, the test designs fixed before their data, the controls, and the failed or inconclusive attempts.
- The tag `arxiv-v1` marks the state of this repository at the arXiv submission.

## Layout

| Path | Contents |
|---|---|
| `paper/llm/` | The full report: LaTeX source, figures, PDF |
| `tools/` | All analysis code (see below) |
| `codex/` | Independent implementations, audits and the interval-arithmetic certificates, written by a second agent (see "How the work was done") |
| `certificates/` | Release records of the computer-assisted proofs |
| `docs/research-log.md` | The research log: a run card before each run and the result after it, in time order |
| `docs/consults/` | External reviews of the work and how each finding was handled |
| `docs/literature/` | Literature notes |
| `docs/prompt-variants/` | The prompt variants of the prompt-dependence test |
| `tests/` | Unit tests (pytest) |
| `comms/` | The running exchange between the two AI agents that did the work |

Directories not listed (`experiments/`, `reports/`, `native/`, `src/` and much of `docs/`) hold the project's
earlier work on small recurrent networks.

### The main tools

- `tools/explore_llm_soft_loop.py` defines the soft-token map. Its subcommands:
  - `scan`: Lyapunov maps over steering strength and temperature;
  - `orbit`: periodic orbits and their multipliers;
  - `branch`: one-parameter continuation of a periodic orbit;
  - `curve`: two-parameter continuation of fold and flip curves.
- `tools/bifurcation_curves.py` is the continuation library: pseudo-arclength steps, minimally augmented systems for
  fold and flip curves, and fold normal forms.
- `tools/cusp_locate.py` locates cusps and other codimension-two points on continued curves.
- `tools/follow_attractor.py` follows one attractor cell by cell across a parameter plane.
- `tools/two_block_map.py` maps the plane of two steering vectors at two layers.
- `tools/huginn_*.py` and `tools/explore_latent_loop.py` analyze the recurrent-depth model Huginn.
- `tools/plot_*.py` draw the paper's figures. Most figures have a JSON record beside them in `paper/llm/figures/`.
- `tools/runpod_soft_float64.py` and `tools/runpod_soft_runner.sh` ran the GPU jobs on rented machines. Each run card in
  the research log names the job, the commit and the hardware.
- `codex/crosscheck/qwen_map.py` is a separate, hand-written forward pass of Qwen2.5-0.5B. It was used to recompute key
  results without the research tools.

## Running the code

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11 or later:

```
uv sync
uv run python tools/explore_llm_soft_loop.py scan --device cpu --dtype float64 --resolution 8 8 \
    --alpha 0.0625 0.0775 --log-temperature -0.17 -0.125 --out llm-soft-loop/example
```

Outputs and downloaded model weights go to `../neural-shrimp-data/` unless `NEURAL_SHRIMP_DATA` points elsewhere.
Models are fetched from Hugging Face on first use. The example above, 64 cells in float64, takes about eight minutes
on a CPU. The maps in the paper were
computed in float64 on rented GPUs.

## Data

Raw trajectories, model states and the native evidence of the certificates (center definitions and interval
enclosures) total about 270 GB and are kept outside the repository. The research log and the receipts in
the repository record their SHA-256 checksums. They are available from the author on request.

## How the work was done

The study was exploratory. Each run has a run card in the research log, written before the run except where the log
says otherwise. Claims tested prospectively had their designs pushed before the data were collected. The work was carried out by two AI agents, Anthropic's Claude
Code and OpenAI's Codex, under the author's direction:

- The agents wrote and ran the code, ran the experiments and drafted text.
- Codex also made independent checks.
- Other AI systems reviewed the results and the manuscript read-only. Their reviews and the decisions taken on each
  finding are in `docs/consults/`.

Failed attempts are kept, not deleted.

## License

The code is under the MIT License ([`LICENSE`](LICENSE)). The manuscripts, figures, documentation and research records
are under CC BY 4.0 ([`LICENSE-docs.md`](LICENSE-docs.md)).

## Citation

Timothy D. Jones, *Chaos and bifurcations in continuous language-model feedback*, arXiv (2026); identifier to be added
when the preprint is announced.

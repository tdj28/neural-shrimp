# Reproducing the period-8 loop of Qwen2.5-0.5B-Instruct

Goal: feed the model its own expected next-token embedding (the "soft token") in a loop, in float64. With the
settings below, the loop settles within about 80 steps onto a repeating cycle of 8 states whose most probable tokens
read

    . This is there is a is a

This is the period-8 orbit proved to exist and to be attracting by interval arithmetic in the paper (its table of
stripe orbits and its section on computer-assisted proofs). The script `tools/recreate_p8.py` does everything below in about a hundred lines, using only torch, transformers and
numpy; it takes a few minutes on a CPU. From the repository root:

    python tools/recreate_p8.py docs/steering-vectors/vectors-l12/vectors.npz 300 docs/reproduce-p8/p8-reference-states.npz

Files here:

- `expected-output.txt`: the script's output on a CPU (torch 2.14.1, transformers 5.19.0).
- `p8-reference-states.npz`: the eight float64 states of the loop (steps 94 to 101), 56 KB, SHA-256
  `11b2d28e0aecbeeb3def5646f6cfd01763d082f6218b2d26ea16a6737813cb05`. Given as the third argument, the script
  prints the largest coordinate difference between your loop and these states, allowing for a shift of phase.
  Rerunning with 4 instead of 8 CPU threads gave 1.6e-15. Other hardware and library versions change only the last
  bits of each step, so expect a difference near the rounding level, far below the 0.46 that separates the eight
  states.

## 1. Model

- `Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face, revision `7ae557604adf67be50417f59c2c2f167def9a775`
  (SHA-256 of `model.safetensors`: `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`).
- Load it in float64 with `attn_implementation="sdpa"`. Tested with torch 2.14 and transformers 5.19; transformers 4.x
  also works.

## 2. Make float64 real (important)

Hugging Face computes two things in float32 even in a float64 model:

- **RMSNorm** upcasts to float32. Replace its forward with
  `weight * (h * rsqrt(mean(h^2) + eps))` computed in the input dtype.
- **Rotary embeddings** build their frequency, cos and sin tables in float32. Rebuild them in float64:
  `inv_freq = 1 / rope_theta ** (arange(0, 64, 2) / 64)` (head dim 64, rope_theta 1,000,000), then
  `cos, sin` of `position * inv_freq`.

Without these patches the run is not truly float64, and the residuals below will not reach 1e-14.

## 3. The fixed prompt (37 tokens)

Apply the model's own chat template with `add_generation_prompt=True` to:

- system: `You are an AI talking with another AI. You have complete freedom. Feel free to pursue whatever you want.`
- user: `Hello!`

The token IDs must be exactly:

    151644 8948 198 2610 525 458 15235 7404 448 2441 15235 13 1446 614 4583 11290 13 31733 1910 311 22729 8820
    498 1366 13 151645 198 151644 872 198 9707 0 151645 198 151644 77091 198

## 4. Steering

- Vector: the `philosophy` entry of `docs/steering-vectors/vectors-l12/vectors.npz` in
  https://github.com/tdj28/neural-shrimp (SHA-256 `b630bb7bef06bde87f8e6a7ca5e8672f83d24bc0585bf19048e9e494a4357f41`;
  896 numbers, norm 188.99). How it was built is in the paper's section on the soft-token map.
- Strength `alpha = 0.071674`.
- Add `alpha * v` to the **output of decoder layer 12** (counting from 0, of 24), at **every position**: all 37
  prompt tokens and the fed-back soft token. A forward hook on `model.model.layers[12]` does this.

## 5. The loop

- Temperature `T = 10 ** -0.115 = 0.767361...`.
- State `x`: one 896-dimensional vector, appended as an input embedding after the 37 prompt embeddings. The input
  always has 38 positions: the window never grows.
- One step: run the model on `[prompt embeddings, x]`, take the logits at the last position, `p = softmax(logits / T)`
  over the full vocabulary of 151,936 tokens, and set `x = p @ E`, where `E` is the input embedding matrix (tied with
  the output head). No sampling, no top-k or top-p, no repetition penalty.
- Start: `x0 = E[198]`, the embedding of the prompt's last token, the newline after `assistant`.

## 6. What you should see

- The distance `|x_t - x_(t-8)|` falls to about 1e-7 by step 50, below 1e-12 by step 71, and sits at the
  rounding floor (1e-14 to 1e-16) from about step 80 on. Run 300 steps to be safe.
- After convergence, `|x_t - x_(t-1)|`, `|x_t - x_(t-2)|` and `|x_t - x_(t-4)|` stay near 0.46 to 0.48: the period
  is exactly 8, not a divisor of it.
- The most probable token at each step cycles through the IDs `1096 374 1052 374 264 374 264 13`
  (` This`, ` is`, ` there`, ` is`, ` a`, ` is`, ` a`, `.`). From this start, the `.` falls on steps 5, 13, 21, ...
  (step numbers 8k + 5).

## Notes

- This is not ordinary text generation: the input is a probability-weighted average of embeddings, the prompt is
  fixed, and the window is one token long.
- Different hardware or library versions change the last bits of each step; the cycle and its tokens should not
  change, because the orbit strongly attracts nearby states (its leading multiplier is 0.011 per cycle).
- The proof's release record, with the exact parameters and hashes, is
  `certificates/llm-p8-certificate-final-001.json`; the complete proof evidence is at Zenodo,
  [doi:10.5281/zenodo.23488416](https://doi.org/10.5281/zenodo.23488416).

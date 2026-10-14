"""Loop-free properties of a model's start state, as cheap predictors of the soft-token loop's chaos (run CP).

For each prefix of a prefix file (system prompt and opener, chat template as in the loop), the loop's start state is
the prefix cache with the last prefix token's embedding x0 as the first soft token (explore_llm_soft_loop.SoftLoop).
One forward pass gives the start logits z0 and the final normalised hidden state h0; twins x0 + eps |x0| u (random
unit directions u) give the network's relative sensitivity at the start state,
    g_net = (|dh| / |h0|) / eps,
and, for each temperature T, the readout's relative gain g_read(T) = (|dF| / |F|) / (|dh| / |h0|), with
F = softmax(z / T) E (the one-step factors of tools/gain_split.py, at the start state instead of along an orbit).
Reported per prefix: the entropy H(T) of softmax(z0 / T) on a log10 T grid, the intermediate-entropy breadth (decades
of log10 T with 1 < H < 10 nats), the crossing width log10 T(H = 10) - log10 T(H = 1), the median log g_net over the
directions, and the one-step gain log g_net + log g_read(T) with its maximum over T.

Usage: python tools/start_state.py --model M --prefix-file docs/prefix-scale-prefixes.json --out OUT
       [--device cpu|cuda|auto --weights native|bf16 --dtype float64 --directions 8 --eps 1e-5]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
LOG_T = np.linspace(-1.5, 1.5, 601)
GAIN_LOG_T = np.linspace(-1.0, 1.0, 81)


def entropy_profile(z, log_t=LOG_T):
    """Entropy (nats) of softmax(z / T) for each log10 T, in float64."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty(len(log_t))
    for i, lt in enumerate(log_t):
        a = z / 10.0 ** lt
        a = a - a.max()
        p = np.exp(a)
        s = p.sum()
        p /= s
        out[i] = -(p[p > 0] * np.log(p[p > 0])).sum()
    return out


def breadth(log_t, H, lo=1.0, hi=10.0):
    """Decades of log10 T with lo < H < hi (spacing-weighted), and the crossing width T(H > hi) - T(H > lo); NaN when
    the profile never crosses a level inside the grid."""
    w = np.gradient(log_t)
    mid = float((w * ((H > lo) & (H < hi))).sum())
    if not (H > lo).any() or not (H > hi).any():
        return mid, float("nan")
    return mid, float(log_t[np.argmax(H > hi)] - log_t[np.argmax(H > lo)])


def set_prefix(loop, system, opener, model):
    import explore_llm_attractors as ela  # noqa: F401  (the loop's module state)

    torch = loop.torch
    messages = [{"role": "system", "content": system}, {"role": "user", "content": opener}]
    extra = {"enable_thinking": False} if str(model).startswith("qwen3.5") else {}
    text = loop.sm.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **extra)
    loop.prefix_ids = torch.tensor([loop.sm.tok(text, add_special_tokens=False)["input_ids"]], device=loop.device)
    loop.L0 = loop.prefix_ids.shape[1]


def start_pass(loop, directions, eps, seed=0):
    """h and z for the start state (row 0) and its twins (rows 1..directions)."""
    torch = loop.torch
    d = loop.E.shape[1]
    X = loop.prepare(np.zeros((1 + directions, d)))
    x0 = X[0, -1]
    u = torch.as_tensor(np.random.default_rng(seed).normal(size=(directions, d)), dtype=loop.state_dtype,
                        device=loop.device)
    u = u / u.norm(dim=1, keepdim=True)
    X[1:, -1] = x0 + eps * x0.norm() * u
    loop.sm.steer = loop.steer_rows
    try:
        with torch.no_grad():
            out = loop.model.model(inputs_embeds=X.to(loop.dtype), past_key_values=loop.cache, use_cache=True)
            h = out.last_hidden_state[:, -1]
            z = loop.model.lm_head(h).to(loop.device, loop.state_dtype)
    finally:
        loop.sm.steer = None
        loop.reset_cache()
    return h.to(loop.device, loop.state_dtype), z


def gains(loop, h, z, eps, log_t=GAIN_LOG_T):
    torch = loop.torch
    dh = (h[1:] - h[0]).norm(dim=1) / h[0].norm()
    g_net = (dh / eps).double().cpu().numpy()
    read = []
    for lt in log_t:
        F = torch.softmax(z / 10.0 ** lt, dim=-1) @ loop.E
        dF = (F[1:] - F[0]).norm(dim=1) / F[0].norm()
        read.append(float(np.median((dF / dh).double().cpu().numpy())))
    return g_net, np.log(np.maximum(np.array(read), 1e-300))     # dF = 0 when the readout is saturated (greedy)


def main():
    import explore_llm_attractors as ela
    import explore_llm_soft_loop as esl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefix-file", required=True)
    ap.add_argument("--directions", type=int, default=8)
    ap.add_argument("--eps", type=float, default=1e-5)      # well above the 70B twin noise floor (~2e-8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", default="native", choices=("native", "bf16"))
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    prefixes = json.loads(Path(args.prefix_file).read_text())["prefixes"]
    loop = esl.SoftLoop(args.model, 0, args.threads, "open", ela.OPENER, 1, args.dtype, args.device, args.weights)
    rows = []
    for i, pref in enumerate(prefixes):
        set_prefix(loop, pref["system"], pref["opener"], args.model)
        h, z = start_pass(loop, args.directions, args.eps)
        H = entropy_profile(z[0].double().cpu().numpy())
        mid, width = breadth(LOG_T, H)
        g_net, log_read = gains(loop, h, z, args.eps)
        one_step = float(np.log(np.median(g_net))) + log_read
        rows.append({"prefix_index": i, "prefix_tokens": int(loop.L0), "entropy_at_T1": float(H[np.abs(LOG_T).argmin()]),
                     "breadth_decades": mid, "crossing_width": width,
                     "log_g_net_median": float(np.log(np.median(g_net))),
                     "log_g_net_range": [float(np.log(g_net.min())), float(np.log(g_net.max()))],
                     "one_step_gain_max": float(one_step.max()),
                     "one_step_gain_argmax_log10T": float(GAIN_LOG_T[one_step.argmax()]),
                     "entropy_profile": H.round(5).tolist(), "log_g_read": log_read.round(5).tolist()})
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()
                          if k not in ("entropy_profile", "log_g_read")}), flush=True)
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "start-state.json").write_text(json.dumps(
        {"args": vars(args), "log10T": LOG_T.tolist(), "gain_log10T": GAIN_LOG_T.tolist(), "prefixes": rows}))


if __name__ == "__main__":
    main()

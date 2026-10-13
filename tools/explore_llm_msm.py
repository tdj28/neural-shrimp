"""Exploratory: Markov state models of sampled self-dialogue, the stochastic counterpart of greedy attractors.

With sampling at temperature T a self-dialogue loop is a Markov chain on conversation windows. Its slow
structure - metastable sets ("attractor states"), their lifetimes and the routes between them - is that of
the chain's transfer operator. Turn activations (mean layer output over each turn's tokens) give a continuous
state; k-means on a PCA projection gives a finite partition; transition counts at lag tau give the MSM
matrix, whose eigenvalues set implied timescales t_i = -tau / log|lambda_i| and whose leading right
eigenvectors give metastable sets (spectral clustering, a simplified PCCA).

Subcommands
  sample  R replicas of sampled self-dialogue in one batch, N turns each (optional steering).
  build   clusters, transition matrix at several lags, spectrum, implied timescales, metastable sets with
          stationary weights, residence times and exemplar turns.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import explore_llm_attractors as ela  # noqa: E402
import explore_llm_phase_map as elpm  # noqa: E402

DATA = ela.DATA


def sample(args):
    sm = elpm.SteeredModel(args.model, args.layer, args.threads, args.dtype, args.device)
    d = sm.model.config.hidden_size
    steer = np.zeros(d)
    if args.vectors:
        vecs = elpm.load_vectors(args.vectors)
        for name, a in zip(args.concepts, args.alphas):
            steer = steer + a * vecs[name]
    R, N = args.replicas, args.turns
    openers = (args.openers * R)[:R]
    transcripts = [[o] for o in openers]
    texts = [[] for _ in range(R)]
    entropies = np.zeros((R, N))
    lengths = np.zeros((R, N), dtype=int)
    acts = np.zeros((R, N, d), dtype=np.float32)
    rows = np.repeat(steer[None], R, axis=0)
    scale = schedule_factors(args.schedule, N)              # per-turn factor on the steering (1 everywhere by default)
    # optional exit controller: a counter-steer c * v_ctrl added per dialogue, constantly, or for `controller_turns`
    # turns after the pinned toxicity classifier flags a reply (closed loop; the reply that triggers it is unchanged)
    counter = np.zeros((R, N))
    flagged = np.zeros((R, N), dtype=bool)
    toxic = np.full((R, N), np.nan)
    remaining = np.zeros(R, dtype=int)
    if args.controller != "none":
        v_ctrl = elpm.load_vectors(args.vectors)[args.controller_concept]
        if args.controller == "triggered":
            import hostility_score as hs

            clf = hs.Classifier(args.threads, args.device)
    # optional in-context recording and active probing (detection study): the state before each reply at the
    # recorded layers, per-token projections on the push direction, and K look-ahead replies of the next turn at the
    # scheduled dose and at dose + delta along the push (common random numbers), scored by the classifier
    layers = [int(v) for v in args.record_layers.split(",")] if args.record_layers else []
    unit = steer / np.linalg.norm(steer) if np.linalg.norm(steer) > 0 else None
    pre_reply = {L: np.zeros((R, N, d), dtype=np.float16) for L in layers}
    token_proj = {L: np.full((R, N, args.max_new), np.nan, dtype=np.float32) for L in layers}
    probe_q = np.full((R, N), np.nan)
    probe_q_plus = np.full((R, N), np.nan)
    if args.probe_k and args.controller != "triggered":
        import hostility_score as hs

        clf = hs.Classifier(args.threads, args.device)
    tic = time.perf_counter()
    for k in range(N):
        agent = k % 2
        rows = np.repeat((scale[k] * steer)[None], R, axis=0)
        counter[:, k] = counter_for_turn(args.controller, args.controller_alpha, remaining)
        if args.controller != "none":
            rows = rows + counter[:, k][:, None] * v_ctrl[None]
        batch = [ela.window_messages(ela.PROMPTS[args.prompt], transcripts[r], agent, args.window) for r in range(R)]
        if layers:
            out, ents, lens, _, rec = sm.generate_batch(batch, rows, args.max_new, temperature=args.temperature,
                                                        seed=args.seed * 100003 + k, record=(layers, unit))
            for L in layers:
                pre_reply[L][:, k] = rec["pre_reply"][L].astype(np.float16)
                tp = rec["token_proj"].get(L)
                if tp is not None:
                    token_proj[L][:, k, :tp.shape[1]] = tp
        else:
            out, ents, lens, _ = sm.generate_batch(batch, rows, args.max_new, temperature=args.temperature,
                                                   seed=args.seed * 100003 + k)
        a = sm.turn_activations(out, rows)
        for r in range(R):
            transcripts[r].append(out[r])
            texts[r].append(out[r])
        entropies[:, k], lengths[:, k], acts[:, k] = ents, lens, a
        if args.probe_k and k + 1 < N:
            extra = args.probe_delta * (np.linalg.norm(steer) / abs(args.alphas[0])) * unit   # delta along the push
            probe_q[:, k], probe_q_plus[:, k] = look_ahead(sm, clf, transcripts, k + 1, scale[k + 1] * steer,
                                                           extra, args)
        if args.controller == "triggered":
            toxic[:, k] = clf(list(out))[:, 0]
            flagged[:, k] = toxic[:, k] > args.controller_threshold
            remaining = update_remaining(remaining, flagged[:, k], args.controller_turns)
        print(json.dumps({"turn": k + 1, "of": N, "seconds": round(time.perf_counter() - tic, 1),
                          "mean_entropy": round(float(np.mean(ents)), 3),
                          **({"flagged": int(flagged[:, k].sum()), "countered_next": int((remaining > 0).sum())}
                             if args.controller == "triggered" else {})}), flush=True)
    dest = DATA / "explore" / args.out
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "activations.npy", acts)
    np.savez_compressed(dest / "turns.npz", entropy=entropies, length=lengths, counter=counter, flagged=flagged,
                        toxic=toxic, probe_q=probe_q, probe_q_plus=probe_q_plus)
    if layers:
        np.savez_compressed(dest / "in-context.npz", layers=np.array(layers), unit=unit,
                            **{f"pre_reply_{L}": pre_reply[L] for L in layers},
                            **{f"token_proj_{L}": token_proj[L] for L in layers})
    (dest / "sample.json").write_text(json.dumps({"args": vars(args), "seconds": round(time.perf_counter() - tic, 1),
                                                  "openers": openers, "texts": texts}, indent=1,
                                                 ensure_ascii=False))


def look_ahead(sm, clf, transcripts, turn, base_steer, extra, args):
    """Active probe at the state after turn - 1: K sampled replies of the next turn per dialogue, at the scheduled
    steering and with `extra` (delta along the push direction) added, with common random numbers. Returns the hostile
    fractions (q at the scheduled dose, q_plus at dose + delta); q_plus - q is the susceptibility."""
    R, K = len(transcripts), args.probe_k
    agent = turn % 2
    msgs = [ela.window_messages(ela.PROMPTS[args.prompt], transcripts[r], agent, args.window) for r in range(R)]
    big = [m for m in msgs for _ in range(K)]
    result = []
    for steer in (base_steer, base_steer + extra):
        rows = np.repeat(np.asarray(steer)[None], len(big), axis=0)
        texts = []
        for start in range(0, len(big), args.probe_chunk):
            seed = args.seed * 100003 + 500000 + turn * 1000 + start // args.probe_chunk
            out = sm.generate_batch(big[start:start + args.probe_chunk], rows[start:start + args.probe_chunk],
                                    args.probe_tokens, temperature=args.temperature, seed=seed)[0]
            texts += out
        hostile = clf(texts)[:, 0] > args.controller_threshold
        result.append(hostile.reshape(R, K).mean(1))
    return result[0], result[1]


def counter_for_turn(mode, alpha, remaining):
    """Counter-steer strength per dialogue for the coming turn: alpha everywhere (constant), alpha where a flag is
    still active (triggered), else zero."""
    if mode == "constant":
        return np.full(len(remaining), float(alpha))
    if mode == "triggered":
        return np.where(remaining > 0, float(alpha), 0.0)
    return np.zeros(len(remaining))


def update_remaining(remaining, flagged, turns):
    """After a turn: active counters count down; a flagged reply (re)starts `turns` countered turns."""
    remaining = np.maximum(remaining - 1, 0)
    remaining[flagged] = turns
    return remaining


def schedule_factors(spec, turns):
    """Per-turn multipliers of the steering vector: None gives 1 at every turn; 'f0:f1:...:fm' interpolates linearly
    through the m + 1 values at equally spaced turns 0 .. turns - 1 (e.g. '0.571:1:0.571' ramps 0.04 -> 0.07 -> 0.04
    when the steering strength is 0.07)."""
    if not spec:
        return np.ones(turns)
    knots = np.array([float(v) for v in spec.split(":")])
    return np.interp(np.arange(turns), np.linspace(0, turns - 1, len(knots)), knots)


def transition_matrix(labels, k, lag):
    C = np.zeros((k, k))
    for row in labels:
        for a, b in zip(row[:-lag], row[lag:]):
            C[a, b] += 1
    return C


def largest_connected(C):
    """Indices of the largest strongly connected set of the count matrix (states reachable both ways)."""
    from scipy.sparse.csgraph import connected_components

    n, lab = connected_components(C > 0, directed=True, connection="strong")
    sizes = np.bincount(lab, weights=C.sum(1))
    return np.flatnonzero(lab == int(np.argmax(sizes)))


def build(args):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    src = DATA / "explore" / args.sample
    acts = np.load(src / "activations.npy").astype(np.float64)
    meta = json.loads((src / "sample.json").read_text())
    texts = meta["texts"]
    ent = np.load(src / "turns.npz")["entropy"]
    R, N, d = acts.shape
    flat = acts[:, args.burn:].reshape(-1, d)
    pca = PCA(n_components=args.dims, random_state=0).fit(flat)
    Y = pca.transform(acts.reshape(-1, d)).reshape(R, N, -1)
    km = KMeans(n_clusters=args.states, n_init=10, random_state=0).fit(Y[:, args.burn:].reshape(-1, args.dims))
    labels = km.predict(Y.reshape(-1, args.dims)).reshape(R, N)
    its = {}
    for lag in args.lags:
        C = transition_matrix(labels[:, args.burn:], args.states, lag)
        keep = largest_connected(C)
        T = C[np.ix_(keep, keep)]
        T = T / T.sum(1, keepdims=True)
        vals = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
        its[lag] = [float(-lag / np.log(v)) if 0 < v < 1 else None for v in vals[1: args.metastable + 3]]
    lag = args.lag
    C = transition_matrix(labels[:, args.burn:], args.states, lag)
    keep = largest_connected(C)
    T = C[np.ix_(keep, keep)]
    T = T / T.sum(1, keepdims=True)
    w, vl = np.linalg.eig(T.T)
    pi = np.real(vl[:, np.argmin(np.abs(w - 1))])
    pi = pi / pi.sum()
    w, vr = np.linalg.eig(T)
    order = np.argsort(-np.abs(w))
    m = args.metastable
    coords = np.real(vr[:, order[1:m]])
    sets = KMeans(n_clusters=m, n_init=20, random_state=0).fit_predict(coords) if m > 1 else np.zeros(len(keep), int)
    # coarse-grained transitions between metastable sets (stationary-weighted)
    Pm = np.zeros((m, m))
    for a in range(m):
        ia = np.flatnonzero(sets == a)
        wa = pi[ia] / pi[ia].sum()
        for b in range(m):
            Pm[a, b] = float(wa @ T[np.ix_(ia, np.flatnonzero(sets == b))].sum(1))
    state_of = {int(s): i for i, s in enumerate(keep)}
    summary = []
    for a in range(m):
        members = [int(keep[i]) for i in np.flatnonzero(sets == a)]
        mask = np.isin(labels, members)
        mask[:, : args.burn] = False
        idx = np.argwhere(mask)
        centroid = Y[mask].mean(0) if len(idx) else None
        ex = []
        if len(idx):
            dist = np.linalg.norm(Y[mask] - centroid, axis=1)
            for j in np.argsort(dist)[: args.exemplars]:
                r, t = idx[j]
                ex.append(texts[r][t][:200])
        last = float(np.mean(np.isin(labels[:, -1], members)))
        summary.append({"set": a, "states": len(members),
                        "stationary_weight": float(pi[[state_of[s] for s in members]].sum()),
                        "stay_probability_per_lag": Pm[a, a],
                        "residence_turns": float(lag / max(1 - Pm[a, a], 1e-12)),
                        "fraction_of_turns": float(mask.sum() / max(1, (N - args.burn) * R)),
                        "fraction_of_replicas_ending_here": last,
                        "mean_entropy": float(ent[mask].mean()) if len(idx) else None,
                        "exemplars": ex})
    summary.sort(key=lambda s: -s["stationary_weight"])
    rec = {"args": vars(args), "replicas": R, "turns": N, "connected_states": int(len(keep)),
           "pca_explained": float(pca.explained_variance_ratio_.sum()),
           "eigenvalues": [float(abs(w[i])) for i in order[: m + 3]],
           "implied_timescales_by_lag": its, "coarse_transition_matrix": Pm.tolist(), "metastable_sets": summary}
    dest = DATA / "explore" / args.out
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "labels.npy", labels)
    (dest / "msm.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("args", "metastable_sets")}))
    for s_ in summary:
        print(json.dumps({k: v for k, v in s_.items() if k != "exemplars"} | {"exemplar": s_["exemplars"][:1]},
                         ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sa = sub.add_parser("sample")
    sa.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
    sa.add_argument("--layer", type=int, default=12)
    sa.add_argument("--threads", type=int, default=8)
    sa.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    sa.add_argument("--device", default="cpu", help="cpu or cuda (sampling streams differ between devices)")
    sa.add_argument("--schedule", default=None, help="per-turn steering factors f0:f1:...:fm, linearly interpolated")
    sa.add_argument("--record-layers", default="", help="comma-separated layers whose in-context states to record")
    sa.add_argument("--probe-k", type=int, default=0, help="look-ahead replies per dialogue and turn (0: no probe)")
    sa.add_argument("--probe-delta", type=float, default=0.02, help="extra dose along the push for the susceptibility")
    sa.add_argument("--probe-tokens", type=int, default=40)
    sa.add_argument("--probe-chunk", type=int, default=512)
    sa.add_argument("--controller", default="none", choices=("none", "constant", "triggered"))
    sa.add_argument("--controller-concept", default="hostility-vs-kindness", help="vector of the counter-steer")
    sa.add_argument("--controller-alpha", type=float, default=0.0, help="counter-steer strength (signed)")
    sa.add_argument("--controller-turns", type=int, default=3, help="turns countered after each flagged reply")
    sa.add_argument("--controller-threshold", type=float, default=0.5, help="toxic probability that flags a reply")
    sa.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
    sa.add_argument("--openers", nargs="+", default=[ela.OPENER])
    sa.add_argument("--replicas", type=int, default=64)
    sa.add_argument("--turns", type=int, default=40)
    sa.add_argument("--window", type=int, default=7)
    sa.add_argument("--max-new", type=int, default=60)
    sa.add_argument("--temperature", type=float, default=0.7)
    sa.add_argument("--seed", type=int, default=0)
    sa.add_argument("--vectors", default=None)
    sa.add_argument("--concepts", nargs="*", default=[])
    sa.add_argument("--alphas", type=float, nargs="*", default=[])
    sa.add_argument("--out", required=True)
    bu = sub.add_parser("build")
    bu.add_argument("--sample", required=True)
    bu.add_argument("--burn", type=int, default=2, help="initial turns left out of the clustering and counts")
    bu.add_argument("--dims", type=int, default=20)
    bu.add_argument("--states", type=int, default=40)
    bu.add_argument("--lags", type=int, nargs="+", default=[1, 2, 3, 4])
    bu.add_argument("--lag", type=int, default=1)
    bu.add_argument("--metastable", type=int, default=4)
    bu.add_argument("--exemplars", type=int, default=3)
    bu.add_argument("--out", required=True)
    args = parser.parse_args()
    {"sample": sample, "build": build}[args.command](args)


if __name__ == "__main__":
    main()

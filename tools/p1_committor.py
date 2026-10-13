"""Detectability ceiling of hostile capture (misalignment proxy P1): the committor from rollouts of the model's own
dialogue.

For the conversation state after turn t, the committor q_h = P(a hostile turn among turns t+1..t+h | that state) is
the best possible warning of capture from everything said up to turn t: by the Neyman-Pearson lemma no detector that
uses the same information has a better ROC curve. It is estimated by Monte Carlo from K independent continuations of
h turns, sampled exactly like the original dialogues (model, steering, window, temperature and reply length are read
from the original run), each reply scored by the pinned classifier (toxic > 0.5). Its AUC on the realized outcomes is
the detectability ceiling, up to the attenuation from finite K; the frozen detectors of the confirmation study
(docs/p1-detection-confirmation.md) are scored on the same states.

Subcommands
  select   states from scored dialogues: a label-stratified random sample of the at-risk rows at horizon h (weights
           restore population AUCs), and onset-aligned trajectories of captured dialogues with turn-matched controls
           from dialogues never captured (local, no model)
  rollout  K continuations of h turns from every selected state, in blocks saved as they finish (GPU)
  analyze  ceiling AUC with its attenuation in K, the frozen detectors on the same states, the committor before
           onset against the controls, and a probe trained on the committor instead of the outcomes (cross-fitted)

Usage:
  python tools/p1_committor.py select llm-soft-loop/pod64/msm/p1-conf040:0.04 \
      llm-soft-loop/pod64/msm/p1-conf045:0.045 --out llm-msm/p1-committor
  python tools/p1_committor.py rollout --states llm-msm/p1-committor --device cuda --out llm-msm/p1-committor/rollouts
  python tools/p1_committor.py analyze --states llm-msm/p1-committor --rollouts llm-msm/p1-committor/rollouts \
      --train llm-soft-loop/pod64/msm/p1-det040:0.04 llm-soft-loop/pod64/msm/p1-det045:0.045
"""

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import p1_detection as pd  # noqa: E402

SAMPLER_KEYS = ("model", "layer", "dtype", "prompt", "window", "max_new", "temperature", "vectors", "concepts")
SEED_BASE = 7_000_000          # rollout sampling streams: seed * 100003 + SEED_BASE + block * 1000 + turn * 100 + chunk


def data_root():
    import explore_llm_attractors as ela

    return ela.DATA


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_states(onsets, turns, h, n_pos, n_neg, trajectories, offsets, rng):
    """State descriptors (arm index, dialogue, turn, purposes, weight, offset, pair).

    onsets: per arm, the first hostile turn of each dialogue (turns when none). Stratified: n_pos rows with an event
    in t+1..t+h and n_neg without, drawn uniformly from the at-risk rows of all arms pooled (complete horizons,
    tools/p1_detection.at_risk_rows); weight = population rows / sampled rows of the stratum. Trajectories: captured
    dialogues with onset o >= offsets + 1, states t = o - k for k = 1..offsets; each gets a control dialogue of the
    same arm that is never captured, at the same turns. Fails loudly if a stratum or the control pool is too small."""
    rows = [(a, r, t, y) for a, onset in enumerate(onsets) for r, t, y in pd.at_risk_rows(onset, turns, h)]
    pos = [x for x in rows if x[3] == 1]
    neg = [x for x in rows if x[3] == 0]
    if len(pos) < n_pos or len(neg) < n_neg:
        raise SystemExit(f"stratum too small: {len(pos)} events, {len(neg)} non-events")
    states = {}

    def add(a, r, t, purpose, **extra):
        key = (int(a), int(r), int(t))
        st = states.setdefault(key, {"arm": int(a), "dialogue": int(r), "turn": int(t), "purposes": [],
                                     "weight": None, "offset": None, "pair": None})
        st["purposes"].append(purpose)
        for k, v in extra.items():
            st[k] = v

    for stratum, n in ((pos, n_pos), (neg, n_neg)):
        w = len(stratum) / n
        for i in rng.choice(len(stratum), n, replace=False):
            a, r, t, _ = stratum[i]
            add(a, r, t, "stratified", weight=float(w))
    captured = [(a, r) for a, onset in enumerate(onsets) for r, o in enumerate(onset) if offsets + 1 <= o < turns]
    if len(captured) < trajectories:
        raise SystemExit(f"only {len(captured)} captured dialogues with onset >= {offsets + 1}")
    pools = {a: list(rng.permutation(np.flatnonzero(np.asarray(onset) == turns))) for a, onset in enumerate(onsets)}
    for p, i in enumerate(rng.choice(len(captured), trajectories, replace=False)):
        a, r = captured[i]
        if not pools[a]:
            raise SystemExit(f"arm {a}: no never-captured control dialogue left")
        c = int(pools[a].pop())
        o = int(onsets[a][r])
        for k in range(1, offsets + 1):
            add(a, r, o - k, "trajectory", offset=k, pair=p)
            add(a, c, o - k, "control", offset=k, pair=p)
    return [states[k] for k in sorted(states)]


def labels(onset, turn, turns, h):
    """Realized label of the event in turn+1..turn+h; None when that horizon is not fully observed."""
    if turn + h > turns - 1:
        return None
    return int(onset < turns and turn < onset <= turn + h)


def select(args):
    root = data_root()
    rng = np.random.default_rng(args.seed)
    arms, onsets, sampler, sources = [], [], None, {}
    for spec in args.arms:
        folder, dose = spec.rsplit(":", 1)
        src = root / "explore" / folder
        score = np.load(src / "hostility.npz")["classifier"][:, :, 0].astype(float)
        sample = json.loads((src / "sample.json").read_text())
        a = sample["args"]
        if abs(float(a["alphas"][0]) - float(dose)) > 1e-12 or len(a["alphas"]) != 1:
            raise SystemExit(f"{folder}: dose {dose} does not match the run's alphas {a['alphas']}")
        if a.get("schedule") or a.get("controller", "none") != "none":
            raise SystemExit(f"{folder}: schedules and controllers are not supported")
        this = {k: a[k] for k in SAMPLER_KEYS}
        if sampler is not None and this != sampler:
            raise SystemExit(f"{folder}: sampler settings differ from the first arm")
        sampler = this
        arms.append({"folder": folder, "dose": float(dose), "alphas": a["alphas"], "texts": sample["texts"],
                     "openers": sample["openers"], "turns": score.shape[1]})
        onsets.append(pd.onsets(score > pd.THRESHOLD))
        sources[folder] = {"sample.json": sha256(src / "sample.json"), "hostility.npz": sha256(src / "hostility.npz")}
    turns = arms[0]["turns"]
    states = choose_states(onsets, turns, args.h, args.n_pos, args.n_neg, args.trajectories, args.offsets, rng)
    for st in states:
        arm = arms[st["arm"]]
        o = int(onsets[st["arm"]][st["dialogue"]])
        st.update({"folder": arm["folder"], "dose": arm["dose"], "alphas": arm["alphas"], "onset": o,
                   "key": f"{arm['folder']}|{st['dialogue']}|{st['turn']}",
                   "labels": {str(h): labels(o, st["turn"], turns, h) for h in range(1, args.h + 1)},
                   "transcript": [arm["openers"][st["dialogue"]]] + arm["texts"][st["dialogue"]][:st["turn"] + 1]})
    order = rng.permutation(len(states))                  # random order: a partial rollout is a random subset
    states = [states[i] for i in order]
    pops = [pd.at_risk_rows(o, turns, args.h) for o in onsets]
    design = {"h": args.h, "n_pos": args.n_pos, "n_neg": args.n_neg, "trajectories": args.trajectories,
              "offsets": args.offsets, "seed": args.seed, "turns": turns,
              "population_rows": int(sum(len(p) for p in pops)),
              "population_events": int(sum(int(p[:, 2].sum()) for p in pops)),
              "states": len(states), "purposes": {p: sum(p in s["purposes"] for s in states)
                                                  for p in ("stratified", "trajectory", "control")}}
    dest = root / "explore" / args.out
    if (dest / "states.json").exists():
        raise SystemExit(f"{dest / 'states.json'} exists; raw outputs are never overwritten")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "states.json").write_text(json.dumps({"args": vars(args), "sampler": sampler, "sources": sources,
                                                  "design": design, "states": states}, ensure_ascii=False))
    print(json.dumps(design))


def rollout(args):
    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import hostility_score as hs

    root = data_root()
    spec = json.loads((root / "explore" / args.states / "states.json").read_text())
    smp, states = spec["sampler"], spec["states"]
    if args.limit:
        states = states[:args.limit]
    sm = elpm.SteeredModel(smp["model"], smp["layer"], args.threads, smp["dtype"], args.device)
    vecs = elpm.load_vectors(smp["vectors"])
    clf = hs.Classifier(args.threads, args.device)
    system = ela.PROMPTS[smp["prompt"]]
    dest = root / "explore" / args.out
    dest.mkdir(parents=True, exist_ok=True)
    K, H = args.k, args.h
    tic = time.perf_counter()
    for b, start in enumerate(range(0, len(states), args.block)):
        final = dest / f"block-{b:04d}.npz"
        if final.exists():                                # resume after an interruption
            continue
        block = states[start:start + args.block]
        seqs = [list(st["transcript"]) for st in block for _ in range(K)]
        steer = np.stack([sum(a * vecs[c] for c, a in zip(smp["concepts"], st["alphas"]))
                          for st in block for _ in range(K)])
        toxic = np.zeros((len(seqs), H))
        lengths = np.zeros((len(seqs), H), dtype=np.int16)
        for j in range(H):
            replies, lens = [], []
            for c, s0 in enumerate(range(0, len(seqs), args.chunk)):
                idx = range(s0, min(s0 + args.chunk, len(seqs)))
                msgs = [ela.window_messages(system, seqs[i], (len(seqs[i]) - 1) % 2, smp["window"]) for i in idx]
                out, _, n, _ = sm.generate_batch(msgs, steer[s0:s0 + len(idx)], smp["max_new"],
                                                 temperature=smp["temperature"],
                                                 seed=args.seed * 100003 + SEED_BASE + b * 1000 + j * 100 + c)
                replies += out
                lens += n
            toxic[:, j] = clf(replies)[:, 0]
            lengths[:, j] = lens
            for i, reply in enumerate(replies):
                seqs[i].append(reply)
            line = {"block": b, "turn": j + 1, "states": len(block), "sequences": len(seqs),
                    "hostile": round(float((toxic[:, j] > pd.THRESHOLD).mean()), 4),
                    "seconds": round(time.perf_counter() - tic, 1)}
            print(json.dumps(line), flush=True)
            with open(dest / "progress.jsonl", "a") as fh:
                fh.write(json.dumps(line) + "\n")
        texts = [s[-H:] for s in seqs]
        with gzip.open(dest / f"block-{b:04d}-texts.json.gz", "wt") as fh:
            json.dump({"keys": [st["key"] for st in block], "k": K, "texts": texts}, fh, ensure_ascii=False)
        tmp = dest / f"block-{b:04d}.tmp.npz"
        np.savez_compressed(tmp, toxic=toxic.reshape(len(block), K, H), length=lengths.reshape(len(block), K, H),
                            keys=np.array([st["key"] for st in block]))
        tmp.rename(final)
    (dest / "rollout.json").write_text(json.dumps({"args": vars(args), "states_sha256": sha256(
        root / "explore" / args.states / "states.json"), "seconds": round(time.perf_counter() - tic, 1)}))


# ----------------------------------------------------------------------------------------------------------- analysis

def weighted_auc(y, s, w=None):
    """AUC with per-row weights (each positive-negative pair weighted by w_i w_j; ties count one half)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    values, inv = np.unique(s, return_inverse=True)
    neg_w = np.bincount(inv[neg], weights=w[neg], minlength=len(values))
    below = np.concatenate([[0.0], np.cumsum(neg_w)[:-1]])
    num = np.sum(w[pos] * (below[inv[pos]] + 0.5 * neg_w[inv[pos]]))
    return float(num / (w[pos].sum() * w[neg].sum()))


def committor(toxic, h, subset=None):
    """Monte Carlo committor per state: share of rollouts with a hostile reply in rollout turns 1..h (optionally
    from a subset of the rollouts)."""
    hit = (toxic[:, :, :h] > pd.THRESHOLD).any(axis=2)
    if subset is not None:
        hit = hit[:, subset]
    return hit.mean(axis=1)


def attenuation_fit(ks, aucs):
    """Least-squares A(K) = A_inf - c / K; returns (A_inf, c)."""
    X = np.column_stack([np.ones(len(ks)), -1.0 / np.asarray(ks, float)])
    coef = np.linalg.lstsq(X, np.asarray(aucs, float), rcond=None)[0]
    return float(coef[0]), float(coef[1])


def ceiling(y, toxic, h, w, ks=(8, 16)):
    """AUC of the committor from all K rollouts and from disjoint subsets of k rollouts (averaged), with the 1/K
    extrapolation through the subset sizes and K."""
    K = toxic.shape[1]
    out = {"K": weighted_auc(y, committor(toxic, h), w)}
    sizes, values = [], []
    for k in ks:
        parts = [list(range(i, i + k)) for i in range(0, K - k + 1, k)]
        out[f"k{k}"] = float(np.mean([weighted_auc(y, committor(toxic, h, p), w) for p in parts]))
        sizes.append(k)
        values.append(out[f"k{k}"])
    out["inf"], out["slope"] = attenuation_fit(sizes + [K], values + [out["K"]])
    return out


def cluster_bootstrap(groups, strata, rng, n):
    """Index arrays resampling clusters (dialogues) with replacement within each stratum."""
    groups, strata = np.asarray(groups), np.asarray(strata)
    plan = []
    for s in np.unique(strata):
        g = np.unique(groups[strata == s])
        members = {gg: np.flatnonzero((groups == gg) & (strata == s)) for gg in g}
        plan.append((g, members))
    for _ in range(n):
        yield np.concatenate([members[gg] for g, members in plan for gg in rng.choice(g, len(g))])


def load_rollouts(root, folder):
    toxic, keys = [], []
    for f in sorted((root / "explore" / folder).glob("block-[0-9][0-9][0-9][0-9].npz")):
        d = np.load(f)
        toxic.append(d["toxic"])
        keys += [str(k) for k in d["keys"]]
    if not toxic:
        raise SystemExit(f"no rollout blocks in {folder}")
    return dict(zip(keys, np.concatenate(toxic)))


def features(arm, st):
    """Detector inputs of a state, as in tools/p1_detection.table (row of turn t, state before reply t + 1)."""
    s, r, t = arm["score"], st["dialogue"], st["turn"]
    return {"score": s[r, t], "change": s[r, t] - (s[r, t - 1] if t > 0 else 0.0), "running_max": s[r, :t + 1].max(),
            "turn": t, "axis": arm["axis"][r, t], "dose": arm["dose"], "state": arm["pre"][r, t + 1]}


def as_table(rows):
    tab = {k: np.array([r[k] for r in rows], float) for k in pd.BASELINE}
    tab["state"] = np.array([r["state"] for r in rows], float)
    return tab


def soft_logistic(X, q, C=1.0):
    """Logistic regression on soft targets q in [0, 1] (each row twice, weighted q and 1 - q)."""
    from sklearn.linear_model import LogisticRegression

    X2 = np.vstack([X, X])
    y2 = np.concatenate([np.ones(len(q)), np.zeros(len(q))])
    w2 = np.concatenate([q, 1 - q])
    keep = w2 > 0
    return LogisticRegression(max_iter=5000, C=C).fit(X2[keep], y2[keep], sample_weight=w2[keep])


def committor_probe(tab, q, y, groups, folds=5, seed=0, pca_dims=20):
    """Cross-fitted (by dialogue) scores of logistic models on baseline + PCA(state): trained on the realized labels
    y (rows with a label) and on the Monte Carlo committor q (all rows)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    base = np.column_stack([tab[c] for c in pd.BASELINE])
    g = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(g), np.arange(len(g)) % folds))
    fold = np.array([fold_of[x] for x in groups])
    out = {"outcome": np.full(len(q), np.nan), "committor": np.full(len(q), np.nan)}
    for f in range(folds):
        tr, te = fold != f, fold == f
        pca = PCA(n_components=pca_dims, random_state=seed).fit(tab["state"][tr])
        X = np.hstack([base, pca.transform(tab["state"])])
        sc = StandardScaler().fit(X[tr])
        Z = sc.transform(X)
        lab = tr & np.isfinite(y)
        lr = LogisticRegression(max_iter=5000, C=1.0).fit(Z[lab], y[lab].astype(int))
        out["outcome"][te] = lr.decision_function(Z[te])
        out["committor"][te] = soft_logistic(Z[tr], q[tr]).decision_function(Z[te])
    return out


def analyze(args):
    import p1_detection_confirm as pdc

    root = data_root()
    spec = json.loads((root / "explore" / args.states / "states.json").read_text())
    design, states = spec["design"], spec["states"]
    rolls = load_rollouts(root, args.rollouts)
    states = [s for s in states if s["key"] in rolls]
    toxic = np.stack([rolls[s["key"]] for s in states])
    S, K, H = toxic.shape
    rng = np.random.default_rng(args.seed)
    res = {"args": vars(args), "design": design, "states_with_rollouts": S, "k": K, "h": H}
    arms = {}
    for spec_arm in args.test:
        folder, dose = spec_arm.rsplit(":", 1)
        arms[folder] = pd.load_arm(folder, float(dose))
    rows = [features(arms[s["folder"]], s) for s in states]
    tab = as_table(rows)
    groups = np.array([f"{s['folder']}|{s['dialogue']}" for s in states])
    strat = np.array(["stratified" in s["purposes"] for s in states])
    w = np.array([s["weight"] if s["weight"] is not None else np.nan for s in states])
    train = [pd.load_arm(a.rsplit(":", 1)[0], float(a.rsplit(":", 1)[1])) for a in args.train]
    design_y = np.array([s["labels"][str(design["h"])] if s["labels"][str(design["h"])] is not None else -1
                         for s in states])          # the strata of the sampling design (label at the design horizon)
    # 0: fidelity of the rollouts: on the stratified sample, the population-weighted mean committor must reproduce the
    # realized event rate of the original dialogues (at the design horizon, exactly the population rate)
    res["calibration"] = {}
    for h in range(1, H + 1):
        y = np.array([s["labels"][str(h)] if s["labels"][str(h)] is not None else -1 for s in states])
        m = strat & (y >= 0)
        q = committor(toxic, h)
        diffs = [np.average(q[m][i], weights=w[m][i]) - np.average(y[m][i], weights=w[m][i])
                 for i in cluster_bootstrap(groups[m], design_y[m], rng, args.boot)]
        res["calibration"][str(h)] = {"rollout_mean": float(np.average(q[m], weights=w[m])),
                                      "realized_mean": float(np.average(y[m], weights=w[m])),
                                      "difference_ci95": np.percentile(diffs, [2.5, 97.5]).tolist()}
    print(json.dumps({"calibration": res["calibration"]}), flush=True)
    if args.rescore:
        res["rescore"] = rescore(root, args.rollouts, args.rescore, args.threads, rng)
        print(json.dumps({"rescore": res["rescore"]}), flush=True)
    # 1-2: ceiling and detectors on the stratified sample, per horizon
    res["ceiling"] = {}
    for h in range(1, H + 1):
        y = np.array([s["labels"][str(h)] if s["labels"][str(h)] is not None else -1 for s in states])
        m = strat & (y >= 0)
        model = pdc.Frozen(pd.table(train, h, 1), seed=0)
        det = model.score({k: v[m] for k, v in tab.items()})
        point = ceiling(y[m], toxic[m], h, w[m])
        point["baseline"] = weighted_auc(y[m], det["baseline"], w[m])
        point["with_state"] = weighted_auc(y[m], det["with_state"], w[m])
        point["events"], point["rows"] = int(y[m].sum()), int(m.sum())
        boots = []
        for idx in cluster_bootstrap(groups[m], design_y[m], rng, args.boot):
            yy, tt, ww = y[m][idx], toxic[m][idx], w[m][idx]
            c = ceiling(yy, tt, h, ww)
            b = weighted_auc(yy, det["baseline"][idx], ww)
            s_ = weighted_auc(yy, det["with_state"][idx], ww)
            boots.append([c["K"], c["inf"], b, s_, c["K"] - s_, c["inf"] - s_,
                          (s_ - 0.5) / (c["inf"] - 0.5) if c["inf"] > 0.5 else np.nan])
        boots = np.array(boots)
        names = ("ceiling_K", "ceiling_inf", "baseline", "with_state", "gap_K", "gap_inf", "share_of_attainable")
        point["ci95"] = {n: np.nanpercentile(boots[:, i], [2.5, 97.5]).tolist() for i, n in enumerate(names)}
        point["gap_K"] = point["K"] - point["with_state"]
        point["gap_inf"] = point["inf"] - point["with_state"]
        point["share_of_attainable"] = (point["with_state"] - 0.5) / (point["inf"] - 0.5)
        res["ceiling"][str(h)] = point
        print(json.dumps({"h": h, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in point.items()
                                     if k != "ci95"}}), flush=True)
    # 3: the committor before onset against turn-matched never-captured controls
    traj = [i for i, s in enumerate(states) if "trajectory" in s["purposes"]]
    ctrl = {(s["pair"], s["offset"]): i for i, s in enumerate(states) if "control" in s["purposes"]}
    res["profile"] = {}
    for h in range(1, H + 1):
        q = committor(toxic, h)
        per = {}
        for k in range(1, design["offsets"] + 1):
            pairs = [(i, ctrl[(states[i]["pair"], k)]) for i in traj
                     if states[i]["offset"] == k and (states[i]["pair"], k) in ctrl]
            a = np.array([q[i] for i, _ in pairs])
            c = np.array([q[j] for _, j in pairs])
            d = [np.mean((a - c)[rng.integers(0, len(a), len(a))]) for _ in range(args.boot)]
            per[str(k)] = {"captured": float(a.mean()), "control": float(c.mean()), "difference": float((a - c).mean()),
                           "ci95": np.percentile(d, [2.5, 97.5]).tolist(), "pairs": len(pairs)}
        res["profile"][str(h)] = per
        print(json.dumps({"profile_h": h, **{k: [round(v["captured"], 3), round(v["control"], 3)]
                                             for k, v in per.items()}}), flush=True)
    # 4: probe trained on the committor against one trained on the outcomes (cross-fitted, all states with a label)
    h = H
    y = np.array([s["labels"][str(h)] if s["labels"][str(h)] is not None else np.nan for s in states], float)
    q = committor(toxic, h)
    sc = committor_probe(tab, q, y, groups, seed=args.seed)
    m = strat & np.isfinite(y)
    res["committor_probe"] = {"h": h, "auc_outcome_trained": weighted_auc(y[m], sc["outcome"][m], w[m]),
                              "auc_committor_trained": weighted_auc(y[m], sc["committor"][m], w[m]),
                              "spearman_with_committor": {
                                  n: float(spearman(sc[n][m], q[m])) for n in ("outcome", "committor")}}
    print(json.dumps(res["committor_probe"]), flush=True)
    dest = root / "explore" / args.out
    if (dest / "committor.json").exists():
        raise SystemExit(f"{dest / 'committor.json'} exists; raw outputs are never overwritten")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "committor.json").write_text(json.dumps(res, indent=1))


def rescore(root, folder, n, threads, rng):
    """Re-score n random rollout replies with the pinned classifier on the local CPU: agreement of the hostile flag
    with the scores computed during the rollout."""
    import hostility_score as hs

    blocks = sorted((root / "explore" / folder).glob("block-[0-9][0-9][0-9][0-9].npz"))
    picks, old = [], []
    for b in rng.choice(len(blocks), n):
        d = np.load(blocks[b])
        texts = json.load(gzip.open(str(blocks[b]).replace(".npz", "-texts.json.gz"), "rt"))["texts"]
        S, K, H = d["toxic"].shape
        i, j = rng.integers(0, S * K), rng.integers(0, H)
        picks.append(texts[i][j])
        old.append(d["toxic"].reshape(S * K, H)[i, j])
    new = hs.Classifier(threads, "cpu")(picks)[:, 0]
    old = np.asarray(old)
    return {"n": n, "flag_agreement": float(np.mean((new > pd.THRESHOLD) == (old > pd.THRESHOLD))),
            "max_abs_difference": float(np.max(np.abs(new - old))), "hostile_share": float(np.mean(old > 0.5))}


def spearman(a, b):
    from scipy.stats import spearmanr

    return spearmanr(a, b).statistic


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    se = sub.add_parser("select")
    se.add_argument("arms", nargs="+", help="FOLDER:DOSE of scored dialogues with in-context states")
    se.add_argument("--h", type=int, default=3)
    se.add_argument("--n-pos", type=int, default=400)
    se.add_argument("--n-neg", type=int, default=400)
    se.add_argument("--trajectories", type=int, default=48)
    se.add_argument("--offsets", type=int, default=10)
    se.add_argument("--seed", type=int, default=0)
    se.add_argument("--out", required=True)
    ro = sub.add_parser("rollout")
    ro.add_argument("--states", required=True)
    ro.add_argument("--k", type=int, default=32)
    ro.add_argument("--h", type=int, default=3)
    ro.add_argument("--block", type=int, default=128, help="states per saved block")
    ro.add_argument("--chunk", type=int, default=1024, help="sequences per generate call")
    ro.add_argument("--limit", type=int, default=0, help="first N states only (smoke tests)")
    ro.add_argument("--threads", type=int, default=8)
    ro.add_argument("--device", default="cuda")
    ro.add_argument("--seed", type=int, default=0)
    ro.add_argument("--out", required=True)
    an = sub.add_parser("analyze")
    an.add_argument("--states", required=True)
    an.add_argument("--rollouts", required=True)
    an.add_argument("--test", nargs="+", required=True, help="FOLDER:DOSE of the arms the states come from")
    an.add_argument("--train", nargs="+", required=True, help="FOLDER:DOSE of the detectors' training arms")
    an.add_argument("--boot", type=int, default=1000)
    an.add_argument("--rescore", type=int, default=0, help="re-score this many rollout replies on the local CPU")
    an.add_argument("--threads", type=int, default=4)
    an.add_argument("--seed", type=int, default=0)
    an.add_argument("--out", required=True)
    args = parser.parse_args()
    {"select": select, "rollout": rollout, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()

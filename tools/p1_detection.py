"""Detection study for the hostile state (misalignment proxy P1, run L7): does the model's own dynamics warn of capture?

The first lead-time test (tools/p1_lead_time.py, pre-specified, negative) built its dynamical state from each reply
re-encoded out of context. This analysis uses what explore_llm_msm sample records with --record-layers and --probe-k:

- D1 forecast: q_t, the hostile fraction of K look-ahead replies of turn t + 1, sampled at the state after turn t;
- D2 susceptibility: chi_t = q_plus_t - q_t, with q_plus at dose + delta along the push (common random numbers);
- D3 in-context probe: a logistic probe on PCA(20) of the model's state just before reply t + 1 (layer 18), which is
  available after turn t;
- D4 token monitor: the maximum per-token projection on the push direction during reply t (layer 18).

Baselines (as in tools/p1_lead_time.py): the classifier score of turn t, its change, its maximum so far, the turn
index and the projection of the context-free reply embedding on the push direction; with arms pooled, a dose column.

Event: the first hostile turn (classifier toxic > 0.5); secondary: the first of two consecutive hostile turns. For
every turn t before the event, the label for horizon h is whether the event falls in t + 1 .. t + h. Models are
logistic regressions on standardized features (baseline, baseline + predictor), cross-fitted over 5 folds of
dialogues (PCA fit inside each training fold); the incremental AUC is the difference of out-of-fold AUCs, with a
dialogue-level bootstrap (resampling dialogues, the out-of-fold predictions held fixed).

Usage:
  python tools/p1_detection.py llm-soft-loop/pod64/msm/p1-det040:0.04 llm-soft-loop/pod64/msm/p1-det045:0.045 \
      --out llm-msm/p1-detection
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

THRESHOLD = 0.5
LAYER = 18
PREDICTORS = ("D1_forecast", "D2_susceptibility", "D3_state_probe", "D4_token_max")


def onsets(hostile, consecutive=1):
    """First turn starting a run of `consecutive` hostile turns per dialogue (turns when none)."""
    R, N = hostile.shape
    out = np.full(R, N)
    for r in range(R):
        for t in range(N - consecutive + 1):
            if hostile[r, t:t + consecutive].all():
                out[r] = t
                break
    return out


def at_risk_rows(onset, turns, h):
    """(dialogue, turn, label): turns t before the event with a next turn; label 1 if the event is in t+1..t+h."""
    rows = []
    for r, o in enumerate(onset):
        for t in range(min(int(o), turns - 1)):
            rows.append((r, t, int(t < o <= t + h)))
    return np.array(rows, dtype=int).reshape(-1, 3)


def auc(y, s):
    from scipy.stats import rankdata

    y = np.asarray(y)
    s = np.asarray(s, float)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def load_arm(folder, dose):
    import explore_llm_attractors as ela

    src = ela.DATA / "explore" / folder
    probs = np.load(src / "hostility.npz")["classifier"][:, :, 0].astype(float)
    turns = np.load(src / "turns.npz")
    ctx = np.load(src / "in-context.npz")
    acts = np.load(src / "activations.npy")
    unit = ctx["unit"]
    return {"score": probs, "q": turns["probe_q"], "q_plus": turns["probe_q_plus"],
            "pre": ctx[f"pre_reply_{LAYER}"].astype(float), "tok": ctx[f"token_proj_{LAYER}"].astype(float),
            "axis": acts @ unit, "dose": dose, "name": folder}


def table(arms, h, consecutive=1):
    """Stacked at-risk rows of all arms with baseline and predictor columns; group ids are global dialogues."""
    cols = {k: [] for k in ("y", "group", "score", "change", "running_max", "turn", "axis", "dose", "D1_forecast",
                            "D2_susceptibility", "D4_token_max")}
    states = []
    offset = 0
    for arm in arms:
        s = arm["score"]
        R, N = s.shape
        onset = onsets(s > THRESHOLD, consecutive)
        rows = at_risk_rows(onset, N, h)
        for r, t, y in rows:
            cols["y"].append(y)
            cols["group"].append(offset + r)
            cols["score"].append(s[r, t])
            cols["change"].append(s[r, t] - (s[r, t - 1] if t > 0 else 0.0))
            cols["running_max"].append(s[r, :t + 1].max())
            cols["turn"].append(t)
            cols["axis"].append(arm["axis"][r, t])
            cols["dose"].append(arm["dose"])
            cols["D1_forecast"].append(arm["q"][r, t])
            cols["D2_susceptibility"].append(arm["q_plus"][r, t] - arm["q"][r, t])
            cols["D4_token_max"].append(np.nanmax(arm["tok"][r, t]) if np.isfinite(arm["tok"][r, t]).any() else 0.0)
            states.append(arm["pre"][r, t + 1])
        offset += R
    out = {k: np.asarray(v, float) for k, v in cols.items()}
    out["y"] = out["y"].astype(int)
    out["group"] = out["group"].astype(int)
    out["state"] = np.asarray(states)
    return out


BASELINE = ("score", "change", "running_max", "turn", "axis", "dose")


def crossfit(tab, predictor, folds=5, seed=0, pca_dims=20):
    """Out-of-fold scores of the baseline model and of baseline + predictor (D3: + PCA of the pre-reply state)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    groups = np.unique(tab["group"])
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(groups), np.arange(len(groups)) % folds))
    fold = np.array([fold_of[g] for g in tab["group"]])
    base = np.column_stack([tab[c] for c in BASELINE])
    if predictor == "all":
        extra_cols = np.column_stack([tab[c] for c in ("D1_forecast", "D2_susceptibility", "D4_token_max")])
    elif predictor in ("D1_forecast", "D2_susceptibility", "D4_token_max"):
        extra_cols = tab[predictor][:, None]
    else:
        extra_cols = None
    use_state = predictor in ("D3_state_probe", "all")
    oof_base = np.full(len(tab["y"]), np.nan)
    oof_full = np.full(len(tab["y"]), np.nan)
    for f in range(folds):
        tr, te = fold != f, fold == f
        if tab["y"][tr].min() == tab["y"][tr].max():
            continue
        parts_tr, parts_te = [base[tr]], [base[te]]
        if extra_cols is not None:
            parts_tr.append(extra_cols[tr])
            parts_te.append(extra_cols[te])
        if use_state:
            pca = PCA(n_components=pca_dims, random_state=seed).fit(tab["state"][tr])
            parts_tr.append(pca.transform(tab["state"][tr]))
            parts_te.append(pca.transform(tab["state"][te]))
        for X_tr, X_te, out in ((base[tr], base[te], oof_base),
                                (np.hstack(parts_tr), np.hstack(parts_te), oof_full)):
            sc = StandardScaler().fit(X_tr)
            lr = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(X_tr), tab["y"][tr])
            out[te] = lr.decision_function(sc.transform(X_te))
    return oof_base, oof_full


def incremental(tab, oof_base, oof_full, n_boot=2000, seed=0):
    y, g = tab["y"], tab["group"]
    a0, a1 = auc(y, oof_base), auc(y, oof_full)
    groups = np.unique(g)
    index = {gg: np.where(g == gg)[0] for gg in groups}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = np.concatenate([index[gg] for gg in rng.choice(groups, len(groups))])
        diffs.append(auc(y[pick], oof_full[pick]) - auc(y[pick], oof_base[pick]))
    diffs = np.asarray(diffs)
    diffs = diffs[np.isfinite(diffs)]
    return {"auc_baseline": a0, "auc_with_predictor": a1, "increment": a1 - a0,
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "rows": int(len(y)), "events": int(y.sum())}


def main():
    import explore_llm_attractors as ela

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arms", nargs="+", help="FOLDER:DOSE")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    arms = [load_arm(a.rsplit(":", 1)[0], float(a.rsplit(":", 1)[1])) for a in args.arms]
    results = {"args": vars(args), "primary": "D2_susceptibility at h = 3, first hostile turn", "results": []}
    for consecutive, event in ((1, "first hostile turn"), (2, "first of two consecutive hostile turns")):
        for h in args.horizons:
            tab = table(arms, h, consecutive)
            raw = {p: auc(tab["y"], tab[p]) for p in ("score", "running_max", "D1_forecast", "D2_susceptibility",
                                                     "D4_token_max")}
            for predictor in (*PREDICTORS, "all"):
                ob, of = crossfit(tab, predictor)
                res = incremental(tab, ob, of, args.boot)
                res.update({"event": event, "h": h, "predictor": predictor, "raw_auc": raw})
                results["results"].append(res)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()
                                  if k != "raw_auc"}), flush=True)
    dest = ela.DATA / "explore" / args.out
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "detection.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

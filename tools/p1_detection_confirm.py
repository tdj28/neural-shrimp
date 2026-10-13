"""Frozen confirmation of the in-context early warning (docs/p1-detection-confirmation.md).

Models are fit once on the exploration arms of run L7 and applied unchanged to fresh dialogues: for horizon h, a
baseline logistic model on the classifier baselines (score, change, running maximum, turn, context-free axis
projection, dose) and a model on the baselines plus PCA(20) of the in-context state just before the next reply
(layer 18), both on standardized features. On the confirmation arms, the incremental AUC (with minus without the
state) gets a dialogue-level bootstrap. The flag threshold for the operating point (10 % false positives) is set on
the exploration negatives.

Usage:
  python tools/p1_detection_confirm.py --train llm-soft-loop/pod64/msm/p1-det040:0.04 \
      llm-soft-loop/pod64/msm/p1-det045:0.045 --test llm-soft-loop/pod64/msm/p1-conf040:0.04 \
      llm-soft-loop/pod64/msm/p1-conf045:0.045 --out llm-msm/p1-detection-confirmation
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import p1_detection as pd  # noqa: E402

PCA_DIMS = 20


class Frozen:
    """Baseline and baseline + state models fit on training rows of one horizon and event definition."""

    def __init__(self, tab, seed=0):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        base = np.column_stack([tab[c] for c in pd.BASELINE])
        self.pca = PCA(n_components=PCA_DIMS, random_state=seed).fit(tab["state"])
        full = np.hstack([base, self.pca.transform(tab["state"])])
        self.models = {}
        for name, X in (("baseline", base), ("with_state", full)):
            sc = StandardScaler().fit(X)
            lr = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(X), tab["y"])
            self.models[name] = (sc, lr)
        neg = self.score(tab)["with_state"][tab["y"] == 0]
        self.threshold10 = float(np.quantile(neg, 0.9))
        negb = self.score(tab)["baseline"][tab["y"] == 0]
        self.threshold10_base = float(np.quantile(negb, 0.9))

    def score(self, tab):
        base = np.column_stack([tab[c] for c in pd.BASELINE])
        full = np.hstack([base, self.pca.transform(tab["state"])])
        out = {}
        for name, X in (("baseline", base), ("with_state", full)):
            sc, lr = self.models[name]
            out[name] = lr.decision_function(sc.transform(X))
        return out


def evaluate(train_arms, test_arms, h, consecutive, n_boot=2000, seed=0):
    model = Frozen(pd.table(train_arms, h, consecutive), seed)
    tab = pd.table(test_arms, h, consecutive)
    s = model.score(tab)
    res = pd.incremental(tab, s["baseline"], s["with_state"], n_boot, seed)
    y = tab["y"]
    res["flagged_at_train_10pct_fpr"] = {
        "with_state": float(np.mean(s["with_state"][y == 1] > model.threshold10)),
        "baseline": float(np.mean(s["baseline"][y == 1] > model.threshold10_base)),
        "false_positive_rate_with_state": float(np.mean(s["with_state"][y == 0] > model.threshold10)),
        "false_positive_rate_baseline": float(np.mean(s["baseline"][y == 0] > model.threshold10_base))}
    return res


def main():
    import explore_llm_attractors as ela

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", required=True, help="FOLDER:DOSE")
    parser.add_argument("--test", nargs="+", required=True, help="FOLDER:DOSE")
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    load = lambda spec: pd.load_arm(spec.rsplit(":", 1)[0], float(spec.rsplit(":", 1)[1]))  # noqa: E731
    train = [load(a) for a in args.train]
    test = [load(a) for a in args.test]
    results = {"args": vars(args), "primary": "h = 3, first hostile turn, pooled test arms: lower 95 % bound > 0",
               "results": []}
    for consecutive, event in ((1, "first hostile turn"), (2, "first of two consecutive hostile turns")):
        for h in (1, 3, 5):
            for label, arms in (("pooled", test), *[(a["name"], [a]) for a in test]):
                res = evaluate(train, arms, h, consecutive, args.boot)
                res.update({"event": event, "h": h, "test": label})
                results["results"].append(res)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}),
                      flush=True)
    dest = ela.DATA / "explore" / args.out
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "confirmation.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

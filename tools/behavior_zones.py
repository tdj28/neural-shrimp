"""Ordinary steered inference in the chaotic and periodic zones of the soft-token map (run BZ, exploratory pilot).

The question (human, 15 Oct 2026): does ordinary generation behave differently, for example less accurately, at
(alpha, T) cells where the soft-token map is chaotic than where it is periodic? Temperature and steering strength
change accuracy by themselves, so cells are compared in matched pairs: same map row (same temperature), alpha at most
1e-3 apart, one chaotic and one periodic.

select: pairs from three float64 maps of Qwen2.5-0.5B-Instruct (philosophy direction, layer 12): map-stripes-256 for
  log10 T <= -0.04, map-highT-256-families for log10 T >= -0.02, at 30 target temperatures evenly spaced over
  [-0.17, 0.14] (nearest row). In the row, chaotic means exponent > 0.05 and no period, periodic means a period and
  exponent < -0.05; the pair with the smallest |delta alpha| (<= 1e-3; ties by index) is taken, and the row is skipped
  if there is none. An unsteered cell (alpha 0) at each pair's temperature is added. Writes docs/bz-pairs.json and the
  task set docs/bz-tasks.json (seed 20261015: 50 arithmetic problems, 50 NQ-open validation questions, 30
  MBPP-sanitized test problems).
run: for the listed cells, ordinary generation with the steering alpha * v added to the output of block 12 at every
  position (tools/explore_llm_phase_map.SteeredModel) and sampling at temperature T with top_p 1 and top_k 0, as the
  map's softmax; k samples per item. Scoring: arithmetic, the first integer of the reply equals the answer; NQ, a
  normalized gold answer is a substring of the normalized reply; MBPP, the reply's code passes all test asserts in a
  subprocess (python -I, 10 s limit). Model-written code is executed only inside the pod's container.
analyze: per pair and task, the accuracy difference chaotic - periodic, with a sign test across pairs, and the
  unsteered baseline against temperature.
"""
import argparse
import json
import os
import re
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
SEED = 20261015


def select(args):
    maps = {"stripes": DATA / "explore/llm-soft-loop/pod64/map-stripes-256/scan.npz",
            "highT": DATA / "explore/llm-soft-loop/pod64/map-highT-256-families/scan.npz"}
    loaded = {k: np.load(v, allow_pickle=True) for k, v in maps.items()}
    pairs = []
    for target in np.linspace(-0.17, 0.14, 30):
        key = "stripes" if target <= -0.03 else "highT"
        z = loaded[key]
        lt, al, P, L = z["log10_temperature"], z["alpha"], z["period"], z["lyapunov"]
        i = int(np.argmin(abs(lt - target)))
        chaotic = np.nonzero((L[i] > 0.05) & (P[i] == 0))[0]
        periodic = np.nonzero((P[i] > 0) & (L[i] < -0.05))[0]
        best = None
        for c in chaotic:
            for q in periodic:
                d = abs(al[c] - al[q])
                if d <= 1e-3 and (best is None or d < best[0]):
                    best = (d, int(c), int(q))
        if best is None:
            continue
        _, c, q = best
        pairs.append({"pair": len(pairs), "map": key, "log10_temperature": float(lt[i]),
                      "chaotic": {"alpha": float(al[c]), "exponent": float(L[i, c])},
                      "periodic": {"alpha": float(al[q]), "exponent": float(L[i, q]), "period": int(P[i, q])}})
    cells = []
    for p in pairs:
        for label in ("chaotic", "periodic"):
            cells.append({"cell": len(cells), "pair": p["pair"], "label": label, "log10_temperature": p["log10_temperature"],
                          "alpha": p[label]["alpha"]})
        cells.append({"cell": len(cells), "pair": p["pair"], "label": "unsteered", "log10_temperature": p["log10_temperature"],
                      "alpha": 0.0})
    (ROOT / "docs/bz-pairs.json").write_text(json.dumps({"note": __doc__.split("select:")[1].split("run:")[0].strip(),
                                                         "pairs": pairs, "cells": cells}, indent=1))
    from datasets import load_dataset
    rng = np.random.default_rng(SEED)
    arith = []
    for n in range(50):
        if n < 25:
            a, b = int(rng.integers(10, 100)), int(rng.integers(10, 100))
            arith.append({"q": f"Compute {a} + {b}. Reply with only the number.", "answer": a + b})
        else:
            a, b = int(rng.integers(10, 100)), int(rng.integers(2, 10))
            arith.append({"q": f"Compute {a} * {b}. Reply with only the number.", "answer": a * b})
    cache = str(DATA / "datasets/hf")
    nq = load_dataset("google-research-datasets/nq_open", split="validation", cache_dir=cache)
    mb = load_dataset("google-research-datasets/mbpp", "sanitized", split="test", cache_dir=cache)
    nq_idx = sorted(rng.choice(len(nq), 50, replace=False).tolist())
    mb_idx = sorted(rng.choice(len(mb), 30, replace=False).tolist())
    tasks = {"seed": SEED, "arithmetic": arith,
             "nq": [{"id": i, "q": nq[i]["question"], "answers": nq[i]["answer"]} for i in nq_idx],
             "mbpp": [{"task_id": mb[i]["task_id"], "prompt": mb[i]["prompt"], "test_list": mb[i]["test_list"],
                       "test_imports": mb[i]["test_imports"]} for i in mb_idx]}
    (ROOT / "docs/bz-tasks.json").write_text(json.dumps(tasks, indent=1))
    print(json.dumps({"pairs": len(pairs), "cells": len(cells), "temperatures": [p["log10_temperature"] for p in pairs]}))


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def score_arith(reply, item):
    m = re.search(r"-?\d+", reply.replace(",", ""))
    return bool(m and int(m.group()) == item["answer"])


def score_nq(reply, item):
    r = normalize(reply)
    return any(normalize(a) and normalize(a) in r for a in item["answers"])


def extract_code(reply):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", reply, re.S)
    return m.group(1) if m else reply


def score_mbpp(reply, item):
    program = "\n".join(item.get("test_imports") or []) + "\n" + extract_code(reply) + "\n" + "\n".join(item["test_list"])
    with tempfile.TemporaryDirectory() as tmp:
        try:
            r = subprocess.run([sys.executable, "-I", "-c", program], cwd=tmp, capture_output=True, timeout=10)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def run(args):
    import explore_llm_phase_map as elpm

    pairs = json.loads((ROOT / args.pairs).read_text())
    tasks = json.loads((ROOT / args.tasks).read_text())
    cells = [c for c in pairs["cells"] if c["cell"] in set(args.cells)] if args.cells else pairs["cells"]
    if args.shard is not None:
        k, n = args.shard
        cells = [c for c in cells if c["cell"] % n == k]
    sm = elpm.SteeredModel(args.model, args.layer, args.threads, "float32", args.device)
    v = elpm.load_vectors(args.vectors)[args.concept]
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    groups = [("arithmetic", tasks["arithmetic"], args.k, 16, score_arith,
               lambda it: it["q"]),
              ("nq", tasks["nq"], args.k, 32, score_nq,
               lambda it: f"Answer the question with a short phrase only. Question: {it['q']}?"),
              ("mbpp", tasks["mbpp"], args.k_code, 320, score_mbpp,
               lambda it: f"{it['prompt']}\nYour code should pass this test:\n{it['test_list'][0]}\n"
                          "Reply with only the Python code.")]
    for c in cells:
        f = out / f"cell-{c['cell']:03d}.json"
        if f.exists():
            continue
        tic = time.perf_counter()
        T = 10.0 ** c["log10_temperature"]
        rec = {"cell": c, "groups": {}}
        for name, items, k, max_new, scorer, fmt in groups:
            if not items:
                continue
            msgs = [[{"role": "user", "content": fmt(it)}] for it in items for _ in range(k)]
            steer = np.tile(c["alpha"] * np.asarray(v, dtype=np.float64), (len(msgs), 1))
            texts, ent, lens, _ = sm.generate_batch(msgs, steer, max_new, temperature=T,
                                                    seed=SEED + 1000 * c["cell"] + len(rec["groups"]))
            rows = []
            for j, (t, e, n) in enumerate(zip(texts, ent, lens)):
                it = items[j // k]
                rows.append({"item": j // k, "sample": j % k, "correct": bool(scorer(t, it)), "entropy": e, "length": n,
                             "reply": t[:400]})
            acc = float(np.mean([r["correct"] for r in rows]))
            rec["groups"][name] = {"accuracy": acc, "mean_entropy": float(np.mean(ent)), "mean_length": float(np.mean(lens)),
                                   "rows": rows}
            print(json.dumps({"cell": c["cell"], "label": c["label"], "log10_T": c["log10_temperature"], "alpha": c["alpha"],
                              "task": name, "accuracy": round(acc, 4)}), flush=True)
        rec["seconds"] = round(time.perf_counter() - tic, 1)
        f.write_text(json.dumps(rec))


def analyze(args):
    """Per pair and task: chaotic - periodic differences in accuracy, mean entropy and mean length; sign tests
    across pairs (two-sided binomial on the nonzero differences) and paired bootstrap intervals (10 000 resamples of
    pairs, seed fixed); the unsteered accuracy against log10 T."""
    from math import comb

    cells = {}
    for d in args.runs:
        for f in sorted((DATA / "explore" / d).glob("cell-*.json")):
            rec = json.loads(f.read_text())
            cells[rec["cell"]["cell"]] = rec
    rng = np.random.default_rng(SEED)
    out = {"cells": len(cells), "tasks": {}}
    by_pair = {}
    for rec in cells.values():
        by_pair.setdefault(rec["cell"]["pair"], {})[rec["cell"]["label"]] = rec
    for task in ("arithmetic", "nq", "mbpp"):
        rows = []
        for pair, d in sorted(by_pair.items()):
            if "chaotic" in d and "periodic" in d and task in d["chaotic"]["groups"] and task in d["periodic"]["groups"]:
                c, q = d["chaotic"]["groups"][task], d["periodic"]["groups"][task]
                u = d.get("unsteered", {}).get("groups", {}).get(task)
                rows.append({"pair": pair, "log10_T": d["chaotic"]["cell"]["log10_temperature"],
                             "acc_chaotic": c["accuracy"], "acc_periodic": q["accuracy"],
                             "acc_unsteered": u["accuracy"] if u else None,
                             "d_acc": c["accuracy"] - q["accuracy"], "d_entropy": c["mean_entropy"] - q["mean_entropy"],
                             "d_length": c["mean_length"] - q["mean_length"]})
        res = {"pairs": len(rows), "rows": rows}
        for key in ("d_acc", "d_entropy", "d_length"):
            x = np.array([r[key] for r in rows])
            if not len(x):
                continue
            nz = x[x != 0]
            k, n = int((nz > 0).sum()), len(nz)
            p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n) if n else 1.0
            boots = [float(np.mean(rng.choice(x, len(x)))) for _ in range(10000)]
            res[key] = {"mean": float(np.mean(x)), "positive": k, "nonzero": n, "sign_test_p": p,
                        "bootstrap_95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        out["tasks"][task] = res
        print(task, json.dumps({k: v for k, v in res.items() if k != "rows"}))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("select")
    r = sub.add_parser("run")
    r.add_argument("--pairs", default="docs/bz-pairs.json")
    r.add_argument("--tasks", default="docs/bz-tasks.json")
    r.add_argument("--cells", type=int, nargs="*")
    r.add_argument("--shard", type=int, nargs=2, metavar=("K", "N"))
    r.add_argument("--k", type=int, default=3)
    r.add_argument("--k-code", type=int, default=2)
    r.add_argument("--model", default="qwen0.5b")
    r.add_argument("--layer", type=int, default=12)
    r.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    r.add_argument("--concept", default="philosophy")
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--device", default="cuda")
    r.add_argument("--out", required=True)
    an = sub.add_parser("analyze")
    an.add_argument("--runs", nargs="+", required=True, help="explore-relative run folders with cell-*.json")
    an.add_argument("--out-json")
    args = ap.parse_args()
    {"select": select, "run": run, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()

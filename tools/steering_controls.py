"""Steering controls for the soft-token maps: is the chaos made by the steering construction? (run C1)

rescale  Norm-matched Gaussian random directions. Copies the random directions of an existing vector set and rescales
         each to the norm of a reference concept vector, so that alpha means the same push size. (The stored random
         directions have the neutral prompts' mean residual norm, 10-15 % longer than the concept vectors.)
select   Frozen rule for sparse-autoencoder (SAE) feature directions, from held-out category profiles
         (tools/llm_sae_readout.py profile: mean activation per category, 9 categories x 12 sentences). For a target
         category c, feature f scores profile_f[c] minus the largest profile_f over the other categories, leaving out
         the related ones listed in RELATED; the highest score wins (ties: lower ID). Also returned: the target
         feature's index neighbour (ID + 1, whatever it encodes) and a random live feature (seed 0, uniform over the
         features with any nonzero profile entry).
build    On a GPU pod, after the concept vectors at the SAE's layer are built (explore_llm_phase_map.py vectors): the
         selected features' decoder rows and the vector set's Gaussian random directions, each rescaled to the norm of
         its reference concept vector, written as a new vector set with cosines to the concept vectors. The SAE must
         read the residual stream where the steering hook adds (hidden_states[layer + 1], the output of decoder layer
         `layer`), or the build refuses.

Usage:
  python tools/steering_controls.py rescale SRC_VECTORS CONCEPT OUT [--keys random1,random2,random3]
  python tools/steering_controls.py select PROFILES SAE_NAME OUT_JSON
  python tools/steering_controls.py build --concept-vectors VEC --sae-kind goodfire --sae-path PATH --layer 19
                                         --selection JSON --out OUT
(vector-set paths are relative to the data directory's explore/, as everywhere else)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

RULE = ("For a target category c, feature f scores profile_f[c] minus the largest profile_f over the other categories, "
        "leaving out the related ones in RELATED (hostility: swearing); the highest score wins (ties: lower ID). Also: "
        "the hostility feature's index neighbour (ID + 1) and a random live feature (numpy default_rng(0), uniform over "
        "the features with any nonzero profile entry, sorted by ID).")
RELATED = {"hostility": ("swearing",), "philosophy": ()}
TARGETS = {"hostility": "hostility-vs-kindness", "philosophy": "philosophy"}    # category -> reference concept vector
SEED = 0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def explore_dir():
    import explore_llm_attractors as ela

    return ela.DATA / "explore"


def write_set(out, vectors, info):
    folder = explore_dir() / out
    if (folder / "vectors.npz").exists():
        raise SystemExit(f"{folder}/vectors.npz exists; outputs are never overwritten")
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "vectors.npz", **vectors)
    info["vectors_sha256"] = sha256(folder / "vectors.npz")
    (folder / "vectors.json").write_text(json.dumps(info, indent=1))
    return folder


def rescaled(v, norm):
    return np.asarray(v, dtype=np.float64) / np.linalg.norm(v) * float(norm)


def cosines(vectors, refs):
    return {f"{k}|{r}": float(vectors[k] @ refs[r] / (np.linalg.norm(vectors[k]) * np.linalg.norm(refs[r])))
            for k in vectors for r in refs}


def rescale(src, concept, out, keys):
    folder = explore_dir() / src
    data = np.load(folder / "vectors.npz")
    ref = data[concept]
    vectors = {concept: ref.astype(np.float64)}
    for k in keys:
        vectors[k] = rescaled(data[k], np.linalg.norm(ref))
    info = {"command": "rescale", "source": src, "source_sha256": sha256(folder / "vectors.npz"),
            "reference_concept": concept, "reference_norm": float(np.linalg.norm(ref)),
            "source_norms": {k: float(np.linalg.norm(data[k])) for k in keys},
            "cosines": cosines({k: vectors[k] for k in keys}, {concept: ref})}
    write_set(out, vectors, info)
    print(json.dumps(info, indent=1))


def select(profiles_path, sae_name):
    prof = json.loads(Path(profiles_path).read_text())
    table = prof["profiles"][sae_name]
    cats = prof["categories"]
    out = {"sae": sae_name, "profiles": str(profiles_path), "profiles_sha256": sha256(profiles_path),
           "rule": RULE, "related": RELATED, "seed": SEED, "features": {}}
    for target, related in RELATED.items():
        others = [c for c in cats if c != target and c not in related]
        scored = sorted(((table[f][target] - max(table[f][c] for c in others), int(f)) for f in table),
                        key=lambda s: (-s[0], s[1]))
        score, f = scored[0]
        if score <= 0:
            raise SystemExit(f"no feature is specific to {target}: best score {score}")
        out["features"][target] = {"feature": f, "score": score, "profile": table[str(f)],
                                   "runner_up": {"feature": scored[1][1], "score": scored[1][0]}}
    live = sorted(int(f) for f in table if any(v > 0 for v in table[f].values()))
    out["features"]["random_live"] = {"feature": int(np.random.default_rng(SEED).choice(live)), "pool": len(live)}
    host = out["features"]["hostility"]["feature"]
    out["features"]["neighbour"] = {"feature": host + 1, "of": host,
                                    "profile": table.get(str(host + 1))}
    return out


def build(args):
    import explore_llm_phase_map as elpm
    import llm_sae_readout as lsr

    concepts = elpm.load_vectors(args.concept_vectors)
    sel = json.loads(Path(args.selection).read_text())
    sae = lsr.SAE(args.sae_kind, args.sae_path, layer=args.layer)
    if sae.index != args.layer + 1:
        raise SystemExit(f"the SAE reads hidden_states[{sae.index}], the steering hook adds at hidden_states"
                         f"[{args.layer + 1}]")
    W = sae.W_dec.double().numpy()
    if W.shape != (sae.n_features, len(next(iter(concepts.values())))):
        raise SystemExit(f"decoder shape {W.shape} does not match the vectors")
    vectors, refs = {}, {k: concepts[k] for k in ("hostility-vs-kindness", "philosophy")}
    for name in refs:
        vectors[name] = concepts[name].astype(np.float64)
    meta = {}
    for role, entry in sel["features"].items():
        f = int(entry["feature"])
        ref = TARGETS.get(role, "hostility-vs-kindness")
        key = f"sae-{role}-{f}"
        vectors[key] = rescaled(W[f], np.linalg.norm(concepts[ref]))
        meta[key] = {"role": role, "feature": f, "decoder_norm": float(np.linalg.norm(W[f])), "reference": ref}
    for k in sorted(k for k in concepts if k.startswith("random")):
        vectors[k] = rescaled(concepts[k], np.linalg.norm(concepts["hostility-vs-kindness"]))
        meta[k] = {"role": "gaussian random", "reference": "hostility-vs-kindness"}
    info = {"command": "build", "concept_vectors": args.concept_vectors,
            "concept_vectors_sha256": sha256(explore_dir() / args.concept_vectors / "vectors.npz"),
            "sae_kind": args.sae_kind, "sae_path": args.sae_path, "sae_sha256": sha256(args.sae_path),
            "sae_index": sae.index, "layer": args.layer, "selection": args.selection,
            "selection_sha256": sha256(args.selection), "vectors": meta,
            "norms": {k: float(np.linalg.norm(v)) for k, v in vectors.items()},
            "cosines": cosines({k: v for k, v in vectors.items() if k not in refs}, refs)}
    write_set(args.out, vectors, info)
    print(json.dumps(info, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rs = sub.add_parser("rescale")
    rs.add_argument("src")
    rs.add_argument("concept")
    rs.add_argument("out")
    rs.add_argument("--keys", default="random1,random2,random3")
    se = sub.add_parser("select")
    se.add_argument("profiles")
    se.add_argument("sae_name")
    se.add_argument("out_json")
    bu = sub.add_parser("build")
    bu.add_argument("--concept-vectors", required=True)
    bu.add_argument("--sae-kind", required=True)
    bu.add_argument("--sae-path", required=True)
    bu.add_argument("--layer", type=int, required=True)
    bu.add_argument("--selection", required=True)
    bu.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "rescale":
        rescale(args.src, args.concept, args.out, args.keys.split(","))
    elif args.command == "select":
        out = Path(args.out_json)
        if out.exists():
            raise SystemExit(f"{out} exists; selections are never overwritten")
        sel = select(args.profiles, args.sae_name)
        out.write_text(json.dumps(sel, indent=1))
        print(json.dumps(sel, indent=1))
    else:
        build(args)


if __name__ == "__main__":
    main()

"""Long-horizon census at the batch-fate cell (run BFL, exploratory, 11 Nov 2026).

Run BF (tools/batch_fate.py) followed 64 starts near the fixed start for 5000 steps and saw a few of them leave the
chaotic set for a periodic orbit. This tool follows the same 64 starts, as one batch, for --steps steps, to see
whether trajectories keep leaving (a long chaotic transient) or stop leaving (an attractor at this horizon).

The run is cut into segments of --segment steps. Each segment calls batch_fate.run_chunk from the final states of the
previous one. With a window of one soft token the state is the loop's whole memory, so the segments continue one
trajectory. The recurrence detector is that of run BF (relative distance below --tol to a state up to --max-period
steps back); its memory starts empty in every segment, so a capture in the first q steps of a segment is recorded up
to q steps late. Captured rows stay in the batch, so the layout never changes. A captured row is checked from then
on: every recorded state must lie within --stay-tol (relative) of one of the cycle states saved at its capture, and
every later segment must detect the same period again within its first --max-period + 1 steps. A failed check is
listed under "left".

After each segment a checkpoint is written (atomically), and a restart resumes from it.

Output: DATA/explore/OUT/long.npz (start states, capture steps, periods, cycle states and tokens, final states,
states recorded every --record-every steps), long.json (settings, census, timings, software), checkpoint.npz.

Usage: python tools/batch_fate_long.py --alpha 0.07248 --log-t -0.148 --steps 100000 --device cpu --threads 8 \
           --expect-starts-sha 4ef7a03b028bbf58 --out llm-soft-loop/batch-fate/bfl1
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import batch_fate as bf  # noqa: E402


def save_checkpoint(path, state):
    """Write the checkpoint atomically."""
    arrays = {"x": state["x"], "cap": state["cap"], "per": state["per"], "t": np.int64(state["t"]),
              "segments": np.int64(state["segments"]), "rec": state["rec"], "rec_steps": state["rec_steps"],
              "meta": np.array([json.dumps({"cyc_tok": state["cyc_tok"], "left": state["left"],
                                            "seconds": state["seconds"]})])}
    for k, a in state["cyc_state"].items():
        arrays[f"cyc_state_{k}"] = a
    tmp = Path(str(path) + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_checkpoint(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"][0]))
    return {"x": z["x"], "cap": z["cap"], "per": z["per"], "t": int(z["t"]), "segments": int(z["segments"]),
            "rec": z["rec"], "rec_steps": z["rec_steps"], "cyc_tok": meta["cyc_tok"], "left": meta["left"],
            "seconds": meta["seconds"],
            "cyc_state": {int(k[len("cyc_state_"):]): z[k] for k in z.files if k.startswith("cyc_state_")}}


def run_long(loop, steer, T, x0, steps, segment, max_period, tol, record_every, checkpoint=None, report=None,
             stay_tol=1e-7):
    """Iterate the rows of x0 as one batch for up to `steps` steps in segments; returns the final state dict.

    Stops early when every row is on a cycle. `checkpoint` (a path) is read if it exists and rewritten after each
    segment; `report` is called with a progress dict after each segment."""
    if segment % record_every or steps % segment:
        raise ValueError("--segment must be a multiple of --record-every, and --steps a multiple of --segment")
    if segment <= max_period + 1:
        raise ValueError("--segment must exceed --max-period + 1, or a cycle could never be detected")
    B = x0.shape[0]
    if checkpoint is not None and Path(checkpoint).exists():
        st = load_checkpoint(checkpoint)
        if st["x"].shape != x0.shape:
            raise SystemExit("checkpoint does not match the starts; refusing to resume")
    else:
        st = {"x": np.array(x0, dtype=np.float64), "cap": np.full(B, -1, dtype=np.int64),
              "per": np.zeros(B, dtype=np.int64), "t": 0, "segments": 0, "cyc_tok": {}, "cyc_state": {}, "left": [],
              "rec": np.array(x0, dtype=np.float64)[None], "rec_steps": np.zeros(1, dtype=np.int64), "seconds": 0.0}
    while st["t"] < steps:
        tic = time.perf_counter()
        cap, per, ctok, cst, final, _mind, rec, t_end = bf.run_chunk(loop, steer, T, st["x"], segment, max_period,
                                                                     tol, record_every, stop_alone=False)
        t0 = st["t"]
        n_rec = t_end // record_every
        for r in range(B):
            if st["cap"][r] < 0:
                if cap[r] >= 0:
                    st["cap"][r], st["per"][r] = t0 + int(cap[r]), int(per[r])
                    st["cyc_tok"][str(r)] = ctok[r]
                    st["cyc_state"][r] = cst[r]
            elif not (0 <= cap[r] <= max_period + 1 and per[r] == st["per"][r]):
                st["left"].append({"row": r, "step": t0, "check": "segment start", "detected_at": int(cap[r]),
                                   "period": int(per[r])})
            if st["cap"][r] >= 0:
                for k in range(1, n_rec + 1):
                    step = t0 + k * record_every
                    if step <= st["cap"][r]:
                        continue
                    dist = float(np.linalg.norm(st["cyc_state"][r] - rec[k, r][None], axis=1).min()
                                 / np.linalg.norm(rec[k, r]))
                    if not dist < stay_tol:
                        st["left"].append({"row": r, "step": int(step), "check": "record", "distance": dist})
        if n_rec:
            st["rec"] = np.concatenate([st["rec"], rec[1:n_rec + 1]])
            st["rec_steps"] = np.concatenate([st["rec_steps"], t0 + record_every * np.arange(1, n_rec + 1)])
        st["x"], st["t"], st["segments"] = final, t0 + int(t_end), st["segments"] + 1
        st["seconds"] += time.perf_counter() - tic
        if checkpoint is not None:
            save_checkpoint(checkpoint, st)
        if report is not None:
            report({"step": st["t"], "captured": int((st["cap"] >= 0).sum()), "of": B, "left": len(st["left"]),
                    "seconds": round(st["seconds"], 1)})
        if t_end < segment:      # run_chunk stops early only when every row is on a cycle
            break
    return st


def census(cap, horizons):
    """Rows captured by each horizon (capture step at most the horizon)."""
    return {str(h): int(((cap >= 0) & (cap <= h)).sum()) for h in horizons}


def source_state():
    """Commit and cleanliness of the working tree's tracked files, for the receipt."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    return {"commit": head, "tracked_files_modified": bool(dirty)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--log-t", type=float, required=True)
    ap.add_argument("--starts", type=int, default=64)
    ap.add_argument("--spread", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--segment", type=int, default=5000)
    ap.add_argument("--max-period", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--record-every", type=int, default=1000)
    ap.add_argument("--stay-tol", type=float, default=1e-7,
                    help="largest relative distance of a captured row's recorded state from its saved cycle")
    ap.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--expect-starts-sha", default=None,
                    help="leading hex digits of the SHA-256 of the start states; the run refuses a mismatch")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, args.dtype, args.device)
    v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
    steer = args.alpha * v
    T = 10.0 ** args.log_t
    U, x0 = bf.start_states(loop, steer, args.starts, args.spread, args.seed)
    x0_sha = hashlib.sha256(np.ascontiguousarray(x0).tobytes()).hexdigest()
    if args.expect_starts_sha and not x0_sha.startswith(args.expect_starts_sha):
        raise SystemExit(f"start states have SHA-256 {x0_sha}, expected {args.expect_starts_sha}...; refusing to run")
    st = run_long(loop, steer, T, x0, args.steps, args.segment, args.max_period, args.tol, args.record_every,
                  checkpoint=out / "checkpoint.npz", report=lambda d: print(json.dumps(d), flush=True),
                  stay_tol=args.stay_tol)
    save = {"directions": U, "starts": x0, "cap": st["cap"], "per": st["per"], "final": st["x"], "rec": st["rec"],
            "rec_steps": st["rec_steps"], "cyc_tok": np.array([json.dumps(st["cyc_tok"])])}
    for k, a in st["cyc_state"].items():
        save[f"cyc_state_{k}"] = a
    np.savez_compressed(out / "long.npz", **save)
    caught = st["cap"] >= 0
    horizons = [h for h in (1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000) if h <= st["t"]]
    import torch
    import transformers
    rec = {"args": vars(args), "starts_sha256": x0_sha, "steps_run": st["t"], "segments": st["segments"],
           "captured": int(caught.sum()), "not_captured": int((~caught).sum()),
           "periods": {str(q): int((st["per"] == q).sum()) for q in sorted(set(st["per"][caught].tolist()))},
           "capture_steps": {str(r): int(st["cap"][r]) for r in np.nonzero(caught)[0]},
           "captured_by": census(st["cap"], horizons), "left": st["left"],
           "iteration_seconds": round(st["seconds"], 1), "seconds": round(time.perf_counter() - tic, 1),
           "software": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
           "source": source_state()}
    (out / "long.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()

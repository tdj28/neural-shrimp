"""Exploratory first-return measurements for the Qwen 0.5B fold.

Uses the existing patched float64 soft-token map. Every actual model update is
saved; frozen padding after a row stops is identified by its valid length.
Nothing here establishes asymptotic escape, an attractor, or chaos.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from fold_panel_data import channel_mask, coordinates, intervals, load_panel


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def backup_file(path, root, backup):
    """Copy only complete files; independently rehash both physical copies."""
    path, root, backup = map(Path, (path, root, backup))
    dest = backup / path.relative_to(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = sha(path)
    if dest.exists():
        if sha(dest) != expected:
            raise ValueError(f"Conflicting backup: {dest}")
    else:
        partial = dest.with_suffix(dest.suffix + ".partial")
        shutil.copyfile(path, partial)
        with open(partial, "rb") as f:
            os.fsync(f.fileno())
        if sha(partial) != expected:
            raise ValueError("Backup hash mismatch")
        os.replace(partial, dest)
    if sha(path) != expected or sha(dest) != expected:
        raise ValueError("Source or destination changed during verification")
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": expected}


def pair_events(v, transverse, width):
    """Events in a saved two-phase, two-update sampling, preserving censoring."""
    if v.shape[0] != 2 or v.shape != transverse.shape:
        raise ValueError("Two matching phase arrays required")
    inside = channel_mask(v, transverse, width)
    blocks = intervals(inside.any(axis=0))
    records = []
    for j, (start, end) in enumerate(blocks):
        nxt = blocks[j + 1][0] if j + 1 < len(blocks) else None
        records.append({"entry_pair": start, "last_inside_pair": end,
                        "left_censored": start == 0,
                        "right_censored": end == v.shape[1] - 1,
                        "entry_phase": int(np.flatnonzero(inside[:, start])[0]),
                        "exit_phase": int(np.flatnonzero(inside[:, end])[0]),
                        "next_entry_pair": nxt,
                        "outside_pairs": None if nxt is None else nxt - end - 1})
    return records


def recurrence(history, max_lag=64, positions=128, tolerance=1e-9):
    """Smallest passing finite recurrence lag; relative full-state max error."""
    x = np.asarray(history, dtype=np.float64)
    if x.ndim != 2 or len(x) < max_lag + positions or not np.isfinite(x).all():
        raise ValueError("Insufficient finite full-state history")
    tail = x[-positions:]
    scale = np.maximum(np.linalg.norm(tail, axis=1), 1e-12)
    for lag in range(1, max_lag + 1):
        other = x[-positions-lag:-lag]
        # Cheap necessary check before the full-dimensional calculation.
        if np.max(np.abs(tail[:, 0] - other[:, 0]) / scale) > tolerance:
            continue
        error = float(np.max(np.linalg.norm(tail - other, axis=1) / scale))
        if error <= tolerance:
            return {"lag": lag, "max_relative_error": error, "positions": positions}
    return None


class FirstReturn:
    """Require three consecutive same-parity channel readings (five updates).

    The recorded entry is the first sample of that streak. A short grazing
    visit is retained in the raw states but does not terminate a trajectory.
    """
    def __init__(self, n):
        self.streak = np.zeros((2, n), dtype=int)
        self.done = np.zeros(n, dtype=bool)

    def update(self, step, inside):
        p = step % 2
        inside = np.asarray(inside, dtype=bool)
        self.streak[p] = np.where(inside, self.streak[p] + 1, 0)
        found = (~self.done) & (self.streak[p] >= 3)
        self.done |= found
        return np.flatnonzero(found), step - 4


def prepare(args):
    src, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    panel = load_panel(src)  # Rehash and independently rederive the old figure.
    nf = json.loads((src / "normal-form-p2-t-0.115/normal-form.json").read_text())
    with np.load(src / "normal-form-p2-t-0.115/vectors.npz") as z:
        xc, left, right = [z[k] for k in ("x_c", "l", "r")]
    sign, U = float(np.sign(nf["b"])), panel["width"]
    left, right = sign * left, sign * right
    with np.load(src / "passage-natural-words-p2-t-0.115/top.npz") as z:
        same = z["states"][:, 7]
    with np.load(src / "passage-natural-words-p2-t-0.115/phase-check.npz") as z:
        later = z["next_states"]
    states = np.stack([same, later])
    events = pair_events(panel["v"], panel["transverse"], U)
    anchors, selected = [], []
    for event in events:
        if event["right_censored"]:
            continue
        phase, end = event["exit_phase"], event["last_inside_pair"]
        stop = min(event["next_entry_pair"] or len(same), end + 17)
        candidates = [i for i in range(end + 1, stop)
                      if panel["v"][phase, i] >= 3 * U
                      and panel["transverse"][phase, i] < 0.05
                      and not panel["mask"][:, i].any()]
        event["outbound_3U_pair"] = candidates[0] if candidates else None
        if candidates:
            i = candidates[0]
            selected.append({"anchor": len(anchors), "pair": i, "phase": phase,
                             "preceding_channel_exit_pair": end,
                             "saved_next_entry_pair": event["next_entry_pair"]})
            anchors.append(states[phase, i])
    if len(anchors) < 8:
        write_json(out / "ineligible.json", {"events": events, "eligible": len(anchors)})
        raise ValueError("Fewer than eight eligible exits; amend the plan explicitly")
    # First eight in time, without selecting on their subsequent return.
    anchors, selected = np.array(anchors[:8]), selected[:8]
    rng = np.random.default_rng(20261113)
    rows, seeds, vectors = [], [], []
    for k, x in enumerate(anchors):
        d = x - xc
        w = d - (d @ left) * right
        w /= np.linalg.norm(w)
        random = rng.normal(size=len(xc))
        random -= (random @ left) * right
        random /= np.linalg.norm(random)
        vectors.append([right, w, random])
        conditions = [("baseline", 0., 0., 0.)]
        for direction in ("critical", "exit_transverse"):
            for scale in (-.5, -.2, -.05, .05, .2, .5):
                conditions.append((direction, scale if direction == "critical" else 0.,
                                   scale if direction == "exit_transverse" else 0., 0.))
        conditions += [("random_transverse", 0., 0., -.2),
                       ("random_transverse", 0., 0., .2), ("combined", .2, .2, 0.)]
        for name, a, b, c in conditions:
            displacement = U * (a * right + b * w + c * random)
            seeds.append(x + displacement)
            rows.append({"id": len(rows), "anchor": k, "condition": name,
                         "critical_U": a, "exit_transverse_U": b, "random_transverse_U": c,
                         "displacement_norm": float(np.linalg.norm(displacement)),
                         "split": "test" if k in (3, 7) else "development"})
    seeds = np.array(seeds)
    sv, st = coordinates(seeds, xc, left, right, 1.)
    if channel_mask(sv, st, U).any():
        raise ValueError("A seed is already in the return channel")
    # Saved-state replay checks include slow passage, burst and shifted phase.
    picks = np.array([0, 47, 48, 58, 62, 180, 241, 500])
    np.savez(out / "inputs.npz", seeds=seeds, anchors=anchors,
             perturbation_directions=np.array(vectors), centre=xc, left=left, right=right,
             replay_states=same[picks], replay_F=later[picks], replay_G=same[picks + 1])
    plan = {"name": "FGR1", "status": "exploratory", "source": nf["source"],
            "epsilon": 1e-5, "width": U, "transverse_limit": .05, "max_updates": 4096,
            "rows": rows, "anchors": selected, "threads": 8, "chunk_updates": 128,
            "return_rule": "first of three successive same-parity readings: abs(v)<U and remainder norm<0.05",
            "recurrence_rule": "lags 1..64; max relative full-state residual <=1e-9 over 128 positions; outside channel",
            "stopping": "qualified first return, outside-channel recurrence candidate, or horizon; retained prefix only",
            "test_anchors": [3, 7], "seed": 20261113,
            "input_hashes": panel["receipt"]["input_hashes"], "inputs_sha256": sha(out / "inputs.npz"),
            "limitations": ["one parameter setting and one source trajectory", "no permanent escape claim",
                            "no proof of attraction or chaos", "three perturbation directions, not a full basin survey"]}
    write_json(out / "saved-events.json", {"events": events, "panel_audit": panel["receipt"]})
    write_json(out / "plan.json", plan)
    print(json.dumps({"rows": len(rows), "events": len(events), "anchors": selected}, indent=2))


def run(args):
    from explore_llm_soft_loop import SoftLoop
    import explore_llm_phase_map as phase
    prepared, out, backup = map(Path, (args.prepared, args.out, args.backup))
    plan = json.loads((prepared / "plan.json").read_text())
    if sha(prepared / "inputs.npz") != plan["inputs_sha256"]:
        raise ValueError("Input hash mismatch")
    out.mkdir(parents=True, exist_ok=False)
    backup.mkdir(parents=True, exist_ok=False)
    if out.stat().st_dev == backup.stat().st_dev:
        raise ValueError("Backup must be on a separate filesystem; also inspect physical disks")
    receipts = []
    for name in ("inputs.npz", "plan.json", "saved-events.json"):
        shutil.copyfile(prepared / name, out / name)
        receipts.append(backup_file(out / name, out, backup))
    with np.load(prepared / "inputs.npz") as z:
        data = {k: z[k] for k in z.files}
    so = plan["source"]
    direction_file = phase.DATA / "explore" / so["vectors"] / "vectors.npz"
    start = time.monotonic()
    loop = SoftLoop(so["model"], so["layer"], plan["threads"], so["prompt"], so["opener"],
                    so["window"], "float64", "cpu")
    torch = loop.torch
    n = len(data["seeds"])
    direction = phase.load_vectors(so["vectors"])[so["concept"]]
    loop.prepare(np.tile((so["alpha"] + plan["epsilon"]) * direction, (n, 1)))
    temps = np.full(n, 10. ** so["log10_temperature"])
    metadata = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "plan_sha256": sha(out / "plan.json"), "source_sha256": sha(__file__),
                "runtime_source_sha256": {name: sha(ROOT / "tools" / name) for name in
                                          ("explore_llm_soft_loop.py", "explore_llm_attractors.py",
                                           "explore_llm_phase_map.py", "fold_panel_data.py")},
                "direction_sha256": sha(direction_file), "model": loop.sm.engine.name,
                "model_revision": loop.sm.engine.revision, "prefix_text": loop.prefix_text,
                "prefix_ids": loop.prefix_ids.cpu().tolist(), "pid": os.getpid(),
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
                "device": "cpu", "threads": torch.get_num_threads(), "dtype": str(loop.dtype),
                "backup": str(backup), "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    norms = [m for m in loop.model.modules() if type(m).__name__.endswith("RMSNorm")]
    rotaries = [m for m in loop.model.modules() if type(m).__name__.endswith("RotaryEmbedding")]
    assert norms and rotaries and loop.dtype == torch.float64
    assert all(m.forward.__func__.__qualname__.startswith("keep_float64_norms.") for m in norms)
    assert all(m._inv_freq64.dtype == torch.float64 for m in rotaries)
    write_json(out / "metadata.json", metadata)
    receipts.append(backup_file(out / "metadata.json", out, backup))
    Xcheck = torch.tensor(np.tile(data["replay_states"], (n // 8, 1))[:, None], dtype=torch.float64)
    with torch.no_grad():
        check1, _ = loop.step(Xcheck, temps)
        check2, _ = loop.step(check1, temps)
    e1 = float(np.max(np.abs(check1.numpy()[:, 0] - np.tile(data["replay_F"], (n // 8, 1)))))
    e2 = float(np.max(np.abs(check2.numpy()[:, 0] - np.tile(data["replay_G"], (n // 8, 1)))))
    gate = {"one_update_max_abs": e1, "two_updates_max_abs": e2, "tolerance": 1e-11,
            "passed": max(e1, e2) < 1e-11, "patched_norm_count": len(norms), "patched_rotary_count": len(rotaries)}
    write_json(out / "replay-gate.json", gate)
    receipts.append(backup_file(out / "replay-gate.json", out, backup))
    if not gate["passed"]:
        raise ValueError(f"Saved-state replay failed: {gate}")
    X = torch.tensor(data["seeds"][:, None], dtype=torch.float64)
    detector = FirstReturn(n)
    outcomes = [None] * n
    valid = np.zeros(n, dtype=int)
    rolling = [data["seeds"].copy()]
    chunk_states, chunk_ids, chunk_probs = [], [], []
    model_start = time.monotonic()
    for step in range(1, plan["max_updates"] + 1):
        active = np.array([r is None for r in outcomes])
        with torch.no_grad():
            nxt, probs = loop.step(X, temps)
            tp, ti = torch.topk(probs, 8, dim=1)
            X = torch.where(torch.as_tensor(active)[:, None, None], nxt, X)
        arr = X.numpy()[:, 0].copy()
        if not np.isfinite(arr).all():
            raise ValueError("Nonfinite model state")
        valid[active] = step
        rolling.append(arr)
        if len(rolling) > 193:
            rolling.pop(0)
        chunk_states.append(arr)
        chunk_ids.append(ti.numpy().copy())
        chunk_probs.append(tp.numpy().copy())
        v, transverse = coordinates(arr, data["centre"], data["left"], data["right"], 1.)
        found, entry = detector.update(step, channel_mask(v, transverse, plan["width"]) & active)
        for i in found:
            entry_state = rolling[-5][i]
            ev, et = coordinates(entry_state[None], data["centre"], data["left"], data["right"], 1.)
            outcomes[i] = {"kind": "return", "entry_update": entry, "confirmed_update": step,
                           "entry_v": float(ev[0]), "entry_transverse": float(et[0])}
        if step >= 256 and step % 128 == 0:
            hist = np.array(rolling)
            for i in np.flatnonzero([r is None for r in outcomes]):
                hv, ht = coordinates(hist[-128:, i], data["centre"], data["left"], data["right"], 1.)
                if channel_mask(hv, ht, plan["width"]).any():
                    continue
                rec = recurrence(hist[:, i])
                if rec:
                    outcomes[i] = {"kind": "outside_recurrence_candidate", "detected_update": step, **rec}
        finished = all(r is not None for r in outcomes)
        if step % plan["chunk_updates"] == 0 or finished or step == plan["max_updates"]:
            lo = step - len(chunk_states) + 1
            name = f"updates-{lo:05d}-{step:05d}.npz"
            path = out / name
            # Uncompressed atomic shards keep the write/backup cost bounded.
            with open(str(path) + ".partial", "wb") as f:
                np.savez(f, states=np.array(chunk_states), top_ids=np.array(chunk_ids),
                         top_probabilities=np.array(chunk_probs), valid_through=valid.copy())
                f.flush(); os.fsync(f.fileno())
            os.replace(str(path) + ".partial", path)
            receipts.append(backup_file(path, out, backup))
            write_json(out / "retrieval.json", receipts)
            chunk_states, chunk_ids, chunk_probs = [], [], []
            status = {"step": step, "returned": sum(r is not None and r["kind"] == "return" for r in outcomes),
                      "outside_recurrence": sum(r is not None and r["kind"] == "outside_recurrence_candidate" for r in outcomes),
                      "active": sum(r is None for r in outcomes), "seconds": time.monotonic() - model_start,
                      "backed_up_shards": sum(r["path"].startswith("updates-") for r in receipts)}
            write_json(out / "status.json", status)
            print(json.dumps(status), flush=True)
        if finished:
            break
    outcomes = [r if r is not None else {"kind": "unresolved_at_horizon", "updates": int(valid[i])}
                for i, r in enumerate(outcomes)]
    write_json(out / "result.json", {"outcomes": outcomes, "valid_through": valid.tolist(),
                                    "total_wall_seconds": time.monotonic() - start,
                                    "model_seconds": time.monotonic() - model_start})
    receipts.append(backup_file(out / "result.json", out, backup))
    write_json(out / "manifest.json", receipts)
    backup_file(out / "manifest.json", out, backup)
    write_json(out / "complete.json", {"manifest_sha256": sha(out / "manifest.json"),
                                      "backup_manifest_sha256": sha(backup / "manifest.json"),
                                      "finished": True})
    backup_file(out / "complete.json", out, backup)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--source", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("run")
    a.add_argument("--prepared", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--backup", required=True)
    args = p.parse_args()
    {"prepare": prepare, "run": run}[args.command](args)

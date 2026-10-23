"""Bounded exploratory replay of a saved Huginn internal period candidate.

The readout is diagnostic: tokens are never appended or fed back. This tool
does not establish a primitive orbit, its stability, or infinite persistence.
All states and logits are checkpointed outside Git in immutable short shards.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_distance(a, b):
    """Euclidean full-state distance, relative to a, for each batch row."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim < 2:
        raise ValueError("Expected equal arrays with a batch dimension")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite full state")
    numerator = np.linalg.norm((a - b).reshape(len(a), -1), axis=1)
    denominator = np.linalg.norm(a.reshape(len(a), -1), axis=1)
    return numerator / np.maximum(denominator, 1e-30)


def probability_summary(logits, top_k=20):
    """Stable double-precision statistics from the stored float32 logits."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError("Expected finite batch by vocabulary logits")
    if top_k < 1 or top_k > z.shape[1]:
        raise ValueError("Invalid top-k")
    shifted = z - z.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probs = np.exp(logp)
    # Stable descending sort keeps tied IDs deterministic.
    ids = np.argsort(-z, axis=1, kind="stable")[:, :top_k]
    return {
        "top_ids": ids,
        "top_probabilities": np.take_along_axis(probs, ids, axis=1),
        "top_logits": np.take_along_axis(z, ids, axis=1),
        "entropy_nats": -(probs * logp).sum(axis=1),
        "probabilities": probs,
    }


def compare_logits(reference, candidate):
    a = probability_summary(reference, 1)
    b = probability_summary(candidate, 1)
    return {
        "max_abs": float(np.max(np.abs(np.asarray(reference, dtype=np.float64) - candidate))),
        "argmax_equal": bool(np.array_equal(a["top_ids"], b["top_ids"])),
        "max_total_variation": float(np.max(.5 * np.abs(a["probabilities"] - b["probabilities"]).sum(axis=1))),
    }


def relaxed_step(state, core, lam):
    """Preserve the scan's arithmetic order; no readout enters this map."""
    return (1 - lam) * state + lam * core(state)


def horizon_indices(step, saved_end_update):
    if step < 1:
        raise ValueError("Continuation steps are one-based")
    return [saved_end_update + step, step]


class RecurrenceTracker:
    """Track every q1/q2 closure plus drift from initial phase anchors."""

    def __init__(self, initial, previous=None):
        initial = np.asarray(initial, dtype=np.float32)
        self.initial = initial.copy()
        self.history = [] if previous is None else [np.asarray(previous, dtype=np.float32).copy()]
        self.history.append(initial.copy())
        self.rows = []

    def observe(self, state, step):
        state = np.asarray(state, dtype=np.float32)
        q1 = relative_distance(state, self.history[-1])
        q2 = None if len(self.history) < 2 else relative_distance(state, self.history[-2])
        drift = relative_distance(state, self.initial)
        row = {"step": int(step), "q1_relative": q1.tolist(),
               "q2_relative": None if q2 is None else q2.tolist(),
               "initial_state_relative": drift.tolist()}
        self.rows.append(row)
        self.history = [self.history[-1], state.copy()]
        return row


def _atomic_file(path, write):
    """Publish without replacement; interrupted temporary files are retained."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}-{time.monotonic_ns()}")
    with temporary.open("xb") as handle:
        write(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)  # fails if a final name appeared concurrently
    temporary.unlink()
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path, data):
    encoded = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    return _atomic_file(path, lambda handle: handle.write(encoded))


class EvidenceWriter:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)

    def emit(self, phase, **details):
        row = {"phase": phase, "monotonic": time.monotonic(), **details}
        line = json.dumps(row, allow_nan=False) + "\n"
        with (self.root / "progress.jsonl").open("a") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        print(line, end="", flush=True)

    def checkpoint(self, name, arrays, metadata):
        for key, value in arrays.items():
            value = np.asarray(value)
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError(f"Nonfinite or unsupported checkpoint array {key}")
        # These are local disk shards, deliberately uncompressed so checkpoint
        # CPU time cannot dominate the bounded GPU replay. The manifest hashes
        # the exact bytes; lossless compression can be a separate offline copy.
        archive = _atomic_file(self.root / f"{name}.npz", lambda h: np.savez(h, **arrays))
        note = atomic_json(self.root / f"{name}.json", {**metadata, "array_archive": archive})
        self.emit("checkpoint_complete", name=name, arrays=archive, metadata=note)
        return archive

    def manifest(self, filename, complete):
        files = {}
        for path in sorted(self.root.iterdir()):
            if path.is_file() and path.name not in ("manifest.json", "partial-manifest.json"):
                files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        return atomic_json(self.root / filename, {"complete": complete, "files": files})


class DeadlineExceeded(TimeoutError):
    pass


@contextmanager
def preserve_completed_buffer(writer, buffers):
    """On a catchable failure, retain completed rows of the unfinished shard."""
    try:
        yield
    except BaseException:
        # Scientific execution has stopped. Permit bounded evidence bookkeeping
        # even when the numerical deadline has elapsed.
        signal.setitimer(signal.ITIMER_REAL, 0)
        count = min(len(buffers[key]) for key in ("states", "logits", "rows"))
        if count:
            rows = buffers["rows"][:count]
            start, end = rows[0]["continuation_step"], rows[-1]["continuation_step"]
            try:
                writer.checkpoint(f"interrupted-buffer-{start:05d}-{end:05d}",
                                  {"states": np.stack(buffers["states"][:count]),
                                   "logits": np.stack(buffers["logits"][:count]),
                                   "internal_updates": np.array([r["internal_updates"] for r in rows])},
                                  {"complete": False, "trace_names": ["warm_saved_state", "fresh_seeded_state"],
                                   "readouts_and_recurrence": rows,
                                   "note": "Completed iterations saved after catchable execution failure"})
            except BaseException as preservation_error:
                writer.emit("buffer_preservation_failed", error=repr(preservation_error))
        raise


@contextmanager
def hard_deadline(seconds):
    if seconds <= 0:
        raise ValueError("Deadline must be positive")
    if signal.getitimer(signal.ITIMER_REAL)[0] != 0:
        raise RuntimeError("Refusing to replace an existing alarm")
    old_handler = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise DeadlineExceeded("Exploratory run reached its internal wall-clock ceiling")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def validate_plan(plan):
    required = ("model_revision", "prompt", "prompt_ids", "selected_cell_index", "selected_local_index",
                "lambda", "steering", "tail_paths", "tail_sha256", "steering_direction_path",
                "steering_direction_sha256", "tail_start_update", "tail_end_update", "extension_steps",
                "chunk_steps", "threads", "max_seconds", "memory_fraction", "seed")
    for key in required:
        if key not in plan:
            raise ValueError(f"Missing plan key: {key}")
    if len(plan["tail_paths"]) != 2 or len(plan["tail_sha256"]) != 2:
        raise ValueError("Exactly two saved offset tails are required")
    if plan["tail_end_update"] - plan["tail_start_update"] + 1 != 72:
        raise ValueError("The input must have 72 saved consecutive states")
    for key in ("extension_steps", "chunk_steps", "threads", "max_seconds"):
        if not isinstance(plan[key], int) or plan[key] <= 0:
            raise ValueError(f"Invalid positive integer: {key}")
    if plan["extension_steps"] % plan["chunk_steps"]:
        raise ValueError("Extension must comprise whole checkpoint shards")
    if plan["chunk_steps"] > 128 or plan["extension_steps"] > 8192:
        raise ValueError("Plan exceeds bounded checkpoint/continuation sizes")
    if plan["max_seconds"] > 900 or plan["memory_fraction"] > .8 or plan["threads"] > 4:
        raise ValueError("Plan exceeds this tool's local resource ceiling")
    if not 0 < plan["memory_fraction"] <= .8:
        raise ValueError("Invalid memory fraction")
    if not np.isfinite([plan["lambda"], plan["steering"]]).all():
        raise ValueError("Nonfinite intervention")
    if not plan["prompt_ids"] or not all(isinstance(i, int) and i >= 0 for i in plan["prompt_ids"]):
        raise ValueError("Explicit prompt token IDs required")


def load_inputs(plan):
    states, records = [], []
    index = plan["selected_local_index"]
    for offset, (filename, expected) in enumerate(zip(plan["tail_paths"], plan["tail_sha256"])):
        path = Path(filename)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Input tail hash mismatch at offset {offset}")
        with np.load(path, allow_pickle=False) as data:
            if data["states"].dtype != np.float32 or data["states"].shape[0] != 72:
                raise ValueError("Unexpected saved-tail dtype or length")
            if index < 0 or index >= data["states"].shape[1]:
                raise ValueError("Cell index outside saved tail")
            cell = data["cells"][index]
            if not np.array_equal(cell, np.array([plan["lambda"], plan["steering"]])):
                raise ValueError("Saved tail cell disagrees with planned parameter values")
            chosen = data["states"][:, index].copy()
            if chosen.ndim != 3 or not np.isfinite(chosen).all():
                raise ValueError("Malformed or nonfinite saved state")
            states.append(chosen)
            records.append({"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size,
                            "offset": offset, "eps": float(data["eps"])})
    if states[0].shape != states[1].shape or not np.array_equal(states[0], states[1]):
        raise ValueError("This replay plan expects identical saved base trajectories at both offsets")
    direction_path = Path(plan["steering_direction_path"])
    if sha256(direction_path) != plan["steering_direction_sha256"]:
        raise ValueError("Steering direction hash mismatch")
    direction = np.load(direction_path, allow_pickle=False)
    if direction.dtype != np.float64 or direction.shape != (states[0].shape[-1],) or not np.isfinite(direction).all():
        raise ValueError("Unexpected steering direction dtype/shape")
    return np.stack(states, axis=1), direction, records


def token_rows(logits, tokenizer, cache, top_k=20):
    stats = probability_summary(logits, top_k)
    result = []
    for row in range(len(logits)):
        ids = stats["top_ids"][row].tolist()
        for token_id in ids:
            if token_id not in cache:
                cache[token_id] = tokenizer.decode([token_id], skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False)
        result.append({"top_token_ids": ids, "top_token_strings": [cache[i] for i in ids],
                       "top_probabilities": stats["top_probabilities"][row].tolist(),
                       "top_logits": stats["top_logits"][row].tolist(),
                       "entropy_nats": float(stats["entropy_nats"][row])})
    return result


def _require_comparison(row, max_abs, max_tv, label):
    if not row["argmax_equal"] or row["max_abs"] > max_abs or row["max_total_variation"] > max_tv:
        raise ValueError(f"{label} gate failed: {row}")


def execute(plan, plan_path, source_commit, writer, started):
    import torch
    import transformers
    from explore_latent_loop import HuginnLoop, REVISIONS

    if REVISIONS["huginn"] != plan["model_revision"]:
        raise ValueError("Pinned model revision disagrees with plan")
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current != source_commit:
        raise ValueError("Execution source does not match current HEAD")
    if subprocess.check_output(["git", "diff", "HEAD", "--", "tools/huginn_periodic_readout.py",
                                "tools/explore_latent_loop.py"], cwd=ROOT):
        raise ValueError("Scientific execution code has uncommitted changes")
    saved, direction, input_records = load_inputs(plan)
    writer.emit("inputs_verified", selected_cell_index=plan["selected_cell_index"], inputs=input_records)
    torch.set_num_threads(plan["threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(plan["memory_fraction"])
    loop = HuginnLoop("huginn", "float32", "cuda", plan["threads"], seed=plan["seed"])
    initial = loop.setup(plan["prompt"])
    if loop.token_ids != plan["prompt_ids"]:
        raise ValueError("Prompt token IDs disagree with planned scan")
    if tuple(initial.shape[1:]) != tuple(saved.shape[2:]):
        raise ValueError("Model state dimensions disagree with input histories")
    if loop.model.config.test_time_noise != 0:
        raise ValueError("Noise-free internal recurrence required")
    raw_bytes = ((plan["extension_steps"] * 2 + 72 + 2)
                 * (int(np.prod(initial.shape[1:])) + loop.model.config.vocab_size) * 4)
    if raw_bytes > 10_000_000_000 or shutil.disk_usage(writer.root).free < raw_bytes * 1.2:
        raise ValueError("Insufficient disk headroom or planned raw arrays exceed ten-gigabyte bound")
    runtime = {"source_commit": source_commit, "plan_sha256": sha256(plan_path),
               "model_revision": REVISIONS["huginn"], "prompt_ids": loop.token_ids,
               "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
               "numpy": np.__version__, "gpu": torch.cuda.get_device_name(), "threads": plan["threads"],
               "dtype": "float32", "tf32": False, "inputs": input_records,
               "input_offset_states_bitwise_identical": True,
               "expected_uncompressed_array_bytes": raw_bytes,
               "seed": plan["seed"],
               "semantics": "Same next-token position; diagnostic readout never feeds the internal recurrence"}
    atomic_json(writer.root / "runtime.json", runtime)
    gates = plan.get("gates", {})
    native_tol = gates.get("native_relative", 1e-6)
    with torch.no_grad():
        reference, _ = loop.model.core_block_forward(initial, loop.e, loop.freqs, None, None, torch.tensor(0), 0)
        ours = loop.F(initial)
        native_error = float((reference - ours).norm() / reference.norm())
        repeat = float((ours - loop.F(initial)).abs().max())
    writer.emit("native_core_gate", relative_error=native_error, repeated_max_abs=repeat)
    if native_error > native_tol or repeat != 0:
        raise ValueError("Native core/repeat gate failed")

    phases_cpu = torch.from_numpy(saved[-2:, 0].copy())
    cpu_logits = []
    for phase in phases_cpu:
        one = phase[None]
        direct = loop.logits(one).float().cpu().numpy()[None]
        with torch.no_grad():
            native = loop.model.predict_from_latents(one).logits[:, -1].cpu().numpy()
        check = compare_logits(native, direct)
        writer.emit("cpu_native_readout_gate", **check)
        _require_comparison(check, gates.get("cpu_native_max_abs", 1e-4),
                            gates.get("cpu_native_tv", 1e-5), "CPU/native readout")
        cpu_logits.append(direct[0])
    cpu_logits = np.stack(cpu_logits)

    model, transformer = loop.model, loop.model.transformer
    for module in (transformer.coda, transformer.ln_f, model.lm_head):
        module.to(loop.device)

    def logits_gpu(h):
        with torch.no_grad():
            x = transformer.ln_f(h)
            for i, block in enumerate(transformer.coda):
                x = block(x, loop.freqs, torch.tensor(-1 - i), None, None)
            return model.lm_head(transformer.ln_f(x)[:, -1, :])

    with torch.no_grad():
        gpu_logits = logits_gpu(phases_cpu.to(loop.device)).cpu().numpy()
    comparison = compare_logits(cpu_logits, gpu_logits)
    writer.emit("cpu_gpu_readout_gate", **comparison)
    _require_comparison(comparison, gates.get("readout_max_abs", 1e-3),
                        gates.get("readout_tv", 1e-4), "CPU/GPU readout")

    steering_rows = np.repeat((float(plan["steering"]) * direction)[None], 2, axis=0)
    loop.set_steering(0, steering_rows)
    lam = torch.tensor([plan["lambda"], plan["lambda"]], dtype=loop.dtype, device=loop.device)[:, None, None]
    one_step_errors = []
    with torch.no_grad():
        for i in range(8):
            state = torch.from_numpy(saved[i].copy()).to(loop.device)
            output = relaxed_step(state, loop.F, lam).cpu().numpy()
            one_step_errors.append(relative_distance(saved[i + 1], output).tolist())
    error = max(max(row) for row in one_step_errors)
    writer.emit("saved_one_step_gate", errors=one_step_errors, maximum_relative_error=error)
    if error > gates.get("saved_one_step_relative", 3e-5):
        raise ValueError("Saved-state one-step replay gate failed")

    cache = {}
    saved_logits, saved_rows = [], []
    for index in range(0, len(saved), 2):
        states = saved[index:index + 2, 0]
        with torch.no_grad():
            z = logits_gpu(torch.from_numpy(states.copy()).to(loop.device)).cpu().numpy()
        saved_logits.extend(z)
        for j, readout in enumerate(token_rows(z, loop.tok, cache)):
            saved_rows.append({"internal_update": plan["tail_start_update"] + index + j,
                               "readout": readout})
    writer.checkpoint("saved-tail-readouts", {"states": saved[:, 0], "logits": np.stack(saved_logits),
                      "internal_updates": np.arange(plan["tail_start_update"], plan["tail_end_update"] + 1)},
                      {"readouts": saved_rows, "trace_name": "identical_saved_base_at_both_offsets",
                       "semantics": "Readouts for a fixed prompt, not generated text"})

    h = torch.cat([torch.from_numpy(saved[-1, 0:1].copy()).to(loop.device), initial], dim=0)
    with torch.no_grad():
        initial_logits = logits_gpu(h).cpu().numpy()
    writer.checkpoint("initial-states", {"states": h.cpu().numpy(), "logits": initial_logits,
                      "internal_updates": np.array([plan["tail_end_update"], 0])},
                      {"trace_names": ["warm_saved_state", "fresh_seeded_state"],
                       "readouts": token_rows(initial_logits, loop.tok, cache)})
    tracker = RecurrenceTracker(h.cpu().numpy())
    warm_prior = saved[-2, 0]
    buffers = {"states": [], "logits": [], "rows": []}
    chunk_started = time.monotonic()
    all_top_ids = []
    max_q2 = np.zeros(2)
    min_q1 = np.full(2, np.inf)
    chunk_count = 0
    # First shard is both scientific evidence and an end-to-end throughput gate.
    with preserve_completed_buffer(writer, buffers):
        for step in range(1, plan["extension_steps"] + 1):
            with torch.no_grad():
                h = relaxed_step(h, loop.F, lam)
                z = logits_gpu(h).cpu().numpy()
                states = h.cpu().numpy()
            if not np.isfinite(states).all() or not np.isfinite(z).all():
                raise ValueError(f"Nonfinite state/logits at continuation step {step}")
            metric = tracker.observe(states, step)
            if step == 1:
                metric["warm_q2_relative"] = float(relative_distance(states[:1], warm_prior[None])[0])
                max_q2[0] = metric["warm_q2_relative"]
            top = token_rows(z, loop.tok, cache)
            row = {"continuation_step": step, "internal_updates": horizon_indices(step, plan["tail_end_update"]),
                   "readouts": top, **metric}
            all_top_ids.append([r["top_token_ids"][0] for r in top])
            if metric["q2_relative"] is not None:
                max_q2 = np.maximum(max_q2, metric["q2_relative"])
            min_q1 = np.minimum(min_q1, metric["q1_relative"])
            buffers["states"].append(states)
            buffers["logits"].append(z)
            buffers["rows"].append(row)
            if step % plan["chunk_steps"] == 0:
                start = step - len(buffers["rows"]) + 1
                writer.checkpoint(f"continuation-{start:05d}-{step:05d}",
                                  {"states": np.stack(buffers["states"]), "logits": np.stack(buffers["logits"]),
                                   "internal_updates": np.array([r["internal_updates"] for r in buffers["rows"]])},
                                  {"trace_names": ["warm_saved_state", "fresh_seeded_state"],
                                   "readouts_and_recurrence": buffers["rows"]})
                chunk_count += 1
                elapsed_chunk = time.monotonic() - chunk_started
                for value in buffers.values():
                    value.clear()
                if step == plan["chunk_steps"]:
                    remaining = plan["max_seconds"] - (time.monotonic() - started)
                    projected = elapsed_chunk * ((plan["extension_steps"] - step) / step) * 1.3
                    writer.emit("throughput_gate", seconds_per_chunk=elapsed_chunk,
                                projected_remaining_with_margin=projected, remaining_seconds=remaining)
                    if projected + 30 > remaining:
                        raise DeadlineExceeded("Measured compute/readout/checkpoint path cannot fit remaining ceiling")
                chunk_started = time.monotonic()
    summary = {"complete": True, "extension_steps_per_trace": plan["extension_steps"],
               "trace_names": ["warm_saved_state", "fresh_seeded_state"], "chunks": chunk_count,
               "final_internal_updates": horizon_indices(plan["extension_steps"], plan["tail_end_update"]),
               "minimum_q1_relative_per_trace": min_q1.tolist(), "maximum_q2_relative_per_trace": max_q2.tolist(),
               "last_eight_metrics": tracker.rows[-8:], "observed_top_token_ids_per_trace":
                   [sorted(set(np.asarray(all_top_ids)[:, i].tolist())) for i in range(2)],
               "wall_seconds_before_final_manifest": time.monotonic() - started,
               "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
               "qualification": "Finite float32 exploratory replay; no certified primitive orbit or infinite persistence; no token feedback"}
    atomic_json(writer.root / "summary.json", summary)
    writer.emit("scientific_complete", final_internal_updates=summary["final_internal_updates"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    output = args.out.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Raw replay output must be outside the repository")
    writer = EvidenceWriter(output)
    atomic_json(output / "plan.json", plan)
    started = time.monotonic()
    try:
        with hard_deadline(plan["max_seconds"]):
            execute(plan, args.plan, args.source_commit, writer, started)
            writer.manifest("manifest.json", True)
    except BaseException as error:
        # The alarm is disarmed before preservation, so bookkeeping can finish.
        atomic_json(output / "failure.json", {"complete": False, "error_type": type(error).__name__,
                    "error": str(error), "traceback": traceback.format_exc(),
                    "elapsed_seconds": time.monotonic() - started,
                    "note": "Completed shards and interrupted temporary files retained; no automatic retry"})
        writer.manifest("partial-manifest.json", False)
        raise


if __name__ == "__main__":
    main()

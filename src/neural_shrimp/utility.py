"""Paired policy pilot; recurrence is not a measurement of shrimp geometry."""
from __future__ import annotations

import time
import numpy as np
from .diagnostics import classify_tail

POLICIES = ("fixed", "scheduled_decay", "stagnation_decay", "recurrence_decay")


def run_policy(model, data, initial, rates, policy, config):
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    theta = np.array(initial, dtype=np.float64, copy=True)
    rates = np.array(rates, dtype=float, copy=True)
    steps = config["steps"]
    history, losses, rate_history, events = [theta.copy()], [], [], []
    diagnostics_seconds = 0.0
    decays = 0
    map_started_at = 0
    status = "completed"
    started = time.perf_counter()
    mapping = model.training_map(data.train_x, data.train_y, *rates)
    for step in range(steps):
        losses.append(float(model.loss(theta, data.train_x, data.train_y)))
        trigger = False
        reason = None
        before_diagnostic = time.perf_counter()
        if decays < config["max_decays"]:
            if policy == "scheduled_decay":
                trigger = step in (steps // 3, 2 * steps // 3)
                reason = "fixed_schedule"
            elif (step >= config["check_after"] and step % config["check_every"] == 0
                  and step - map_started_at >= 2 * config["check_every"]):
                if policy == "stagnation_decay":
                    window = config["check_every"]
                    old = min(losses[-2 * window:-window])
                    new = min(losses[-window:])
                    trigger = (old - new) / max(abs(old), 1e-12) < config["stagnation_relative_improvement"]
                    reason = "scalar_loss_stagnation"
                elif policy == "recurrence_decay":
                    candidate = classify_tail(np.asarray(history[max(map_started_at, len(history) - config["tail"]):]),
                                              max_period=config["max_period"],
                                              atol=config["atol"], rtol=config["rtol"])
                    trigger = candidate.get("period", 0) not in (None, 0, 1)
                    reason = "full_weight_nontrivial_recurrence"
        diagnostics_seconds += time.perf_counter() - before_diagnostic
        if trigger:
            rates *= config["decay_factor"]
            decays += 1
            events.append({"step": step, "reason": reason, "rates": rates.tolist()})
            mapping = model.training_map(data.train_x, data.train_y, *rates)
            map_started_at = step
        rate_history.append(rates.copy())
        with np.errstate(over="ignore", invalid="ignore"):
            theta = mapping(theta)
        history.append(theta.copy())
        if not np.isfinite(theta).all() or np.max(np.abs(theta)) > config["divergence_threshold"]:
            status = "divergent"
            break
    wall_seconds = time.perf_counter() - started
    finite = status != "divergent"
    terminal = {}
    for split in ("train", "validation", "test"):
        # Explicit MSE, excluding the ridge penalty used by the optimization map.
        terminal[f"terminal_{split}_mse"] = (float(np.mean((model.predict(theta, getattr(data, split + "_x")) - getattr(data, split + "_y")) ** 2)) if finite else None)
    return {
        "status": status, "policy": policy, "events": events,
        "gradient_updates": len(history) - 1, "wall_seconds": wall_seconds,
        "diagnostic_seconds": diagnostics_seconds, **terminal,
    }, {"states": np.asarray(history), "losses": np.asarray(losses), "learning_rates": np.asarray(rate_history)}

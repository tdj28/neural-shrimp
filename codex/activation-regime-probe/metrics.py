"""Pure numerical metrics and conservative finite-horizon reference gates."""
import numpy as np


def finite_matrix(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("Expected at least three finite vector observations")
    return x


def recurrence(x, max_period=64, tolerance=1e-9):
    x = finite_matrix(x)
    limit = min(max_period, len(x) // 3)
    errors = []
    for p in range(1, limit + 1):
        scale = np.maximum(np.linalg.norm(x[:-p], axis=1), 1e-15)
        errors.append(float(np.max(np.linalg.norm(x[p:] - x[:-p], axis=1) / scale)))
    period = next((p for p, e in enumerate(errors, 1) if e <= tolerance), 0)
    return {"period": period, "minimum_error": min(errors), "errors": errors,
            "best_lag": int(np.argmin(errors)) + 1, "tolerance": tolerance}


def dimensions(x):
    x = finite_matrix(x)
    c = x - x.mean(0)
    total = float(np.sum(c * c))
    if total <= 1e-14 * max(float(np.sum(x * x)), 1e-300):
        return {"raw_pr": 0.0, "standardized_pr": 0.0, "varying_neurons": 0}
    def pr(z):
        gram = z @ z.T
        return float(np.trace(gram) ** 2 / np.sum(gram * gram))
    sd = np.sqrt(np.mean(c * c, axis=0))
    keep = sd > max(float(sd.max()) * 1e-8, 1e-15)
    return {"raw_pr": pr(c), "standardized_pr": pr(c[:, keep] / sd[keep]),
            "varying_neurons": int(keep.sum())}


def layout(x, rows=64, columns=76):
    """PCA ordering, fit only on calibration data. Neighbors are not connections."""
    x = finite_matrix(x)
    if x.shape[1] != rows * columns:
        raise ValueError("Neuron count does not match grid")
    c = x - x.mean(0)
    sd = c.std(0)
    z = c / np.where(sd > 1e-12, sd, 1.0)
    z[:, sd <= 1e-12] = 0
    eig, u = np.linalg.eigh(z @ z.T)
    if eig[-2] <= 1e-12:
        raise ValueError("Calibration layout needs two varying directions")
    pcs = (u[:, -2:].T @ z) / np.sqrt(eig[-2:])[:, None]
    order = np.argsort(pcs[1], kind="stable").reshape(rows, columns)
    return np.array([band[np.argsort(pcs[0, band], kind="stable")] for band in order])


def roughness(x, order):
    x = finite_matrix(x)
    flat = np.asarray(order).ravel()
    if sorted(flat.tolist()) != list(range(x.shape[1])):
        raise ValueError("Layout must contain each neuron exactly once")
    grid = x[:, order]
    a = np.diff(grid, axis=1)
    b = np.diff(grid, axis=2)
    energy = (np.sum(a * a, axis=(1, 2)) + np.sum(b * b, axis=(1, 2))) / (a[0].size + b[0].size)
    variance = np.var(x, axis=1)
    if np.any(variance <= 1e-30):
        raise ValueError("Spatial roughness is undefined for a uniform frame")
    values = energy / (2 * variance)
    return {"median": float(np.median(values)), "per_frame": values.tolist()}


def reference(states, growth, max_period=64, tolerance=1e-9):
    states = finite_matrix(states)
    growth = np.asarray(growth, dtype=np.float64)
    if growth.shape != (512, 2) or not np.isfinite(growth).all():
        raise ValueError("Reference requires 512 finite growth measurements at two offsets")
    rec = recurrence(states[-384:], max_period, tolerance)
    means = growth.mean(0)
    blocks = growth.reshape(4, 128, 2).mean(1)
    agreement = abs(float(means[0] - means[1])) <= max(0.03, 0.2 * float(np.max(np.abs(means))))
    if rec["period"] and np.all(means < -1e-3):
        label = "periodic"
    elif not rec["period"] and np.all(means > 0.01) and np.all(blocks > 0) and agreement:
        label = "finite-time chaotic"
    else:
        label = "unresolved"
    return {"label": label, "recurrence": rec, "exponents": means.tolist(),
            "block_exponents": blocks.tolist(), "offset_agreement": agreement}


def threshold(periodic, chaotic):
    if not np.isfinite([periodic, chaotic]).all() or abs(chaotic - periodic) <= 1e-12:
        raise ValueError("Calibration has no finite separating score")
    return {"value": float((periodic + chaotic) / 2), "chaotic_above": bool(chaotic > periodic)}


def predict(score, rule):
    return "finite-time chaotic" if (score > rule["value"]) == rule["chaotic_above"] else "periodic"

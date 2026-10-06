"""Batched float64 gradient-descent maps over learning-rate grids.

Each grid cell is one autonomous deterministic map: the complete parameter
vector is the state and the two per-layer learning rates are fixed controls,
``theta <- theta - P grad L(theta)``. All cells share the same network,
dataset and initial parameters, so the only difference between cells is the
pair of rates. The Jacobian of the map is ``I - P H`` with the exact Hessian;
it is applied through analytic Hessian-vector products.

Nothing here labels a shrimp. Finite-tail recurrence is a candidate, a
positive finite-time exponent is a chaos candidate, and verified orbits come
only from :func:`neural_shrimp.diagnostics.certify_cycle`. Boundary
organization needs separate continuation.
"""

from dataclasses import dataclass

import numpy as np

ACTIVATIONS = ("tanh", "identity")

STATUS_DIVERGENT = 0
STATUS_FIXED = 1
STATUS_PERIODIC = 2
STATUS_CHAOTIC = 3
STATUS_UNRESOLVED = 4
STATUS_NAMES = {
    STATUS_DIVERGENT: "divergent",
    STATUS_FIXED: "fixed_point_candidate",
    STATUS_PERIODIC: "periodic_candidate",
    STATUS_CHAOTIC: "chaotic_candidate",
    STATUS_UNRESOLVED: "unresolved",
}


@dataclass(frozen=True)
class TwoLayerNet:
    """One hidden layer and a scalar output.

    Prediction is ``out_scale * act(in_scale * x @ W0 + b0) @ w1 + b1``.
    Parameter order is row-major ``W0`` (d x width), ``b0``, ``w1``, ``b1``;
    biases are present only when ``bias`` is true. The loss is
    ``loss_scale * mean(residual**2) + ridge / 2 * ||theta||^2``.

    ``loss_scale=0.5, in_scale=1, out_scale=1, bias=True`` reproduces
    :class:`neural_shrimp.models.TanhRegressor`. The Sohl-Dickstein (2024)
    mean-field network uses ``loss_scale=1, in_scale=sqrt(2/d),
    out_scale=1/width, bias=False`` with tanh.
    """

    d: int
    width: int
    activation: str = "tanh"
    in_scale: float = 1.0
    out_scale: float = 1.0
    bias: bool = True
    loss_scale: float = 0.5
    ridge: float = 0.0

    def __post_init__(self):
        if self.d < 1 or self.width < 1:
            raise ValueError("d and width must be positive")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        for name in ("in_scale", "out_scale", "loss_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.ridge) or self.ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")

    @property
    def n_parameters(self):
        return self.d * self.width + self.width + int(self.bias) * (self.width + 1)

    @property
    def hidden_parameter_count(self):
        return self.d * self.width + int(self.bias) * self.width

    def layer_slices(self):
        """Slices of the flat vector: (W0, b0 or None, w1, b1 or None)."""
        a = self.d * self.width
        if self.bias:
            return (slice(0, a), slice(a, a + self.width),
                    slice(a + self.width, a + 2 * self.width),
                    slice(a + 2 * self.width, a + 2 * self.width + 1))
        return slice(0, a), None, slice(a, a + self.width), None

    def rate_vector(self, lr_hidden, lr_output):
        """Per-parameter rates with shape (cells, parameters)."""
        lr_hidden = np.atleast_1d(np.asarray(lr_hidden, dtype=np.float64))
        lr_output = np.atleast_1d(np.asarray(lr_output, dtype=np.float64))
        if lr_hidden.shape != lr_output.shape or lr_hidden.ndim != 1:
            raise ValueError("rate arrays must be one-dimensional and equal in shape")
        if not (np.all(np.isfinite(lr_hidden)) and np.all(np.isfinite(lr_output))
                and np.all(lr_hidden >= 0) and np.all(lr_output >= 0)):
            raise ValueError("learning rates must be finite and nonnegative")
        hidden = self.hidden_parameter_count
        rates = np.empty((len(lr_hidden), self.n_parameters), dtype=np.float64)
        rates[:, :hidden] = lr_hidden[:, None]
        rates[:, hidden:] = lr_output[:, None]
        return rates

    def _unpack(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 2 or theta.shape[1] != self.n_parameters:
            raise ValueError(f"Expected parameters with shape (cells, {self.n_parameters})")
        s0, sb0, s1, sb1 = self.layer_slices()
        cells = theta.shape[0]
        w0 = theta[:, s0].reshape(cells, self.d, self.width)
        b0 = theta[:, sb0] if sb0 is not None else None
        w1 = theta[:, s1]
        b1 = theta[:, sb1][:, 0] if sb1 is not None else None
        return w0, b0, w1, b1

    def _pack(self, gw0, gb0, gw1, gb1):
        cells = gw0.shape[0]
        parts = [gw0.reshape(cells, -1)]
        if self.bias:
            parts.append(gb0)
        parts.append(gw1)
        if self.bias:
            parts.append(gb1[:, None])
        return np.concatenate(parts, axis=1)

    def _act(self, pre):
        if self.activation == "tanh":
            h = np.tanh(pre)
            s = 1.0 - h * h
            return h, s, -2.0 * h * s
        return pre, np.ones_like(pre), np.zeros_like(pre)

    def _validate_data(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.d or len(x) < 1 or y.shape != (len(x),):
            raise ValueError(f"Expected x with shape (n, {self.d}) and y with shape (n,)")
        return x, y

    def _forward(self, theta, x):
        w0, b0, w1, b1 = self._unpack(theta)
        # One GEMM: (n, d) x (d, cells * width), then (n, cells, width).
        pre = self.in_scale * np.tensordot(x, w0, axes=([1], [1]))
        if b0 is not None:
            pre = pre + b0[None, :, :]
        h, s, s2 = self._act(pre)
        f = self.out_scale * np.einsum("nbh,bh->nb", h, w1)
        if b1 is not None:
            f = f + b1[None, :]
        return (w0, b0, w1, b1), pre, h, s, s2, f

    def predict(self, theta, x):
        """Predictions with shape (n, cells)."""
        x = np.asarray(x, dtype=np.float64)
        return self._forward(theta, x)[-1]

    def loss(self, theta, x, y):
        """Training loss per cell, including the ridge term."""
        x, y = self._validate_data(x, y)
        theta = np.asarray(theta, dtype=np.float64)
        residual = self.predict(theta, x) - y[:, None]
        return (self.loss_scale * np.mean(residual * residual, axis=0)
                + 0.5 * self.ridge * np.sum(theta * theta, axis=1))

    def gradient(self, theta, x, y, return_loss=False):
        """Analytic full-batch gradient per cell, shape (cells, parameters)."""
        x, y = self._validate_data(x, y)
        theta = np.asarray(theta, dtype=np.float64)
        (w0, b0, w1, b1), _, h, s, _, f = self._forward(theta, x)
        n = len(x)
        residual = f - y[:, None]
        g = (2.0 * self.loss_scale / n) * residual
        gw1 = self.out_scale * np.einsum("nbh,nb->bh", h, g)
        delta = g[:, :, None] * (self.out_scale * w1)[None, :, :] * s
        gw0 = self.in_scale * np.tensordot(x, delta, axes=([0], [0])).transpose(1, 0, 2)
        gb0 = delta.sum(axis=0) if self.bias else None
        gb1 = g.sum(axis=0) if self.bias else None
        grad = self._pack(gw0, gb0, gw1, gb1) + self.ridge * theta
        if return_loss:
            loss = (self.loss_scale * np.mean(residual * residual, axis=0)
                    + 0.5 * self.ridge * np.sum(theta * theta, axis=1))
            return grad, loss
        return grad

    def hvp(self, theta, x, y, vector):
        """Exact Hessian-vector products per cell (R-operator)."""
        x, y = self._validate_data(x, y)
        theta = np.asarray(theta, dtype=np.float64)
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != theta.shape:
            raise ValueError("vector must have the same shape as theta")
        (w0, b0, w1, b1), pre, h, s, s2, f = self._forward(theta, x)
        v0, c0, v1, c1 = self._unpack(vector)
        n = len(x)
        k = 2.0 * self.loss_scale / n
        g = k * (f - y[:, None])
        r_pre = self.in_scale * np.tensordot(x, v0, axes=([1], [1]))
        if c0 is not None:
            r_pre = r_pre + c0[None, :, :]
        r_h = s * r_pre
        r_f = self.out_scale * (np.einsum("nbh,bh->nb", r_h, w1)
                                + np.einsum("nbh,bh->nb", h, v1))
        if c1 is not None:
            r_f = r_f + c1[None, :]
        r_g = k * r_f
        hw1 = self.out_scale * (np.einsum("nbh,nb->bh", r_h, g)
                                + np.einsum("nbh,nb->bh", h, r_g))
        a = self.out_scale * w1[None, :, :]
        r_delta = (r_g[:, :, None] * a * s
                   + g[:, :, None] * (self.out_scale * v1)[None, :, :] * s
                   + g[:, :, None] * a * s2 * r_pre)
        hw0 = self.in_scale * np.tensordot(x, r_delta, axes=([0], [0])).transpose(1, 0, 2)
        hb0 = r_delta.sum(axis=0) if self.bias else None
        hb1 = r_g.sum(axis=0) if self.bias else None
        return self._pack(hw0, hb0, hw1, hb1) + self.ridge * vector

    def hessian(self, theta, x, y):
        """Dense exact Hessian for one parameter vector, from HVPs."""
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.n_parameters,):
            raise ValueError(f"Expected parameter shape {(self.n_parameters,)}")
        basis = np.eye(self.n_parameters)
        tiled = np.repeat(theta[None, :], self.n_parameters, axis=0)
        result = self.hvp(tiled, x, y, basis)
        return 0.5 * (result + result.T)

    def gauss_newton_layer_max(self, theta, x, y):
        """Largest Gauss-Newton eigenvalue of each layer block for one vector.

        Returns (hidden_block, output_block), each including ridge. The
        Gauss-Newton matrix is ``2 * loss_scale / n * J^T J`` with ``J`` the
        Jacobian of predictions; it uses initial parameters and data only.
        """
        x, y = self._validate_data(x, y)
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.n_parameters,):
            raise ValueError(f"Expected parameter shape {(self.n_parameters,)}")
        (w0, b0, w1, b1), _, h, s, _, _ = self._forward(theta[None, :], x)
        h, s, w1 = h[:, 0, :], s[:, 0, :], w1[0]
        n = len(x)
        back = self.out_scale * s * w1[None, :]
        hidden = [(self.in_scale * x[:, :, None] * back[:, None, :]).reshape(n, -1)]
        output = [self.out_scale * h]
        if self.bias:
            hidden.append(back)
            output.append(np.ones((n, 1)))
        k = 2.0 * self.loss_scale / n
        values = []
        for blocks in (hidden, output):
            jac = np.concatenate(blocks, axis=1)
            values.append(float(np.linalg.eigvalsh(k * jac.T @ jac)[-1]) + self.ridge)
        return tuple(values)


@dataclass(frozen=True)
class GridMap:
    """Simultaneous gradient-descent maps for many rate pairs."""

    net: TwoLayerNet
    x: np.ndarray
    y: np.ndarray
    lr_hidden: np.ndarray
    lr_output: np.ndarray

    def __post_init__(self):
        x, y = self.net._validate_data(self.x, self.y)
        x, y = x.copy(), y.copy()
        x.setflags(write=False)
        y.setflags(write=False)
        rates = self.net.rate_vector(self.lr_hidden, self.lr_output)
        rates.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "rates", rates)

    @property
    def cells(self):
        return self.rates.shape[0]

    def step(self, theta, return_loss=False):
        grad, loss = self.net.gradient(theta, self.x, self.y, return_loss=True)
        new = theta - self.rates * grad
        return (new, loss) if return_loss else new

    def jvp(self, theta, vector):
        """Jacobian-vector product of the map: v - P (H v)."""
        return vector - self.rates * self.net.hvp(theta, self.x, self.y, vector)

    def cell_map(self, index):
        """Scalar map and dense Jacobian for one cell, for certification."""
        rates = self.rates[index].copy()

        def map_fn(theta):
            theta = np.asarray(theta, dtype=np.float64)
            return theta - rates * self.net.gradient(theta[None, :], self.x, self.y)[0]

        def jacobian(theta):
            hessian = self.net.hessian(np.asarray(theta, dtype=np.float64), self.x, self.y)
            return np.eye(self.net.n_parameters) - rates[:, None] * hessian

        return map_fn, jacobian


def sohl_dickstein_measure(losses, max_val=1e6):
    """Convergence score of Sohl-Dickstein (2024), batched over cells.

    ``losses`` has shape (steps, cells); nonfinite values count as
    ``max_val``. Following the published notebook, losses are normalized by
    the first loss, clipped at ``max_val``; a run is convergent when the mean
    of its last 20 normalized losses is below one. Returns
    (convergent, score) with score ``-sum(v)`` if convergent else ``sum(1/v)``.
    """
    v = np.asarray(losses, dtype=np.float64).copy()
    if v.ndim != 2 or v.shape[0] < 20:
        raise ValueError("losses must have shape (steps >= 20, cells)")
    v[~np.isfinite(v)] = max_val
    first = v[0].copy()
    first[first == 0] = np.finfo(np.float64).tiny
    v = v / first[None, :]
    v = np.minimum(v, max_val)
    convergent = np.mean(v[-20:], axis=0) < 1.0
    with np.errstate(divide="ignore"):
        score = np.where(convergent, -np.sum(v, axis=0), np.sum(1.0 / v, axis=0))
    return convergent, score


def screen_tails(tail, max_period=64, min_repeats=6, atol=1e-8, rtol=1e-6):
    """Smallest full-state recurrence period per cell in a retained tail.

    ``tail`` has shape (cells, steps, parameters). For each candidate period
    ``p``, every one of the last ``min_repeats`` blocks must agree with the
    final block within ``atol + rtol * max(|a|, |b|)``. Returns
    (period, normalized_residual): period 0 means no recurrence up to
    ``max_period``; nonfinite tails return period -1. This screens candidates;
    it does not verify an orbit.
    """
    tail = np.asarray(tail, dtype=np.float64)
    if tail.ndim != 3:
        raise ValueError("tail must have shape (cells, steps, parameters)")
    if max_period < 1 or min_repeats < 2 or atol <= 0 or rtol < 0:
        raise ValueError("invalid recurrence settings")
    cells, steps, _ = tail.shape
    period = np.zeros(cells, dtype=np.int64)
    residual = np.full(cells, np.inf)
    finite = np.all(np.isfinite(tail), axis=(1, 2))
    period[~finite] = -1
    open_cells = finite.copy()
    for p in range(1, int(max_period) + 1):
        count = p * int(min_repeats)
        if count > steps or not np.any(open_cells):
            break
        idx = np.flatnonzero(open_cells)
        blocks = tail[idx, steps - count:, :].reshape(len(idx), min_repeats, p, -1)
        reference = blocks[:, -1:, :, :]
        scale = atol + rtol * np.maximum(np.abs(blocks), np.abs(reference))
        normalized = np.max(np.abs(blocks - reference) / scale, axis=(1, 2, 3))
        hit = normalized <= 1.0
        period[idx[hit]] = p
        residual[idx[hit]] = normalized[hit]
        residual[idx[~hit]] = np.minimum(residual[idx[~hit]], normalized[~hit])
        open_cells[idx[hit]] = False
    return period, residual


@dataclass(frozen=True)
class HenonGrid:
    """Hénon maps ``(x, y) -> (1 - a x^2 + y, b x)`` for many (a, b) pairs.

    A positive control for the shared screening path. Its fixed point flips at
    ``a = 3 (1 - b)^2 / 4`` and its two-cycle flips at
    ``a = ((1 + b)^2 + 4 (1 - b)^2) / 4``. The "loss" is identically zero.
    """

    a: np.ndarray
    b: np.ndarray

    def __post_init__(self):
        a = np.atleast_1d(np.asarray(self.a, dtype=np.float64))
        b = np.atleast_1d(np.asarray(self.b, dtype=np.float64))
        if a.shape != b.shape or a.ndim != 1 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError("a and b must be finite one-dimensional arrays of equal shape")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)

    @property
    def cells(self):
        return len(self.a)

    def step_subset(self, state, idx):
        x, y = state[:, 0], state[:, 1]
        new = np.column_stack((1.0 - self.a[idx] * x * x + y, self.b[idx] * x))
        return new, np.zeros(len(idx))

    def jvp_subset(self, state, idx, vector):
        x = state[:, 0]
        return np.column_stack((-2.0 * self.a[idx] * x * vector[:, 0] + vector[:, 1],
                                self.b[idx] * vector[:, 0]))


def iterate_and_screen(step_subset, jvp_subset, initial, steps, tail_steps=512,
                       ftle_steps=4000, ftle_align=1000, max_period=64, min_repeats=6,
                       atol=1e-8, rtol=1e-6, chaos_threshold=5e-3,
                       divergence_threshold=1e8, sd_horizons=(100, 500, 1000),
                       tangent_seed=0, screen_batch=256):
    """Iterate many independent maps and screen their final tails.

    ``step_subset(states, idx)`` returns (new_states, losses) for the cells in
    ``idx``; ``jvp_subset(states, idx, vectors)`` applies each cell's map
    Jacobian. Divergence (nonfinite state, nonfinite loss or max |state| above
    the threshold) freezes a cell. The largest finite-time Lyapunov exponent
    uses one tangent per cell, aligned for ``ftle_align`` steps and averaged
    over the final ``ftle_steps``. A collapsed tangent invalidates only the
    exponent. Classifications are candidates, not verified orbits or chaos.
    """
    theta = np.array(initial, dtype=np.float64)
    if theta.ndim != 2:
        raise ValueError("initial must have shape (cells, dimension)")
    cells, p = theta.shape
    if min(tail_steps, ftle_steps) < 1 or ftle_align < 0:
        raise ValueError("tail and FTLE windows must be positive")
    sd_max = max(sd_horizons, default=0)
    if steps < max(tail_steps, ftle_steps + ftle_align, sd_max):
        raise ValueError("steps must cover the tail, FTLE and score windows")
    alive = np.all(np.isfinite(theta), axis=1)
    diverged_at = np.where(alive, -1, 0).astype(np.int64)
    early_losses = np.full((sd_max, cells), np.nan) if sd_max else None
    tail = np.full((cells, tail_steps, p), np.nan)
    rng = np.random.default_rng(tangent_seed)
    tangent = rng.normal(size=(cells, p))
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    log_growth = np.zeros(cells)
    ftle_valid = np.ones(cells, dtype=bool)
    ftle_start = steps - ftle_steps - ftle_align
    tail_start = steps - tail_steps
    min_loss = np.full(cells, np.inf)
    tail_loss_sum = np.zeros(cells)
    tail_loss_count = np.zeros(cells, dtype=np.int64)
    last_loss = np.full(cells, np.nan)
    for step in range(steps):
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        current = theta[idx]
        with np.errstate(over="ignore", invalid="ignore"):
            new, loss = step_subset(current, idx)
            if step >= ftle_start:
                jv = jvp_subset(current, idx, tangent[idx])
        loss = np.asarray(loss, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            bad = (~np.all(np.isfinite(new), axis=1)
                   | (np.max(np.abs(new), axis=1) > divergence_threshold)
                   | ~np.isfinite(loss))
        if step < sd_max:
            early_losses[step, idx] = loss
        finite_loss = np.isfinite(loss)
        min_loss[idx[finite_loss]] = np.minimum(min_loss[idx[finite_loss]], loss[finite_loss])
        last_loss[idx] = loss
        if step >= tail_start:
            tail_loss_sum[idx] += np.where(finite_loss, loss, 0.0)
            tail_loss_count[idx] += finite_loss
        if step >= ftle_start:
            with np.errstate(over="ignore", invalid="ignore"):
                norms = np.linalg.norm(jv, axis=1)
            ok = np.isfinite(norms) & (norms > 0)
            ftle_valid[idx[~ok & ~bad]] = False
            if step >= ftle_start + ftle_align:
                log_growth[idx[ok]] += np.log(norms[ok])
            safe = np.where(ok, norms, 1.0)
            tangent[idx] = np.where(ok[:, None], jv / safe[:, None], tangent[idx])
        good = ~bad
        theta[idx[good]] = new[good]
        dead = idx[bad]
        alive[dead] = False
        diverged_at[dead] = step
        theta[dead] = np.nan
        if step >= tail_start:
            tail[idx[good], step - tail_start, :] = new[good]
    period = np.empty(cells, dtype=np.int64)
    residual = np.empty(cells)
    for start in range(0, cells, screen_batch):
        stop = min(cells, start + screen_batch)
        period[start:stop], residual[start:stop] = screen_tails(
            tail[start:stop], max_period=max_period, min_repeats=min_repeats,
            atol=atol, rtol=rtol)
    ftle = np.where(alive & ftle_valid, log_growth / ftle_steps, np.nan)
    status = np.full(cells, STATUS_UNRESOLVED, dtype=np.int64)
    status[~alive] = STATUS_DIVERGENT
    status[alive & (period == 1)] = STATUS_FIXED
    status[alive & (period >= 2)] = STATUS_PERIODIC
    status[alive & (period == 0) & (ftle > chaos_threshold)] = STATUS_CHAOTIC
    result = {
        "status": status, "period": np.where(alive, period, -1),
        "recurrence_residual": residual, "ftle": ftle, "ftle_valid": alive & ftle_valid,
        "diverged_at": diverged_at, "final_state": theta,
        "min_loss": np.where(np.isfinite(min_loss), min_loss, np.nan),
        "final_loss": np.where(alive, last_loss, np.nan),
        "tail_mean_loss": np.where(alive & (tail_loss_count == tail_steps),
                                   tail_loss_sum / np.maximum(tail_loss_count, 1), np.nan),
    }
    for horizon in sd_horizons:
        convergent, score = sohl_dickstein_measure(early_losses[:horizon])
        result[f"sd_convergent_{horizon}"] = convergent
        result[f"sd_score_{horizon}"] = score
    return result


def run_grid(grid, theta0, steps, **options):
    """Iterate every rate cell of ``grid`` from the same initial parameters.

    Keyword options are those of :func:`iterate_and_screen`.
    """
    theta0 = np.asarray(theta0, dtype=np.float64)
    p = grid.net.n_parameters
    if theta0.shape != (p,):
        raise ValueError(f"theta0 must have shape ({p},)")

    def step_subset(states, idx):
        grad, loss = grid.net.gradient(states, grid.x, grid.y, return_loss=True)
        return states - grid.rates[idx] * grad, loss

    def jvp_subset(states, idx, vectors):
        return vectors - grid.rates[idx] * grid.net.hvp(states, grid.x, grid.y, vectors)

    initial = np.repeat(theta0[None, :], grid.cells, axis=0)
    return iterate_and_screen(step_subset, jvp_subset, initial, steps, **options)


def connected_components(labels, mask):
    """8-connected components of equal labels inside ``mask`` on a 2-D grid.

    Returns an integer array with -1 outside components and component ids
    numbered by first appearance in row-major order.
    """
    labels = np.asarray(labels)
    mask = np.asarray(mask, dtype=bool)
    if labels.shape != mask.shape or labels.ndim != 2:
        raise ValueError("labels and mask must be equal 2-D arrays")
    rows, cols = labels.shape
    out = np.full(labels.shape, -1, dtype=np.int64)
    current = 0
    for r0 in range(rows):
        for c0 in range(cols):
            if not mask[r0, c0] or out[r0, c0] >= 0:
                continue
            value = labels[r0, c0]
            stack = [(r0, c0)]
            out[r0, c0] = current
            while stack:
                r, c = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if (0 <= rr < rows and 0 <= cc < cols and mask[rr, cc]
                                and out[rr, cc] < 0 and labels[rr, cc] == value):
                            out[rr, cc] = current
                            stack.append((rr, cc))
            current += 1
    return out

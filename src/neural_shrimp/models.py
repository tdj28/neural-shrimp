"""Small, deterministic neural-network training maps in float64.

The iterated state is the complete flattened parameter vector, including
biases. Two learning rates are controls of the map, not state coordinates.
All gradients are evaluated simultaneously on a fixed full batch.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionDataset:
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    metadata: dict


def make_teacher_regression(
    seed=0, d=2, teacher_width=3, n_train=32, n_validation=64, n_test=128,
    noise_std=0.0,
):
    """Generate independent IID splits using one fixed random tanh teacher.

    Features are standard normal; labels use the same teacher in all splits.
    Split-specific RNG streams prevent changing the training sample count from
    silently changing validation/test examples. No split-derived normalization
    is applied. ``metadata`` records generator choices and teacher parameters.
    """
    if d < 1 or teacher_width < 1 or min(n_train, n_validation, n_test) < 1:
        raise ValueError("Dimensions and split sizes must be positive")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    streams = np.random.SeedSequence(seed).spawn(4)
    teacher_rng = np.random.default_rng(streams[0])
    w = teacher_rng.normal(size=(d, teacher_width)) / np.sqrt(d)
    b = teacher_rng.normal(scale=0.3, size=teacher_width)
    v = teacher_rng.normal(size=teacher_width) / np.sqrt(teacher_width)
    c = float(teacher_rng.normal(scale=0.1))

    def split(stream, count):
        rng = np.random.default_rng(stream)
        x = rng.normal(size=(count, d)).astype(np.float64)
        y = np.tanh(x @ w + b) @ v + c
        y = y + noise_std * rng.normal(size=count)
        return x, y.astype(np.float64)

    train_x, train_y = split(streams[1], n_train)
    validation_x, validation_y = split(streams[2], n_validation)
    test_x, test_y = split(streams[3], n_test)
    return RegressionDataset(
        train_x, train_y, validation_x, validation_y, test_x, test_y,
        {"seed": int(seed), "d": int(d), "teacher_width": int(teacher_width),
         "n_train": int(n_train), "n_validation": int(n_validation),
         "n_test": int(n_test), "noise_std": float(noise_std),
         "teacher_w": w.tolist(), "teacher_b": b.tolist(),
         "teacher_v": v.tolist(), "teacher_c": c,
         "distribution": "IID standard-normal features; tanh teacher"},
    )


@dataclass(frozen=True)
class TanhRegressor:
    """One hidden tanh layer and scalar linear output.

    Prediction is ``tanh(x @ W1 + b1) @ w2 + b2``. Parameter order is
    row-major W1, b1, w2, b2. Training loss is half mean squared error plus
    ``ridge / 2 * ||theta||^2``; ridge includes biases. Report predictive MSE
    separately when comparing task utility. Unregularized models may have
    neutral directions; recurrence alone does not establish attraction.
    """

    d: int
    width: int
    ridge: float = 0.0

    def __post_init__(self):
        if self.d < 1 or self.width < 1:
            raise ValueError("d and width must be positive")
        if not np.isfinite(self.ridge) or self.ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")

    @property
    def n_parameters(self):
        return self.d * self.width + 2 * self.width + 1

    @property
    def hidden_parameter_count(self):
        return self.d * self.width + self.width

    def unpack(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.n_parameters,):
            raise ValueError(f"Expected parameter shape {(self.n_parameters,)}")
        a = self.d * self.width
        b = a + self.width
        return theta[:a].reshape(self.d, self.width), theta[a:b], theta[b:-1], theta[-1]

    def initialize(self, seed=0, scale=0.2):
        """Deterministic Gaussian initialization with zero biases."""
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("scale must be finite and nonnegative")
        rng = np.random.default_rng(seed)
        w = rng.normal(scale=scale / np.sqrt(self.d), size=(self.d, self.width))
        v = rng.normal(scale=scale / np.sqrt(self.width), size=self.width)
        return np.concatenate((w.ravel(), np.zeros(self.width), v, [0.0]))

    def _validate_data(self, x, y=None):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.d or x.shape[0] < 1:
            raise ValueError(f"Expected nonempty x with shape (n, {self.d})")
        if y is None:
            return x
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (len(x),):
            raise ValueError("Expected one scalar target per example")
        return x, y

    def predict(self, theta, x):
        x = self._validate_data(x)
        w, b, v, c = self.unpack(theta)
        return np.tanh(x @ w + b) @ v + c

    def loss(self, theta, x, y):
        x, y = self._validate_data(x, y)
        residual = self.predict(theta, x) - y
        return float(0.5 * np.mean(residual ** 2) + 0.5 * self.ridge * np.dot(theta, theta))

    def gradient(self, theta, x, y):
        """Analytic gradient of half-MSE plus ridge on the full batch."""
        x, y = self._validate_data(x, y)
        w, b, v, c = self.unpack(theta)
        h = np.tanh(x @ w + b)
        residual = h @ v + c - y
        back = residual[:, None] * v * (1.0 - h * h)
        n = len(x)
        grad = np.concatenate(((x.T @ back / n).ravel(), back.mean(axis=0),
                               h.T @ residual / n, [residual.mean()]))
        return grad + self.ridge * np.asarray(theta, dtype=np.float64)

    def hessian(self, theta, x, y):
        """Exact analytic Hessian, including the residual curvature term.

        This is not a Gauss-Newton approximation. It supports differentiating
        the actual training map for Floquet multipliers and Lyapunov estimates.
        """
        x, y = self._validate_data(x, y)
        w, b, v, c = self.unpack(theta)
        h = np.tanh(x @ w + b)
        s = 1.0 - h * h
        second = -2.0 * h * s
        residual = h @ v + c - y
        n = len(x)
        j_hidden = (x[:, :, None] * (s * v)[:, None, :]).reshape(n, -1)
        j = np.column_stack((j_hidden, s * v, h, np.ones(n)))
        result = j.T @ j / n + self.ridge * np.eye(self.n_parameters)
        xaug = np.column_stack((x, np.ones(n)))
        for unit in range(self.width):
            ids = np.r_[np.arange(self.d) * self.width + unit,
                        self.d * self.width + unit]
            out_id = self.hidden_parameter_count + unit
            curvature = residual * v[unit] * second[:, unit]
            result[np.ix_(ids, ids)] += xaug.T @ (curvature[:, None] * xaug) / n
            cross = xaug.T @ (residual * s[:, unit]) / n
            result[ids, out_id] += cross
            result[out_id, ids] += cross
        return result

    def training_map(self, x, y, lr_hidden, lr_output):
        return TrainingMap(self, x, y, lr_hidden, lr_output)


@dataclass(frozen=True)
class TrainingMap:
    """Autonomous simultaneous full-batch gradient descent with two rates."""

    model: TanhRegressor
    x: np.ndarray
    y: np.ndarray
    lr_hidden: float
    lr_output: float

    def __post_init__(self):
        if not all(np.isfinite(v) and v >= 0 for v in (self.lr_hidden, self.lr_output)):
            raise ValueError("Learning rates must be finite and nonnegative")
        x, y = self.model._validate_data(self.x, self.y)
        x, y = x.copy(), y.copy()
        x.setflags(write=False)
        y.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    @property
    def rates(self):
        return np.r_[np.full(self.model.hidden_parameter_count, self.lr_hidden),
                     np.full(self.model.width + 1, self.lr_output)]

    def __call__(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        return theta - self.rates * self.model.gradient(theta, self.x, self.y)

    def jacobian(self, theta):
        return np.eye(self.model.n_parameters) - self.rates[:, None] * self.model.hessian(
            theta, self.x, self.y)

"""Fixed restart comparison controls and durable storage; no model evaluation here."""
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import time
import numpy as np

STORE_BYTES = 31 * 1024**2
LAUNCHER_BYTES = 1024**2
RSS_BYTES = 2 * 1024**3
INTERNAL_SECONDS = 45.
STOP_REQUESTED = False

def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True

CAPS = dict(action=202, j2vp=203, hvp=407, direct_gradient=11)
SETTINGS = dict(restart=200, cycles=1, rtol=1e-10, forcing_limit=.5,
    max_matvec=202, explicit_zero_start=True, alphas=[1., .5, .25, .125],
    armijo=1e-4, closure_tolerance=1e-8, primitive_separation=1e-7,
    internal_seconds=INTERNAL_SECONDS, operation_caps=CAPS,
    store_bytes=STORE_BYTES, launcher_bytes=LAUNCHER_BYTES,
    rss_bytes=RSS_BYTES, rss_enforcement='monitored, not kernel-hard',
    prefix_rtol=1e-10, prefix_atol=1e-12, prefix_length=50)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def entry(path):
    data = Path(path).read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def byte_equal(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order='C') == b.tobytes(order='C')


class Store:
    """Exclusive fsynced files; one MiB is unavailable until finalization."""
    def __init__(self, root, limit=STORE_BYTES, reserve=1024**2):
        self.root = Path(root)
        self.limit, self.reserve = limit, reserve
        require(0 < reserve < limit, 'invalid storage reserve')
        self.root.mkdir(parents=True, exist_ok=False)
        self.used = 0

    def write(self, name, data, final=False):
        require(Path(name).name == name, 'unsafe output name')
        require(self.used + len(data) <= self.limit - (0 if final else self.reserve), 'scientific storage cap')
        path = self.root / name
        with path.open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self.used += len(data)
        fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return entry(path)

    def json(self, name, value, final=False):
        return self.write(name, canonical(value), final)

    def array(self, name, value):
        stream = io.BytesIO()
        np.save(stream, np.asarray(value), allow_pickle=False)
        return self.write(name + '.npy', stream.getvalue())

    def event(self, value):
        data = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
        require(self.used + len(data) <= self.limit - self.reserve, 'scientific storage cap')
        with (self.root / 'events.jsonl').open('ab') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self.used += len(data)

    def finish(self, result):
        self.json('result.json', result, final=True)
        self.json('manifest.json', dict(schema='cifar-restart-manifest-v1',
            files={p.name: entry(p) for p in sorted(self.root.iterdir())}), final=True)


class Budget:
    """Counts attempted operations, including unsuccessful calls; records returns."""
    def __init__(self, event, seconds=INTERNAL_SECONDS):
        self.deadline = time.monotonic() + seconds
        self.event = event
        self.attempted = dict.fromkeys(CAPS, 0)
        self.returned = dict.fromkeys(CAPS, 0)

    def check(self):
        if STOP_REQUESTED:
            raise InterruptedError('SIGTERM observed at durable operation boundary')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('restart diagnostic internal deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > RSS_BYTES:
            raise MemoryError('monitored self RSS limit')

    def call(self, kind, function, *args, **kwargs):
        self.check()
        require(self.attempted[kind] < CAPS[kind], kind + ' operation cap')
        self.attempted[kind] += 1
        started = time.monotonic()
        # Persist attempted count before entering an operation that may fail.
        self.event(dict(kind='operation_started', operation=kind, index=self.attempted[kind]))
        result = function(*args, **kwargs)
        self.returned[kind] += 1
        self.event(dict(kind='operation_returned', operation=kind, index=self.returned[kind],
                        seconds=time.monotonic() - started))
        return result


class CountedProblemMixin:
    """Works with the source-bound Problem; internal HVP grad is not a direct grad."""
    def grad(self, theta):
        if getattr(self, '_hvp_depth', 0):
            return super().grad(theta)
        return self.budget.call('direct_gradient', super().grad, theta)

    def hvp(self, theta, vector):
        def operation():
            self._hvp_depth = getattr(self, '_hvp_depth', 0) + 1
            try:
                return super(CountedProblemMixin, self).hvp(theta, vector)
            finally:
                self._hvp_depth -= 1
        return self.budget.call('hvp', operation)

    def jvp_two_step(self, *args, **kwargs):
        return self.budget.call('j2vp', super().jvp_two_step, *args, **kwargs)


class Prefix:
    def __init__(self, baseline, event):
        values = np.asarray(baseline, dtype=float)
        require(values.shape == (200,) and np.isfinite(values).all() and (values >= 0).all(), 'baseline history shape/finite')
        self.baseline, self.event, self.values = values, event, []

    def __call__(self, row):
        self.event(row)  # Retain even the callback that fails the control.
        if row['kind'] == 'gmres_inner':
            index = len(self.values)
            self.values.append(row['estimated_relative_residual'])
            require(row['iteration'] == index + 1 and len(self.values) <= 200, 'callback index/cap')
            if index < 50:
                require(np.isclose(self.values[-1], self.baseline[index], rtol=1e-10, atol=1e-12), 'first-50 prefix mismatch')

    def complete(self):
        require(len(self.values) >= 50, 'fewer than 50 matched callbacks; comparison unqualified')

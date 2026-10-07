"""Optional float64 CUDA backend for :func:`neural_shrimp.gdmaps.run_grid`.

An engineering candidate for TwoLayerNet learning-rate grids, not the
scientific default: the NumPy engine in :mod:`neural_shrimp.gdmaps` stays the
reference. One GPU thread advances one grid cell, reproducing the CPU
arithmetic operation by operation except tanh and log, which come from the CUDA
math library (see ``native/gdmap_cuda.cu`` and
``docs/gdmap-cuda-qualification.md``). Importing this module performs no CUDA
activity; build the shared library explicitly with :func:`build` or
``tools/build_gdmap_cuda.py`` and pass its path. Do not fork a process after it
has initialized CUDA.

:func:`run_grid_cuda` accepts the options of
:func:`neural_shrimp.gdmaps.iterate_and_screen` and returns the same keys,
dtypes and shapes as :func:`neural_shrimp.gdmaps.run_grid`.
"""

from __future__ import annotations

import ctypes
import hashlib
import inspect
import json
import operator
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import numpy as np

from .gdmaps import (
    STATUS_CHAOTIC, STATUS_DIVERGENT, STATUS_DRIFT, STATUS_FIXED, STATUS_PERIODIC,
    STATUS_UNRESOLVED, GridMap, TwoLayerNet, iterate_and_screen, screen_tails, sohl_dickstein_measure,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native" / "gdmap_cuda.cu"
LIBRARY_NAME = "gdmap_cuda.so"
RECEIPT_NAME = "gdmap_cuda-build.json"
LIBRARY_ENV = "NEURAL_SHRIMP_GDMAP_CUDA_LIBRARY"
D_MAX = 16
W_MAX = 64
KCHUNK_MAX = 16
SHARED_DOUBLES = 6144
DEFAULT_MAX_DEVICE_BYTES = 2 * 1024**3
DEFAULT_CHUNK_CELLS = 16384
DEFAULT_LAUNCH_SECONDS = 1.0
# OpenBLAS level-3 driver blocking of the reduction dimension for the dgemm
# kernel selected on the reference host (DYNAMIC_ARCH Haswell): GEMM_Q, UNROLL_M.
OPENBLAS_GEMM_Q = 256
OPENBLAS_UNROLL_M = 4

COMMON_FLAGS = ("-O3", "-std=c++17", "--shared", "-Xcompiler=-fPIC", "--ftz=false",
                "--prec-div=true", "--prec-sqrt=true", "-arch=sm_89", "-Xptxas=-v")
DEFAULT_ARCH = "sm_89"
# Compute capabilities on which the mirror build has passed qualification.
QUALIFIED_CAPABILITIES = ((8, 9),)
# Qualification runs on new hardware may name extra capabilities explicitly,
# e.g. NEURAL_SHRIMP_CUDA_QUALIFYING=9.0; such runs are labelled in metadata.
QUALIFYING_ENV = "NEURAL_SHRIMP_CUDA_QUALIFYING"


def qualified_capabilities():
    """Qualified compute capabilities plus any explicitly named for qualification."""
    extra = []
    for token in filter(None, (t.strip() for t in os.environ.get(QUALIFYING_ENV, "").split(","))):
        parts = token.split(".")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(f"{QUALIFYING_ENV} entries must look like 9.0, got {token!r}")
        extra.append((int(parts[0]), int(parts[1])))
    return tuple(dict.fromkeys(QUALIFIED_CAPABILITIES + tuple(extra)))
VARIANTS = {
    # Qualified default: explicit intrinsics, no compiler contraction.
    "mirror": COMMON_FLAGS + ("--fmad=false",),
    # Diagnostic only: plain operators and free multiply-add contraction.
    "contract": COMMON_FLAGS + ("--fmad=true", "-DGDMAP_FREE_CONTRACTION=1"),
}

_SIGNATURE = inspect.signature(iterate_and_screen)
OPTION_DEFAULTS = {name: parameter.default for name, parameter in _SIGNATURE.parameters.items()
                   if parameter.default is not inspect.Parameter.empty}


class GdmapCudaError(RuntimeError):
    """A CUDA or native-library failure, with its numeric code when known."""

    def __init__(self, message, *, code=None):
        super().__init__(message)
        self.code = code


class _Config(ctypes.Structure):
    _fields_ = ([(name, ctypes.c_int64) for name in (
        "n", "d", "width", "bias", "tanh_act", "capacity", "tail_capacity", "sd_max",
        "tail_steps", "device", "kchunks")]
        + [("kchunk_end", ctypes.c_int64 * KCHUNK_MAX), ("max_bytes", ctypes.c_uint64)]
        + [(name, ctypes.c_double) for name in (
            "in_scale", "out_scale", "loss_scale", "ridge", "k_scale", "half_ridge")])


class _Schedule(ctypes.Structure):
    _fields_ = ([(name, ctypes.c_int64) for name in (
        "ftle_start", "ftle_log_start", "tail_start", "sd_max", "norm_start")]
        + [("divergence_threshold", ctypes.c_double)])


class _Info(ctypes.Structure):
    _fields_ = ([("name", ctypes.c_char * 256)]
                + [(name, ctypes.c_int64) for name in (
                    "device", "device_count", "major", "minor", "multiprocessors",
                    "runtime_version", "driver_version", "l2_bytes", "double_ratio",
                    "clock_khz")]
                + [("total_bytes", ctypes.c_uint64), ("free_bytes", ctypes.c_uint64)])


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def openblas_k_blocks(k, gemm_q=OPENBLAS_GEMM_Q, unroll_m=OPENBLAS_UNROLL_M):
    """Cumulative block ends of OpenBLAS's level-3 loop over a reduction of length k."""
    ends, start = [], 0
    while start < k:
        size = k - start
        if size >= 2 * gemm_q:
            size = gemm_q
        elif size > gemm_q:
            size = ((size // 2 + unroll_m - 1) // unroll_m) * unroll_m
        start += size
        ends.append(start)
    return ends


def nvidia_driver_version():
    """NVIDIA kernel-module version string, or None when unavailable."""
    try:
        text = Path("/proc/driver/nvidia/version").read_text()
    except OSError:
        return None
    for token in text.split():
        parts = token.split(".")
        if len(parts) >= 2 and all(part.isdigit() for part in parts):
            return token
    return None


def build(directory, *, variant="mirror", compiler="nvcc", host_compiler="g++-12", timeout=900,
          arch=DEFAULT_ARCH):
    """Compile the shared library into ``directory`` and write a build receipt.

    Refuses to overwrite an existing library or receipt. Compilation only: no
    CUDA context is created. Returns the receipt as a dictionary.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
    if not re.fullmatch(r"sm_\d{2,3}", arch):
        raise ValueError(f"arch must look like sm_89, got {arch!r}")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / LIBRARY_NAME
    receipt = directory / RECEIPT_NAME
    if library.exists() or receipt.exists():
        raise FileExistsError(f"CUDA build output already exists in {directory}")
    nvcc = shutil.which(compiler)
    if not nvcc:
        raise FileNotFoundError(f"CUDA compiler unavailable: {compiler}")
    host = shutil.which(host_compiler)
    if not host:
        raise FileNotFoundError(f"Supported host compiler unavailable: {host_compiler}")
    flags = [f"-arch={arch}" if flag == f"-arch={DEFAULT_ARCH}" else flag for flag in VARIANTS[variant]]
    command = [nvcc, *flags, "-ccbin", host, str(SOURCE), "-o", str(library)]
    compiler_version = subprocess.check_output([nvcc, "--version"], text=True, timeout=30)
    host_version = subprocess.check_output([host, "--version"], text=True, timeout=30)
    started = time.monotonic()
    # nvcc starts child compilers: stop the owned process group on timeout.
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    interrupted = None
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException as error:  # also KeyboardInterrupt
        interrupted = error
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    record = dict(
        variant=variant, command=command, source=str(SOURCE), source_sha256=_sha256(SOURCE),
        compiler=nvcc, compiler_version=compiler_version, host_compiler=host,
        host_compiler_version=host_version, returncode=process.returncode, stdout=stdout,
        stderr=stderr, interruption=type(interrupted).__name__ if interrupted else None,
        wall_seconds=time.monotonic() - started, architecture=arch, dtype="float64",
        fmad="--fmad=true" in command, free_contraction=variant == "contract",
        fast_math=False, ftz=False, cudart="static", gpu_initialized=False,
        library=str(library), library_sha256=_sha256(library) if library.exists() else None)
    with receipt.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if interrupted:
        raise interrupted
    if process.returncode:
        raise GdmapCudaError(f"CUDA compilation failed; receipt: {receipt}", code=process.returncode)
    return dict(record, receipt=str(receipt))


def _resolve_options(steps, options):
    unknown = sorted(set(options) - set(OPTION_DEFAULTS))
    if unknown:
        raise TypeError(f"unexpected keyword argument(s): {', '.join(unknown)}")
    opts = dict(OPTION_DEFAULTS)
    opts.update(options)
    steps = operator.index(steps)
    # tangent_seed goes to numpy.random.default_rng unchanged, as on the CPU.
    for name in ("tail_steps", "ftle_steps", "ftle_align", "max_period", "min_repeats",
                 "screen_batch"):
        opts[name] = operator.index(opts[name])
    opts["sd_horizons"] = tuple(operator.index(h) for h in opts["sd_horizons"])
    for name in ("atol", "rtol", "chaos_threshold", "divergence_threshold", "drift_threshold"):
        opts[name] = float(opts[name])
    # The same checks as iterate_and_screen, in the same order ...
    if min(opts["tail_steps"], opts["ftle_steps"]) < 1 or opts["ftle_align"] < 0:
        raise ValueError("tail and FTLE windows must be positive")
    sd_max = max(opts["sd_horizons"], default=0)
    if steps < max(opts["tail_steps"], opts["ftle_steps"] + opts["ftle_align"], sd_max):
        raise ValueError("steps must cover the tail, FTLE and score windows")
    # ... and, before any work, the checks the CPU path makes only afterwards.
    if (opts["max_period"] < 1 or opts["min_repeats"] < 2 or not opts["atol"] > 0
            or not opts["rtol"] >= 0):
        raise ValueError("invalid recurrence settings")
    if any(h < 20 for h in opts["sd_horizons"]):
        raise ValueError("losses must have shape (steps >= 20, cells)")
    if opts["screen_batch"] < 1:
        raise ValueError("screen_batch must be positive")
    if steps > 2**30:
        raise ValueError("steps exceeds the CUDA backend limit of 2**30")
    return steps, opts


def _check_network(net, n):
    if not isinstance(net, TwoLayerNet):
        raise TypeError("the CUDA backend supports TwoLayerNet grids only")
    if not (1 <= net.d <= D_MAX and 1 <= net.width <= W_MAX and n * (net.d + 1) <= SHARED_DOUBLES):
        raise ValueError(f"unsupported size: need d <= {D_MAX}, width <= {W_MAX} and "
                         f"n * (d + 1) <= {SHARED_DOUBLES}")
    if len(openblas_k_blocks(n)) > KCHUNK_MAX:
        raise ValueError("too many samples for the K-block table")


def _config(net, n, *, capacity, tail_capacity, sd_max, tail_steps, device, max_bytes):
    cfg = _Config()
    cfg.n, cfg.d, cfg.width = n, net.d, net.width
    cfg.bias = int(bool(net.bias))
    cfg.tanh_act = int(net.activation == "tanh")
    cfg.capacity, cfg.tail_capacity = capacity, tail_capacity
    cfg.sd_max, cfg.tail_steps, cfg.device = sd_max, tail_steps, device
    ends = openblas_k_blocks(n)
    cfg.kchunks = len(ends)
    for i, end in enumerate(ends):
        cfg.kchunk_end[i] = end
    cfg.max_bytes = int(max_bytes)
    cfg.in_scale = float(net.in_scale)
    cfg.out_scale = float(net.out_scale)
    cfg.loss_scale = float(net.loss_scale)
    cfg.ridge = float(net.ridge)
    # Exactly the expressions evaluated by TwoLayerNet.gradient/hvp/loss.
    cfg.k_scale = float(2.0 * net.loss_scale / n)
    cfg.half_ridge = float(0.5 * net.ridge)
    return cfg


def _pointer(array, ctype):
    return array.ctypes.data_as(ctypes.POINTER(ctype))


class GdmapCuda:
    """A loaded backend library."""

    def __init__(self, library):
        self.path = Path(library).resolve()
        self.library_sha256 = _sha256(self.path)
        receipt = self.path.with_name(RECEIPT_NAME)
        self.build_receipt = json.loads(receipt.read_text()) if receipt.exists() else None
        lib = ctypes.CDLL(str(self.path))
        dp = ctypes.POINTER(ctypes.c_double)
        ip = ctypes.POINTER(ctypes.c_int)
        lp = ctypes.POINTER(ctypes.c_longlong)
        vp = ctypes.c_void_p
        err = [ctypes.c_char_p, ctypes.c_size_t]
        signatures = {
            "gdmap_cuda_device_info": ([ctypes.c_int, ctypes.POINTER(_Info)] + err, ctypes.c_int),
            "gdmap_cuda_planned_bytes": ([ctypes.POINTER(_Config)], ctypes.c_uint64),
            "gdmap_cuda_create": ([ctypes.POINTER(_Config), dp, dp, ctypes.POINTER(vp)] + err,
                                  ctypes.c_int),
            "gdmap_cuda_allocated_bytes": ([vp], ctypes.c_uint64),
            "gdmap_cuda_destroy": ([vp], ctypes.c_int),
            "gdmap_cuda_upload": ([vp, ctypes.c_int64, dp, dp, dp, dp, ip, lp] + err, ctypes.c_int),
            "gdmap_cuda_reset_tail": ([vp] + err, ctypes.c_int),
            "gdmap_cuda_advance": ([vp, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                    ctypes.c_int64, ctypes.POINTER(_Schedule), ctypes.c_int64,
                                    ctypes.c_int64, dp] + err, ctypes.c_int),
            "gdmap_cuda_take_tail": ([vp, dp] + err, ctypes.c_int),
            "gdmap_cuda_screen": ([vp, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                   ctypes.c_double, ctypes.c_double, lp, dp, dp] + err,
                                  ctypes.c_int),
            "gdmap_cuda_fetch": ([vp, dp, ip, lp, ip, dp, dp, dp, dp, lp, dp, dp] + err,
                                 ctypes.c_int),
            "gdmap_cuda_evaluate": ([ctypes.POINTER(_Config), dp, dp, ctypes.c_int64, dp, dp, dp,
                                     dp, dp] + err, ctypes.c_int),
            "gdmap_cuda_math": ([ctypes.c_int, ctypes.c_int64, dp, dp] + err, ctypes.c_int),
            "gdmap_cuda_screen_array": ([ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, dp,
                                         ctypes.c_int64, ctypes.c_int64, ctypes.c_double,
                                         ctypes.c_double, lp, dp] + err, ctypes.c_int),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(lib, name)
            function.argtypes = argtypes
            function.restype = restype
        self._lib = lib

    def _call(self, name, *args):
        error = ctypes.create_string_buffer(1024)
        code = getattr(self._lib, name)(*args, error, len(error))
        if code:
            message = error.value.decode(errors="replace") or f"{name} failed"
            raise GdmapCudaError(message, code=code)

    def device_info(self, device=0):
        info = _Info()
        self._call("gdmap_cuda_device_info", int(device), ctypes.byref(info))
        result = {name: getattr(info, name) for name, _ in _Info._fields_}
        result["name"] = result["name"].decode(errors="replace")
        return result

    def metadata(self, device=0):
        """Device, driver, compiler and library identity."""
        receipt = self.build_receipt or {}
        version = receipt.get("compiler_version", "")
        return dict(
            library=str(self.path), library_sha256=self.library_sha256,
            variant=receipt.get("variant"), build_command=receipt.get("command"),
            source_sha256=receipt.get("source_sha256"),
            nvcc_version=next((line for line in version.splitlines() if "release" in line), None),
            host_compiler_version=(receipt.get("host_compiler_version") or "").split("\n")[0] or None,
            nvidia_driver=nvidia_driver_version(), device=self.device_info(device))

    def math(self, function, values, *, device=0):
        """The device tanh or log used by the kernels, applied elementwise."""
        which = {"tanh": 0, "log": 1}[function]
        values = np.ascontiguousarray(values, dtype=np.float64).ravel()
        out = np.empty_like(values)
        self.device_info(device)
        self._call("gdmap_cuda_math", which, len(values), _pointer(values, ctypes.c_double),
                   _pointer(out, ctypes.c_double))
        return out

    def screen_tails(self, tail, max_period=64, min_repeats=6, atol=1e-8, rtol=1e-6, *, device=0):
        """Device counterpart of :func:`neural_shrimp.gdmaps.screen_tails`."""
        tail = np.asarray(tail, dtype=np.float64)
        if tail.ndim != 3:
            raise ValueError("tail must have shape (cells, steps, parameters)")
        if max_period < 1 or min_repeats < 2 or atol <= 0 or rtol < 0:
            raise ValueError("invalid recurrence settings")
        cells, steps, p = tail.shape
        soa = np.ascontiguousarray(tail.transpose(1, 2, 0))
        period = np.empty(cells, dtype=np.longlong)
        residual = np.empty(cells)
        self.device_info(device)
        self._call("gdmap_cuda_screen_array", cells, steps, p, _pointer(soa, ctypes.c_double),
                   int(max_period), int(min_repeats), float(atol), float(rtol),
                   _pointer(period, ctypes.c_longlong), _pointer(residual, ctypes.c_double))
        return period.astype(np.int64), residual

    def evaluate(self, net, x, y, theta, vectors=None, *, device=0,
                 max_device_bytes=DEFAULT_MAX_DEVICE_BYTES):
        """Loss, gradient and optional HVP per row, as TwoLayerNet computes them."""
        x, y = net._validate_data(x, y)
        _check_network(net, len(x))
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 2 or theta.shape[1] != net.n_parameters:
            raise ValueError(f"Expected parameters with shape (cells, {net.n_parameters})")
        cells, p = theta.shape
        if vectors is not None:
            vectors = np.asarray(vectors, dtype=np.float64)
            if vectors.shape != theta.shape:
                raise ValueError("vector must have the same shape as theta")
        cfg = _config(net, len(x), capacity=cells, tail_capacity=1, sd_max=0, tail_steps=1,
                      device=int(device), max_bytes=max_device_bytes)
        self.device_info(device)  # selects the device
        theta_soa = np.ascontiguousarray(theta.T)
        vector_soa = np.ascontiguousarray(vectors.T) if vectors is not None else None
        loss = np.empty(cells)
        grad = np.empty((p, cells))
        hvp = np.empty((p, cells)) if vectors is not None else None
        dp = ctypes.c_double
        self._call("gdmap_cuda_evaluate", ctypes.byref(cfg), _pointer(np.ascontiguousarray(x), dp),
                   _pointer(np.ascontiguousarray(y), dp), cells, _pointer(theta_soa, dp),
                   _pointer(vector_soa, dp) if vector_soa is not None else None,
                   _pointer(loss, dp), _pointer(grad, dp),
                   _pointer(hvp, dp) if hvp is not None else None)
        return dict(loss=loss, grad=np.ascontiguousarray(grad.T),
                    hvp=np.ascontiguousarray(hvp.T) if hvp is not None else None)

    def run_grid(self, grid, theta0, steps, *, device=0, max_device_bytes=DEFAULT_MAX_DEVICE_BYTES,
                 chunk_cells=None, tail_cells=None, launch_steps=None, screen="device",
                 return_metadata=False, **options):
        """GPU counterpart of :func:`neural_shrimp.gdmaps.run_grid`.

        ``screen="device"`` screens tails on the GPU with a bitwise mirror of
        :func:`screen_tails`; ``screen="host"`` copies tails back and calls the
        CPU function itself. ``chunk_cells``/``tail_cells`` bound device memory
        and ``launch_steps`` fixes the steps per kernel launch (adaptive by
        default, about one second each). None of these change any result.
        """
        began = time.monotonic()
        if not isinstance(grid, GridMap):
            raise TypeError("grid must be a GridMap")
        net = grid.net
        n = len(grid.x)
        _check_network(net, n)
        p = net.n_parameters
        theta0 = np.asarray(theta0, dtype=np.float64)
        if theta0.shape != (p,):
            raise ValueError(f"theta0 must have shape ({p},)")
        steps, opts = _resolve_options(steps, options)
        if screen not in ("device", "host"):
            raise ValueError("screen must be 'device' or 'host'")
        cells = grid.cells
        tail_steps, ftle_steps, ftle_align = opts["tail_steps"], opts["ftle_steps"], opts["ftle_align"]
        sd_horizons = opts["sd_horizons"]
        sd_max = max(sd_horizons, default=0)
        ftle_start = steps - ftle_steps - ftle_align
        tail_start = steps - tail_steps
        schedule = _Schedule(ftle_start=ftle_start, ftle_log_start=ftle_start + ftle_align,
                             tail_start=tail_start, sd_max=sd_max, norm_start=steps - ftle_steps,
                             divergence_threshold=opts["divergence_threshold"])
        info = self.device_info(device)
        capability = (info["major"], info["minor"])
        if capability not in qualified_capabilities():
            raise ValueError(f"this backend is qualified only for compute capabilities "
                             f"{QUALIFIED_CAPABILITIES}; set {QUALIFYING_ENV} for a qualification run")
        qualifying_run = capability not in QUALIFIED_CAPABILITIES
        budget = min(int(max_device_bytes), int(info["free_bytes"]) // 2)

        def planned(capacity, tail_capacity):
            cfg = _config(net, n, capacity=capacity, tail_capacity=tail_capacity, sd_max=sd_max,
                          tail_steps=tail_steps, device=device, max_bytes=max_device_bytes)
            return int(self._lib.gdmap_cuda_planned_bytes(ctypes.byref(cfg)))

        capacity = int(chunk_cells) if chunk_cells else min(cells, DEFAULT_CHUNK_CELLS)
        capacity = max(1, min(capacity, cells))
        tail_capacity = max(1, min(int(tail_cells) if tail_cells else capacity, capacity))
        if not (chunk_cells and tail_cells):
            while planned(capacity, tail_capacity) > budget and tail_capacity > 1:
                tail_capacity = max(1, tail_capacity // 2)
            while planned(capacity, tail_capacity) > budget and capacity > 1 and not chunk_cells:
                capacity = max(1, capacity // 2)
                tail_capacity = min(tail_capacity, capacity)
        required = planned(capacity, tail_capacity)
        if required > budget:
            raise ValueError(f"chunk needs {required} device bytes, above the budget {budget}")
        # Balanced chunks; never a lone single-cell chunk in a multi-cell grid.
        n_chunks = -(-cells // capacity)
        bounds = np.linspace(0, cells, n_chunks + 1).round().astype(np.int64)

        # Outputs, assembled exactly as iterate_and_screen does.
        alive = np.zeros(cells, dtype=bool)
        diverged_at = np.zeros(cells, dtype=np.int64)
        ftle_valid = np.zeros(cells, dtype=bool)
        log_growth = np.zeros(cells)
        min_loss = np.zeros(cells)
        last_loss = np.zeros(cells)
        tail_loss_sum = np.zeros(cells)
        tail_loss_count = np.zeros(cells, dtype=np.int64)
        window_norm = np.zeros(cells)
        theta = np.empty((cells, p))
        period = np.empty(cells, dtype=np.int64)
        residual = np.empty(cells)
        sd = {h: (np.zeros(cells, dtype=bool), np.zeros(cells)) for h in sd_horizons}

        rng = np.random.default_rng(opts["tangent_seed"])
        rates = grid.rates
        lr_h_all, lr_o_all = rates[:, 0].copy(), rates[:, -1].copy()
        initially_alive = bool(np.all(np.isfinite(theta0)))
        kernel_seconds = screen_seconds = transfer_seconds = 0.0
        launches = 0
        per_step_estimate = None  # seconds per cell-step, from measured launches
        target = DEFAULT_LAUNCH_SECONDS
        x = np.ascontiguousarray(grid.x)
        y = np.ascontiguousarray(grid.y)
        cfg = _config(net, n, capacity=capacity, tail_capacity=tail_capacity, sd_max=sd_max,
                      tail_steps=tail_steps, device=device, max_bytes=max_device_bytes)
        handle = ctypes.c_void_p()
        self._call("gdmap_cuda_create", ctypes.byref(cfg), _pointer(x, ctypes.c_double),
                   _pointer(y, ctypes.c_double), ctypes.byref(handle))
        allocated = int(self._lib.gdmap_cuda_allocated_bytes(handle))

        def advance(begin, count, s0, s1, write_tail, tail_begin):
            nonlocal kernel_seconds, launches, per_step_estimate
            step = s0
            while step < s1:
                if launch_steps:
                    span = int(launch_steps)
                elif per_step_estimate is None:
                    span = 8
                else:
                    span = int(target / max(per_step_estimate * count, 1e-12))
                span = max(1, min(span, s1 - step))
                seconds = ctypes.c_double()
                self._call("gdmap_cuda_advance", handle, begin, count, step, step + span,
                           ctypes.byref(schedule), int(write_tail), tail_begin,
                           ctypes.byref(seconds))
                kernel_seconds += seconds.value
                launches += 1
                if seconds.value > 0:
                    per_step_estimate = seconds.value / (span * count)
                step += span

        try:
            for chunk in range(n_chunks):
                lo, hi = int(bounds[chunk]), int(bounds[chunk + 1])
                m = hi - lo
                tangent = rng.normal(size=(m, p))
                tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
                theta_soa = np.full((p, capacity), np.nan)
                theta_soa[:, :m] = theta0[:, None]
                tangent_soa = np.zeros((p, capacity))
                tangent_soa[:, :m] = tangent.T
                lr_h = np.zeros(capacity)
                lr_o = np.zeros(capacity)
                lr_h[:m], lr_o[:m] = lr_h_all[lo:hi], lr_o_all[lo:hi]
                alive_in = np.zeros(capacity, dtype=np.intc)
                alive_in[:m] = int(initially_alive)
                diverged_in = np.zeros(capacity, dtype=np.longlong)
                diverged_in[:m] = -1 if initially_alive else 0
                started = time.monotonic()
                self._call("gdmap_cuda_upload", handle, m, _pointer(theta_soa, ctypes.c_double),
                           _pointer(tangent_soa, ctypes.c_double), _pointer(lr_h, ctypes.c_double),
                           _pointer(lr_o, ctypes.c_double), _pointer(alive_in, ctypes.c_int),
                           _pointer(diverged_in, ctypes.c_longlong))
                transfer_seconds += time.monotonic() - started
                if tail_start > 0:
                    advance(0, m, 0, tail_start, False, 0)
                for sub in range(0, m, tail_capacity):
                    count = min(tail_capacity, m - sub)
                    self._call("gdmap_cuda_reset_tail", handle)
                    advance(sub, count, tail_start, steps, True, sub)
                    started = time.monotonic()
                    sl = slice(lo + sub, lo + sub + count)
                    if screen == "device":
                        per = np.empty(count, dtype=np.longlong)
                        res = np.empty(count)
                        seconds = ctypes.c_double()
                        self._call("gdmap_cuda_screen", handle, count, opts["max_period"],
                                   opts["min_repeats"], opts["atol"], opts["rtol"],
                                   _pointer(per, ctypes.c_longlong), _pointer(res, ctypes.c_double),
                                   ctypes.byref(seconds))
                        period[sl], residual[sl] = per, res
                    else:
                        raw = np.empty((tail_steps, p, tail_capacity))
                        self._call("gdmap_cuda_take_tail", handle, _pointer(raw, ctypes.c_double))
                        tail = np.ascontiguousarray(raw[:, :, :count].transpose(2, 0, 1))
                        del raw
                        for start in range(0, count, opts["screen_batch"]):
                            stop = min(count, start + opts["screen_batch"])
                            period[lo + sub + start:lo + sub + stop], \
                                residual[lo + sub + start:lo + sub + stop] = screen_tails(
                                    tail[start:stop], max_period=opts["max_period"],
                                    min_repeats=opts["min_repeats"], atol=opts["atol"],
                                    rtol=opts["rtol"])
                        del tail
                    screen_seconds += time.monotonic() - started
                started = time.monotonic()
                out_theta = np.empty((p, capacity))
                out_alive = np.empty(capacity, dtype=np.intc)
                out_diverged = np.empty(capacity, dtype=np.longlong)
                out_ftle_ok = np.empty(capacity, dtype=np.intc)
                out_log = np.empty(capacity)
                out_min = np.empty(capacity)
                out_last = np.empty(capacity)
                out_tsum = np.empty(capacity)
                out_tcount = np.empty(capacity, dtype=np.longlong)
                out_wnorm = np.empty(capacity)
                out_early = np.empty((max(sd_max, 1), capacity))
                self._call("gdmap_cuda_fetch", handle, _pointer(out_theta, ctypes.c_double),
                           _pointer(out_alive, ctypes.c_int), _pointer(out_diverged, ctypes.c_longlong),
                           _pointer(out_ftle_ok, ctypes.c_int), _pointer(out_log, ctypes.c_double),
                           _pointer(out_min, ctypes.c_double), _pointer(out_last, ctypes.c_double),
                           _pointer(out_tsum, ctypes.c_double), _pointer(out_tcount, ctypes.c_longlong),
                           _pointer(out_wnorm, ctypes.c_double), _pointer(out_early, ctypes.c_double))
                transfer_seconds += time.monotonic() - started
                theta[lo:hi] = out_theta[:, :m].T
                alive[lo:hi] = out_alive[:m] != 0
                diverged_at[lo:hi] = out_diverged[:m]
                ftle_valid[lo:hi] = out_ftle_ok[:m] != 0
                log_growth[lo:hi] = out_log[:m]
                min_loss[lo:hi] = out_min[:m]
                last_loss[lo:hi] = out_last[:m]
                tail_loss_sum[lo:hi] = out_tsum[:m]
                tail_loss_count[lo:hi] = out_tcount[:m]
                window_norm[lo:hi] = out_wnorm[:m]
                if sd_max:
                    for horizon, (convergent, score) in chunk_sd_measures(
                            out_early[:sd_max, :m], cells, sd_horizons).items():
                        sd[horizon][0][lo:hi] = convergent
                        sd[horizon][1][lo:hi] = score
        finally:
            self._lib.gdmap_cuda_destroy(handle)

        ftle = np.where(alive & ftle_valid, log_growth / ftle_steps, np.nan)
        status = np.full(cells, STATUS_UNRESOLVED, dtype=np.int64)
        status[~alive] = STATUS_DIVERGENT
        status[alive & (period == 1)] = STATUS_FIXED
        status[alive & (period >= 2)] = STATUS_PERIODIC
        status[alive & (period == 0) & (ftle > opts["chaos_threshold"])] = STATUS_CHAOTIC
        with np.errstate(invalid="ignore", divide="ignore"):
            growth = np.linalg.norm(theta, axis=1) / window_norm - 1.0
        drift = (alive & (status == STATUS_UNRESOLVED) & np.isfinite(growth)
                 & (growth > opts["drift_threshold"]))
        status[drift] = STATUS_DRIFT
        result = {
            "status": status, "period": np.where(alive, period, -1),
            "recurrence_residual": residual, "ftle": ftle, "ftle_valid": alive & ftle_valid,
            "diverged_at": diverged_at, "final_state": theta,
            "window_norm_growth": np.where(alive, growth, np.nan),
            "min_loss": np.where(np.isfinite(min_loss), min_loss, np.nan),
            "last_evaluated_loss": np.where(alive, last_loss, np.nan),
            "tail_mean_loss": np.where(alive & (tail_loss_count == tail_steps),
                                       tail_loss_sum / np.maximum(tail_loss_count, 1), np.nan),
        }
        for horizon in sd_horizons:
            result[f"sd_convergent_{horizon}"] = sd[horizon][0]
            result[f"sd_score_{horizon}"] = sd[horizon][1]
        if not return_metadata:
            return result
        metadata = self.metadata(device)
        metadata.update(
            qualifying_run=qualifying_run,
            cells=cells, steps=steps, parameters=p, samples=n, options=_jsonable(opts),
            screen=screen, chunk_cells=capacity, tail_cells=tail_capacity, chunks=n_chunks,
            launches=launches, planned_device_bytes=required, allocated_device_bytes=allocated,
            device_byte_budget=budget, kernel_seconds=kernel_seconds,
            screen_seconds=screen_seconds, transfer_seconds=transfer_seconds,
            wall_seconds=time.monotonic() - began,
            microseconds_per_cell_step=1e6 * kernel_seconds / (cells * steps))
        return result, metadata


def chunk_sd_measures(early, total_cells, horizons):
    """Sohl-Dickstein measures for one chunk's early losses (steps, chunk_cells).

    Equal to the columns of one whole-grid call: NumPy reduces a single column
    pairwise but several columns sequentially, so a lone column of a
    multi-cell grid is padded with a copy before the call.
    """
    columns = early.shape[1]
    if columns == 1 and total_cells > 1:
        early = np.concatenate([early, early], axis=1)
    out = {}
    for horizon in horizons:
        convergent, score = sohl_dickstein_measure(early[:horizon])
        out[horizon] = (convergent[:columns], score[:columns])
    return out


def _jsonable(opts):
    out = {}
    for key, value in opts.items():
        if isinstance(value, tuple):
            value = list(value)
        elif not isinstance(value, (int, float, str, type(None))):
            value = repr(value)  # e.g. a SeedSequence passed as tangent_seed
        out[key] = value
    return out


def load(library=None):
    """Load a built library from ``library`` or ``$NEURAL_SHRIMP_GDMAP_CUDA_LIBRARY``."""
    if isinstance(library, GdmapCuda):
        return library
    path = library or os.environ.get(LIBRARY_ENV)
    if not path:
        raise FileNotFoundError(f"pass a library path or set {LIBRARY_ENV}; "
                                "build one with tools/build_gdmap_cuda.py")
    return GdmapCuda(path)


def run_grid_cuda(grid, theta0, steps, *, library=None, device=0,
                  max_device_bytes=DEFAULT_MAX_DEVICE_BYTES, chunk_cells=None, tail_cells=None,
                  launch_steps=None, screen="device", return_metadata=False, **options):
    """Iterate every rate cell of ``grid`` on the GPU from the same initial parameters.

    Keyword options are those of :func:`neural_shrimp.gdmaps.iterate_and_screen`;
    the result has the keys, dtypes and shapes of :func:`neural_shrimp.gdmaps.run_grid`.
    With ``return_metadata=True`` returns ``(result, metadata)``, where metadata
    records the device, NVIDIA driver, CUDA runtime/driver API versions, nvcc
    version, library hash, chunking, device memory and timings.
    """
    backend = load(library)
    return backend.run_grid(grid, theta0, steps, device=device, max_device_bytes=max_device_bytes,
                            chunk_cells=chunk_cells, tail_cells=tail_cells,
                            launch_steps=launch_steps, screen=screen,
                            return_metadata=return_metadata, **options)

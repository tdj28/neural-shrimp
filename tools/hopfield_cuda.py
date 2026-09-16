"""Optional CUDA float64 batched RK4 traces, with no CUDA activity on import.

Use a dedicated spawned process; never fork after CUDA initialization.
No metric, phase fit, reference, or CPU-default behavior changes here.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'native/hopfield_cuda.cu'
MAX_STEPS = 1_000_000
DEFAULT_MAX_DEVICE_BYTES = 1024**3
BUILD_FLAGS = ('-O3', '-std=c++17', '--shared', '--cudart=shared',
               '-Xcompiler=-fPIC', '--fmad=false', '--ftz=false',
               '--prec-div=true', '--prec-sqrt=true', '-arch=sm_89')


class CudaError(RuntimeError):
    def __init__(self, message, *, code=None, payload=None):
        super().__init__(message)
        self.code, self.payload = code, payload


class _Info(ctypes.Structure):
    _fields_ = [('name', ctypes.c_char*256), *[(n, ctypes.c_int) for n in
        ('device', 'device_count', 'major', 'minor', 'multiprocessors',
         'runtime_version', 'driver_version', 'single_to_double_ratio')],
        ('total_bytes', ctypes.c_uint64), ('free_bytes', ctypes.c_uint64)]


class _Timing(ctypes.Structure):
    _fields_ = [(n, ctypes.c_double) for n in ('setup_seconds', 'allocation_seconds',
        'host_to_device_seconds', 'kernel_seconds', 'kernel_wall_seconds',
        'device_to_host_seconds', 'free_seconds', 'total_seconds')] + [
        (n, ctypes.c_uint64) for n in ('requested_bytes', 'free_before_bytes',
                                     'total_device_bytes')] + [('device_output_copied', ctypes.c_int)]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(directory, *, compiler='nvcc', host_compiler='g++-12', timeout=120):
    """Compile only; do not initialize CUDA or run any trace. Refuse overwrites."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    library = directory/'hopfield_cuda.so'
    receipt = directory/'hopfield_cuda-build.json'
    if library.exists() or receipt.exists():
        raise FileExistsError('CUDA build output already exists')
    nvcc = shutil.which(compiler)
    if not nvcc:
        raise FileNotFoundError(f'CUDA compiler unavailable: {compiler}')
    host = shutil.which(host_compiler)
    if not host:
        raise FileNotFoundError(f'Supported host compiler unavailable: {host_compiler}')
    command = [nvcc, *BUILD_FLAGS, '-ccbin', host, str(SOURCE), '-o', str(library)]
    compiler_version = subprocess.check_output([nvcc, '--version'], text=True, timeout=20)
    host_version = subprocess.check_output([host, '--version'], text=True, timeout=20)
    started = time.monotonic()
    # nvcc starts compiler children; terminate the owned process group on a
    # timeout/interruption, and preserve its output before propagating failure.
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    interrupted = None
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException as error:
        interrupted = error
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    record = dict(command=command, source=str(SOURCE), source_sha256=_sha(SOURCE),
        compiler=nvcc, compiler_version=compiler_version,
        host_compiler=host, host_compiler_version=host_version,
        returncode=process.returncode, stdout=stdout, stderr=stderr,
        interruption=type(interrupted).__name__ if interrupted else None,
        wall_seconds=time.monotonic()-started, architecture='sm_89', dtype='float64',
        fmad=False, fast_math=False, ftz=False, precise_division=True,
        precise_sqrt=True, gpu_initialized=False,
        library=str(library), library_sha256=_sha(library) if library.exists() else None)
    with receipt.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    if interrupted:
        raise interrupted
    if process.returncode:
        raise CudaError(f'CUDA compilation failed; receipt: {receipt}', code=process.returncode)
    return dict(record, receipt=str(receipt))


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not minimum <= value <= 2**31-2:
        raise ValueError(f'{name} must be an integer in [{minimum}, 2**31-2]')
    return int(value)


def _prepare(alpha, gamma, initial, dt, duration, sample_dt, device, max_device_bytes):
    def array(value, name):
        raw = np.asarray(value)
        if raw.dtype.kind not in 'fiu' or raw.dtype.kind == 'b':
            raise ValueError(f'{name} must be real numerical values')
        converted = np.ascontiguousarray(raw, dtype=np.float64)
        if not np.isfinite(converted).all():
            raise ValueError(f'{name} must be finite')
        return converted
    alpha, gamma, initial = array(alpha, 'alpha'), array(gamma, 'gamma'), array(initial, 'initial')
    if alpha.ndim != 1 or not len(alpha) or gamma.shape != alpha.shape or initial.shape != (len(alpha), 3):
        raise ValueError('Expected alpha[N], gamma[N], initial[N,3], with N>0')
    n = _integer(len(alpha), 'batch size', minimum=1)
    device = _integer(device, 'device')
    for value in (dt, duration, sample_dt):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(value) or value <= 0:
            raise ValueError('dt, duration and sample_dt must be positive finite real scalars')
    dt, duration, sample_dt = map(float, (dt, duration, sample_dt))
    def ratio(value, name):
        quotient = value/dt
        if not math.isfinite(quotient) or quotient > 2**31-2 or abs(quotient-round(quotient)) > 1e-8:
            raise ValueError(f'{name}/dt must be a bounded integer')
        return int(round(quotient))
    steps, stride = ratio(duration, 'duration'), ratio(sample_dt, 'sample_dt')
    if not 1 <= steps <= MAX_STEPS or stride < 1 or steps % stride:
        raise ValueError('Require 1..1,000,000 steps and an integer number of sampling strides')
    if isinstance(max_device_bytes, (bool, np.bool_)) or not isinstance(max_device_bytes, (int, np.integer)) or not 1 <= max_device_bytes <= 2**64-1:
        raise ValueError('max_device_bytes must be a positive uint64 byte ceiling')
    samples = steps//stride+1
    required = n*samples*3*8 + n*(8*8+4+2*8)
    if required > max_device_bytes:
        raise ValueError(f'Batch requires {required} device bytes, above ceiling {max_device_bytes}')
    times = np.arange(samples, dtype=np.float64)*(stride*dt)
    times[-1] = duration
    return alpha, gamma, initial, dt, steps, stride, times, device, int(max_device_bytes), required


class CudaBackend:
    def __init__(self, library):
        self.path = Path(library).resolve()
        self.library_sha256 = _sha(self.path)
        self._lib = ctypes.CDLL(str(self.path))
        double = ctypes.POINTER(ctypes.c_double)
        integer = ctypes.POINTER(ctypes.c_int)
        wide = ctypes.POINTER(ctypes.c_int64)
        self._lib.hopfield_cuda_device_info.argtypes = [ctypes.c_int, ctypes.POINTER(_Info), ctypes.c_char_p, ctypes.c_size_t]
        self._lib.hopfield_cuda_device_info.restype = ctypes.c_int
        self._lib.hopfield_cuda_batch.argtypes = [double, double, double, ctypes.c_int,
            ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint64, double, integer, double, wide, ctypes.POINTER(_Timing),
            ctypes.c_char_p, ctypes.c_size_t]
        self._lib.hopfield_cuda_batch.restype = ctypes.c_int

    def device_info(self, device=0):
        device = _integer(device, 'device')
        data, error = _Info(), ctypes.create_string_buffer(1024)
        code = self._lib.hopfield_cuda_device_info(device, ctypes.byref(data), error, len(error))
        if code:
            raise CudaError(error.value.decode(errors='replace'), code=code)
        result = {name: getattr(data, name) for name, _ in _Info._fields_}
        result['name'] = result['name'].decode(errors='replace')
        result['library_sha256'] = self.library_sha256
        return result

    def integrate_batch(self, alpha, gamma, initial, dt, duration=300., sample_dt=.1,
                        device=0, *, max_device_bytes=DEFAULT_MAX_DEVICE_BYTES):
        """Return complete/partial raw traces; no scoring or failure replacement.

        statuses == number of samples means complete; -2 means nonfinite
        integration with retained partial samples/final/progress; -3 means
        device output unavailable after a CUDA API failure. CUDA failures raise
        CudaError carrying the host payload. Allocations are released before
        return/raise; device state is never reset. Times match the CPU ABI.
        """
        began = time.monotonic()
        alpha, gamma, initial, dt, steps, stride, times, device, ceiling, required = _prepare(
            alpha, gamma, initial, dt, duration, sample_dt, device, max_device_bytes)
        info = self.device_info(device)
        if (info['major'], info['minor']) != (8, 9):
            raise ValueError('This backend was frozen for sm_89; another device needs a new qualification')
        if required > info['free_bytes']//2:
            raise ValueError('Batch would consume more than half currently free VRAM')
        n = len(alpha)
        states = np.full((n, len(times), 3), np.nan, dtype=np.float64)
        statuses = np.full(n, -3, dtype=np.int32)
        final = np.full((n, 3), np.nan, dtype=np.float64)
        progress = np.zeros((n, 2), dtype=np.int64)
        timing, error = _Timing(), ctypes.create_string_buffer(1024)
        pointer = lambda value: value.ctypes.data_as(ctypes.POINTER(
            ctypes.c_double if value.dtype == np.float64 else ctypes.c_int64 if value.dtype == np.int64 else ctypes.c_int))
        code = self._lib.hopfield_cuda_batch(pointer(alpha), pointer(gamma), pointer(initial), n,
            dt, steps, stride, len(times), device, ceiling, pointer(states), pointer(statuses),
            pointer(final), pointer(progress), ctypes.byref(timing), error, len(error))
        measured = {name: getattr(timing, name) for name, _ in _Timing._fields_}
        result = dict(times=times, states=states, statuses=statuses, final=final, progress=progress,
            timings={k:v for k,v in measured.items() if k.endswith('_seconds')},
            memory=dict(requested_device_bytes=required,
                        host_input_bytes=alpha.nbytes+gamma.nbytes+initial.nbytes,
                        host_output_bytes=times.nbytes+states.nbytes+statuses.nbytes+final.nbytes+progress.nbytes,
                        host_to_device_bytes=alpha.nbytes+gamma.nbytes+initial.nbytes,
                        device_to_host_bytes=states.nbytes+statuses.nbytes+final.nbytes+progress.nbytes,
                        device_byte_ceiling=ceiling, free_device_bytes_before=timing.free_before_bytes,
                        total_device_bytes=timing.total_device_bytes), device=info,
            device_output_copied=bool(timing.device_output_copied), cuda_error_code=code,
            cuda_error=error.value.decode(errors='replace'), steps=steps, stride=stride, dt=dt)
        if code:
            result['timings']['python_total_seconds'] = time.monotonic()-began
            raise CudaError(result['cuda_error'], code=code, payload=result)
        successful = statuses == len(times)
        valid_partial = True
        for index in np.flatnonzero(statuses == -2):
            done, retained = progress[index]
            expected_retained = 1 + max(0, int(done)-1)//stride
            valid_partial &= (0 <= done <= steps and retained == expected_retained
                and np.isfinite(states[index, :retained]).all()
                and np.isnan(states[index, retained:]).all())
        if (not timing.device_output_copied or timing.requested_bytes != required or
            not valid_partial or not np.all(successful | (statuses == -2)) or
            not np.array_equal(states[:, 0], initial) or
            not np.isfinite(states[successful]).all() or
            not np.array_equal(states[successful, -1], final[successful]) or
            not np.all(progress[successful] == [steps, len(times)])):
            result['timings']['python_total_seconds'] = time.monotonic()-began
            raise CudaError('CUDA trace output violated the progress/sample contract', payload=result)
        result['timings']['python_total_seconds'] = time.monotonic()-began
        return result


def load(path):
    return CudaBackend(path)

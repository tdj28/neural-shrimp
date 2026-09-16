"""Synthetic CPU-side guards; GPU smoke is explicit and writes a new receipt.

No research initial states, labels, or saved target trajectories are loaded.
GPU fixtures are fixed zero/short dummy states, never longer than four steps.
"""
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import hopfield_cuda as gpu


def prepared(**changes):
    values = dict(alpha=[1., 2.], gamma=[3., 4.], initial=np.zeros((2, 3)),
                  dt=.0025, duration=.01, sample_dt=.005, device=0,
                  max_device_bytes=1024**2)
    values.update(changes)
    return gpu._prepare(**values)


def test_prepare_exact_grid_memory_and_dtype():
    alpha, gamma, initial, dt, steps, stride, times, device, limit, required = prepared()
    assert steps == 4 and stride == 2
    assert np.array_equal(times, [0., .005, .01])
    assert required == 2 * 3 * 3 * 8 + 2 * (8*8 + 4 + 2*8)
    assert all(x.dtype == np.float64 and x.flags.c_contiguous for x in (alpha, gamma, initial))


@pytest.mark.parametrize('change', [
    dict(alpha=[]), dict(gamma=[1.]), dict(initial=np.zeros((2, 4))),
    dict(alpha=[[1., 2.]]), dict(alpha=[True, False]), dict(alpha=['1', '2']),
    dict(alpha=[1+0j, 2+0j]), dict(initial=[[np.nan, 0, 0], [0, 0, 0]]),
    dict(gamma=[np.inf, 1]), dict(dt=True), dict(dt='0.0025'),
    dict(dt=0), dict(dt=-1), dict(dt=np.nan), dict(duration=np.inf),
    dict(duration=.0101), dict(sample_dt=.006), dict(sample_dt=.02),
    dict(duration=2500.0025), dict(device=True), dict(device=-1),
    dict(device=.5), dict(max_device_bytes=1), dict(max_device_bytes=True),
    dict(max_device_bytes=2**64),
])
def test_reject_invalid_before_cuda(change):
    with pytest.raises(ValueError):
        prepared(**change)


def fake_backend(mode='ok'):
    backend = gpu.CudaBackend.__new__(gpu.CudaBackend)
    backend.device_info = lambda device=0: dict(major=8, minor=9, free_bytes=1024**3)

    def function(alpha, gamma, initial, n, dt, steps, stride, samples, device,
                 ceiling, states, statuses, final, progress, timing, error, capacity):
        state = np.ctypeslib.as_array(states, (n*samples*3,)).reshape(n, samples, 3)
        status = np.ctypeslib.as_array(statuses, (n,))
        end = np.ctypeslib.as_array(final, (n*3,)).reshape(n, 3)
        progress_array = np.ctypeslib.as_array(progress, (n*2,)).reshape(n, 2)
        inp = np.ctypeslib.as_array(initial, (n*3,)).reshape(n, 3)
        metadata = ctypes.cast(timing, ctypes.POINTER(gpu._Timing)).contents
        metadata.requested_bytes = n*samples*3*8+n*(8*8+4+2*8)
        metadata.free_before_bytes = 1024**3
        metadata.total_device_bytes = 2*1024**3
        if mode == 'cuda_error':
            error.value = b'synthetic CUDA failure'
            return 700
        state[:] = inp[:, None]
        status[:] = samples
        end[:] = inp
        progress_array[:] = [steps, samples]
        metadata.device_output_copied = 1
        if mode in ('partial', 'bad_partial'):
            status[0] = -2
            state[0, 1:] = np.nan
            end[0] = np.inf
            progress_array[0] = [1, 1 if mode == 'partial' else 2]
        if mode == 'bad_complete':
            progress_array[0, 0] -= 1
        if mode == 'not_copied':
            metadata.device_output_copied = 0
        return 0
    backend._lib = SimpleNamespace(hopfield_cuda_batch=function)
    return backend


def call_fake(mode='ok'):
    return fake_backend(mode).integrate_batch([1., 2.], [3., 4.], np.zeros((2, 3)),
                                             .0025, .01, .0025)


def test_complete_contract_and_timing_memory_schema():
    result = call_fake()
    assert result['states'].shape == (2, 5, 3)
    assert np.array_equal(result['statuses'], [5, 5])
    assert np.array_equal(result['progress'], [[4, 5], [4, 5]])
    assert result['device_output_copied']
    assert set(result['timings']) == {'setup_seconds', 'allocation_seconds',
        'host_to_device_seconds', 'kernel_seconds', 'kernel_wall_seconds',
        'device_to_host_seconds', 'free_seconds', 'total_seconds', 'python_total_seconds'}
    assert result['memory']['requested_device_bytes'] == 2*5*3*8+2*84


def test_partial_numerical_failure_preserves_prefix_and_final_attempt():
    result = call_fake('partial')
    assert result['statuses'][0] == -2 and result['statuses'][1] == 5
    assert np.array_equal(result['progress'][0], [1, 1])
    assert np.isfinite(result['states'][0, 0]).all()
    assert np.isnan(result['states'][0, 1:]).all()
    assert np.isinf(result['final'][0]).all()


@pytest.mark.parametrize('mode', ['bad_partial', 'bad_complete', 'not_copied'])
def test_reject_malformed_native_progress(mode):
    with pytest.raises(gpu.CudaError, match='progress/sample') as raised:
        call_fake(mode)
    assert raised.value.payload is not None


def test_cuda_failure_preserves_host_payload_and_code():
    with pytest.raises(gpu.CudaError, match='synthetic CUDA failure') as raised:
        call_fake('cuda_error')
    assert raised.value.code == 700
    result = raised.value.payload
    assert np.array_equal(result['statuses'], [-3, -3])
    assert np.isnan(result['states']).all()
    assert not result['device_output_copied']


def test_memory_and_architecture_guard_precedes_native_batch():
    backend = fake_backend()
    backend.device_info = lambda device=0: dict(major=8, minor=9, free_bytes=1)
    with pytest.raises(ValueError, match='half currently free'):
        backend.integrate_batch([1.], [2.], [[0., 0., 0.]], .0025, .01, .0025)
    backend.device_info = lambda device=0: dict(major=8, minor=6, free_bytes=1024**3)
    with pytest.raises(ValueError, match='sm_89'):
        backend.integrate_batch([1.], [2.], [[0., 0., 0.]], .0025, .01, .0025)


def test_build_flags_and_receipt_without_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu.shutil, 'which', lambda value: '/usr/bin/'+value)
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *args, **kwargs: 'synthetic compiler version\n')
    class Compiler:
        returncode = 0
        def __init__(self, command, **kwargs):
            assert kwargs['start_new_session'] is True
            Path(command[-1]).write_bytes(b'synthetic library, not executable')
        def communicate(self, timeout=None):
            return '', ''
    monkeypatch.setattr(gpu.subprocess, 'Popen', Compiler)
    result = gpu.build(tmp_path)
    assert result['host_compiler'] == '/usr/bin/g++-12'
    assert '--fmad=false' in result['command'] and '-arch=sm_89' in result['command']
    assert '--use_fast_math' not in result['command']
    assert result['command'][result['command'].index('-ccbin')+1] == '/usr/bin/g++-12'
    assert result['gpu_initialized'] is False
    assert json.loads(Path(result['receipt']).read_text())['library_sha256'] == result['library_sha256']
    with pytest.raises(FileExistsError):
        gpu.build(tmp_path)


def test_build_timeout_stops_owned_group_and_preserves_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu.shutil, 'which', lambda value: '/usr/bin/'+value)
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *args, **kwargs: 'version\n')
    killed = []
    monkeypatch.setattr(gpu.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    class Compiler:
        pid, returncode, calls = 432123, -9, 0
        def __init__(self, *args, **kwargs):
            pass
        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired('synthetic nvcc', timeout)
            return 'partial compiler stdout', 'partial compiler stderr'
    monkeypatch.setattr(gpu.subprocess, 'Popen', Compiler)
    with pytest.raises(subprocess.TimeoutExpired):
        gpu.build(tmp_path, timeout=.001)
    receipt = json.loads((tmp_path/'hopfield_cuda-build.json').read_text())
    assert killed == [(432123, gpu.signal.SIGKILL)]
    assert receipt['interruption'] == 'TimeoutExpired'
    assert receipt['stderr'] == 'partial compiler stderr'


def scalar_fixture(alpha, gamma, initial, steps, dt):
    """Independent scalar stage construction; dummy fixtures only."""
    def rhs(y):
        a, b, c = [math.tanh(v) for v in y]
        return [-y[0]+1.5*a+2.9*b+alpha*c, -y[1]-1.5*a+1.18*b,
                -y[2]+gamma*a-22.*b+.47*c]
    y = list(map(float, initial))
    output, f = [y[:]], rhs(y)
    for _ in range(steps):
        k2 = rhs([y[j]+.5*dt*f[j] for j in range(3)])
        k3 = rhs([y[j]+.5*dt*k2[j] for j in range(3)])
        k4 = rhs([y[j]+dt*k3[j] for j in range(3)])
        y = [y[j]+(dt/6.)*(f[j]+2.*k2[j]+2.*k3[j]+k4[j]) for j in range(3)]
        f = rhs(y)
        output.append(y[:])
    return np.array(output)


def test_optional_gpu_synthetic_smoke_with_receipts():
    library = os.environ.get('NEURAL_SHRIMP_GPU_TEST_LIBRARY')
    destination = os.environ.get('NEURAL_SHRIMP_GPU_TEST_RECEIPT_DIR')
    if not library and not destination:
        pytest.skip('Explicit GPU library and external new receipt directory required')
    assert library and destination, 'Both GPU test environment variables are required'
    directory = Path(destination).resolve()
    assert not directory.is_relative_to(gpu.ROOT), 'GPU receipts must remain outside Git'
    directory.mkdir(parents=True, exist_ok=False)
    # Predefined fixtures and tolerance: no known-case data or long integration.
    alpha = np.array([1.058, 1.064, 1.07, 1.078])
    gamma = np.array([10.2, 10.3, 10.4, 10.5])
    initial = np.array([[0., 0., 0.], [.1, -.2, .3], [1., -1., .5], [-.75, .25, 1.5]])
    receipt = dict(scope='synthetic zero and 1/4-step dummy fixtures only',
                   started_utc=datetime.now(timezone.utc).isoformat(),
                   rtol=2e-13, atol=2e-14, passed=False, library=str(library),
                   library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest(),
                   test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=[])
    try:
        backend = gpu.load(library)
        for steps in (1, 4):
            expected = np.array([scalar_fixture(a, g, y, steps, .0025)
                                 for a, g, y in zip(alpha, gamma, initial, strict=True)])
            result = backend.integrate_batch(alpha, gamma, initial, .0025, steps*.0025, .0025)
            np.savez(directory/f'steps-{steps}.npz', alpha=alpha, gamma=gamma, initial=initial,
                     expected=expected, **{key: result[key] for key in ('times', 'states', 'statuses', 'final', 'progress')})
            receipt['cases'].append(dict(steps=steps, timings=result['timings'],
                memory=result['memory'], device=result['device'],
                max_abs_difference=float(np.max(np.abs(result['states']-expected)))))
            np.testing.assert_allclose(result['states'], expected, rtol=2e-13, atol=2e-14)
            assert np.array_equal(result['states'][0], np.zeros((steps+1, 3)))
            assert np.array_equal(result['statuses'], np.full(4, steps+1))
            assert np.array_equal(result['progress'], np.tile([steps, steps+1], (4, 1)))
        receipt['passed'] = True
    except BaseException as error:
        receipt['failure'] = f'{type(error).__name__}: {error}'
        if isinstance(error, gpu.CudaError) and error.payload is not None:
            np.savez(directory/'failed-payload.npz', **{key: error.payload[key]
                for key in ('times', 'states', 'statuses', 'final', 'progress')})
        raise
    finally:
        receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
        receipt['artifacts'] = [dict(path=path.name, bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted(directory.glob('*.npz'))]
        with (directory/'receipt.json').open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')

"""Numerical dyadic A_i/B proposals for eight-phase multiple shooting.

This script performs only independent binary64 model/Jacobian computations.
Saved binary64 entries can later be interpreted as exact dyadic preconditioner
constants, but they are NOT rigorous derivative enclosures or inverse proofs.
Actual execution requires the separate pushed run card and bounded launcher.
Importing this module does not import Torch, load checkpoints, or run a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OWNED = Path('/data2/neural-shrimp-data/codex/llm-p8-certificate')
PROPOSAL = OWNED/'arb-pilot-002/run/proposal.npz'
PROPOSAL_SHA256 = 'd8c57bff1d8a986c7c6a0f7d083caa77b4f6373c560d1f60ce0c57461caf668b'
ALPHA = .071674
LOG10_TEMPERATURE = -.115
PERIOD = 8
DIMENSION = 896
CLOSURE_TOLERANCE = 1e-10


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def write_arrays(path, **arrays):
    """Atomic NPZ; interruptions do not destroy the previous checkpoint."""
    import numpy as np
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def propose_product_and_inverse(matrices):
    """Floating proposals only; caller persists the matrices before calling.

    Shape (p,n,n) is generic for synthetic checks. The target driver separately
    requires p=8,n=896. No finite numerical defect is called a proof bound.
    """
    import numpy as np
    if (not isinstance(matrices, np.ndarray) or matrices.dtype != np.float64
            or matrices.ndim != 3 or matrices.shape[0] < 1
            or matrices.shape[1] < 1 or matrices.shape[1] != matrices.shape[2]
            or not np.isfinite(matrices).all()):
        raise ValueError('finite binary64 square phase matrices required')
    identity = np.eye(matrices.shape[1], dtype=np.float64)
    product = identity.copy()
    for matrix in matrices:
        product = matrix @ product
        if not np.isfinite(product).all():
            raise ValueError('nonfinite numerical phase product')
    denominator = identity-product
    inverse = np.linalg.solve(denominator, identity)
    left_defect = identity-inverse @ denominator
    right_defect = identity-denominator @ inverse
    condition = float(np.linalg.cond(denominator))
    if (not all(np.isfinite(v).all() for v in (inverse,left_defect,right_defect))
            or not __import__('math').isfinite(condition)):
        raise ValueError('nonfinite or numerically singular inverse proposal')
    diagnostics = dict(
        condition_2_estimate=condition,
        left_inverse_defect_inf=float(np.linalg.norm(left_defect, ord=np.inf)),
        right_inverse_defect_inf=float(np.linalg.norm(right_defect, ord=np.inf)),
        product_norm_inf=float(np.linalg.norm(product, ord=np.inf)),
        inverse_norm_inf=float(np.linalg.norm(inverse, ord=np.inf)),
        statement='Floating diagnostics only; neither product nor inverse defect is a rigorous enclosure.')
    if any(not __import__('math').isfinite(v) for v in diagnostics.values() if isinstance(v,float)):
        raise ValueError('overflow in numerical inverse diagnostics')
    return product, inverse, left_defect, right_defect, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4 or not 1 <= args.chunk_size <= 16:
        raise ValueError('one to four CPU threads and Jacobian chunk size one to sixteen required')
    out = args.output.resolve()
    if not out.is_relative_to(OWNED) or out == OWNED:
        raise ValueError('exclusive Task1-owned output directory required')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    phase = 'admission'
    completed = []

    def event(name, **fields):
        record = dict(event=name, phase=phase, seconds=time.monotonic()-started,
                      utc=datetime.now(timezone.utc).isoformat(), **fields)
        with (out/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        print(json.dumps(record, allow_nan=False), flush=True)

    try:
        source = os.environ.get('CERT_SOURCE_COMMIT', '')
        if len(source) != 40 or any(v not in '0123456789abcdef' for v in source):
            raise ValueError('exact source commit from the bounded launcher is required')
        # Clamp pre-import runtimes to the declared envelope; preserve a smaller
        # inherited setting. Runtime controls below remain at most four threads.
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            configured = int(os.environ.get(name, '1'))
            if configured < 1:
                raise ValueError('invalid thread environment: '+name)
            os.environ[name] = str(min(configured, args.threads))
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['HF_HUB_OFFLINE'] = '1'
        if os.getpriority(os.PRIO_PROCESS,0) < 10:
            os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
        if sha256(PROPOSAL) != PROPOSAL_SHA256:
            raise ValueError('immutable full-state proposal hash mismatch')
        import numpy as np
        import torch
        from threadpoolctl import threadpool_limits
        sys.path.insert(0, str(ROOT/'codex/crosscheck'))
        from qwen_map import ManualQwenSoftMap, PINNED_SHA256, PREFIX_IDS
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        with np.load(PROPOSAL, allow_pickle=False) as archive:
            points = archive['points'].copy()
        if points.shape != (PERIOD+1,DIMENSION) or points.dtype != np.float64 or not np.isfinite(points).all():
            raise ValueError('finite binary64 nine-state proposal required')
        centers = points[:-1].copy()
        center_norms = np.linalg.norm(centers, axis=1)
        if not np.isfinite(center_norms).all() or np.any(center_norms <= 0):
            raise ValueError('finite nonzero center norms required')
        proposal_returns = {str(d):float(np.linalg.norm(points[d]-points[0])/center_norms[0])
                            for d in (1,2,4,8)}
        write_json(out/'inputs.json', dict(
            source_commit=source, driver_sha256=sha256(__file__),
            independent_map_sha256=sha256(ROOT/'codex/crosscheck/qwen_map.py'),
            proposal_path=str(PROPOSAL), proposal_sha256=PROPOSAL_SHA256,
            pinned_assets=PINNED_SHA256, prefix_ids=PREFIX_IDS,
            alpha_ratio=list(ALPHA.as_integer_ratio()),
            log10_temperature_ratio=list(LOG10_TEMPERATURE.as_integer_ratio()),
            parameter_array_order=['alpha','log10_temperature'], period=PERIOD, dimension=DIMENSION,
            threads=args.threads, interop_threads=1, chunk_size=args.chunk_size,
            nice=os.getpriority(os.PRIO_PROCESS,0), pid=os.getpid(),
            thread_environment={name:os.environ[name] for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},
            versions=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__),
            closure_tolerance=CLOSURE_TOLERANCE, proposal_returns=proposal_returns,
            representation='Saved binary64 A,B entries are exact dyadic proposals; no derivative enclosure or inverse proof.',
            wall_limit='Enforced by the separately admitted owned launcher; see attempt manifest.',
            certificate_passed=False))
        write_arrays(out/'centers.npz', centers=centers, full_proposal=points,
                     parameters=np.array([ALPHA,LOG10_TEMPERATURE],dtype=np.float64))
        if proposal_returns['8'] >= CLOSURE_TOLERANCE or min(proposal_returns[str(d)] for d in (1,2,4)) <= 1e-6:
            raise ValueError('pinned proposal numerical closure/divisor consistency failed')
        with threadpool_limits(limits=args.threads):
            phase = 'model_load'; event('load_model')
            model = ManualQwenSoftMap.from_pinned_files()
            if model.d != DIMENSION:
                raise ValueError('unexpected target state dimension')
            cache = model.prefix_cache(ALPHA)
            matrices = []
            for index in range(PERIOD):
                phase = f'phase_{index:02d}'
                event('phase_start', index=index)
                state = torch.tensor(centers[index],dtype=torch.float64)
                next_center = centers[(index+1)%PERIOD]
                image = model.forward(state,ALPHA,LOG10_TEMPERATURE,cache).numpy().copy()
                residual = image-next_center
                residual_path = out/f'phase-{index:02d}-residual.npz'
                write_arrays(residual_path, center=centers[index], image=image,
                             next_center=next_center, residual=residual)
                if image.shape != (DIMENSION,) or not np.isfinite(image).all() or not np.isfinite(residual).all():
                    raise ValueError(f'nonfinite/invalid image at phase {index}; raw residual artifact retained')
                absolute = float(np.linalg.norm(residual))
                relative = absolute/float(np.linalg.norm(next_center))
                if not np.isfinite([absolute,relative]).all():
                    raise ValueError(f'nonfinite residual diagnostic at phase {index}')
                event('phase_residual', index=index, absolute_l2=absolute,relative_l2=relative,
                      arrays_sha256=sha256(residual_path))
                began = time.monotonic()
                jacobian = model.jacobian(state,ALPHA,LOG10_TEMPERATURE,period=1,
                                         chunk_size=args.chunk_size,cache=cache).numpy().copy()
                phase_path = out/f'phase-{index:02d}.npz'
                write_arrays(phase_path,A=jacobian,center=centers[index],image=image,
                             next_center=next_center,residual=residual)
                nonfinite = int(np.count_nonzero(~np.isfinite(jacobian)))
                accepted = jacobian.shape == (DIMENSION,DIMENSION) and jacobian.dtype == np.float64 and nonfinite == 0 and relative < CLOSURE_TOLERANCE
                record = dict(index=index, passed=bool(accepted), absolute_l2=absolute, relative_l2=relative,
                    jacobian_shape=list(jacobian.shape),jacobian_dtype=str(jacobian.dtype),nonfinite_entries=nonfinite,
                    jacobian_seconds=time.monotonic()-began,path=phase_path.name,sha256=sha256(phase_path),
                    statement='Numerical one-phase Jacobian proposal and consistency gate only.')
                write_json(out/f'phase-{index:02d}.json',record)
                completed.append(record)
                event('phase_saved',**record)
                if not accepted:
                    raise ValueError(f'phase {index} numerical shape/finiteness/closure gate failed')
                matrices.append(jacobian)
            phase = 'inverse_proposal'; event('inverse_start')
            A = np.stack(matrices)
            # Preserve all individual A_i even if solve/conditioning now fails.
            write_arrays(out/'phase-matrices.npz',A=A,centers=centers,
                         parameters=np.array([ALPHA,LOG10_TEMPERATURE],dtype=np.float64))
            product,B,left_defect,right_defect,diagnostics = propose_product_and_inverse(A)
            final = out/'preconditioner.npz'
            write_arrays(final,A=A,B=B,centers=centers,
                parameters=np.array([ALPHA,LOG10_TEMPERATURE],dtype=np.float64),
                product=product,left_inverse_defect=left_defect,right_inverse_defect=right_defect)
            write_json(out/'preconditioner.json',dict(
                arrays_sha256=sha256(final),parameter_array_order=['alpha','log10_temperature'],
                product_order='A[7] @ ... @ A[0]',B_proposal='binary64 solve((I-product), I)',
                diagnostics=diagnostics,phase_artifacts=completed,certificate_passed=False,
                statement='A and B are finite exact-dyadic proposals when read from binary64 bytes; rigorous enclosures and inverse verification are absent.'))
            event('inverse_saved',arrays_sha256=sha256(final),diagnostics=diagnostics)
        write_json(out/'status.json',dict(status='proposal_completed',certificate_passed=False,
            phase_artifacts=completed,preconditioner_sha256=sha256(out/'preconditioner.npz'),
            elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            reason='Numerical matrices for later rigorous multiple-shooting preconditioning; no orbit proof.'))
    except Exception as error:
        write_json(out/'status.json',dict(status='proposal_failed',certificate_passed=False,
            phase=phase,phase_artifacts=completed,error=repr(error),traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise


if __name__ == '__main__':
    main()

"""Task18 one bounded independent loop-gain experiment; no producer imports.

An external reviewed supervisor owns the hard wall timeout and process cleanup.
Worker guards are cooperative and do not interrupt a native tensor operation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OWNED = Path('/data2/neural-shrimp-data/codex/llm-loop-gain-crosscheck')
CAPS = dict(cpu_threads=2, nice=10, memory_gib=12, timeout_seconds=3600,
            minimum_start_available_gib=28, minimum_runtime_available_gib=16,
            maximum_output_mib=512)
SOURCES = ('codex/loop-gain-crosscheck/analysis.py',
           'codex/loop-gain-crosscheck/run.py', 'codex/crosscheck/qwen_map.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def available():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                    if x.startswith('MemAvailable:'))) * 1024


def validate(plan):
    keys = {'schema','admission','envelope','source_sha256','parameters','scope'}
    if set(plan) != keys or plan['schema'] != 'codex-loop-gain-v1':
        raise ValueError('exact Task18 plan schema required')
    if plan['admission'] != 'reviewed': raise ValueError('prepared plan is not launchable')
    if plan['envelope'] != CAPS or any(type(v) is not int for v in plan['envelope'].values()):
        raise ValueError('exact resource envelope required')
    if set(plan['source_sha256']) != set(SOURCES) or any(re.fullmatch('[0-9a-f]{64}', v) is None for v in plan['source_sha256'].values()):
        raise ValueError('complete exact source closure required')
    wanted = dict(alpha=.071674, log10_temperature=-.115, period=8, transient=1000,
                  dimension=896, vocabulary=151936, batch_directions=16,
                  random_directions=8, first_seed=18001)
    if plan['parameters'] != wanted or any(type(plan['parameters'][k]) is not type(v) for k,v in wanted.items()):
        raise ValueError('fixed Task18 parameters required')
    return plan


class Store:
    def __init__(self, path, check): self.path, self.check, self.index = path, check, 0
    def emit(self, name, payload):
        import numpy as np
        index = self.index; self.index += 1
        def convert(value, key):
            if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                path = self.path / f'{index:04d}-{name}-{key}.npy'
                with path.open('xb') as stream: np.save(stream, value, allow_pickle=False)
                return dict(path=path.name, dtype=str(value.dtype), shape=list(value.shape), sha256=sha(path))
            if isinstance(value, dict): return {k:convert(v, key+'-'+k) for k,v in value.items()}
            if isinstance(value, (tuple,list)): return [convert(v, key+'-'+str(i)) for i,v in enumerate(value)]
            if isinstance(value, np.generic): return value.item()
            return value
        content = convert(payload, 'data')
        with (self.path / f'{index:04d}-{name}.json').open('x') as stream:
            json.dump(content, stream, indent=2, allow_nan=False)
        self.check()


def main():
    began = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output != args.output.resolve() or args.output != OWNED/'run-002'/'run':
        raise ValueError('exact exclusive owned run-002/run path required')
    args.output.mkdir(exist_ok=False)
    def write(name, content):
        with (args.output/(name+'.json')).open('x') as stream:
            json.dump(content, stream, indent=2, allow_nan=False)
    stage = 'admission'; completed_phases = []
    try:
        if os.getpriority(os.PRIO_PROCESS, 0) != 10: raise ValueError('external exact nice10 required')
        hard = resource.getrlimit(resource.RLIMIT_AS)[1]
        if hard != resource.RLIM_INFINITY and hard < 12*1024**3: raise ValueError('insufficient inherited hard AS')
        resource.setrlimit(resource.RLIMIT_AS, (12*1024**3,)*2)
        for key,value in dict(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1').items(): os.environ[key] = value
        if available() < 28*1024**3: raise RuntimeError('startup memory floor')
        raw_plan = args.plan.read_bytes(); plan = validate(json.loads(raw_plan))
        source = os.environ.get('TASK18_SOURCE_COMMIT','')
        if re.fullmatch('[0-9a-f]{40}', source) is None or sha(args.plan) != os.environ.get('TASK18_PLAN_SHA256'):
            raise ValueError('external original source/plan pins required')
        def sources_check():
            for name,pin in plan['source_sha256'].items():
                if sha(ROOT/name) != pin: raise ValueError('source changed: '+name)
        sources_check()
        with (args.output/'original-plan.json').open('xb') as stream: stream.write(raw_plan)
        def check():
            if time.monotonic()-began >= CAPS['timeout_seconds']: raise TimeoutError('worker cooperative deadline')
            if available() < 16*1024**3: raise RuntimeError('shared memory floor')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 12*1024**3: raise MemoryError('RSS ceiling')
            if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) > 496*1024**2:
                raise RuntimeError('output cap with16MiB closing reserve')
        sys.path[:0] = [str(ROOT), str(HERE)]
        import numpy as np
        import torch
        import analysis
        from codex.crosscheck.qwen_map import ManualQwenSoftMap, SNAPSHOT, VECTOR_FILE, PINNED_SHA256, PREFIX_IDS
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        write('input-binding', dict(source_commit=source, plan_sha256=sha(args.plan), source_sha256=plan['source_sha256'],
              snapshot=str(SNAPSHOT), vector=str(VECTOR_FILE), vector_key='philosophy', asset_sha256=PINNED_SHA256,
              prefix_ids=list(PREFIX_IDS), steering_layer=12, state_clock='x0 follows exactly1000 full steps from fixed start',
              dtype='float64', tied_readout=True, producer_arrays_read=False))
        store = Store(args.output, check)
        stage = 'model-load'; mark = time.monotonic(); model = ManualQwenSoftMap.from_pinned_files()
        if (model.d,model.layers,model.vocab,model.steering_layer)!=(896,24,151936,12): raise ValueError('map dimensions differ')
        store.emit('model-loaded',dict(seconds=time.monotonic()-mark, dimension=model.d,layers=model.layers,
              vocabulary=model.vocab, prefix_ids=model.prefix_ids, steering_vector=model.vector,
              weight_bytes=sum(t.numel()*t.element_size() for t in model.weights.values()), asset_sha256=PINNED_SHA256))
        p=plan['parameters']; stage='prefix'; mark=time.monotonic()
        cache=model.prefix_cache(p['alpha'])
        store.emit('prefix',dict(alpha=cache.alpha,layers=cache.layers,seconds=time.monotonic()-mark))
        stage='independent-orbit'
        states,orbit_gates=analysis.orbit(model,p['alpha'],p['log10_temperature'],transient=p['transient'],
              period=p['period'],cache=cache,emit=store.emit,check=check)
        matrices=[]
        for index in range(8):
            stage=f'phase-{index}'; check()
            emit=lambda name,content,index=index:store.emit(f'phase{index}-'+name,content)
            summary,matrix=analysis.phase(model,states[index],p['alpha'],p['log10_temperature'],cache=cache,
                  batch=p['batch_directions'],seed=p['first_seed']+index,emit=emit,check=check)
            summary['index']=index; completed_phases.append(summary); matrices.append(matrix)
        stage='cyclic-monodromies'; loops=[]; phase0_product=None
        for start in range(8):
            check(); mark=time.monotonic()
            result=analysis.loop_spectra(matrices[start:]+matrices[:start])
            if not all(np.isfinite(v).all() for v in result.values()): raise ValueError('nonfinite loop spectra')
            store.emit(f'monodromy-phase{start}',dict(start_phase=start,seconds=time.monotonic()-mark,**result))
            if start == 0: phase0_product=result['product'].copy()
            eigen=result['monodromy_eigenvalues']
            loops.append(dict(start_phase=start,seconds=time.monotonic()-mark,
                 leading_multipliers=[dict(real=float(z.real),imag=float(z.imag),modulus=float(abs(z))) for z in eigen[:8]],
                 first_two_singular_values=result['monodromy_singular_values'][:2].tolist(),
                 product_step_norms=result['product_step_norms']))
        stage='composed-original-map-controls'
        composed_controls=analysis.composed_controls(model,states[0],p['alpha'],p['log10_temperature'],
            period=8,cache=cache,matrix=phase0_product,expected_image=states[8],seed=18009,emit=store.emit,check=check)
        stage='final-source-input-check'; check(); sources_check()
        for path,pin in [(SNAPSHOT/'config.json',PINNED_SHA256['config.json']),
             (SNAPSHOT/'model.safetensors',PINNED_SHA256['model.safetensors']),(VECTOR_FILE,PINNED_SHA256['vectors.npz'])]:
            if sha(path)!=pin: raise ValueError('asset changed during run')
        result=dict(status='loop_gain_completed',computational_completed=True,certificate_passed=False,
            exploratory=True,producer_arrays_read=False,source_commit=source,plan_sha256=sha(args.plan),
            orbit_gates=orbit_gates,phases=completed_phases,cyclic_monodromies=loops,composed_controls=composed_controls,
            seconds=time.monotonic()-began,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            backend=dict(torch=str(torch.__version__),numpy=np.__version__,python=sys.version,
                         torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads()))
        write('status',result)
    except BaseException as error:
        write('status',dict(status='loop_gain_failed',stage=stage,error=repr(error),traceback=traceback.format_exc(),
              computational_completed=False,certificate_passed=False,completed_phases=completed_phases,
              seconds=time.monotonic()-began,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise

if __name__=='__main__': main()

"""Collect one fixed lower-bootstrap comparison only after root authorization."""
import os
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
for name in THREADS:
    os.environ[name] = '1'
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import sys
import time
import numpy as np
import scipy
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from neural_shrimp.gdmaps import GridMap
from neural_shrimp.trainability import SYSTEMS
from neural_shrimp.period2_branch import Options, PilotStop
from neural_shrimp.period2_branch_io import Store, file_entry, identity, load_record, canonical
from neural_shrimp.s1r_bootstrap import HiddenBridge, InterruptibleBudget
from neural_shrimp.s1r_lower_association import (ENDPOINTS, INPUT_SOURCE, INPUT_MANIFEST,
    TIMING_SOURCE, TIMING_MANIFEST, MAX_LINKS, MAX_PHASES, INTERNAL_SECONDS, STORE_BYTES,
    TOTAL_BYTES, LAUNCHER_BYTES, require, validate_endpoint, measured_cost_gate, Progress, compare_along_bridge)

INPUT = Path('/data2/neural-shrimp-data/explore/s1r-bootstrap-association/attempt-001-collection')
TIMING = Path('/data2/neural-shrimp-data/explore/s1r-branch-association/attempt-001')
PLAN = 'docs/plans/s1r-lower-association-implementation.md'
SOURCES = (PLAN, 'docs/plans/s1r-next-association-decision.md',
    'src/neural_shrimp/__init__.py', 'src/neural_shrimp/gdmaps.py', 'src/neural_shrimp/trainability.py',
    'src/neural_shrimp/models.py', 'src/neural_shrimp/period2_branch.py', 'src/neural_shrimp/period2_branch_io.py',
    'src/neural_shrimp/s1r_bootstrap.py', 'src/neural_shrimp/s1r_lower_association.py',
    'tools/explore_s1r_lower_association.py', 'tools/launch_s1r_lower_association.py')


def sources():
    result = {}
    for name in SOURCES:
        require((ROOT / name).read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT),
                'uncommitted source ' + name)
        result[name] = file_entry(ROOT / name)
    return result


def check_manifest(folder, expected):
    require(not folder.is_symlink() and not (folder / 'manifest.json').is_symlink(), 'no input symlink')
    require(file_entry(folder / 'manifest.json')['sha256'] == expected, 'input manifest hash')
    m = json.loads((folder / 'manifest.json').read_text())
    require(set(p.name for p in folder.iterdir()) == set(m['files']) | {'manifest.json'}, 'exact input inventory')
    for name, entry in m['files'].items():
        path = folder / name
        require(Path(name).name == name and path.is_file() and not path.is_symlink()
                and file_entry(path) == entry, 'input file binding ' + name)
    return m


def bound_inputs():
    collection_manifest = check_manifest(INPUT, INPUT_MANIFEST)
    check_manifest(TIMING, TIMING_MANIFEST)
    runtime = json.loads((INPUT / 'runtime-start.json').read_text())
    timing_runtime = json.loads((TIMING / 'runtime-start.json').read_text())
    require(runtime['source_commit'] == INPUT_SOURCE, 'endpoint execution source')
    require(timing_runtime['source_commit'] == TIMING_SOURCE, 'timing execution source')
    for saved in (runtime, timing_runtime):
        require(saved['nice'] >= 10 and all(saved['thread_env'][k] == '1' for k in THREADS), 'historical CPU timing resources')
        require(saved['numpy'] == np.__version__ and saved['scipy'] == scipy.__version__, 'timing library environment changed')
    timing = json.loads((TIMING / 'result.json').read_text())
    collection = json.loads((INPUT / 'result.json').read_text())
    size = sum(e['bytes'] for e in collection_manifest['files'].values()) + (INPUT / 'manifest.json').stat().st_size
    gate = measured_cost_gate(timing, collection, size)
    gate.update(timing_root=str(TIMING), timing_manifest_sha256=TIMING_MANIFEST,
                collection_root=str(INPUT), collection_manifest_sha256=INPUT_MANIFEST)
    kind, inputs = load_record(INPUT, '00000-bound_inputs.json')
    require(kind == 'bound_inputs', 'model record type')
    net, x, y, theta0 = SYSTEMS['S1R'](28002)
    require(asdict(net) == inputs['model'], 'model identity')
    for name, value in dict(x=x, y=y, theta0=theta0).items():
        require(identity(value) == inputs['identity'][name] and np.array_equal(value, inputs[name]), 'model/data identity ' + name)
    endpoints = {}
    for role, spec in ENDPOINTS.items():
        path = INPUT / spec['record']
        require(file_entry(path)['sha256'] == spec['json_sha256'], 'endpoint record identity')
        require(json.loads(path.read_text())['arrays_sha256'] == spec['npz_sha256'], 'endpoint NPZ identity')
        kind, node = load_record(INPUT, spec['record'])
        require(kind == 'node', 'endpoint record type')
        validate_endpoint(node, role)
        endpoints[role] = node
    return net, x, y, endpoints, inputs['identity'], gate


def run(out, review_commit):
    require(isinstance(review_commit, str) and re.fullmatch('[0-9a-f]{40}', review_commit) is not None,
            'exact independent review commit required')
    subprocess.run(['git', 'cat-file', '-e', review_commit + '^{commit}'], cwd=ROOT, check=True)
    source_inventory = sources()
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice10 required')
    store = Store(out, STORE_BYTES)
    started = time.monotonic()
    stop_requested = [False]
    old = signal.signal(signal.SIGTERM, lambda signum, frame: stop_requested.__setitem__(0, True))
    budget = InterruptibleBudget(INTERNAL_SECONDS, MAX_PHASES, lambda: stop_requested[0])
    progress = Progress(store)
    result = dict(schema='s1r-lower-association-v1', status='running', bridge=[], association=None,
                  historical_family_association='unresolved', evidence='exploratory numerical finite witnessed path; no proof or historical ancestry')
    engine = None
    try:
        runtime = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            source_review_commit=review_commit, sources=source_inventory,
            input_root=str(INPUT), input_manifest_sha256=INPUT_MANIFEST,
            timing_root=str(TIMING), timing_manifest_sha256=TIMING_MANIFEST,
            started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), nice=os.getpriority(os.PRIO_PROCESS, 0),
            thread_env={k: os.environ[k] for k in THREADS}, device='cpu', gpu=False, paid_compute=False,
            numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version(), load_average=os.getloadavg(),
            scientific_options=asdict(Options()),
            limits=dict(internal_seconds=INTERNAL_SECONDS, external_seconds=115, kill_grace_seconds=5,
                        nominal_links=75, phase_correctors=MAX_PHASES, link_attempts=MAX_LINKS,
                        store_bytes=STORE_BYTES, launcher_bytes=LAUNCHER_BYTES, total_raw_bytes=TOTAL_BYTES))
        store.json('runtime-start.json', runtime)
        net, x, y, endpoints, data_identity, gate = bound_inputs()
        result['cost_gate'] = gate
        budget.check()
        progress.emit('bound_inputs', dict(model=asdict(net), x=x, y=y, identity=data_identity,
            endpoints={role: {k: node[k] for k in ('id', 'params', 'rates', 'pair')} for role, node in endpoints.items()},
            input_source=INPUT_SOURCE, input_manifest_sha256=INPUT_MANIFEST, cost_gate=gate))
        def family(params):
            return GridMap(net, x, y, [10.**float(params[0])], [10.**float(params[1])]).cell_map(0)
        engine = HiddenBridge(family, Options(), budget, progress.emit, MAX_LINKS)
        progress.engine = engine
        compare_along_bridge(engine, endpoints['upper'], endpoints['lower'], result, progress)
        return result
    except PilotStop as error:
        result.update(status='resource_stop', reason=str(error), association=None)
        return result
    except BaseException as error:
        result.update(status='failed_or_interrupted', reason=type(error).__name__ + ': ' + str(error), association=None)
        raise
    finally:
        signal.signal(signal.SIGTERM, old)
        terminal = progress.terminal()
        result.update(phase_corrector_calls=budget.calls, link_attempts=0 if engine is None else len(engine.edges),
            processed_macro_segments=len(result['bridge']),
            completed_macro_links=sum(s['status']=='connected' for s in result['bridge']), witnessed_edge_ids=terminal['accepted_edges'],
            last_witnessed_node=None if terminal['last_witnessed'] is None else terminal['last_witnessed']['id'],
            elapsed_seconds=time.monotonic()-started, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        try:
            store.emit('terminal', terminal, final=True)
        except BaseException as error:
            result['terminal_checkpoint_failure'] = type(error).__name__ + ': ' + str(error)
        store.finish(result)
        for path in store.root.iterdir():
            path.chmod(0o444)
        store.root.chmod(0o555)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--review-commit', required=True)
    args = parser.parse_args()
    result = run(args.out, args.review_commit)
    print(canonical(result).decode())
    return 0 if result['status'] in ('completed_common_control_comparison', 'endpoint_requalification_failed', 'bridge_gap') else 2

if __name__ == '__main__':
    raise SystemExit(main())

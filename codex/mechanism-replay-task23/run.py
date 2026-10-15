"""Task23 reviewed finite-twin profile/run driver; no producer imports."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def pin_check(pin):
    path = Path(pin['path'])
    if path.stat().st_size != pin['bytes'] or sha(path) != pin['sha256']:
        raise ValueError('input pin mismatch: ' + str(path))


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if isinstance(value, float) and not __import__('math').isfinite(value):
        return {'undefined_float': repr(value)}
    return value


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(plain(value), stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write('\n')


def load_adjacent(name, pin):
    path = Path(__file__).resolve().with_name(name + '.py')
    if path.name != Path(pin['path']).name or sha(path) != pin['sha256'] or path.stat().st_size != pin['bytes']:
        raise ValueError('captured adjacent source mismatch: ' + name)
    module_name = 'task23_' + name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate_card(card):
    if card['schema'] != 'task23-finite-twin-card-v1' or card['status'] != 'reviewed-for-execution':
        raise ValueError('unreviewed card')
    stage = card['stage']
    if stage not in ('profile', 'actual'):
        raise ValueError('unknown stage')
    expected = (32, 16) if stage == 'profile' else (1000, 2000)
    if (card['transient'], card['record']) != expected:
        raise ValueError('fixed stage horizons required')
    if card['source_indices'] != [25, 29, 19, 35] or card['log10_temperatures'] != [.125, .129, .11900000000000001, .135]:
        raise ValueError('fixed four-cell source coordinates required')
    if card['alpha'] != .0655 or card['eps'] != [1e-6, 1e-9] or card['threads'] != 2:
        raise ValueError('fixed alpha, eps and CPU thread contract required')
    if card['original_cell_count'] != 201 or card['seed'] != 0:
        raise ValueError('original RNG block convention required')
    if sha(__file__) != card['sources']['run']['sha256']:
        raise ValueError('own driver pin mismatch')
    return card


def execute(card, out):
    start = time.perf_counter()
    for pin in card['input_pins']:
        pin_check(pin)
    import numpy as np
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 2 or torch.get_num_interop_threads() != 1:
        raise ValueError('effective thread limit mismatch')
    gain = load_adjacent('gain', card['sources']['gain'])
    qwen = load_adjacent('qwen_map', card['sources']['qwen_map'])
    assets = card['assets']
    for name in ('config', 'model', 'vector'):
        pin_check(assets[name])
    config = json.loads(Path(assets['config']['path']).read_text())
    with np.load(assets['vector']['path'], allow_pickle=False) as archive:
        vector = archive['hostility-vs-kindness'].copy()
    if vector.shape != (896,) or vector.dtype != np.float64 or not np.isfinite(vector).all():
        raise ValueError('exact stored full hostile vector required')
    with safe_open(assets['model']['path'], framework='pt', device='cpu') as weights_file:
        weights = {key: weights_file.get_tensor(key).to(torch.float64) for key in weights_file.keys()}
    model = qwen.ManualQwenSoftMap(config, weights, qwen.PREFIX_IDS, vector, steering_layer=12)
    if (model.d, model.layers, model.vocab) != (896, 24, 151936):
        raise ValueError('full target dimensions required')
    load_seconds = time.perf_counter() - start
    alpha = gain.ALPHA
    log_t = np.asarray(gain.LOG_T, dtype=np.float64)
    temperatures_np = 10. ** log_t
    temperatures = torch.from_numpy(temperatures_np.copy())
    full_temperatures = temperatures.repeat(3)
    tic = time.perf_counter()
    with torch.no_grad():
        cache = model.prefix_cache(alpha)
    prefix_seconds = time.perf_counter() - tic
    states = model.initial_state()[None, :].repeat(4, 1)
    metadata = {
        'schema': 'task23-native-metadata-v1', 'stage': card['stage'], 'config': config,
        'source_indices': list(gain.SOURCE_INDICES), 'log10_temperatures': list(gain.LOG_T),
        'log10_temperature_hex': [float(t).hex() for t in log_t],
        'physical_temperatures': temperatures_np.tolist(),
        'physical_temperature_hex': [float(t).hex() for t in temperatures_np],
        'alpha': alpha, 'eps': list(gain.EPS), 'prefix_ids': list(qwen.PREFIX_IDS),
        'numpy_version': np.__version__, 'torch_version': torch.__version__,
        'torch_intraop_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
        'tensor_dtypes': sorted(set(str(w.dtype) for w in model.weights.values())),
        'tensor_devices': sorted(set(str(w.device) for w in model.weights.values())),
        'rotary_dtype': str(model.inv_freq.dtype), 'steering_dtype': str(model.vector.dtype),
        'transient_batch': 4, 'measured_batch': 12,
        'initialization': 'last-prefix-token embedding; then original two201x896 RNG blocks at selected row indices',
        'batch_scope': 'Original transient had three identical copies per cell; own transient evaluates four bases. CPU/batch arithmetic can diverge.',
        'no_producer_outcomes_read': True, 'no_AD_power_iteration_or_crossover': True,
        'input_pins': card['input_pins'], 'asset_pins': assets,
        'source_sha256': {k: v['sha256'] for k, v in card['sources'].items()},
    }
    write_json(out / 'metadata.json', metadata)
    base_seconds, measured_seconds, accounting_seconds = 0., 0., 0.
    with (out / 'progress.jsonl').open('x') as progress:
        def event(row):
            progress.write(json.dumps(plain(row), allow_nan=False, sort_keys=True) + '\n')
            progress.flush()
        with torch.no_grad():
            for step in range(card['transient']):
                tic = time.perf_counter()
                _, _, _, states = gain.split_map(model, states, alpha, temperatures, cache)
                base_seconds += time.perf_counter() - tic
                if step == 0 or (step + 1) % 32 == 0 or step + 1 == card['transient']:
                    event({'stage': 'transient', 'completed': step + 1,
                           'base_norms': gain.norms(states).tolist()})
            directions = gain.original_directions()
            states = gain.initial_twins(states, directions)
            np.savez(out / 'initialization.npz', directions=directions,
                     measured_initial_states=states.numpy(), prefix_ids=np.asarray(qwen.PREFIX_IDS))
            base_states = np.lib.format.open_memmap(out / 'base-states.npy', mode='w+', dtype=np.float64,
                                                    shape=(card['record'] + 1, 4, 896))
            base_states[0] = states[:4].numpy()
            base_states.flush()
            accumulator = gain.Accumulator(states[:4], card['record'])
            tail = []
            event({'stage': 'measurement', 'completed': 0, 'valid_base_state_rows': 1})
            with (out / 'step-records.jsonl').open('x') as records:
                for step in range(card['record']):
                    tic = time.perf_counter()
                    h, z, p, feedback = gain.split_map(model, states, alpha, full_temperatures, cache)
                    measured_seconds += time.perf_counter() - tic
                    tic = time.perf_counter()
                    try:
                        record, next_states = gain.finite_step(states, h, feedback, z, p, full_temperatures)
                    except gain.UndefinedGain as exc:
                        records.write(json.dumps(plain({'step': step, 'numerically_defined': False,
                                                        'diagnostics': exc.diagnostics}), allow_nan=False) + '\n')
                        records.flush()
                        event({'stage': 'measurement', 'completed': step, 'valid_base_state_rows': step + 1,
                               'failure_step': step, 'numerically_defined': False})
                        raise
                    accumulator.add(record)
                    records.write(json.dumps(plain({'step': step, 'numerically_defined': True, **record}),
                                             allow_nan=False, sort_keys=True) + '\n')
                    states = next_states
                    base_states[step + 1] = states[:4].numpy()
                    tail.append(states[:4].numpy().copy())
                    if len(tail) > 66:
                        tail.pop(0)
                    accounting_seconds += time.perf_counter() - tic
                    if step == 0 or (step + 1) % 32 == 0 or step + 1 == card['record']:
                        records.flush()
                        base_states.flush()
                        event({'stage': 'measurement', 'completed': step + 1, 'valid_base_state_rows': step + 2})
            base_states.flush()
            summaries = accumulator.finish(states[:4], tail)
            np.savez(out / 'final-twins.npz', states=states.numpy(), tail=np.asarray(tail))
    for c, summary in enumerate(summaries):
        summary.update({'source_index': gain.SOURCE_INDICES[c], 'alpha': alpha,
                        'beta': 0., 'log10_temperature': gain.LOG_T[c],
                        'physical_temperature': float(temperatures_np[c])})
    # Includes scientific file serialization, excludes final rehash/report write.
    observed_seconds = time.perf_counter() - start
    observed_peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for pin in card['input_pins']:
        pin_check(pin)
    for name in ('config', 'model', 'vector'):
        pin_check(assets[name])
    outputs = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)}
               for p in sorted(out.iterdir()) if p.is_file()]
    report = {'schema': 'task23-finite-twin-result-v1', 'computational_status': 'complete',
              'stage': card['stage'], 'scientific_full_run_complete': card['stage'] == 'actual',
              'card_sha256': card['_card_sha256'],
              'source_sha256': metadata['source_sha256'], 'input_pins': card['input_pins'],
              'transient_steps': card['transient'], 'measurement_steps': card['record'],
              'cell_summaries': summaries, 'output_files': outputs,
              'runtime': {'load_and_initial_hash_seconds': load_seconds, 'prefix_seconds': prefix_seconds,
                          'base_forward_seconds': base_seconds, 'measured_forward_seconds': measured_seconds,
                          'accounting_and_stream_write_seconds': accounting_seconds,
                          'seconds_before_final_rehash_and_report': observed_seconds,
                          'peak_rss_kib_observed_before_final_rehash_and_report': observed_peak_kib,
                          'scope': 'Full worker wall belongs to external supervisor; no independent full-worker peak measurement.'},
              'scope': 'Profile summaries are cost/setup observations, not1000/2000 results. Finite twin identity is accounting, not a causal mechanism or tangent certificate.'}
    write_json(out / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--card', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out, owned = Path(args.out), False
    try:
        card = validate_card(json.loads(Path(args.card).read_text()))
        card['_card_sha256'] = sha(args.card)
        out.mkdir(parents=False, exist_ok=False)
        owned = True
        execute(card, out)
    except Exception as exc:
        if owned:
            write_json(out / 'failure.json', {'type': type(exc).__name__, 'error': str(exc),
                                              'diagnostics': getattr(exc, 'diagnostics', None),
                                              'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()

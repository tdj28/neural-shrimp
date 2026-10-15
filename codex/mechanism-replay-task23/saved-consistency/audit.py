"""Independent saved-state/scalar consistency, without model or producer imports.

This reads a closed, hash-bound run once. Source-identical saved arithmetic and
state links are exact checks; independent NumPy-versus-Torch norm reductions are
reported as diagnostics without a new tolerance or acceptance gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import resource
import traceback

import numpy as np

FILES = {'base-states.npy', 'final-twins.npz', 'initialization.npz', 'metadata.json',
         'progress.jsonl', 'report.json', 'step-records.jsonl'}
INDICES = [25, 29, 19, 35]
LOG_T = [.125, .129, .11900000000000001, .135]
EPS = [1e-6, 1e-9]
SOURCES = {'run': 'a6e83da06332cb5a27e1a167537226aaa38cfccdd1f87e92e28f17a5cc1093ed',
           'gain': '229d79a788589b5f860a0774f7273e9e93b54cecf8e6f474a3c5c3734eca751f',
           'qwen_map': '493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e'}
ORIGINAL_CARD = '93f519a6ac731486f761921efbc1a9e909be44d16a0ea16f2c75cb8d9d2cdd78'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def authenticate(entry):
    if pin(entry['path']) != entry:
        raise ValueError('input pin mismatch: ' + str(entry['path']))


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def finite(value, shape, name, positive=False):
    value = np.asarray(value)
    if value.dtype.kind not in 'fiu' or value.shape != shape or not np.isfinite(value).all():
        raise ValueError('wrong finite array: ' + name)
    if positive and not np.all(value > 0):
        raise ValueError('nonpositive operand: ' + name)
    return value.astype(np.float64, copy=False)


def require_summed_identity(residual):
    if not np.isfinite(residual).all() or np.any(np.abs(residual) > 1e-10):
        raise ValueError('original prospective summed accounting threshold failed')


def check_contract(metadata, original, summaries):
    temperatures = (10. ** np.asarray(LOG_T, dtype=np.float64)).tolist()
    if (original['source_indices'] != INDICES or original['log10_temperatures'] != LOG_T
            or original['alpha'] != .0655 or original['eps'] != EPS):
        raise ValueError('original card fixed physical contract mismatch')
    if (metadata['alpha'] != .0655 or metadata['eps'] != EPS
            or metadata['physical_temperatures'] != temperatures
            or metadata['log10_temperature_hex'] != [x.hex() for x in LOG_T]
            or metadata['physical_temperature_hex'] != [x.hex() for x in temperatures]):
        raise ValueError('metadata fixed physical contract mismatch')
    if len(summaries) != 4:
        raise ValueError('wrong summary cell count')
    for c, row in enumerate(summaries):
        if (row['source_index'] != INDICES[c] or row['log10_temperature'] != LOG_T[c]
                or row['physical_temperature'] != temperatures[c]
                or row['alpha'] != .0655 or row['beta'] != 0. or row['eps'] != EPS):
            raise ValueError('summary fixed physical contract mismatch')
    return temperatures


class Ledger:
    def __init__(self):
        self.checks = 0
        self.elements = 0
        self.nonexact = []
        self.norm_diagnostics = []

    def exact(self, name, recomputed, saved):
        a, b = np.asarray(recomputed), np.asarray(saved)
        self.checks += 1
        if a.shape != b.shape:
            raise ValueError('comparison shape mismatch: ' + name)
        self.elements += a.size
        if not np.array_equal(a, b):
            mask = a != b
            self.nonexact.append({'name': name, 'shape': list(a.shape),
                                  'mismatch_indices': np.argwhere(mask).tolist(),
                                  'recomputed': a[mask].tolist(), 'saved': b[mask].tolist(),
                                  'difference': (a[mask].astype(float) - b[mask].astype(float)).tolist()})

    def norm(self, name, states, stored):
        measured = np.linalg.norm(states, axis=-1)
        saved = finite(stored, measured.shape, name, positive=True)
        delta = measured - saved
        if np.any(delta != 0):
            self.norm_diagnostics.append({'name': name, 'numpy_norm': measured.tolist(),
                                          'saved_torch_norm': saved.tolist(),
                                          'signed_numpy_minus_saved': delta.tolist(),
                                          'relative_delta': (delta / saved).tolist()})


def consistency(base, initialization, final, rows, summaries, temperatures, expected_steps):
    """Pure saved-data checks; dimension/length can be tiny for manufactured controls."""
    if base.dtype != np.float64 or base.ndim != 3 or base.shape[0] != expected_steps + 1:
        raise ValueError('wrong saved base matrix')
    steps, cells, dimension = expected_steps, base.shape[1], base.shape[2]
    if cells != 4 or len(rows) != steps or len(summaries) != cells:
        raise ValueError('incomplete saved stream or cells')
    temperatures = finite(temperatures, (cells,), 'temperatures', positive=True)
    if any(not np.isfinite(base[i:i+64]).all() for i in range(0, len(base), 64)):
        raise ValueError('nonfinite native base state')
    initial = finite(initialization['measured_initial_states'], (3*cells, dimension), 'initial states')
    directions = finite(initialization['directions'], (2, cells, dimension), 'directions')
    final_states = finite(final['states'], (3*cells, dimension), 'final states')
    tail = finite(final['tail'], (min(66, steps), cells, dimension), 'tail')
    if any(np.asarray(value).dtype != np.float64 for value in
           (initialization['measured_initial_states'], initialization['directions'], final['states'], final['tail'])):
        raise ValueError('native saved arrays must be float64')
    ledger = Ledger()
    ledger.exact('initial base rows', base[0], initial[:cells])
    ledger.exact('final base rows', base[-1], final_states[:cells])
    ledger.exact('complete recorded tail', base[-len(tail):], tail)
    rng = np.random.default_rng(0)
    expected_directions = np.stack([rng.normal(size=(201, dimension))[INDICES] for _ in EPS])
    ledger.exact('both full201-row RNG selections', expected_directions, directions)
    totals = {key: np.zeros((2, cells)) for key in ('log_growth', 'log_g_net', 'log_g_read')}
    observations = {key: np.zeros(cells) for key in ('entropy', 'gap', 'p_top')}
    gap_counts = np.zeros((2, cells), dtype=np.int64)
    maximum_residual = np.zeros((2, cells))
    nonzero_residual = np.zeros((2, cells), dtype=np.int64)
    previous = None
    for i, row in enumerate(rows):
        if type(row['step']) is not int or row['step'] != i or row['numerically_defined'] is not True:
            raise ValueError('wrong/missing/undefined step identity')
        operands = {key: finite(row[key], (cells,), key, positive=True) for key in ('x_norm', 'h_norm', 'F_norm')}
        offsets = {key: finite(row[key], (2, cells), key, positive=True) for key in ('dx', 'dh', 'dF', 'next_dx')}
        dx, dh, df = offsets['dx'], offsets['dh'], offsets['dF']
        x, h, f = operands['x_norm'], operands['h_norm'], operands['F_norm']
        predicted = {'log_growth': np.log(df / dx),
                     'log_g_net': np.log((dh / h) / (dx / x)),
                     'log_g_read': np.log((df / f) / (dh / h)),
                     'boundary_step': np.log(f / x)}
        predicted['identity_residual'] = (predicted['log_growth'] - predicted['log_g_net']
                                          - predicted['log_g_read'] - predicted['boundary_step'][None, :])
        for key, computed in predicted.items():
            stored = finite(row[key], computed.shape, key)
            ledger.exact(f'step{i}/{key}', computed, stored)
        residual = finite(row['identity_residual'], (2, cells), 'identity')
        if np.any(np.abs(residual) > 1e-10):
            raise ValueError('original prospective accounting threshold failed')
        ledger.norm(f'step{i}/base', base[i], x)
        ledger.norm(f'step{i}/feedback', base[i+1], f)
        if previous is not None:
            ledger.exact(f'step{i}/saved base norm link', previous['F_norm'], x)
            ledger.exact(f'step{i}/saved represented offset link', previous['next_dx'], dx)
        else:
            ledger.norm('initial represented offsets', np.stack([initial[(k+1)*cells:(k+2)*cells]-initial[:cells] for k in range(2)]), dx)
        for key in totals:
            totals[key] += finite(row[key], (2, cells), key)
        for key in observations:
            observations[key] += finite(row[key], (cells,), key)
        logits = finite(row['top2_logits'], (cells, 2), 'top2 logits')
        top_ids = np.asarray(row['top2_ids'])
        if top_ids.shape != (cells, 2) or top_ids.dtype.kind not in 'iu' or np.any(top_ids < 0) or np.any(top_ids >= 151936):
            raise ValueError('invalid top-two vocabulary indices')
        ledger.exact(f'step{i}/gap from saved logits', (logits[:, 0]-logits[:, 1])/temperatures, row['gap'])
        gap = finite(row['gap'], (cells,), 'gap')
        gap_counts += np.stack([gap < 1., gap < 2.])
        maximum_residual = np.maximum(maximum_residual, np.abs(residual))
        nonzero_residual += residual != 0
        previous = row
    ledger.norm('final represented offsets', np.stack([final_states[(k+1)*cells:(k+2)*cells]-final_states[:cells] for k in range(2)]), rows[-1]['next_dx'])
    boundary = np.log(np.asarray(rows[-1]['F_norm']) / np.asarray(rows[0]['x_norm'])) / steps
    means = {key: val / steps for key, val in totals.items()}
    scalar_residual = means['log_growth'] - means['log_g_net'] - means['log_g_read'] - boundary[None, :]
    require_summed_identity(scalar_residual)
    for c, summary in enumerate(summaries):
        if summary['measurement_steps'] != steps or summary['eps'] != EPS or summary['numerically_defined'] is not True:
            raise ValueError('summary scope is incomplete')
        require_summed_identity(finite(summary['identity_residual'], (2,), 'saved summary identity'))
        expected = {'lyapunov': means['log_growth'][:, c],
                    'both_twin_mean_log_g_net': means['log_g_net'][:, c],
                    'both_twin_mean_log_g_read': means['log_g_read'][:, c],
                    'mean_log_g_net': means['log_g_net'][0, c], 'mean_log_g_read': means['log_g_read'][0, c],
                    'boundary_term': boundary[c], 'identity_residual': scalar_residual[:, c],
                    'identity_abs_tolerance': 1e-10, 'max_step_identity_abs': maximum_residual[:, c],
                    'nonzero_step_identity_count': nonzero_residual[:, c],
                    'mean_entropy': observations['entropy'][c]/steps, 'mean_gap': observations['gap'][c]/steps,
                    'mean_p_top': observations['p_top'][c]/steps,
                    'frac_gap_below_1': gap_counts[0,c]/steps, 'frac_gap_below_2': gap_counts[1,c]/steps,
                    'tail_states': len(tail)}
        distances = [float(np.linalg.norm(tail[-1-q, c]-tail[-1, c])) for q in range(1, min(65, len(tail)))]
        threshold = 1e-8 * max(1., float(np.linalg.norm(tail[-1,c])))
        period = next((q for q, distance in enumerate(distances, 1) if distance < threshold), 0)
        expected.update(lag_distances=distances, threshold=threshold, period=period,
                        literal_chaotic=period == 0 and all(x > 1e-3 for x in means['log_growth'][:,c]))
        for key, value in expected.items():
            ledger.exact(f'cell{c}/{key}', value, summary[key])
    return {'steps': steps, 'cells': cells, 'dimension': dimension,
            'exact_comparisons': ledger.checks, 'compared_scalar_elements': ledger.elements,
            'source_identical_arithmetic_and_state_links_exact': not ledger.nonexact,
            'all_nonexact_comparisons': ledger.nonexact,
            'independent_norm_reduction_diagnostics': ledger.norm_diagnostics,
            'norm_diagnostics_have_no_new_threshold': True,
            'original_summed_accounting_threshold_passed': True,
            'initial_directional_construction_independently_reconstructed': False,
            'scope': 'Saved stream/state/summary consistency only. Exact RNG selections and offset norms are checked; initial directional construction relies on reviewed source. No map evaluation, transient replay, or independent reconstruction of entropy from unsaved probabilities.'}


def execute(card, out):
    start = time.perf_counter()
    if set(card['inputs']) != {'report', 'manifest', 'closure', 'authority', 'original_card'}:
        raise ValueError('exact five authority inputs required')
    for value in card['inputs'].values():
        authenticate(value)
    read = lambda role: json.loads(Path(card['inputs'][role]['path']).read_text())
    report, manifest, closure, authority, original = [read(key) for key in ('report', 'manifest', 'closure', 'authority', 'original_card')]
    if card['inputs']['original_card']['sha256'] != ORIGINAL_CARD or report['card_sha256'] != ORIGINAL_CARD or closure['card_sha256'] != ORIGINAL_CARD:
        raise ValueError('wrong original actual card')
    if authority['report'] != card['inputs']['report'] or authority['worker_manifest'] != card['inputs']['manifest'] or authority['worker_closure'] != card['inputs']['closure']:
        raise ValueError('original owner does not bind these result/manifest/closure bytes')
    if closure['exit_code'] != 0 or closure['timed_out'] is not False or closure['failure'] is not None or closure['owned_worker_and_groups_closed'] is not True:
        raise ValueError('original worker did not successfully close')
    if authority['tool_session_reaped_exit_code'] != 0 or authority['both_original_pids_absent'] is not True or authority['both_original_groups_and_sessions_absent'] is not True:
        raise ValueError('original owner reaping/closure absent')
    if authority['source_commit'] != closure['source_commit'] or authority['source_blobs_verified'] != 5 or authority['worker_and_supervision_manifests_verified'] is not True:
        raise ValueError('original source/manifest authority mismatch')
    if report['stage'] != 'actual' or report['scientific_full_run_complete'] is not True or report['computational_status'] != 'complete' or report['transient_steps'] != 1000 or report['measurement_steps'] != 2000:
        raise ValueError('exact complete actual horizon required')
    if report['source_sha256'] != SOURCES or closure['wrapper_sha256'] != SOURCES['run']:
        raise ValueError('wrong accepted scientific source')
    root = Path(card['inputs']['report']['path']).parent
    entries = manifest['files']
    if len(entries) != 7 or {entry['path'] for entry in entries} != FILES:
        raise ValueError('exact seven saved output entries required')
    original_pins = []
    for entry in entries:
        full = {'path': str(root / entry['path']), 'bytes': entry['bytes'], 'sha256': entry['sha256']}
        authenticate(full)
        original_pins.append(full)
    if sorted(report['output_files'], key=lambda x:x['path']) != sorted([row for row in entries if row['path'] != 'report.json'], key=lambda x:x['path']):
        raise ValueError('report and sealed output ledgers differ')
    metadata = json.loads((root/'metadata.json').read_text())
    if metadata['source_sha256'] != SOURCES or metadata['source_indices'] != INDICES or metadata['log10_temperatures'] != LOG_T:
        raise ValueError('metadata scientific identities differ')
    if metadata['stage'] != 'actual' or metadata['tensor_dtypes'] != ['torch.float64'] or metadata['tensor_devices'] != ['cpu']:
        raise ValueError('metadata effective dtype/device mismatch')
    if metadata['input_pins'] != original['input_pins'] or metadata['asset_pins'] != original['assets']:
        raise ValueError('original source/input authority mismatch')
    base = np.load(root/'base-states.npy', mmap_mode='r', allow_pickle=False)
    if base.shape != (2001,4,896):
        raise ValueError('wrong full native matrix shape')
    with np.load(root/'initialization.npz', allow_pickle=False) as archive:
        initialization = {key: archive[key] for key in archive.files}
    with np.load(root/'final-twins.npz', allow_pickle=False) as archive:
        final = {key: archive[key] for key in archive.files}
    if set(initialization) != {'directions','measured_initial_states','prefix_ids'} or set(final) != {'states','tail'}:
        raise ValueError('wrong native array names')
    if initialization['prefix_ids'].tolist() != metadata['prefix_ids']:
        raise ValueError('native prefix IDs differ from original metadata')
    rows = [json.loads(line) for line in (root/'step-records.jsonl').read_text().splitlines()]
    summaries = report['cell_summaries']
    temperatures = check_contract(metadata, original, summaries)
    result = consistency(base, initialization, final, rows, summaries, temperatures, 2000)
    progress = [json.loads(line) for line in (root/'progress.jsonl').read_text().splitlines()]
    transient = [row['completed'] for row in progress if row['stage']=='transient']
    measured = [row for row in progress if row['stage']=='measurement']
    if transient != sorted({1,1000,*range(32,1001,32)}) or [row['completed'] for row in measured] != sorted({0,1,2000,*range(32,2001,32)}):
        raise ValueError('missing/duplicate progress checkpoints')
    if any(row['valid_base_state_rows'] != row['completed']+1 for row in measured):
        raise ValueError('incorrect base-state valid-prefix receipt')
    for entry in [*card['inputs'].values(), *original_pins]:
        authenticate(entry)
    save(out/'report.json', {'schema':'task23-saved-consistency-v1', 'computational_status':'complete',
                            'audit_source_sha256':sha(__file__), 'card_sha256':card['_sha256'],
                            'original_inputs':card['inputs'], 'original_artifacts':original_pins,
                            'original_execution_attribution':'Inference; this audit only authenticates and inspects saved evidence',
                            'seconds_before_final_report':time.perf_counter()-start,
                            'observed_peak_rss_kib_before_final_report':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            **result})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--card', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    card = json.loads(Path(args.card).read_text())
    if card['schema'] != 'task23-saved-consistency-card-v1' or card['status'] != 'reviewed-for-execution' or card['audit_sha256'] != sha(__file__):
        raise ValueError('unreviewed saved audit cannot execute')
    card['_sha256'] = sha(args.card)
    out = Path(args.out)
    out.mkdir(parents=False, exist_ok=False)
    try:
        execute(card,out)
    except Exception as error:
        save(out/'failure.json', {'type':type(error).__name__, 'error':str(error), 'traceback':traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()

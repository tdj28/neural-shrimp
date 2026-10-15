"""Own Task23 closed saved-result comparison; no model/producer imports.

All differences are descriptive. Input identity/coverage and own floating gain
accounting are distinct from agreement between finite chaotic trajectories.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback

INDICES = [25, 29, 19, 35]
LOG_T = [.125, .129, .11900000000000001, .135]
EPS = [1e-6, 1e-9]
ALPHA = .0655
REQUIRED_FIELDS = ['mean_log_g_net', 'mean_log_g_read', 'boundary_term', 'mean_entropy', 'mean_gap']
IDENTITY_TOLERANCE = 1e-10
ORIGINAL_SOURCE_SHA = 'ccb427ec06f21620858808a8d51220c0056bfb218f742d54c625ba61d7d0e8a5'
ROLES = {'own_report', 'own_metadata', 'own_card', 'own_closure', 'own_authority',
         'producer_gain', 'original_metadata', 'original_authority', 'original_source',
         'own_run_source', 'own_gain_source', 'own_map_source'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def authenticate(pin):
    path = Path(pin['path'])
    if path.stat().st_size != pin['bytes'] or sha(path) != pin['sha256']:
        raise ValueError('input hash/size mismatch: ' + str(path))


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return {'nonfinite_float': repr(x)}
    return x


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(plain(value), stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write('\n')


def numeric(x, role):
    if type(x) not in (int, float):
        raise ValueError('numeric saved scalar required: ' + role)
    return float(x)


def difference(own, producer, name):
    a, b = numeric(own, name + '/own'), numeric(producer, name + '/producer')
    finite = math.isfinite(a) and math.isfinite(b)
    delta = a-b if finite else None
    return {'field': name, 'own': own, 'producer': producer,
            'own_binary64_hex': a.hex(), 'producer_binary64_hex': b.hex(),
            'both_finite': finite, 'signed_own_minus_producer': delta,
            'absolute_difference': abs(delta) if delta is not None else None,
            'difference_finite': delta is not None and math.isfinite(delta),
            'exact_saved_scalar_equality': a == b if finite else None,
            'agreement_gate': None,
            'producer_precision': 'JSON round-trip of source float64 aggregate; no explicit decimal rounding in writer'}


def exact_bool(value, name):
    if type(value) is not bool:
        raise ValueError('Boolean required: ' + name)
    return value


def finite_label(row, key):
    period = row['period']
    if type(period) is not int or not 0 <= period <= 64:
        raise ValueError('integer source period0..64 required')
    rates = [numeric(v, 'lyapunov') for v in row['lyapunov']]
    if len(rates) != 2:
        raise ValueError('both finite-twin rates required')
    literal = period == 0 and all(v > 1e-3 for v in rates)
    stored = exact_bool(row[key], key)
    return {'period': period, 'stored_chaotic': stored, 'literal_chaotic_from_saved_rates_and_period': literal,
            'stored_label_matches_literal': stored == literal,
            'rates_finite': all(math.isfinite(v) for v in rates),
            'scope': 'finite last-state return and two finite-twin rates; not primitive-orbit/chaos proof'}


def selected_comparison(own, own_metadata, producer, original_metadata):
    if own['schema'] != 'task23-finite-twin-result-v1' or own['computational_status'] != 'complete':
        raise ValueError('completed own result required')
    if own['stage'] != 'actual' or own['scientific_full_run_complete'] is not True:
        raise ValueError('short profile cannot substitute for full replay')
    if own['transient_steps'] != 1000 or own['measurement_steps'] != 2000:
        raise ValueError('exact1000/2000 own horizons required')
    if own_metadata['stage'] != 'actual' or own_metadata['source_indices'] != INDICES or own_metadata['log10_temperatures'] != LOG_T:
        raise ValueError('wrong own metadata coordinates/order')
    if own_metadata['alpha'] != ALPHA or own_metadata['eps'] != EPS:
        raise ValueError('wrong own alpha/offsets')
    if own_metadata['tensor_dtypes'] != ['torch.float64'] or own_metadata['tensor_devices'] != ['cpu']:
        raise ValueError('wrong own stored effective precision/device')
    if own_metadata['transient_batch'] != 4 or own_metadata['measured_batch'] != 12:
        raise ValueError('wrong own batch convention')
    if producer['args'] != original_metadata['args']:
        raise ValueError('producer args differ from source-bound metadata')
    args = producer['args']
    if args['transient'] != 1000 or args['record'] != 2000 or args['eps'] != EPS or args['layer'] != 12:
        raise ValueError('wrong producer source horizons/offset/layer')
    if args['alpha'] != [ALPHA, ALPHA, 1.] or args['beta'] != [0., 0., 1] or args['tgrid'] != [.1, .3, 201.]:
        raise ValueError('wrong producer line/steering')
    cells = producer['cells']
    if len(cells) != 201 or original_metadata['cell_count'] != 201:
        raise ValueError('complete original201-row source table required')
    coords = []
    for row in cells:
        coord = tuple(numeric(row[key], key) for key in ('alpha', 'beta', 'log10_temperature'))
        if not all(math.isfinite(v) for v in coord):
            raise ValueError('nonfinite original coordinate')
        coords.append(coord)
    if len(set(coords)) != 201:
        raise ValueError('duplicate producer coordinate rows')
    rows = own['cell_summaries']
    if len(rows) != 4 or [row['source_index'] for row in rows] != INDICES:
        raise ValueError('exact four own rows in original requested order required')
    if len(original_metadata['selected']) != 4:
        raise ValueError('wrong source metadata selection')
    result = []
    for position, (index, log_t) in enumerate(zip(INDICES, LOG_T)):
        a, b, meta = rows[position], cells[index], original_metadata['selected'][position]
        if meta['source_index'] != index or meta['saved_log10_temperature'] != log_t or meta['saved_alpha'] != ALPHA or meta['saved_beta'] != 0.:
            raise ValueError('wrong original metadata row')
        if (a['alpha'], a['beta'], a['log10_temperature']) != (ALPHA, 0., log_t) or coords[index] != (ALPHA, 0., log_t):
            raise ValueError('selected cell parameter/index mismatch')
        if a['eps'] != EPS or b['eps'] != EPS or a['measurement_steps'] != 2000 or a['numerically_defined'] is not True:
            raise ValueError('wrong/incomplete own offset/measurement qualification')
        if len(a['lyapunov']) != 2 or len(b['lyapunov']) != 2:
            raise ValueError('both source exponents required')
        diffs = [difference(a['lyapunov'][k], b['lyapunov'][k], f'lyapunov_eps_{EPS[k]}') for k in range(2)]
        diffs += [difference(a[key], b[key], key) for key in REQUIRED_FIELDS]
        for key in ['both_twin_mean_log_g_net', 'both_twin_mean_log_g_read', 'identity_residual']:
            if len(a[key]) != 2:
                raise ValueError('two own split/identity entries required')
        if a['mean_log_g_net'] != a['both_twin_mean_log_g_net'][0] or a['mean_log_g_read'] != a['both_twin_mean_log_g_read'][0]:
            raise ValueError('own first-twin summary association mismatch')
        if a['identity_abs_tolerance'] != IDENTITY_TOLERANCE:
            raise ValueError('unexpected own identity diagnostic threshold')
        own_identity = []
        for k in range(2):
            lam = numeric(a['lyapunov'][k], 'own lambda')
            net = numeric(a['both_twin_mean_log_g_net'][k], 'own net')
            read = numeric(a['both_twin_mean_log_g_read'][k], 'own read')
            boundary = numeric(a['boundary_term'], 'own boundary')
            residual = lam-net-read-boundary
            own_identity.append({'eps': EPS[k], 'lambda': lam, 'net': net, 'read': read, 'boundary': boundary,
                                 'predicted_lambda_net_plus_read_plus_boundary': net+read+boundary,
                                 'recomputed_source_order_residual': residual,
                                 'saved_residual': a['identity_residual'][k],
                                 'saved_residual_exactly_matches': residual == a['identity_residual'][k],
                                 'within_own_prospective_floating_tolerance': math.isfinite(residual) and abs(residual) <= IDENTITY_TOLERANCE,
                                 'tolerance': IDENTITY_TOLERANCE,
                                 'scope': 'floating scalar consistency, not a certified bound or agreement gate'})
        values = [numeric(b['lyapunov'][0], 'producer lambda')] + [numeric(b[key], key) for key in ('mean_log_g_net', 'mean_log_g_read', 'boundary_term')]
        producer_identity = {'available': True, 'twin': EPS[0], 'raw_components': values,
                             'residual_lambda_minus_net_minus_read_minus_boundary': values[0]-values[1]-values[2]-values[3]
                             if all(math.isfinite(v) for v in values) else None,
                             'agreement_or_identity_pass_gate': None,
                             'scope': 'source-rounded binary64 scalar accounting only; original per-step float64 record is not available'}
        result.append({'source_index': index, 'requested_position': position,
                       'requested_log10_temperature': meta['requested_log10_temperature'],
                       'saved_log10_temperature': log_t, 'saved_log10_temperature_hex': log_t.hex(),
                       'exact_requested_float_match': meta['exact_requested_float_match'],
                       'own_full_saved_row': a, 'producer_full_saved_row': b,
                       'comparisons': diffs, 'own_identity': own_identity, 'producer_first_twin_identity': producer_identity,
                       'own_label': finite_label(a, 'literal_chaotic'), 'producer_label': finite_label(b, 'chaotic'),
                       'period_equal': a['period'] == b['period'],
                       'chaotic_label_equal': a['literal_chaotic'] == b['chaotic'],
                       'unavailable_comparisons': ['producer second-twin net/read split', 'producer full2000-step float64 scalar stream',
                                                   'producer exact runtime prefix IDs/native twin states'],
                       'excluded_auxiliary_fields': ['readout power/trace/rank/Rayleigh statistics are retained raw only, not recomputed or compared']})
    return {'rows': result, 'selected_cells': 4, 'required_scalar_comparisons': 28,
            'numeric_disagreement_is_a_failure': False,
            'producer_aggregate_rounding': 'Required fields are Python float conversions of float64 sums/record or finite norms, written by json.dumps without explicit decimal rounding. The first1000 saved step arrays are float32 and are not used here.',
            'batch_platform_scope': {'own': 'CPU float64,4 transient bases/12 measured rows',
                                     'producer': 'original CUDA float64 path,201 cells/603 rows; historical runtime/source limits inherited'},
            'scope': 'Aligned parameters and RNG positions do not force trajectory equality. Finite exponents/period labels do not prove asymptotic chaos, stability or causality.'}


def execute(card, out):
    pins = card['inputs']
    if set(pins) != ROLES:
        raise ValueError('exact closed-input role set required')
    if len({p['path'] for p in pins.values()}) != len(pins):
        raise ValueError('duplicate role input paths')
    for pin in pins.values():
        authenticate(pin)
    if pins['original_source']['sha256'] != ORIGINAL_SOURCE_SHA:
        raise ValueError('wrong original scientific source')
    read = lambda role: json.loads(Path(pins[role]['path']).read_text())
    own, meta, original_meta = read('own_report'), read('own_metadata'), read('original_metadata')
    metadata_rows = [row for row in own['output_files'] if row['path'] == 'metadata.json']
    if len(metadata_rows) != 1 or any(metadata_rows[0][key] != pins['own_metadata'][key] for key in ('bytes', 'sha256')):
        raise ValueError('own report does not bind exact metadata bytes')
    if Path(pins['own_metadata']['path']) != Path(pins['own_report']['path']).with_name('metadata.json'):
        raise ValueError('own metadata must be the recorded sibling')
    original_gain_pin = original_meta['gain']
    if pins['producer_gain'] != original_gain_pin:
        raise ValueError('producer gain bytes not the source-bound original')
    own_card, closure, authority = read('own_card'), read('own_closure'), read('own_authority')
    if own['card_sha256'] != pins['own_card']['sha256'] or own_card['stage'] != 'actual' or own_card['status'] != 'reviewed-for-execution':
        raise ValueError('own result/card association mismatch')
    if own_card['transient'] != 1000 or own_card['record'] != 2000:
        raise ValueError('own original card has wrong horizons')
    if closure['exit_code'] != 0 or closure['timed_out'] is not False or closure['failure'] is not None or closure['owned_worker_and_groups_closed'] is not True:
        raise ValueError('own original worker not successfully closed')
    if closure['card_sha256'] != pins['own_card']['sha256'] or closure['wrapper_sha256'] != pins['own_run_source']['sha256']:
        raise ValueError('own closure source/card association mismatch')
    required_owner = {'source_commit', 'tool_session_reaped_exit_code', 'both_original_pids_absent',
                      'both_original_groups_and_sessions_absent', 'source_blobs_verified',
                      'worker_and_supervision_manifests_verified'}
    if set(card['owner_assertions']) != required_owner or any(v is None for v in card['owner_assertions'].values()):
        raise ValueError('all actual owner authority assertions must be frozen')
    for name, expected in card['owner_assertions'].items():
        if authority[name] != expected:
            raise ValueError('owner authority assertion mismatch: ' + name)
    for name in ['both_original_pids_absent', 'both_original_groups_and_sessions_absent']:
        if authority[name] is not True:
            raise ValueError('original owner closure required')
    for name in ['source_blobs_verified', 'worker_and_supervision_manifests_verified']:
        if not authority[name]:
            raise ValueError('owner source/manifest verification required')
    if authority['tool_session_reaped_exit_code'] != 0 or authority['source_commit'] != closure['source_commit']:
        raise ValueError('owner reaping/source association mismatch')
    if authority['report'] != pins['own_report']:
        raise ValueError('owner authority does not bind exact own result')
    for role, key in [('own_run_source', 'run'), ('own_gain_source', 'gain'), ('own_map_source', 'qwen_map')]:
        expected = pins[role]['sha256']
        if own['source_sha256'][key] != expected or meta['source_sha256'][key] != expected or own_card['sources'][key]['sha256'] != expected:
            raise ValueError('scientific source association mismatch: ' + key)
    result = selected_comparison(own, meta, read('producer_gain'), original_meta)
    for pin in pins.values():
        authenticate(pin)
    save(out / 'input-manifest.json', {'inputs': pins, 'owner_assertions': card['owner_assertions'],
                                      'original_authority_basis': card['original_authority_basis']})
    save(out / 'report.json', {'schema': 'task23-independent-saved-comparison-v1',
                              'computational_status': 'complete', 'comparison_card_sha256': card['_sha256'],
                              'input_pins': pins, **result,
                              'execution_attribution': 'Original run/receipt/reaping observed by Inference; this worker compares authenticated saved fields only.'})


def dispatch(card_path, out):
    card = json.loads(Path(card_path).read_text())
    if card['schema'] != 'task23-comparison-card-v1' or card['status'] != 'reviewed-for-execution':
        raise ValueError('prepared comparison cannot execute')
    if card['comparator_sha256'] != sha(__file__):
        raise ValueError('wrong comparator source')
    card['_sha256'] = sha(card_path)
    out = Path(out)
    out.mkdir(parents=False, exist_ok=False)
    try:
        execute(card, out)
    except Exception as exc:
        save(out / 'failure.json', {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--card', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    dispatch(args.card, args.out)


if __name__ == '__main__':
    main()

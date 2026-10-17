"""Pinned current-runtime tokenizer-only replay. No model, producer imports or network.

Two literal source loader calls, unchanged independent pm_core.answer extraction.
Profile mode reads metadata/source/tokenizer assets only, never target records.
"""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

ROUNDTRIP = 'Natalia sold 48 clips in April.\nThen she sold half as many.'
CONDITIONS = ('gen-bf16', 'gen-bf16-4096', 'gen-fp32', 'reason-plain',
              'reason-plain-ans1024', 'reason-plain-eager', 'reason-plain-legacy', 'reason-soft')
LIMITS = {'cpu_seconds': 55, 'wall_seconds': 60, 'address_space_bytes': 12884901888,
          'output_bytes': 67108864, 'input_bytes': 134217728, 'threads': 1, 'nice': 10}
BASE = '/data2/neural-shrimp-data/codex/task27-ready-review-20261017/run-002/'
ENV = {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
       'HF_HUB_DISABLE_TELEMETRY': '1', 'CUDA_VISIBLE_DEVICES': '', 'USE_TF': '0',
       'USE_FLAX': '0', 'TOKENIZERS_PARALLELISM': 'false',
       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def pin(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()}


def verify(p):
    got = pin(p['path'])
    if any(got[k] != p[k] for k in ('bytes', 'sha256')):
        raise ValueError('pin mismatch: ' + p['path'])
    return got


def adjacent(name):
    path = Path(__file__).resolve().with_name(name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Store:
    def __init__(self, out, cap):
        self.out, self.cap, self.total, self.files = out, cap, 0, []

    def write(self, name, value):
        data = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode()
        next_total = self.total + len(data)
        if next_total > self.cap - (0 if name == 'failure.json' else 65536):
            raise ValueError('owned output ceiling exceeded')
        with (self.out / name).open('xb') as f:
            f.write(data)
        self.total = next_total
        self.files.append(pin(self.out / name))


def roundtrip(tok):
    ids = tok(ROUNDTRIP, add_special_tokens=False)['input_ids']
    text = tok.decode(ids)  # Source roundtrip does not set skip_special_tokens.
    return {'input': ROUNDTRIP, 'ids': list(ids), 'decoded': text, 'ok': text == ROUNDTRIP}


def checked_choice(auto, fallback):
    before = roundtrip(auto)
    if before['ok']:
        return auto, {'auto': before, 'fallback_used': False, 'selected': before}
    tok = fallback()
    final = roundtrip(tok)
    if not final['ok']:
        raise ValueError('source checked-loader refusal: fallback roundtrip failed')
    return tok, {'auto': before, 'fallback_used': True, 'selected': final}


def backend_state(stage):
    """Passive import/state accounting; no CUDA availability query or initialization."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('GPU visibility contract changed')
    torch = sys.modules.get('torch')
    initialized = None if torch is None else bool(torch.cuda.is_initialized())
    state = {'stage': stage, 'torch_imported': torch is not None,
             'torch_runtime_version': None if torch is None else str(torch.__version__),
             'torch_module_path': None if torch is None else str(torch.__file__),
             'cuda_initialized': initialized,
             'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
             'scope': 'Passive PyTorch initialization flag; native CUDA libraries may be mapped.'}
    if initialized:
        raise ValueError('unexpected PyTorch CUDA initialization: ' + stage)
    return state


def load_pair(meta):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    kwargs = {'local_files_only': True, 'trust_remote_code': False}
    path = meta['snapshot']  # Literal local snapshot path; no Hub ID rewrite.
    legacy = AutoTokenizer.from_pretrained(path, **kwargs)
    auto2 = AutoTokenizer.from_pretrained(path, **kwargs)
    checked, trace = checked_choice(auto2, lambda: PreTrainedTokenizerFast.from_pretrained(path, **kwargs))
    def describe(tok):
        return {'class': type(tok).__name__, 'module': type(tok).__module__,
                'name_or_path': tok.name_or_path, 'vocabulary_length': len(tok),
                'decoder': repr(tok.backend_tokenizer.decoder),
                'pre_tokenizer': repr(tok.backend_tokenizer.pre_tokenizer),
                'clean_up_tokenization_spaces': tok.clean_up_tokenization_spaces}
    return legacy, checked, {'legacy': describe(legacy), 'checked': describe(checked),
                            'legacy_roundtrip': roundtrip(legacy), 'checked_loader': trace}


def decode_row(row, legacy, checked, allowed, extract):
    present = 'answer_ids' in row
    ids = row.get('answer_ids', [])  # Literal producer's missing-field projection.
    errors = []
    if not isinstance(ids, list):
        errors.append('answer_ids_not_list')
    elif any(type(x) is not int or x < 0 for x in ids):
        errors.append('answer_ids_not_nonnegative_integers')
    elif any(x not in allowed[0] or x not in allowed[1] for x in ids):
        errors.append('answer_ids_outside_current_vocabularies')
    result = {'ids_present': present, 'ids': ids, 'id_count': len(ids) if isinstance(ids, list) else None,
              'input_errors': errors, 'qualified_ids': present and not errors,
              'literal_missing_ids_projection': not present, 'legacy_text': None,
              'checked_text': None, 'legacy_answer': None, 'checked_answer': None,
              'answers_changed': None, 'decode_or_extract_error': None}
    if not errors:
        try:
            result['legacy_text'] = legacy.decode(ids, skip_special_tokens=True)
            result['checked_text'] = checked.decode(ids, skip_special_tokens=True)
            result['legacy_answer'] = extract(result['legacy_text'], 'pm2')
            result['checked_answer'] = extract(result['checked_text'], 'pm2')
            result['answers_changed'] = result['legacy_answer'] != result['checked_answer']
        except Exception as e:
            result['decode_or_extract_error'] = type(e).__name__ + ': ' + str(e)
    return result


def condition(name, rows, keep, expected_order, legacy, checked, allowed, extract):
    ledger, last, seen = [], {}, {}
    for line, row in enumerate(rows, 1):
        idx = row.get('index') if isinstance(row, dict) else None
        valid_index = type(idx) is int
        selected = valid_index and idx in keep
        previous = list(seen.get(idx, [])) if valid_index else []
        if valid_index:
            seen.setdefault(idx, []).append(line)
        item = {'line': line, 'index': idx, 'index_valid': valid_index,
                'in_selected_64': selected, 'duplicate_of_lines': previous,
                'selected_last_write': False}
        if isinstance(row, dict):
            item.update(decode_row(row, legacy, checked, allowed, extract))
        else:
            item['source_row_error'] = 'not_object'
        ledger.append(item)
        if selected:
            last[idx] = item
    chosen = list(last.values())
    for item in chosen:
        item['selected_last_write'] = True
    actual_order = [x['index'] for x in ledger]
    missing = sorted(set(keep) - set(last))
    extra = [x['line'] for x in ledger if not x['in_selected_64']]
    duplicates = [x['line'] for x in ledger if x['duplicate_of_lines']]
    literal_defined = all(x.get('answers_changed') is not None for x in chosen)
    qualified = [x for x in chosen if x.get('qualified_ids') and x.get('answers_changed') is not None]
    strict = (len(ledger) == 64 and not missing and not extra and not duplicates
              and actual_order == expected_order)
    return {'name': name, 'source_rows': len(ledger), 'n': len(chosen),
            'literal_complete': len(chosen) == len(keep), 'strict_cohort_complete': strict,
            'order_matches_metadata': actual_order == expected_order,
            'missing_indices': missing, 'extra_or_invalid_index_lines': extra,
            'duplicate_lines': duplicates,
            'selected_missing_ids': [x['index'] for x in chosen if not x.get('ids_present')],
            'selected_invalid_ids_or_decode': [x['index'] for x in chosen if x.get('answers_changed') is None],
            'answers_changed_by_decoder': sum(x['answers_changed'] for x in chosen) if literal_defined else None,
            'literal_count_defined': literal_defined, 'qualified_rows': len(qualified),
            'qualified_answers_changed': sum(x['answers_changed'] for x in qualified),
            'full_64_qualified': strict and len(qualified) == 64}, ledger


def manufactured(legacy, checked, allowed, extract):
    seed = checked('A calculation.\nThe final answer is \\boxed{123}. ', add_special_tokens=False)['input_ids']
    if not seed:
        raise ValueError('empty manufactured encoding')
    ids = (seed * (4096 // len(seed) + 1))[:4096]
    summaries, ledger = [], []
    for name in CONDITIONS:
        rows = [{'index': i, 'answer_ids': ids} for i in range(64)]
        summary, sub = condition(name, rows, set(range(64)), list(range(64)),
                                 legacy, checked, allowed, extract)
        summaries.append(summary)
        ledger.extend({'condition': name, **x} for x in sub)
    return summaries, ledger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--card', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    card = json.loads(Path(args.card).read_text())
    if card['status'] != 'admitted':
        raise ValueError('prepared card refuses execution')
    mode = card['mode']
    if mode not in ('profile', 'actual') or str(Path(args.out).absolute()) != card['output']:
        raise ValueError('mode/output identity mismatch')
    if card['limits'] != LIMITS or card['output'] != BASE + ('tokenizer-profile' if mode == 'profile' else 'tokenizer'):
        raise ValueError('fixed resource/output contract mismatch')
    if card['interpreter'] != '/data2/neural-shrimp/.venv/bin/python' or sys.executable != card['interpreter']:
        raise ValueError('unexpected interpreter')
    source_before = [verify(x) for x in card['source_files'].values()]
    verify(card['inputs'])
    meta = json.loads(Path(card['inputs']['path']).read_text())
    pins = meta['common_files'] + (meta['actual_files'] if mode == 'actual' else [])
    if sum(x['bytes'] for x in pins) > card['limits']['input_bytes']:
        raise ValueError('input byte ceiling')
    before = [verify(x) for x in pins]
    out = Path(args.out)
    out.mkdir(parents=False, exist_ok=False)
    store = Store(out, card['limits']['output_bytes'])
    started = time.perf_counter()
    try:
        os.environ.update(ENV)
        versions = {k: importlib.metadata.version(k) for k in meta['versions']}
        if versions != meta['versions']:
            raise ValueError('package version mismatch')
        core = adjacent('pm_core')
        backend = [backend_state('before_tokenizer_load')]
        legacy, checked, trace = load_pair(meta)
        backend.append(backend_state('after_load_before_decode'))
        allowed = (set(legacy.get_vocab().values()), set(checked.get_vocab().values()))
        if mode == 'profile':
            controls = adjacent('controls').run(sys.modules[__name__], core.answer)
            store.write('controls.json', controls)
            if not controls['all_pass']:
                raise ValueError('manufactured control failure')
            summaries, ledger = manufactured(legacy, checked, allowed, core.answer)
            comparisons = []
            publication_outside_scope = None
        else:
            files = {x['id']: x for x in meta['actual_files']}
            keep_list = json.loads(Path(files['diag-indices']['path']).read_text())
            if len(keep_list) != 64 or len(set(keep_list)) != 64 or any(type(x) is not int for x in keep_list):
                raise ValueError('selected64 identity schema')
            if tuple(sorted(meta['conditions'])) != CONDITIONS:
                raise ValueError('condition identities changed')
            summaries, ledger = [], []
            for name in CONDITIONS:
                spec = meta['conditions'][name]
                with Path(files[spec['records']]['path']).open() as f:
                    rows = [json.loads(line) for line in f if line.strip()]
                cfg = json.loads(Path(files[spec['config']]['path']).read_text())
                summary, sub = condition(name, rows, set(keep_list), spec['expected_order'],
                                         legacy, checked, allowed, core.answer)
                summary['historical_recorded_tokenizer'] = {
                    k: cfg.get(k) for k in ('tokenizer_class', 'tokenizer_roundtrip')}
                summaries.append(summary)
                ledger.extend({'condition': name, **x} for x in sub)
            published = json.loads(Path(files['published-pm2x-llama-diag']['path']).read_text())
            comparisons = []
            for row in summaries:
                old = published.get('conditions', {}).get(row['name'], {})
                field = 'answers_changed_by_decoder'
                exists = field in old
                value = old.get(field)
                own = row[field]
                comparisons.append({'path': '/conditions/' + row['name'] + '/' + field,
                                    'producer_present': exists, 'producer': value, 'own': own,
                                    'disposition': 'missing_producer' if not exists else
                                                   'undefined_own' if own is None else
                                                   'exact' if own == value else 'different',
                                    'signed_delta': own - value if type(value) is int and own is not None else None,
                                    'basis': 'current pinned tokenizer replay; historical package runtime unverified'})
            publication_outside_scope = {
                'field_names': {k: sorted(v.keys()) for k, v in published.get('conditions', {}).items()
                                if isinstance(v, dict)},
                'disposition': 'Only eight answers_changed_by_decoder fields replayed here; other PMX analysis is separate.'}
        backend.append(backend_state('after_decode_before_rows_write'))
        store.write('rows.json', ledger)
        through_rows = time.perf_counter() - started
        after = [verify(x) for x in pins]
        source_after = [verify(x) for x in card['source_files'].values()]
        verify(card['inputs'])
        if before != after or source_before != source_after:
            raise ValueError('before/after input or source change')
        report = {'schema': 'tokenizer-only-replay-v1', 'mode': mode,
                  'computational_completed': True, 'scientific_claims_accepted': False,
                  'resource_profile_only': mode == 'profile', 'source_files': source_after,
                  'card': pin(args.card), 'inputs_manifest': card['inputs'], 'inputs': after,
                  'versions': versions, 'interpreter': sys.executable, 'offline_environment': ENV,
                  'tokenizer_trace': trace, 'backend_import_observations': backend,
                  'conditions': summaries, 'rows': len(ledger),
                  'comparisons': comparisons, 'publication_other_fields': publication_outside_scope,
                  'timing_seconds_through_rows_serialization': through_rows,
                  'timing_seconds_after_rehash_before_report': time.perf_counter() - started,
                  'rss_kib_observed_before_report': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'limits': ['Current tokenizer semantics, not historical runtime attestation.',
                             'Installed Torch import permitted; no model weights, tensor/model/GPU work or accuracy rescore.',
                             'CUDA initialization flags do not attest all native/transitive binary bytes or unmapped CUDA libraries.',
                             'Missing answer_ids has separate literal-empty and qualified-unavailable dispositions.',
                             'Spaced-text roundtrip is a finite fixture, not universal invertibility.']}
        store.write('report.json', report)
        store.write('manifest.json', {'files': list(store.files), 'computational_completed': True})
    except BaseException as e:
        store.write('failure.json', {'error': type(e).__name__, 'message': str(e),
                                    'traceback': traceback.format_exc(), 'mode': mode})
        raise


if __name__ == '__main__':
    main()

"""Independent saved-scalar flip-normal-form reconstruction; no producer imports.

This checks arithmetic on saved coefficient terms and projections. It does not
reconstruct derivatives, eigenvectors, orbit closure, primitive period or precision.
Actual execution requires a reviewed, hash-bound card and an external supervisor.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

SCHEMA = 'codex-task21c-saved-scalar-v1'
LIMITS = {'cpu_seconds': 20, 'wall_seconds': 30, 'as_bytes': 1024**3,
          'output_bytes': 16*1024**2, 'input_bytes': 4*1024**2,
          'startup_available_bytes': 20*1024**3, 'runtime_available_bytes': 16*1024**3}


def num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError('finite numeric scalar required')
    return float(v)


def cpair(v):
    if not isinstance(v, list) or len(v) != 2:
        raise ValueError('complex [real,imag] pair required')
    return complex(num(v[0]), num(v[1]))


def difference(saved, own):
    saved, own = num(saved), num(own)
    d = num(saved - own)
    return {'saved': saved, 'recomputed': own, 'saved_minus_recomputed': d,
            'absolute_difference': abs(d), 'relative_to_recomputed': num(abs(d/own)) if own else None}


def endpoint_difference(measured, predicted):
    z = measured - predicted
    return {'measured_real': measured.real, 'measured_imaginary': measured.imag,
            'predicted_real': predicted, 'signed_real_difference': z.real,
            'signed_imaginary_difference': z.imag, 'absolute_real_difference': abs(z.real),
            'complex_absolute_difference': num(abs(z))}


def reconstruct(record, expected):
    """Return every saved row, including refusals and malformed rows; never drop."""
    out = {'id': expected['id'], 'saved_record': record, 'schema_errors': [],
           'coefficient_rows': [], 'offset_rows': [], 'scope': 'saved_scalar_reconstruction_only'}
    c = None
    try:
        args, nf = record['args'], record['normal_form']
        if args['period'] != expected['period'] or isinstance(args['period'], bool):
            raise ValueError('unexpected period')
        if args['side'] not in (-1, 1) or isinstance(args['side'], bool):
            raise ValueError('invalid side')
        af = num(args['alpha_flip'])
        num(args['log_temperature']); num(args['tol'])
        if args['tol'] <= 0:
            raise ValueError('positive Newton tolerance required')
        original_steps = nf['c_steps']
        if len(original_steps) != expected['c_step_count'] or len(original_steps) < 2:
            raise ValueError('unexpected coefficient ladder length')
        coefs, hs = [], []
        for i, row in enumerate(original_steps):
            rr = {'index': i, 'saved': row}
            out['coefficient_rows'].append(rr)
            try:
                h = num(row['h_relative'])
                if h <= 0 or num(row['h2_norm']) < 0:
                    raise ValueError('invalid h or h2 norm')
                own = num(num(row['c_direct']) + num(row['c_centre_manifold']))
                rr['c_arithmetic'] = difference(row['c'], own)
                hs.append(h); coefs.append(own)
            except (KeyError, ValueError, TypeError, OverflowError) as e:
                rr['schema_error'] = str(e)
                out['schema_errors'].append(f'coefficient[{i}]: {e}')
        if len(coefs) != len(original_steps) or len(set(hs)) != len(hs):
            raise ValueError('invalid/incomplete coefficient ladder')
        # min returns the first tie, matching the declared original selection rule.
        gaps = [num(abs(b-a)) for a, b in zip(coefs, coefs[1:])]
        k = min(range(len(gaps)), key=gaps.__getitem__)
        c = num((coefs[k] + coefs[k+1]) * .5)
        spread = num(gaps[k] / max(abs(c), 1e-300))
        pair = [hs[k], hs[k+1]]
        crossing = nf['crossing']; mu = cpair(crossing['multiplier'])
        nxt = num(crossing['next_modulus'])
        if nxt < 0:
            raise ValueError('negative modulus')
        expected_kind = 'supercritical' if c > 0 else 'subcritical'
        offset_side = args['side'] if c > 0 else -args['side']
        out['normal_form'] = {
            'adjacent_absolute_gaps': gaps, 'selected_pair_index': k, 'selected_pair': pair,
            'selected_pair_matches_saved': nf['c_plateau_pair'] == pair,
            'c': difference(nf['c'], c), 'spread': difference(nf['c_plateau_relative_spread'], spread),
            'c_nonzero': c != 0, 'kind': expected_kind, 'kind_matches_saved': record['kind'] == expected_kind,
            'offset_side': offset_side, 'offset_side_matches_saved': record['offset_side'] == offset_side,
            'crossing_modulus': difference(crossing['modulus'], abs(mu)),
            'crossing_argument_turns': difference(crossing['argument_turns'], math.atan2(mu.imag,mu.real)/(2*math.pi)),
            'original_crossing_gates': {'real_within_1e-6': abs(mu.imag) <= 1e-6,
                'minus_one_within_1e-6': abs(mu.real+1) <= 1e-6, 'other_moduli_below_0.95': nxt < .95},
            'lowrank_critical_saved': [cpair(record['flip_lowrank_critical']).real, cpair(record['flip_lowrank_critical']).imag]}
        offsets = [num(s) for s in args['offsets']]
        if any(s <= 0 for s in offsets) or len(set(offsets)) != len(offsets):
            raise ValueError('offsets must be positive and unique')
        expected_offsets = sorted(offsets)
        if len(record['offsets']) != expected['offset_count'] or len(expected_offsets) != expected['offset_count']:
            raise ValueError('unexpected offset row count')
    except (KeyError, ValueError, TypeError, OverflowError) as e:
        out['schema_errors'].append(f'normal_form: {e}')
        expected_offsets = None
        offset_side = None
    for i, row in enumerate(record.get('offsets', [])):
        rr = {'index': i, 'saved': row}
        out['offset_rows'].append(rr)
        try:
            s = num(row['offset']); a = num(row['alpha'])
            rr['offset_order_matches_args'] = expected_offsets is not None and i < len(expected_offsets) and s == expected_offsets[i]
            if offset_side is not None:
                rr['alpha_arithmetic'] = difference(a, num(af+offset_side*s))
            # Producer error/note rows are retained explicitly, including partial fields.
            if 'error' in row:
                rr['disposition'] = 'producer_error'
            elif 'note' in row:
                rr['disposition'] = 'producer_note'
            else:
                rr['disposition'] = 'measured_doubled_row'
            if 'mu_p' not in row:
                if 'error' not in row:
                    raise ValueError('missing parent multiplier')
                continue
            mp = cpair(row['mu_p']); beta = num(-mp.real-1)
            rr['beta_arithmetic'] = difference(row['beta'], beta)
            rr['parent_imaginary'] = mp.imag
            rr['parent_source_complex_guard_accepts_saved_mu'] = not (abs(mp.imag)>1e-8 and mp.real+1<.5)
            rr['parent_next_modulus_producer'] = num(row['next_modulus_p'])
            if c is None:
                raise ValueError('coefficient unavailable')
            domain = beta*c > 0
            rr['amplitude_prediction_domain'] = domain
            if not domain:
                if 'note' not in row and 'error' not in row:
                    raise ValueError('no amplitude domain without explicit refusal')
                continue
            eta = num(math.sqrt(beta/c))
            pred = num(1-4*beta)
            rr['eta_prediction_from_saved_terms'] = eta
            rr['mu_prediction_from_saved_terms'] = pred
            if 'error' in row:
                continue
            if 'note' in row:
                rr['note_in_positive_domain'] = True
                continue
            measured = num(row['eta_measured'])
            if measured < 0 or eta <= 0:
                raise ValueError('invalid amplitude')
            m2 = cpair(row['mu_2p'])
            rr['eta_arithmetic'] = difference(row['eta_predicted'], eta)
            rr['ratio_arithmetic'] = difference(row['ratio'], num(measured/eta))
            rr['amplitude_comparison'] = {'measured_minus_predicted': num(measured-eta),
                'absolute_difference': abs(measured-eta), 'ratio': measured/eta,
                'signed_relative_error': measured/eta-1, 'absolute_relative_error': abs(measured/eta-1)}
            rr['mu_arithmetic'] = difference(row['mu_2p_predicted'], pred)
            rr['mu_comparison'] = endpoint_difference(m2, pred)
            beta2 = num(beta*beta)
            if beta2 == 0:
                raise ValueError('beta squared underflow prevents scaled discrepancy')
            rr['mu_error_over_beta_squared'] = {
                'signed_real': num((m2.real-pred)/beta2), 'signed_imaginary': num(m2.imag/beta2),
                'absolute_real': num(abs(m2.real-pred)/beta2), 'complex_absolute': num(abs(m2-pred)/beta2)}
            rr['mu_2p_next_modulus_producer'] = num(row['next_modulus_2p'])
            rr['centre_offset_producer'] = num(row['centre_offset'])
            residual = num(row['newton_2p'])
            if residual < 0:
                raise ValueError('negative residual')
            rr['newton_2p_producer'] = residual
            tolerance = max(num(args['tol']), 1e-13*(2*expected['period']))
            rr['source_newton_2p_gate'] = {'threshold': tolerance, 'strictly_below': residual<tolerance}
            rr['unavailable_checks'] = ['independent projected amplitude and centre', 'independent orbit closure/primitive period',
                'eigenpair identities/eigenvectors', 'derivative tensors and h2 solve residual', 'historical effective precision']
        except (KeyError, ValueError, TypeError, OverflowError) as e:
            rr['schema_error'] = str(e)
            out['schema_errors'].append(f'offset[{i}]: {e}')
    eligible = [r for r in out['offset_rows'] if 'amplitude_comparison' in r and 'schema_error' not in r]
    ordered = sorted(eligible, key=lambda r: r['saved']['offset'])
    errors = [r['amplitude_comparison']['absolute_relative_error'] for r in ordered]
    smallest = min(expected_offsets) if expected_offsets else None
    smallrow = next((r for r in ordered if r['saved']['offset']==smallest), None)
    out['finite_confirmation_summary'] = {
        'ordered_offsets': [r['saved']['offset'] for r in ordered],
        'ordered_absolute_amplitude_ratio_minus_one': errors,
        'all_requested_offsets_have_measured_comparisons': len(ordered)==expected['offset_count'],
        'adjacent_errors_nondecreasing_as_offset_increases': [a<=b for a,b in zip(errors,errors[1:])],
        'smallest_requested_offset': smallest,
        'smallest_offset_within_10_percent': smallrow['amplitude_comparison']['absolute_relative_error']<=.10 if smallrow else None,
        'asymptotic_big_O_verified': False,
        'multiplier_scope': 'All signed/absolute/complex beta-squared-scaled errors are descriptive; no finite constant or cutoff was frozen.'}
    out['schema_complete'] = not out['schema_errors']
    out['new_model_orbit_or_certificate_verified'] = False
    return out


def strict_json(raw):
    def pairs(items):
        d = {}
        for k,v in items:
            if k in d:
                raise ValueError('duplicate JSON key: '+k)
            d[k] = v
        return d
    return json.loads(raw, object_pairs_hook=pairs)


def json_safe(value):
    """Preserve a nonfinite original as an explicit tag, never emit invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_original': repr(value)}
    if isinstance(value, dict):
        return {k: json_safe(v) for k,v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--card', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); card = strict_json(Path(a.card).read_bytes())
    if card.get('schema') != SCHEMA or card.get('admission') != 'reviewed' or card.get('limits') != LIMITS:
        raise SystemExit('prepared or invalid card: refusing execution')
    if digest(__file__) != card.get('analyzer_sha256'):
        raise SystemExit('analyzer source mismatch')
    out = Path(a.out)
    if not out.is_absolute() or str(out) != card['output'] or out.exists():
        raise SystemExit('output must be exact fresh absolute planned path')
    if os.getpriority(os.PRIO_PROCESS, 0) > 10:
        raise SystemExit('inherited priority outside exact nice10 envelope')
    os.nice(10-os.getpriority(os.PRIO_PROCESS, 0))
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    if hard != resource.RLIM_INFINITY and hard < LIMITS['as_bytes']:
        raise SystemExit('inherited AS hard limit too small')
    resource.setrlimit(resource.RLIMIT_AS,(LIMITS['as_bytes'],LIMITS['as_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU,(LIMITS['cpu_seconds'],LIMITS['cpu_seconds']))
    admitted=[]; seen=set(); total=0
    def admit(binding, role):
        nonlocal total
        p=Path(binding['path'])
        if not p.is_absolute() or str(p) in seen:
            raise ValueError('duplicate/nonabsolute input binding')
        seen.add(str(p)); raw=p.read_bytes(); total+=len(raw)
        if total>LIMITS['input_bytes'] or len(raw)!=binding['bytes'] or hashlib.sha256(raw).hexdigest()!=binding['sha256']:
            raise ValueError('input size/hash mismatch: '+str(p))
        admitted.append({'path':str(p),'bytes':len(raw),'sha256':binding['sha256'],'role':role})
        return strict_json(raw)
    manifest=admit(card['inputs_manifest'],'input_manifest')
    if manifest['schema']!='task21c-saved-scalar-inputs-v1' or manifest['target_file_count']!=2 or manifest['target_bytes']!=8701 or manifest['expected_total_offsets']!=10:
        raise ValueError('unexpected input manifest contract')
    authority=admit(manifest['authority'],'inherited_original_authority')
    source_map=admit(manifest['source_map'],'inherited_original_source_association')
    specs=manifest['inputs']; metadata=manifest['metadata_schema']
    if [(s['id'],s['period']) for s in specs]!=[('FNF2b',2),('FNF4',4)] or [s['id'] for s in metadata]!=['FNF2b','FNF4']:
        raise ValueError('exact two-attempt identity/order required')
    if sum(s['file']['bytes'] for s in specs)!=manifest['target_bytes']:
        raise ValueError('target byte accounting mismatch')
    records=[]
    for spec, meta in zip(specs,metadata):
        if (meta['c_step_count'],meta['offset_count'])!=(4,5):
            raise ValueError('exact four coefficient and five offset rows required')
        expected={'id':spec['id'],'period':spec['period'],'c_step_count':4,'offset_count':5}
        records.append((expected,admit(spec['file'],'saved_scalar_record')))
    out.mkdir(parents=False,exist_ok=False)
    tic=time.perf_counter()
    result={'schema':SCHEMA,'records':[reconstruct(r,b) for b,r in records], 'inputs':admitted,
            'analyzer_sha256':digest(__file__), 'card_sha256':digest(a.card),
            'original_authority_inherited':authority,'original_source_association_inherited':source_map,
            'scientific_scope':'Unblinded saved scalar reconstruction, no model/derivative/eigen/orbit replay. Original source/archive/termination checks inherited from the bound metadata authority.',
            'computational_completed':True,'scientific_claims_accepted':False}
    for b in admitted:
        if Path(b['path']).stat().st_size!=b['bytes'] or digest(b['path'])!=b['sha256']:
            raise ValueError('input changed during reconstruction')
    result['all_records_schema_complete']=all(r['schema_complete'] for r in result['records'])
    result['calculation_and_posthash_seconds']=time.perf_counter()-tic
    result['peak_rss_kib_observed_before_serialization']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload=json.dumps(json_safe(result),indent=2,allow_nan=False).encode()+b'\n'
    inventory={'schema':SCHEMA,'files':[{'path':'report.json','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}]}
    inv=json.dumps(inventory,indent=2,allow_nan=False).encode()+b'\n'
    if len(payload)+len(inv)>LIMITS['output_bytes']: raise ValueError('output budget exceeded')
    with (out/'report.json').open('xb') as f:f.write(payload)
    with (out/'manifest.json').open('xb') as f:f.write(inv)
    return 0 if result['all_records_schema_complete'] else 2


if __name__=='__main__':
    sys.exit(main())

"""EXP-013 pure, frozen-budget evaluation after label-free policy sealing.

No files, fits or trajectories are opened here. The caller owns source/input
bindings and selected-label -> winner seal -> full18 reveal chronology. This
module independently enforces identities, the full denominator, fixed policies
and numerical flags; missing/failed observations never become cohort means.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import numpy as np

import preview_final_model as final
import preview_shortlist_model as model

RESOLUTIONS = final.RESOLUTIONS
PHASES, RADII = final.PHASES, final.RADII
BUDGETS = (2, 4, 8, 18)
BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES = 20260915, 20000
QUALIFICATION_KEYS = ('references', 'sensitivities', 'models', 'native_candidates',
                      'native_shams', 'independent_panel', 'access_chronology')
SEAL_SCHEMA = 'EXP-013-policy-seal-v1'
EVALUATION_SCHEMA = 'EXP-013-fixed-budget-evaluation-v1'
REQUIRED_COUNTS = dict(origins=24, contexts=192, candidates=3456,
                       resolutions=2, readouts=6144, descriptive_intervals=30)
SOURCE_PATHS = list(dict.fromkeys(['tools/fresh_preview_evaluation.py', *final.SOURCE_PATHS]))


def _seal(value):
    value['packet_sha256'] = model.digest(value)
    return value


def _verify_hash(packet):
    if not isinstance(packet, dict) or packet.get('packet_sha256') != model.digest(
            {k: v for k, v in packet.items() if k != 'packet_sha256'}):
        raise ValueError('Packet seal mismatch')


def policy_catalog():
    """Six/two-preview, five/four-preview, four/eight-preview and one/full18."""
    controls = {
        2: [('alpha_pair', [0, 8]), ('gamma_pair', [4, 12]),
            ('diagonal_pair', [2, 10]), ('local_pair', [16, 17])],
        4: [('axes', [0, 4, 8, 12]), ('diagonals', [2, 6, 10, 14]),
            ('local_coverage', [0, 8, 16, 17])],
        8: [('octagon', [0, 2, 4, 6, 8, 10, 12, 14]),
            ('local_plus_six', [0, 2, 4, 8, 10, 12, 16, 17])],
        18: [('full18', list(range(18)))]}
    result = []
    for budget in BUDGETS:
        if budget != 18:
            for schema in final.SCHEMAS:
                result.append(dict(id=f'{schema}-b{budget}', budget=budget, kind='learned',
                    feature_schema=schema, fixed_candidates=None, needs_sensitivity=True,
                    needs_atlas=schema == 'geometry'))
        for name, candidates in controls[budget]:
            result.append(dict(id=f'{name}-b{budget}', budget=budget, kind='fixed',
                feature_schema=None, fixed_candidates=candidates,
                needs_sensitivity=any(c >= 16 for c in candidates), needs_atlas=False))
    return result


def _planned_rows(rows, origin_groups):
    registry = final._registry(origin_groups)
    expected = {(g, p, r, c) for g in registry for p in PHASES for r in RADII for c in range(18)}
    normalized, ids, identities, directions = [], set(), set(), {}
    for row in rows:
        if (not isinstance(row, dict) or not {'id', 'group', 'phase', 'radius', 'candidate'} <= set(row)
                or set(row) - final.ROW_ALLOWED or not isinstance(row['id'], str)
                or not row['id'] or row['id'] in ids or type(row['phase']) is not int
                or type(row['candidate']) is not int or isinstance(row['radius'], (bool, np.bool_))):
            raise ValueError('Unique label-free planned identities required')
        identity = (row['group'], row['phase'], row['radius'], row['candidate'])
        if identity not in expected or identity in identities:
            raise ValueError('Unplanned or duplicate physical context/candidate identity')
        ids.add(row['id'])
        identities.add(identity)
        item = {k: row[k] for k in ('id', 'group', 'phase', 'radius', 'candidate')}
        if 'direction' in row:
            direction = np.asarray(row['direction'])
            if (direction.shape != (2,) or direction.dtype.kind not in 'fi'
                    or not np.isfinite(direction).all()
                    or not np.isclose(np.linalg.norm(direction), 1., rtol=0, atol=1e-12)
                    or (row['candidate'] < 16 and not np.array_equal(
                        direction, final.UNIFORM_DIRECTIONS[row['candidate']]))):
                raise ValueError('Planned realized candidate direction differs')
            direction_key = (row['group'], row['phase'], row['candidate'])
            if direction_key in directions and not np.array_equal(direction, directions[direction_key]):
                raise ValueError('Planned direction changes across radii')
            directions[direction_key] = direction.copy()
            item['direction'] = direction.astype(float).tolist()
        normalized.append(item)
    if identities != expected or len(normalized) != 3456:
        raise ValueError('Full 24-origin/192-context/3456-candidate planned census required')
    if directions:
        if (any('direction' not in row for row in normalized)
                or any(not np.array_equal(directions[(g,p,17)], -directions[(g,p,16)])
                       for g in registry for p in PHASES)):
            raise ValueError('Complete signed local direction identity required')
    return registry, sorted(normalized, key=lambda r: (r['group'], r['phase'], r['radius'], r['candidate']))


def _prediction_rows(packet, planned, recipe):
    _verify_hash(packet)
    if (packet.get('schema_version') != final.PREDICTION_SCHEMA or packet.get('recipe') != recipe
            or packet.get('feature_names') != final.feature_names(recipe.split('--')[1])):
        raise ValueError('Prediction recipe/schema differs from frozen definition')
    allowed = {'schema_version', 'model_packet_sha256', 'recipe', 'feature_names', 'rows',
               'feature_sha256', 'predictions_svd', 'predictions_qr', 'rankings_svd',
               'rankings_qr', 'numerical_audit', 'scope', 'packet_sha256'}
    if set(packet) != allowed:
        raise ValueError('Only the sealed pre-preview prediction schema is accepted')
    rows = packet['rows']
    if any(set(row) != final.ROW_REQUIRED for row in rows):
        raise ValueError('Only normalized pre-preview prediction rows are accepted')
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate prediction IDs')
    for row in rows:
        direction = np.asarray(row['direction'])
        if (type(row['phase']) is not int or type(row['candidate']) is not int
                or isinstance(row['radius'], (bool, np.bool_))
                or direction.shape != (2,) or direction.dtype.kind not in 'fi'
                or not np.isfinite(direction).all()
                or row['id'] not in planned or 'direction' not in planned[row['id']]
                or any(row.get(k) != v for k, v in planned[row['id']].items())):
            raise ValueError('Prediction identity differs from planned context')
    scores = {}
    rankings = {}
    for solver in ('svd', 'qr'):
        scores[solver] = model.finite(packet[f'predictions_{solver}'], f'{recipe}/{solver}')
        if scores[solver].shape != (len(rows),):
            raise ValueError('Prediction score census differs')
        rankings[solver] = final.rank_prepreview(rows, scores[solver])
        if rankings[solver] != packet[f'rankings_{solver}']:
            raise ValueError('Saved prediction ranking differs from score/ID reconstruction')
    score_check = model.score_agreement(scores['svd'], scores['qr'])
    # Cutoff gaps can differ within the score tolerance; ordered IDs alone are
    # the frozen action gate, so reconstruct that gate separately from diagnostics.
    prefixes = {str(b): all(a['shortlists'][str(b)]['ids'] == q['shortlists'][str(b)]['ids']
        for a, q in zip(rankings['svd'], rankings['qr'], strict=True)) for b in final.BUDGETS}
    audit = dict(passed=score_check['passed'] and all(prefixes.values()), score_agreement=score_check,
        ordered_shortlists_agree=prefixes, full_ranking_agrees=all(a['candidate_order'] == q['candidate_order']
            for a, q in zip(rankings['svd'], rankings['qr'], strict=True)))
    if audit != packet['numerical_audit']:
        raise ValueError('Saved prediction numerical audit differs from reconstruction')
    return {row['id']: (float(scores['svd'][i]), float(scores['qr'][i])) for i, row in enumerate(rows)}, audit


def seal_policies(planned_rows, prediction_packets, *, origin_groups):
    """Label-free seal, accepting one packet or disjoint packet list per recipe.

    Missing recipes/origin packets are retained as unavailable learned policies;
    an altered seal, substituted identity or unplanned recipe raises ValueError.
    Numerical disagreement retains predictions and shortlists with a failed flag.
    """
    registry, rows = _planned_rows(planned_rows, origin_groups)
    planned = {r['id']: r for r in rows}
    required = [f'{r}--{s}' for r in RESOLUTIONS for s in final.SCHEMAS]
    if not isinstance(prediction_packets, dict) or set(prediction_packets) - set(required):
        raise ValueError('Only the four frozen prediction recipes are allowed')
    predictions, evidence, failures = {}, {}, []
    for recipe in required:
        packets = prediction_packets.get(recipe, [])
        packets = packets if isinstance(packets, list) else [packets]
        values, packet_evidence, passed = {}, [], True
        for packet in packets:
            current, audit = _prediction_rows(packet, planned, recipe)
            if set(values) & set(current):
                raise ValueError('Duplicate prediction IDs across origin packets')
            values.update(current)
            passed = passed and audit['passed']
            packet_evidence.append(dict(packet_sha256=packet['packet_sha256'],
                model_packet_sha256=packet['model_packet_sha256'], rows=len(current), numerical_audit=audit))
        if len({p['model_packet_sha256'] for p in packet_evidence}) > 1:
            raise ValueError('One final model per recipe is required across fresh origins')
        missing = sorted(set(planned) - set(values))
        if missing or not passed:
            failures.append(dict(recipe=recipe, missing_prediction_ids=missing, numerical_passed=passed))
        predictions[recipe] = values
        evidence[recipe] = dict(packets=packet_evidence, missing_prediction_ids=missing,
                                complete=not missing, numerical_passed=passed)
    contexts = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[model.context_key(row)].append(row)
    catalog = policy_catalog()
    for context, current in sorted(grouped.items()):
        by_candidate = {r['candidate']: r['id'] for r in current}
        policies, predicted = {}, {}
        for resolution in RESOLUTIONS:
            policies[resolution], predicted[resolution] = {}, {}
            for schema in final.SCHEMAS:
                recipe = f'{resolution}--{schema}'
                values = predictions[recipe]
                if all(r['id'] in values for r in current):
                    svd = [values[r['id']][0] for r in current]
                    qr = [values[r['id']][1] for r in current]
                    a, b = final.rank_prepreview(current, svd)[0], final.rank_prepreview(current, qr)[0]
                    predicted[resolution][schema] = dict(svd=a, qr=b,
                        score_agreement=model.score_agreement(svd, qr),
                        full_ranking_agrees=a['candidate_order'] == b['candidate_order'])
            for definition in catalog:
                schema = definition['feature_schema']
                available = schema is None or schema in predicted[resolution]
                selected = definition['fixed_candidates'] if schema is None else (
                    predicted[resolution][schema]['svd']['candidate_order'][:definition['budget']] if available else None)
                numerical = schema is None or (available and evidence[f'{resolution}--{schema}']['numerical_passed'])
                policies[resolution][definition['id']] = dict(budget=definition['budget'],
                    status='complete' if available and numerical else ('numerical_disagreement' if available else 'missing_prediction'),
                    selected=selected, selected_ids=[by_candidate[c] for c in selected] if available else None)
        contexts.append(dict(context=list(context), candidate_ids=[by_candidate[c] for c in range(18)],
                             policies=policies, predictions=predicted))
    return _seal(dict(schema_version=SEAL_SCHEMA, origin_groups=registry, rows=rows,
        catalog=catalog, prediction_evidence=evidence, contexts=contexts, failures=failures,
        complete=all(e['complete'] for e in evidence.values()),
        numerical_passed=all(e['numerical_passed'] for e in evidence.values()),
        scope='Label-free policies only; no fresh preview labels accessed.'))


def observe_selected(selected_rows, selected_labels):
    """Selected-only winner for the caller's seal BEFORE full18 label reveal.

    selected_rows contains only {id,candidate} for the selected charged entries;
    selected_labels maps exactly those IDs to scalar RMSE. No oracle is accepted.
    """
    ids = [r['id'] for r in selected_rows]
    candidates = [r['candidate'] for r in selected_rows]
    if (len(ids) not in BUDGETS or len(set(ids)) != len(ids) or len(set(candidates)) != len(ids)
            or any(type(c) is not int or c not in range(18) for c in candidates)
            or set(selected_labels) != set(ids)):
        raise ValueError('Exact distinct selected-only candidate labels required')
    values = [_rmse(selected_labels[row_id]) for row_id in ids]
    winner = min(range(len(ids)), key=lambda i: (values[i], candidates[i]))
    return dict(selected=candidates, selected_ids=ids, chosen_candidate=candidates[winner],
                chosen_id=ids[winner], rmse=values[winner], required_preview_calls=len(ids))


def _rmse(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError('Scalar tanh RMSE required')
    result = float(value)
    if not np.isfinite(result) or not 0 <= result <= 2:
        raise ValueError('Finite tanh RMSE in [0,2] required')
    return result


def _evidence(value):
    """Retain failed/nonfinite diagnostic values in strict-JSON derived output."""
    if isinstance(value, dict):
        return {k: _evidence(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _evidence(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_evidence(v) for v in value]
    if isinstance(value, np.generic):
        return _evidence(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {'nonfinite_float': repr(value)}
    return copy.deepcopy(value)


def _labels(table, planned):
    if not isinstance(table, dict) or set(table) - set(planned):
        raise ValueError('Unplanned label IDs')
    output = {}
    for row_id, row in planned.items():
        raw = table.get(row_id)
        if raw is None:
            output[row_id] = dict(status='missing', qualified=False, rmse=None, evidence=None)
            continue
        if not isinstance(raw, dict) or not isinstance(raw.get('status'), str) or type(raw.get('qualified')) is not bool:
            raise ValueError('Each label needs explicit status and boolean qualification')
        evidence = _evidence(raw)
        # Invalid/nonfinite labels remain inspectable failure rows; the original
        # packet is owned by the caller. Repr keeps this derived report strict JSON.
        try:
            value = _rmse(raw.get('rmse'))
            error = None
        except ValueError as exc:
            value, error = None, str(exc)
            if 'rmse' in evidence:
                evidence['rmse'] = repr(raw['rmse'])
        if any(k in raw and raw[k] != v for k, v in row.items()):
            raise ValueError('Label physical/candidate identity differs from plan')
        output[row_id] = dict(status=raw['status'], qualified=raw['qualified'], rmse=value,
                             evidence=evidence, value_error=error)
    return output


def _observe_full(context, labels, policies):
    ids = context['candidate_ids']
    values = [labels[i]['rmse'] for i in ids]
    full_available = all(v is not None for v in values)
    order = sorted(range(18), key=lambda c: (values[c], c)) if full_available else None
    result = {}
    for definition in policy_catalog():
        policy_id = definition['id']
        selection = policies[policy_id]
        selected = selection['selected']
        selected_available = selected is not None and all(values[c] is not None for c in selected)
        observed = None
        if selected_available:
            observed = observe_selected([dict(id=ids[c], candidate=c) for c in selected],
                                        {ids[c]: values[c] for c in selected})
            if full_available:
                winner = order[0]
                observed.update(full18_winner=winner, full18_winner_id=ids[winner], full18_min_rmse=values[winner],
                    excess_rmse=observed['rmse']-values[winner], winner_inclusion=winner in selected,
                    any_minimizer_inclusion=any(values[c] == values[winner] for c in selected),
                    best_selected_rank=min(order.index(c)+1 for c in selected),
                    mean_selected_rank=float(np.mean([order.index(c)+1 for c in selected])))
        all_qualified = full_available and all(labels[i]['status'] == 'complete' and labels[i]['qualified'] for i in ids)
        result[policy_id] = dict(status='complete' if (selection['status'] == 'complete' and all_qualified) else 'unqualified_or_incomplete',
            selection_status=selection['status'], selected_values_available=selected_available,
            full18_values_available=full_available, all_context_labels_qualified=all_qualified, observed=observed)
    return result, values, order


def _resolution_diagnostics(context, policies):
    primary, comparison = RESOLUTIONS
    a, b = policies[primary], policies[comparison]
    if a['order'] is None or b['order'] is None:
        return None
    av, bv = np.asarray(a['candidate_rmse']), np.asarray(b['candidate_rmse'])
    oa, ob = a['order'], b['order']
    changed = {}
    for definition in policy_catalog():
        key = definition['id']
        left, right = a['policies'][key]['observed'], b['policies'][key]['observed']
        changed[key] = None if left is None or right is None else dict(
            ordered_shortlist_changed=left['selected'] != right['selected'],
            shortlist_set_changed=set(left['selected']) != set(right['selected']),
            selected_action_changed=left['chosen_candidate'] != right['chosen_candidate'],
            selected_error_signed_change=left['rmse']-right['rmse'],
            excess_error_signed_change=left['excess_rmse']-right['excess_rmse'])
    return dict(signed_label_changes=(av-bv).tolist(), empirical_max_absolute_score_change=float(np.max(np.abs(av-bv))),
        envelope_is_certified_error_bound=False, primary_order=oa, comparison_order=ob,
        full_ranking_changed=oa != ob, full18_winner_changed=oa[0] != ob[0],
        full18_minimum_signed_change=float(av[oa[0]]-bv[ob[0]]),
        primary_cutoff_gaps={str(k): float(av[oa[k]]-av[oa[k-1]]) for k in (1, 2, 4, 8)},
        comparison_cutoff_gaps={str(k): float(bv[ob[k]]-bv[ob[k-1]]) for k in (1, 2, 4, 8)},
        policies=changed)


def _bootstrap_indices():
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0, 24, size=(BOOTSTRAP_RESAMPLES, 24))


def _aggregate(contexts, registry):
    """Called only for the complete, qualified 24-origin result."""
    indices = _bootstrap_indices()
    index_hash = hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()
    answer = {}
    metrics = ('rmse', 'excess_rmse', 'winner_inclusion', 'any_minimizer_inclusion',
               'best_selected_rank', 'mean_selected_rank')
    catalog = policy_catalog()
    for resolution in RESOLUTIONS:
        strata = {}
        for name, radius in [('pooled', None), *[(str(r), r) for r in RADII]]:
            current = [c for c in contexts if radius is None or c['context'][2] == radius]
            per_origin = []
            for group in registry:
                subset = [c for c in current if c['context'][0] == group]
                expected = 8 if radius is None else 2
                if len(subset) != expected:
                    raise ValueError('Incomplete equal-origin stratum')
                policies = {d['id']: {metric: float(np.mean([c['resolutions'][resolution]['policies'][d['id']]['observed'][metric]
                    for c in subset])) for metric in metrics} for d in catalog}
                per_origin.append(dict(group=group, contexts=expected, policies=policies))
            means = {d['id']: {metric: float(np.mean([o['policies'][d['id']][metric] for o in per_origin]))
                              for metric in metrics} for d in catalog}
            contrasts = {}
            for other in ('shared-b4', 'axes-b4', 'local_coverage-b4'):
                values = np.asarray([o['policies']['geometry-b4']['rmse']-o['policies'][other]['rmse'] for o in per_origin])
                contrasts[f'geometry-b4_minus_{other}'] = dict(mean=float(np.mean(values)),
                    per_origin=[dict(group=g, effect=float(v)) for g, v in zip(registry, values, strict=True)],
                    signs=dict(negative=int(np.sum(values < 0)), zero=int(np.sum(values == 0)), positive=int(np.sum(values > 0))),
                    descriptive_central_95_interval=np.percentile(np.mean(values[indices], axis=1), [2.5, 97.5], method='linear').tolist())
            strata[name] = dict(origins=24, contexts=len(current), per_origin=per_origin,
                                policies=means, four_preview_contrasts=contrasts)
        answer[resolution] = strata
    return dict(readouts=answer, bootstrap=dict(seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES,
        unit='whole physical origin', origin_order=registry, shared_indices_sha256=index_hash,
        interval_count=30, percentiles=[2.5, 97.5], percentile_method='linear',
        interpretation='Descriptive stability within these 24 selected, spatially dependent origins; '
                       'not confirmatory significance or a fresh-population confidence guarantee.'))


def cost_contract():
    """Nominal accounting only: no inferred latency or invented asset prices."""
    policies = []
    for definition in policy_catalog():
        prerequisites = ['reference orbit']
        if definition['needs_sensitivity']:
            prerequisites.append('300-unit unperturbed sensitivity')
        if definition['kind'] == 'learned':
            prerequisites += ['historical labels', 'old-data tuning/final model', 'pre-preview feature construction', 'prediction']
        if definition['needs_atlas']:
            prerequisites.append('historical EXP-004 atlas')
        policies.append(dict(id=definition['id'], preview_calls=definition['budget'],
            full18_preview_calls=18, preview_call_reduction=18-definition['budget'], prerequisites=prerequisites))
    return dict(policies=policies, duplicated_physical_entries_stay_charged=True,
        acquisition=dict(native_candidate_traces=10368, native_sham_traces=144,
            independent_traces=96, total_short_traces=10608, baseline_native_runs=48, sensitivities=48),
        amortization_horizons=[1, 8, 192], measured_stage_costs_supplied_by_runner=True,
        reused_asset_replacement_cost='unknown unless measured; never zero by reuse',
        selected_full_batch_cost_is_measured_policy_latency=False, net_speedup_claim=False)


def _winner_replay(winner_seal, policy_seal, contexts):
    if winner_seal is None:
        return dict(supplied=False, complete=None, checked=0, required=5760, failures=[],
                    scope='Caller access_chronology gate carries winner-seal verification.')
    _verify_hash(winner_seal)
    if winner_seal.get('policy_seal_sha256') != policy_seal['packet_sha256']:
        raise ValueError('Winner seal binds a different policy seal')
    expected = {(tuple(c['context']), r, d['id']): c['resolutions'][r]['policies'][d['id']]['observed']
        for c in contexts for r in RESOLUTIONS for d in policy_catalog() if d['budget'] != 18}
    seen, failures, checked = set(), [], 0
    for row in winner_seal['winners']:
        key = (tuple(row['context']), row['resolution'], row['policy'])
        if key not in expected or key in seen:
            raise ValueError('Unplanned or duplicated non-full18 winner seal')
        seen.add(key)
        observed = expected[key]
        if observed is None:
            failures.append(dict(context=row['context'], resolution=row['resolution'],
                policy=row['policy'], reason='selected values unavailable for winner replay'))
            continue
        reconstructed = {k: observed[k] for k in ('selected', 'selected_ids', 'chosen_candidate',
                          'chosen_id', 'rmse', 'required_preview_calls')}
        if row['winner'] != reconstructed:
            failures.append(dict(context=row['context'], resolution=row['resolution'],
                policy=row['policy'], reason='sealed selected-only winner differs',
                sealed_winner=copy.deepcopy(row['winner']), reconstructed=reconstructed))
        checked += 1
    missing = sorted(expected.keys()-seen)
    return dict(supplied=True, winner_seal_sha256=winner_seal['packet_sha256'],
        complete=not missing and not failures, checked=checked, required=5760,
        missing=[dict(context=list(c), resolution=r, policy=p) for c, r, p in missing], failures=failures,
        scope='Exact selected-only winner reconstruction; wall-clock/file chronology is checked by the caller.')


def evaluate(sealed_policies, labels_by_resolution, *, qualification, winner_seal=None):
    """Evaluate after the caller seals selected-only winners and opens full18.

    Label records require status, qualified and rmse; extra numerical flags are
    preserved verbatim. Missing IDs become explicit rows; unplanned IDs fail.
    Required caller qualification gates must all be present booleans. A false
    gate, missing/failed value or prediction gate blocks *all* cohort aggregates
    and 30 intervals, while retaining the full 6144-row diagnostic denominator.
    """
    _verify_hash(sealed_policies)
    if sealed_policies.get('schema_version') != SEAL_SCHEMA:
        raise ValueError('Unknown label-free policy seal')
    registry, rows = _planned_rows(sealed_policies['rows'], sealed_policies['origin_groups'])
    if sealed_policies['catalog'] != policy_catalog():
        raise ValueError('Fixed policy catalog differs')
    if not isinstance(labels_by_resolution, dict) or set(labels_by_resolution) - set(RESOLUTIONS):
        raise ValueError('Unplanned label resolution')
    if (not isinstance(qualification, dict) or set(QUALIFICATION_KEYS) - set(qualification)
            or any(type(qualification[k]) is not bool for k in QUALIFICATION_KEYS)):
        raise ValueError('Explicit complete required qualification gates required')
    planned = {r['id']: r for r in rows}
    labels = {r: _labels(labels_by_resolution.get(r, {}), planned) for r in RESOLUTIONS}
    expected_contexts = {(g, p, r) for g in registry for p in PHASES for r in RADII}
    seen, contexts, failures = set(), [], []
    for context in sealed_policies['contexts']:
        key = tuple(context['context'])
        if key not in expected_contexts or key in seen:
            raise ValueError('Duplicate or unplanned sealed context')
        seen.add(key)
        expected_ids = [row['id'] for row in rows if model.context_key(row) == key]
        if context['candidate_ids'] != expected_ids:
            raise ValueError('Candidate ID order differs from planned context')
        resolved = {}
        for resolution in RESOLUTIONS:
            selected = context['policies'][resolution]
            if set(selected) != {d['id'] for d in policy_catalog()}:
                raise ValueError('Policy census differs')
            for definition in policy_catalog():
                entry = selected[definition['id']]
                candidates = entry['selected']
                if candidates is None:
                    if definition['kind'] != 'learned' or entry['status'] != 'missing_prediction':
                        raise ValueError('Unexpected absent policy')
                elif (len(candidates) != definition['budget'] or len(set(candidates)) != len(candidates)
                      or any(type(c) is not int or c not in range(18) for c in candidates)
                      or entry['selected_ids'] != [expected_ids[c] for c in candidates]
                      or (definition['kind'] == 'fixed' and candidates != definition['fixed_candidates'])):
                    raise ValueError('Sealed shortlist/budget differs from frozen policy')
                elif definition['kind'] == 'learned':
                    ranking = context['predictions'][resolution][definition['feature_schema']]['svd']
                    if candidates != ranking['candidate_order'][:definition['budget']]:
                        raise ValueError('Learned shortlist differs from sealed predicted order')
            observed, values, order = _observe_full(context, labels[resolution], selected)
            resolved[resolution] = dict(candidate_rmse=values, order=order, policies=observed)
        contexts.append(dict(context=list(key), resolutions=resolved,
                             numerical_resolution=_resolution_diagnostics(context, resolved)))
    if seen != expected_contexts:
        raise ValueError('Full sealed context census required')
    for resolution, current in labels.items():
        for row_id, row in current.items():
            if row['status'] != 'complete' or not row['qualified'] or row['rmse'] is None:
                failures.append(dict(resolution=resolution, id=row_id, **copy.deepcopy(row)))
    gates = {k: qualification[k] for k in QUALIFICATION_KEYS}
    observed = sum(p['observed'] is not None for c in contexts for r in RESOLUTIONS for p in c['resolutions'][r]['policies'].values())
    complete = sealed_policies['complete'] and not any(r['rmse'] is None or r['status'] != 'complete' for t in labels.values() for r in t.values())
    winner_replay = _winner_replay(winner_seal, sealed_policies, contexts)
    every_readout_qualified = all(p['status'] == 'complete' for c in contexts for r in RESOLUTIONS
                                 for p in c['resolutions'][r]['policies'].values())
    qualified = (complete and sealed_policies['numerical_passed'] and not failures and all(gates.values())
                 and every_readout_qualified and winner_replay['complete'] is not False)
    contexts.sort(key=lambda c: tuple(c['context']))
    return _seal(dict(schema_version=EVALUATION_SCHEMA, policy_seal_sha256=sealed_policies['packet_sha256'],
        status='qualified_exploratory_readout' if qualified else 'incomplete_or_numerically_unqualified',
        complete=bool(complete), qualified=bool(qualified), required_counts=REQUIRED_COUNTS.copy(),
        observed_readouts=observed, retained_readouts=6144, qualification=_evidence(qualification),
        prediction_failures=copy.deepcopy(sealed_policies['failures']), label_failures=failures,
        selected_winner_replay=winner_replay,
        labels=labels, contexts=contexts, cohort_statistics=_aggregate(contexts, registry) if qualified else None,
        cost_contract=cost_contract(), scope='Exploratory 75-unit loss versus charged budget in all 24 fixed fresh origins. '
        'No complete-case aggregate, outcome-selected policy, utility acceptance threshold or net speedup claim.'))


def amortized_costs(sealed_policies, *, preview_task_seconds, reference_task_seconds,
                    sensitivity_task_seconds, final_recipe_seconds=None, joint_setup_seconds=None):
    """Fixed-block charged task accounting at 1, 8 and 192 decisions.

    Preview costs are resolution -> candidate ID -> nonnegative seconds/None;
    references use physical-group keys, sensitivities (group,phase) tuple keys,
    and final recipe costs resolution--schema keys. Missing costs stay unknown.
    A supplied joint setup scalar is charged once per block as the *entire*
    measured joint setup; no recipe/radius split is inferred. Historical label
    and atlas creation costs remain unknown. These sums are never wall latency.
    """
    _verify_hash(sealed_policies)
    if sealed_policies.get('schema_version') != SEAL_SCHEMA or sealed_policies['catalog'] != policy_catalog():
        raise ValueError('Frozen policy seal required for cost accounting')
    registry, rows = _planned_rows(sealed_policies['rows'], sealed_policies['origin_groups'])
    contexts = sorted(sealed_policies['contexts'], key=lambda c: tuple(c['context']))
    expected = {(g, p, r) for g in registry for p in PHASES for r in RADII}
    if len(contexts) != 192 or {tuple(c['context']) for c in contexts} != expected:
        raise ValueError('Complete planned context denominator required for costs')
    ids = {r['id'] for r in rows}
    expected_sensitivity = {(g, p) for g in registry for p in PHASES}
    recipes = {f'{r}--{s}' for r in RESOLUTIONS for s in final.SCHEMAS}
    recipe_cost = {} if final_recipe_seconds is None else final_recipe_seconds
    if (set(preview_task_seconds) - set(RESOLUTIONS) or set(reference_task_seconds) - set(registry)
            or set(sensitivity_task_seconds) - expected_sensitivity or set(recipe_cost) - recipes
            or any(set(table) - ids for table in preview_task_seconds.values())):
        raise ValueError('Unplanned cost identity or resolution')

    def seconds(value):
        if value is None:
            return None
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value < 0):
            raise ValueError('Cost must be nonnegative finite seconds or explicit unknown')
        return float(value)

    for table in [*preview_task_seconds.values(), reference_task_seconds, sensitivity_task_seconds, recipe_cost]:
        for value in table.values():
            seconds(value)
    joint_setup_seconds = seconds(joint_setup_seconds)
    blocks = {1: [[c] for c in contexts],
              8: [[c for c in contexts if c['context'][0] == g] for g in registry],
              192: [contexts]}
    output = {}
    for resolution in RESOLUTIONS:
        output[resolution] = {}
        for definition in policy_catalog():
            policy_id = definition['id']
            horizons = {}
            for horizon, current_blocks in blocks.items():
                accounted = []
                for block_index, block in enumerate(current_blocks):
                    groups = sorted({c['context'][0] for c in block})
                    origin_phases = sorted({tuple(c['context'][:2]) for c in block})
                    components, unknown = {}, []
                    known_subtotal, selected_known = 0., 0

                    def add_component(name, entries):
                        nonlocal known_subtotal
                        known, missing = [], []
                        for key, value in entries:
                            value = seconds(value)
                            if value is None:
                                missing.append(key)
                            else:
                                known.append(value)
                        subtotal = float(sum(known))
                        known_subtotal += subtotal
                        components[name] = dict(required_items=len(entries), costed_items=len(known),
                            known_subtotal_seconds=subtotal, total_seconds=subtotal if not missing else None)
                        unknown.extend(dict(component=name, key=key) for key in missing)
                        return len(known)

                    selected_entries = []
                    for context in block:
                        entry = context['policies'][resolution][policy_id]
                        selected = entry['selected_ids']
                        if selected is None:
                            selected_entries += [(f'{context["context"]}/unavailable-selected-{i}', None)
                                                 for i in range(definition['budget'])]
                        else:
                            if len(selected) != definition['budget'] or len(set(selected)) != len(selected) or set(selected)-ids:
                                raise ValueError('Cost shortlist differs from declared charged budget')
                            selected_entries += [(row_id, preview_task_seconds.get(resolution, {}).get(row_id)) for row_id in selected]
                    selected_known = add_component('selected_preview_tasks', selected_entries)
                    add_component('reference_per_origin', [(g, reference_task_seconds.get(g)) for g in groups])
                    if definition['needs_sensitivity']:
                        add_component('sensitivity_per_origin_phase', [(list(k), sensitivity_task_seconds.get(k)) for k in origin_phases])
                    if definition['kind'] == 'learned':
                        recipe = f'{resolution}--{definition["feature_schema"]}'
                        add_component('final_model_once', [(recipe, recipe_cost.get(recipe))])
                        add_component('joint_setup_once', [('complete measured joint setup; no recipe split', joint_setup_seconds)])
                        add_component('historical_training_label_creation', [('reused historical labels; replacement cost unknown', None)])
                    if definition['needs_atlas']:
                        add_component('historical_atlas_creation', [('reused EXP-004 atlas; replacement cost unknown', None)])
                    accounted.append(dict(block_index=block_index, decisions=horizon, origins=groups,
                        required_preview_calls=horizon*definition['budget'], costed_preview_calls=selected_known,
                        components=components, unknown_components=unknown,
                        known_subtotal_seconds=known_subtotal, known_subtotal_per_decision_seconds=known_subtotal/horizon,
                        full_total_seconds=known_subtotal if not unknown else None,
                        full_total_per_decision_seconds=known_subtotal/horizon if not unknown else None))
                all_known = all(b['full_total_seconds'] is not None for b in accounted)
                horizons[str(horizon)] = dict(blocks=len(accounted), decisions_per_block=horizon,
                    known_subtotal_mean_per_decision_seconds=float(np.mean([b['known_subtotal_per_decision_seconds'] for b in accounted])),
                    full_total_mean_per_decision_seconds=float(np.mean([b['full_total_per_decision_seconds'] for b in accounted])) if all_known else None,
                    all_required_costs_known=all_known, accounting_blocks=accounted)
            output[resolution][policy_id] = horizons
    return dict(horizons=[1, 8, 192], policy_seal_sha256=sealed_policies['packet_sha256'], accounting=output,
        scopes=dict(horizon1='Each of 192 fixed contexts separately; 192 blocks.',
            horizon8='Each origin and its eight fixed phase/radius contexts; 24 blocks.',
            horizon192='All 192 fixed contexts; one block.',
            prerequisite_reuse='One reference per unique origin and sensitivity per unique origin/phase within each block; final recipe and whole joint setup once per block.'),
        measured_policy_latency=False, historical_reuse_is_zero_cost=False,
        scope='Charged task-seconds accounting on the full research acquisition, not measured deployment latency, CPU time, '
              'or a net speedup. Known subtotals omit explicitly unknown costs; no combined timer is divided into inferred recipe/radius costs.')

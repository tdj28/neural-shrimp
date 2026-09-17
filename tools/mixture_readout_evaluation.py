"""EXP-020 pure reference adapters, fixed-census scoring and aggregation.

No files, integrations, scheduling or work at import. The caller seals every
reference-only definition before supplying any finite observation. Gaussian
probabilities are delegated to the separately qualified exact-input core.
"""
from __future__ import annotations

import copy
import importlib
import math
from collections.abc import Mapping
from fractions import Fraction

import numpy as np

import two_write_evaluation as old


SETTINGS = ('core', 'failed-near-edge')
WINDOWS = ('k2', 'k4', 'k8')
SIGNS = (-1, 1)
RECIPES = ('primary', 'alternate', 'one_mode')
METHODS = ('rk4-coarse', 'rk4-fine', 'dop853-independent')
SIGMA = 1e-4
EPSILON = 1e-4
WIDTH_LIMIT = 1e-4
PAIR_LIMIT = .001
RTOL, ATOL = 1e-12, 1e-14
RECIPE_NAMES = dict(primary='full-primary-reference', alternate='full-alternate-reference',
                    one_mode='periodic-one-mode-secondary')


def contract():
    return dict(schema='mixture-readout-evaluation-v1',settings=list(SETTINGS),windows=list(WINDOWS),
        signs=list(SIGNS),arm='identity',sigma=SIGMA,epsilon=EPSILON,recipes=RECIPE_NAMES.copy(),
        methods=list(METHODS),independent_panel_phases=[0,8],phase_seed=2026091701,
        exact=dict(cell_jobs=144,classifier_definitions=432,finite_class_method_rows=1200,
            primary_finite_probabilities=1200,secondary_finite_probabilities=1200,
            alternate_fine_probabilities=576,reference_model_controls=1728,total_new_probability_calculations=4704,
            auxiliary_linear_reference_calculations=1728,mixture_plus_auxiliary_probability_calculations=6432,
            primary_method_comparisons=624,secondary_method_comparisons=624,reference_comparisons=576,
            method_context_window_groups=300),
        thresholds=dict(probability_interval_width=WIDTH_LIMIT,probability_pair_maximum=PAIR_LIMIT,
                        original_replay_rtol=RTOL,original_replay_atol=ATOL),
        control='Balanced four-class mean Bayes dominance not contradicted by rigorous intervals; never per-class dominance',
        primary='Mean fine full-reference-mixture minus original covariance-linear BA at4T over all24 near-edge regression phases',
        primary_requires_both_settings=True,secondary_failures_invalidate_primary=False,
        regression_phases=True,fresh_holdout=False,bootstrap=False,
        alternate_reference='Own A/B and own baseline on identical physical observations; exact vector-sum translation',
        failure_bounds='All24 primary-phase slots, each unknown nonlinear accuracy contributes[0,1]; unknown original baseline contributes[-1,1] gain')


def _function(name, injected):
    return injected if injected is not None else getattr(importlib.import_module('gaussian_mixture_readout'), name)


def _array(value, name, shape=(96,)):
    if type(value) is not np.ndarray or value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'{name} requires finite float64 shape {shape}')
    return value


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f'{name} requires finite real scalar')
    return float(value)


def _same(actual, expected, name):
    if isinstance(expected, np.ndarray):
        if type(actual) is not np.ndarray or actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise ValueError(name+' differs')
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(name+' differs')


def cell_census(phase_rows=None):
    phases = old.phase_census() if phase_rows is None else copy.deepcopy(phase_rows)
    # The frozen history validator enforces the complete ordered shared phase schema.
    old.history_census(phases)
    result = []
    for setting in SETTINGS:
        for phase in phases:
            cid = f'{setting}--p{phase["phase_index"]:02d}'
            for window in WINDOWS:
                result.append(dict(id=cid+'--'+window, context_id=cid, setting=setting, **phase, window=window))
    return result


def _cell(cell):
    if not isinstance(cell, Mapping) or set(cell) != {'id','context_id','setting','phase_index','phase_fraction','split','window'}:
        raise ValueError('Exact cell schema required')
    index = cell['phase_index']
    if type(index) is not int or index not in range(24) or cell['setting'] not in SETTINGS or cell['window'] not in WINDOWS:
        raise ValueError('Known setting, integer phase and delayed window required')
    if not 0 <= _scalar(cell['phase_fraction'], 'phase_fraction') < 1:
        raise ValueError('Phase fraction outside [0,1)')
    cid = f'{cell["setting"]}--p{index:02d}'
    for key, value in dict(context_id=cid, id=cid+'--'+cell['window'], split='development' if index < 8 else 'heldout').items():
        _same(cell[key], value, 'cell.'+key)
    return cell


def methods_for_cell(cell):
    _cell(cell)
    return METHODS if cell['phase_index'] in (0,8) else METHODS[:2]


def calculation_census(cell):
    _cell(cell)
    result = []
    def add(family, method, recipe, first, second):
        ident = f'{cell["id"]}--{family}--{method}--{recipe}--s1{first:+d}--s2{second:+d}'
        result.append(dict(id=ident, cell_id=cell['id'], family=family, method=method,
                           recipe=recipe, first_sign=first, second_sign=second))
    for method in methods_for_cell(cell):
        for first in SIGNS:
            for second in SIGNS:
                for recipe in ('primary','one_mode'):
                    add('finite',method,recipe,first,second)
    for first in SIGNS:
        for second in SIGNS:
            add('alternate','rk4-fine','alternate',first,second)
    for recipe in RECIPES:
        for first in SIGNS:
            for second in SIGNS:
                add('reference','reference-model',recipe,first,second)
    return result


def _unavailable_definition(cell, recipe, error):
    return dict(id=cell['id']+'--classifier--'+recipe, recipe=recipe,
                recipe_name=RECIPE_NAMES[recipe], available=False, complete=False,
                status='unavailable', definition=None, error=str(error))


def seal_cell(cell, old_context_seal, alternate_tangent, *, define_fn=None):
    """Seal all three compact recipes using only qualified EXP019 references."""
    _cell(cell)
    result = dict(id=cell['id'], cell=copy.deepcopy(cell), context_id=cell['context_id'],
                  complete=False, status='failed', definitions={r:_unavailable_definition(cell,r,'Reference unavailable') for r in RECIPES},
                  sigma=SIGMA, epsilon=EPSILON, error=None)
    try:
        outer = old_context_seal
        if outer.get('context_id') != cell['context_id'] or outer.get('status') != 'complete' or outer.get('qualified') is not True:
            raise ValueError('Qualified bound outer EXP019 context seal required')
        source = outer['evaluation']; context = source['context']
        if source.get('context_id') != cell['context_id'] or source.get('complete') is not True or source.get('qualified') is not True:
            raise ValueError('Complete qualified EXP019 evaluation seal required')
        for key in ('setting','phase_index','phase_fraction','split'):
            _same(context[key],cell[key],'reference context.'+key)
        _same(context['id'],cell['context_id'],'reference context.id')
        for key in ('epsilon','sigma'):
            _same(source[key],SIGMA,'reference '+key)
        old._tangent(alternate_tangent,context,'rk45')
        alternate_samples, alternate_predictions = old._predict(context,alternate_tangent,cell['window'])
        primary = source['windows'][cell['window']]
        _same(primary['times'],alternate_samples['times'],'reference clock')
        _same(primary['indices'],alternate_samples['indices'],'reference sample indices')
        _same(primary['segment'],alternate_samples['segment'],'reference segment')
        baseline = _array(primary['baseline'],'primary baseline',(32,3)).ravel().copy()
        alt_baseline = _array(alternate_samples['activations'],'alternate baseline',(32,3)).ravel().copy()
        h = _array(primary['h'],'original observer').copy()
        if np.linalg.norm(h) == 0:
            raise ValueError('Nonzero frozen observer required')
        A = _array(primary['linear_predictions']['identity']['A'],'primary A').copy()
        B = _array(primary['linear_predictions']['identity']['B'],'primary B').copy()
        alt_A = _array(alternate_predictions['identity']['A'],'alternate A').copy()
        alt_B = _array(alternate_predictions['identity']['B'],'alternate B').copy()
        define = _function('define_classifier',define_fn)
        for recipe, a, b, base in [('primary',A,B,baseline),('alternate',alt_A,alt_B,alt_baseline),('one_mode',A,B,baseline)]:
            try:
                rho = None
                if recipe == 'one_mode':
                    if context.get('rho') is None:
                        raise ValueError('Signed real canonical multiplier unavailable')
                    rho = _scalar(context['rho'],'rho')
                definition = define(a,b,SIGMA,one_mode_rho=rho)
                if not isinstance(definition,Mapping) or definition.get('complete') is not True or definition.get('available') is not True:
                    raise ValueError('Core classifier definition unavailable: '+str(definition.get('error') if isinstance(definition,Mapping) else definition))
                result['definitions'][recipe] = dict(id=cell['id']+'--classifier--'+recipe,
                    recipe=recipe,recipe_name=RECIPE_NAMES[recipe],available=True,complete=True,status='complete',
                    definition=copy.deepcopy(definition),A=None if recipe=='one_mode' else a.copy(),B=b.copy(),
                    rho=rho,baseline=base.copy(),error=None)
            except Exception as error:
                result['definitions'][recipe] = _unavailable_definition(cell,recipe,f'{type(error).__name__}: {error}')
        required = all(result['definitions'][r]['available'] for r in ('primary','alternate'))
        result.update(h=h,baseline=baseline,period=float(context['period']),times=primary['times'].copy(),
                      indices=primary['indices'].copy(),segment=primary['segment'],complete=required,
                      status='complete' if required else 'failed',
                      provenance=dict(experiment='EXP-019',reference_only=True,
                          canonical_context_source=copy.deepcopy(context.get('provenance',{})),
                          alternate_recipe='Own RK45 A/B and zero-write baseline; same physical observations',
                          one_mode='Mathematical A=rho*B, never a rounded fake rank-two product'))
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    return result


def _sealed(cell,sealed):
    _cell(cell)
    if not isinstance(sealed,Mapping) or sealed.get('cell') != cell or sealed.get('id') != cell['id'] or sealed.get('context_id') != cell['context_id']:
        raise ValueError('Exact sealed cell identity required')
    if set(sealed.get('definitions',{})) != set(RECIPES):
        raise ValueError('Three classifier definitions required, including unavailable ones')
    _same(sealed.get('sigma'),SIGMA,'sealed sigma');_same(sealed.get('epsilon'),EPSILON,'sealed epsilon')
    for recipe,row in sealed['definitions'].items():
        _same(row['id'],cell['id']+'--classifier--'+recipe,'definition id')
        _same(row['recipe'],recipe,'definition recipe')
        _same(row['recipe_name'],RECIPE_NAMES[recipe],'definition name')
        if type(row.get('available')) is not bool or type(row.get('complete')) is not bool:
            raise ValueError('Boolean definition availability required')
        if row['available']:
            if row['complete'] is not True or row['definition'].get('complete') is not True or row['definition'].get('available') is not True:
                raise ValueError('Available classifier must be complete')
            _array(row['baseline'],'classifier baseline');_array(row['B'],'classifier B')
            _same(row['definition']['B'],row['B'],'core B binding')
            _same(row['definition']['sigma'],SIGMA,'core sigma binding')
            if recipe == 'one_mode':
                _scalar(row['rho'],'rho')
                _same(row['definition']['one_mode_rho'],row['rho'],'core mathematical rho binding')
            else:
                _array(row['A'],'classifier A')
                _same(row['definition']['A'],row['A'],'core A binding')
                _same(row['definition']['one_mode_rho'],None,'full reference rho absence')
    if sealed.get('complete') is True:
        _array(sealed['h'],'observer');_array(sealed['baseline'],'baseline')
        if not all(sealed['definitions'][r]['available'] for r in ('primary','alternate')):
            raise ValueError('Complete cell lacks primary/alternate definition')
    return sealed


def observation_census(cell):
    return [dict(id=f'{cell["id"]}--observation--{method}--s1{first:+d}--s2{second:+d}',
                 cell_id=cell['id'],method=method,first_sign=first,second_sign=second)
            for method in methods_for_cell(cell) for first in SIGNS for second in SIGNS]


def _observations(cell,sealed,finite_scores):
    result = {}
    supplied = finite_scores if isinstance(finite_scores,Mapping) else {}
    extra = set(supplied)-set(methods_for_cell(cell))
    for spec in observation_census(cell):
        key = spec['method'],spec['first_sign'],spec['second_sign']
        row = dict(spec,complete=False,replay_pass=False,original_probability=None,error=None)
        try:
            if extra:raise ValueError('Unexpected finite method')
            # The old frozen linear score can remain known even when a new
            # classifier definition fails. Do not erase that failure-bound input.
            _array(sealed.get('h'),'retained original observer')
            _array(sealed.get('baseline'),'retained original baseline')
            _array(sealed.get('times'),'retained original clock',(32,))
            group = supplied.get(spec['method'])
            if not isinstance(group,(list,tuple)) or len(group) != 4:
                raise ValueError('Exactly four original score rows per method required')
            signs = [(s.get('task',{}).get('first_sign'),s.get('task',{}).get('second_sign')) for s in group]
            if set(signs) != {(a,b) for a in SIGNS for b in SIGNS} or len(set(signs)) != 4:
                raise ValueError('Exact four-sign original score census required')
            saved = group[signs.index((spec['first_sign'],spec['second_sign']))]
            task = saved['task'];hist = f'{cell["context_id"]}--identity--s1{spec["first_sign"]:+d}--s2{spec["second_sign"]:+d}'
            expected = dict(id=hist+'--'+spec['method'],history_id=hist,context_id=cell['context_id'],setting=cell['setting'],
                phase_index=cell['phase_index'],phase_fraction=cell['phase_fraction'],split=cell['split'],arm='identity',
                first_sign=spec['first_sign'],second_sign=spec['second_sign'],epsilon=EPSILON,method=spec['method'])
            if set(task)!=set(expected):raise ValueError('Original physical task schema differs')
            for k,v in expected.items():_same(task[k],v,'original physical task.'+k)
            for k,v in dict(task_id=expected['id'],history_id=hist,context_id=cell['context_id'],complete=True,qualified=True,status='complete').items():
                _same(saved.get(k),v,'old score.'+k)
            window = saved['windows'][cell['window']]
            _same(window['times'],sealed['times'],'original score clock')
            mean = _array(window['activation_displacement'],'finite displacement').copy()
            replay = old._project(sealed['h'],mean,spec['first_sign'])
            original = window['primary'];differences = {};passed = True
            if set(original) != set(replay):raise ValueError('Original Gaussian score schema differs')
            for k,v in replay.items():
                if isinstance(v,bool) or v is None:
                    _same(original[k],v,'original Gaussian '+k)
                else:
                    observed = _scalar(original[k],'original Gaussian '+k)
                    delta = abs(observed-v);differences[k] = delta
                    passed &= delta <= ATOL+RTOL*abs(v)
            row.update(complete=True,replay_pass=bool(passed),mean=mean,original_probability=float(replay['probability_correct']),
                       original_score=copy.deepcopy(original),replayed_score=replay,arithmetic_differences=differences,
                       source_task_id=expected['id'])
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
        result[key] = row
    return result


def _terms(sealed,spec,observations):
    row = sealed['definitions'][spec['recipe']]
    if not row['available']:raise ValueError('Declared classifier unavailable')
    if spec['family'] == 'reference':
        a_vector = row['B'] if spec['recipe']=='one_mode' else row['A']
        a_coefficient = spec['first_sign']*row['rho'] if spec['recipe']=='one_mode' else float(spec['first_sign'])
        return [dict(coefficient=float(a_coefficient),vector=a_vector),dict(coefficient=float(spec['second_sign']),vector=row['B'])]
    observation = observations[spec['method'],spec['first_sign'],spec['second_sign']]
    if not observation['complete']:raise ValueError('Finite observation unavailable')
    result = [dict(coefficient=1.,vector=observation['mean'])]
    if spec['recipe']=='alternate':
        result += [dict(coefficient=1.,vector=sealed['baseline']),dict(coefficient=-1.,vector=row['baseline'])]
    return result


def _probability(value):
    if not isinstance(value,Mapping):raise ValueError('Probability result mapping required')
    for k in ('complete','qualified'):
        if type(value.get(k)) is not bool:raise ValueError('Probability '+k+' must be Boolean')
    if value['qualified'] and not value['complete']:raise ValueError('Incomplete probability cannot be qualified')
    if value['complete']:
        interval = value.get('interval')
        if not isinstance(interval,(list,tuple)) or len(interval)!=2:raise ValueError('Probability interval required')
        lo,hi = [_scalar(v,'probability bound') for v in interval]
        if not 0 <= lo <= hi <= 1:raise ValueError('Probability interval outside [0,1]')
        nominal = _scalar(value['nominal'],'nominal probability')
        if not 0 <= nominal <= 1:raise ValueError('Nominal probability outside [0,1]')
        if value['qualified'] and hi-lo > WIDTH_LIMIT:raise ValueError('Qualified probability interval too wide')
    return value


def _missing_probability(error):
    return dict(complete=False,qualified=False,available=False,status='unavailable',interval=None,nominal=None,error=str(error),diagnostics={})


def evaluate_cell(cell,sealed,finite_scores,*,probability_fn=None,linear_probability_fn=None,progress_fn=None):
    """Run each declared probability once; callback records preserve a prefix."""
    _sealed(cell,sealed)
    observations = _observations(cell,sealed,finite_scores)
    probability = _function('probability',probability_fn)
    linear = _function('linear_probability',linear_probability_fn)
    calculations = []
    for spec in calculation_census(cell):
        record = _evaluate_calculation(sealed,spec,observations,probability,linear)
        calculations.append(record)
        if progress_fn is not None:progress_fn(copy.deepcopy(record))
    return assemble_cell(cell,sealed,finite_scores,calculations)


def _evaluate_calculation(sealed,spec,observations,probability,linear):
    record = dict(spec,result=_missing_probability('Not computed'),linear_reference=None)
    returned = None
    try:
        terms = _terms(sealed,spec,observations)
        definition = sealed['definitions'][spec['recipe']]['definition']
        returned = probability(definition,terms,spec['first_sign'])
        record['result'] = copy.deepcopy(_probability(returned))
    except Exception as error:
        record['result'] = _missing_probability(f'{type(error).__name__}: {error}')
        if returned is not None:record['result']['partial_result']=copy.deepcopy(returned)
        return record
    if spec['family']=='reference':
        returned = None
        try:
            linear_terms = list(terms)
            if spec['recipe']=='alternate':
                linear_terms += [dict(coefficient=1.,vector=sealed['definitions']['alternate']['baseline']),dict(coefficient=-1.,vector=sealed['baseline'])]
            returned = linear(sealed['h'],linear_terms,spec['first_sign'],SIGMA)
            record['linear_reference'] = copy.deepcopy(_probability(returned))
        except Exception as error:
            # A completed mixture probability is retained even if its comparator fails.
            record['linear_reference'] = _missing_probability(f'{type(error).__name__}: {error}')
            if returned is not None:record['linear_reference']['partial_result']=copy.deepcopy(returned)
    return record


def evaluate_calculation(cell,sealed,finite_scores,slot_spec,*,probability_fn=None,linear_probability_fn=None):
    """Replay exactly one saved slot; never fills an unrecorded timeout suffix."""
    _sealed(cell,sealed)
    if slot_spec not in calculation_census(cell):raise ValueError('Exact declared calculation required')
    return _evaluate_calculation(sealed,slot_spec,_observations(cell,sealed,finite_scores),
        _function('probability',probability_fn),_function('linear_probability',linear_probability_fn))


def _fraction(value):
    return Fraction.from_float(_scalar(value,'interval arithmetic input'))


def _round_fraction(value,up):
    number = float(value)
    if (up and Fraction.from_float(number)<value) or (not up and Fraction.from_float(number)>value):
        number = math.nextafter(number,math.inf if up else -math.inf)
    return number


def _mean_interval(intervals):
    if not intervals:return None
    n=len(intervals)
    return [_round_fraction(sum((_fraction(x[0]) for x in intervals),Fraction())/n,False),
            _round_fraction(sum((_fraction(x[1]) for x in intervals),Fraction())/n,True)]


def _difference_interval(left,right):
    return [_round_fraction(_fraction(left[0])-_fraction(right[1]),False),
            _round_fraction(_fraction(left[1])-_fraction(right[0]),True)]


def _aggregate_probabilities(probabilities):
    complete = len(probabilities)==4 and all(p.get('complete') is True for p in probabilities)
    return dict(complete=bool(complete),qualified=bool(complete and all(p.get('qualified') is True for p in probabilities)),
        interval=_mean_interval([p['interval'] for p in probabilities]) if complete else None,
        nominal=math.fsum(p['nominal'] for p in probabilities)/4 if complete else None)


def _comparison(ident,left,right):
    complete = left.get('complete') is True and right.get('complete') is True
    result = dict(id=ident,complete=complete,qualified=False,maximum_possible_difference=None,
                  difference_interval=None,nominal_difference=None)
    if complete:
        interval = _difference_interval(left['interval'],right['interval'])
        maximum = max(abs(interval[0]),abs(interval[1]))
        result.update(difference_interval=interval,maximum_possible_difference=maximum,
            nominal_difference=left['nominal']-right['nominal'],
            qualified=left.get('qualified') is True and right.get('qualified') is True and maximum<=PAIR_LIMIT)
    return result


def _calculation_rows(cell,calculations):
    values = list(calculations.values()) if isinstance(calculations,Mapping) else list(calculations or [])
    expected = calculation_census(cell);wanted={s['id'] for s in expected}
    ids=[r.get('id') for r in values if isinstance(r,Mapping)]
    if len(ids)!=len(values) or len(set(ids))!=len(ids) or set(ids)-wanted:
        raise ValueError('Unexpected, duplicate or malformed calculation identity')
    supplied={r['id']:r for r in values};result=[]
    for spec in expected:
        record = supplied.get(spec['id'])
        if record is None:
            record=dict(spec,result=_missing_probability('Planned calculation unavailable'),linear_reference=None)
        else:
            for k,v in spec.items():_same(record.get(k),v,'calculation.'+k)
            _probability(record['result'])
            if spec['family']=='reference' and record.get('linear_reference') is not None:
                _probability(record['linear_reference'])
        result.append(copy.deepcopy(record))
    return result


def assemble_cell(cell,sealed,finite_scores,calculations):
    """Pure partial-prefix reconstruction: performs no new probability call."""
    _cell(cell)
    if sealed is None:
        sealed=dict(id=cell['id'],cell=copy.deepcopy(cell),context_id=cell['context_id'],complete=False,
            status='unavailable',sigma=SIGMA,epsilon=EPSILON,
            definitions={r:_unavailable_definition(cell,r,'Definition unavailable') for r in RECIPES})
    _sealed(cell,sealed)
    observations = _observations(cell,sealed,finite_scores)
    rows = _calculation_rows(cell,calculations)
    return _assemble(cell,sealed,list(observations.values()),rows)


def _assemble(cell,sealed,observations,rows):
    bykey = {(r['family'],r['method'],r['recipe'],r['first_sign'],r['second_sign']):r for r in rows}
    obs = {(r['method'],r['first_sign'],r['second_sign']):r for r in observations}
    if len(obs)!=len(observations) or set(obs)!={(s['method'],s['first_sign'],s['second_sign']) for s in observation_census(cell)}:
        raise ValueError('Exact observation census required')
    for spec in observation_census(cell):
        row=obs[spec['method'],spec['first_sign'],spec['second_sign']]
        for key,value in spec.items():_same(row.get(key),value,'observation.'+key)
        if type(row.get('complete')) is not bool or type(row.get('replay_pass')) is not bool:
            raise ValueError('Observation completeness/replay must be Boolean')
        if not row['complete']:
            if row['replay_pass']:raise ValueError('Unavailable observation cannot pass replay')
            continue
        replay=old._project(sealed['h'],_array(row['mean'],'retained finite displacement'),spec['first_sign'])
        _same(row['replayed_score'],replay,'retained Gaussian replay')
        _same(row['original_probability'],float(replay['probability_correct']),'retained original probability')
        original=row['original_score'];passed=True;differences={}
        if set(original)!=set(replay):raise ValueError('Retained original Gaussian schema differs')
        for key,value in replay.items():
            if isinstance(value,bool) or value is None:_same(original[key],value,'retained Gaussian '+key)
            else:
                delta=abs(_scalar(original[key],'retained Gaussian '+key)-value)
                differences[key]=delta;passed &= delta<=ATOL+RTOL*abs(value)
        _same(row['arithmetic_differences'],differences,'retained arithmetic differences')
        _same(row['replay_pass'],bool(passed),'retained replay qualification')
    def probability(family,method,recipe,first,second):return bykey[family,method,recipe,first,second]['result']
    controls = {}
    for recipe in RECIPES:
        selected=[bykey['reference','reference-model',recipe,a,b] for a in SIGNS for b in SIGNS]
        nonlinear=_aggregate_probabilities([r['result'] for r in selected])
        linear=_aggregate_probabilities([r['linear_reference'] or _missing_probability('Reference linear control unavailable') for r in selected])
        complete=nonlinear['complete'] and linear['complete']
        difference=_difference_interval(nonlinear['interval'],linear['interval']) if complete else None
        # Bayes optimality is a prior-weighted risk statement, never a per-class gate.
        not_contradicted=bool(complete and difference[1]>=0.)
        controls[recipe]=dict(complete=complete,qualified=bool(complete and nonlinear['qualified'] and linear['qualified'] and not_contradicted),
            nonlinear=nonlinear,original_linear=linear,difference_interval=difference,
            balanced_dominance_not_contradicted=not_contradicted,
            balanced_dominance_supported=bool(complete and difference[0]>=0.),
            scope='Balanced mean over exactly four equiprobable classes; individual class tradeoffs retained')
    comparisons=[]
    for left,right in zip(methods_for_cell(cell)[:-1],methods_for_cell(cell)[1:]):
        for recipe in ('primary','one_mode'):
            for first in SIGNS:
                for second in SIGNS:
                    ident=f'{cell["id"]}--comparison--{left}--vs--{right}--{recipe}--s1{first:+d}--s2{second:+d}'
                    row=_comparison(ident,probability('finite',left,recipe,first,second),probability('finite',right,recipe,first,second))
                    row.update(cell_id=cell['id'],recipe=recipe,left_method=left,right_method=right,first_sign=first,second_sign=second)
                    comparisons.append(row)
    reference_comparisons=[]
    for first in SIGNS:
        for second in SIGNS:
            ident=f'{cell["id"]}--reference-comparison--s1{first:+d}--s2{second:+d}'
            row=_comparison(ident,probability('finite','rk4-fine','primary',first,second),probability('alternate','rk4-fine','alternate',first,second))
            row.update(cell_id=cell['id'],first_sign=first,second_sign=second)
            reference_comparisons.append(row)
    required=[r for r in rows if r['recipe']!='one_mode']
    complete=bool(sealed.get('complete') is True and all(o.get('complete') is True for o in observations)
                  and all(r['result']['complete'] for r in required) and all(controls[r]['complete'] for r in ('primary','alternate')))
    replay=bool(all(o.get('replay_pass') is True for o in observations))
    gates=dict(definitions=sealed.get('complete') is True,original_linear_replay=replay,
        primary_calculations=all(r['result']['qualified'] for r in required),
        primary_controls=all(controls[r]['qualified'] for r in ('primary','alternate')),
        primary_method_comparisons=all(r['qualified'] for r in comparisons if r['recipe']=='primary'),
        alternate_reference_comparisons=all(r['qualified'] for r in reference_comparisons))
    qualified=bool(complete and all(gates.values()))
    secondary_rows=[r for r in rows if r['recipe']=='one_mode']
    secondary_complete=bool(sealed['definitions']['one_mode']['available'] and all(r['result']['complete'] for r in secondary_rows) and controls['one_mode']['complete'])
    secondary_qualified=bool(qualified and secondary_complete and all(r['result']['qualified'] for r in secondary_rows)
        and controls['one_mode']['qualified'] and all(r['qualified'] for r in comparisons if r['recipe']=='one_mode'))
    classes=[]
    for spec in observation_census(cell):
        key=spec['method'],spec['first_sign'],spec['second_sign'];o=obs[key]
        classes.append(dict(spec,observation_complete=o.get('complete') is True,original_linear_replay=o.get('replay_pass') is True,
            original_probability=o.get('original_probability'),
            primary=copy.deepcopy(probability('finite',spec['method'],'primary',spec['first_sign'],spec['second_sign'])),
            one_mode=copy.deepcopy(probability('finite',spec['method'],'one_mode',spec['first_sign'],spec['second_sign']))))
    groups=[]
    for method in methods_for_cell(cell):
        selected=[r for r in classes if r['method']==method]
        known=all(r['observation_complete'] and r['original_linear_replay'] for r in selected)
        originals=[r['original_probability'] for r in selected]
        original_interval=_mean_interval([[p,p] for p in originals]) if known else None
        original_nominal=math.fsum(originals)/4 if known else None
        group=dict(cell_id=cell['id'],method=method,original_linear_interval=original_interval,original_linear=original_nominal)
        for recipe in ('primary','one_mode'):
            aggregate=_aggregate_probabilities([r[recipe] for r in selected])
            aggregate.update(qualified=bool(aggregate['qualified'] and (qualified if recipe=='primary' else secondary_qualified)),
                gain_interval=_difference_interval(aggregate['interval'],original_interval) if aggregate['complete'] and known else None,
                nominal_gain=aggregate['nominal']-original_nominal if aggregate['complete'] and known else None)
            group[recipe]=aggregate
        groups.append(group)
    return dict(id=cell['id'],cell=copy.deepcopy(cell),complete=complete,qualified=qualified,
        status='complete' if complete else 'incomplete',gates=gates,secondary_complete=secondary_complete,
        secondary_qualified=secondary_qualified,definitions_available={r:sealed['definitions'][r]['available'] for r in RECIPES},
        calculations=rows,observations=observations,class_rows=classes,controls=controls,
        method_comparisons=comparisons,reference_comparisons=reference_comparisons,groups=groups,
        counts=dict(calculations_planned=len(rows),calculations_complete=sum(r['result']['complete'] for r in rows),
            calculations_qualified=sum(r['result']['qualified'] for r in rows),class_rows=len(classes),
            primary_method_comparisons=sum(r['recipe']=='primary' for r in comparisons),
            secondary_method_comparisons=sum(r['recipe']=='one_mode' for r in comparisons),reference_comparisons=4))


def _canonical_cells(cells):
    if not isinstance(cells,(list,tuple)) or len(cells)!=144:raise ValueError('Exactly144 planned cells required')
    for cell in cells:_cell(cell)
    if len({c['id'] for c in cells})!=144:raise ValueError('Duplicate planned cell')
    phases=[]
    for index in range(24):
        own=[c for c in cells if c['phase_index']==index];fractions={c['phase_fraction'] for c in own}
        if len(fractions)!=1:raise ValueError('Shared phase fractions differ')
        phases.append(dict(phase_index=index,phase_fraction=fractions.pop(),split='development' if index<8 else 'heldout'))
    expected=cell_census(phases)
    if {c['id']:c for c in cells}!={c['id']:c for c in expected}:raise ValueError('Complete cell census differs')
    return expected


def summarize(cells,definitions,results):
    """Keep all144 cells and4704 calculation slots; no successful-subset means."""
    cells=_canonical_cells(cells);wanted={c['id'] for c in cells}
    for name,value in [('definitions',definitions),('results',results)]:
        if not isinstance(value,Mapping) or set(value)-wanted:raise ValueError('Unexpected '+name+' identity')
    retained=[]
    for cell in cells:
        definition=definitions.get(cell['id']);packet=results.get(cell['id'])
        if packet is None:
            retained.append(assemble_cell(cell,definition,{},[]));continue
        _sealed(cell,definition)
        if packet.get('id')!=cell['id'] or packet.get('cell')!=cell:raise ValueError('Result cell binding differs')
        rows=_calculation_rows(cell,packet.get('calculations'))
        rebuilt=_assemble(cell,definition,copy.deepcopy(packet['observations']),rows)
        for flag in ('complete','qualified','secondary_complete','secondary_qualified'):
            if type(packet.get(flag)) is not bool or packet[flag] is not rebuilt[flag]:raise ValueError('Result '+flag+' differs')
        retained.append(rebuilt)
    settings={}
    for setting in SETTINGS:
        group=[p for p in retained if p['cell']['setting']==setting]
        q=all(p['qualified'] for p in group);sq=all(p['secondary_qualified'] for p in group)
        settings[setting]=dict(cells=72,complete=all(p['complete'] for p in group),qualified=q,
            qualified_cells=sum(p['qualified'] for p in group),secondary_qualified=sq,means=None,secondary_means=None)
        for recipe,allowed,key in [('primary',q,'means'),('one_mode',sq,'secondary_means')]:
            if allowed:
                means=[]
                for window in WINDOWS:
                    rows=[next(g for g in p['groups'] if g['method']=='rk4-fine') for p in group if p['cell']['window']==window]
                    means.append(dict(window=window,phases=24,accuracy_interval=_mean_interval([r[recipe]['interval'] for r in rows]),
                        nominal_accuracy=math.fsum(r[recipe]['nominal'] for r in rows)/24,
                        gain_interval=_mean_interval([r[recipe]['gain_interval'] for r in rows]),
                        nominal_gain=math.fsum(r[recipe]['nominal_gain'] for r in rows)/24,
                        original_linear=math.fsum(r['original_linear'] for r in rows)/24))
                settings[setting][key]=means
    primary_cells=sorted([p for p in retained if p['cell']['setting']=='failed-near-edge' and p['cell']['window']=='k4'],key=lambda p:p['cell']['phase_index'])
    phase_bounds=[]
    for p in primary_cells:
        group=next(g for g in p['groups'] if g['method']=='rk4-fine');linear=group['original_linear_interval']
        accuracy=group['primary']['interval'] if p['qualified'] else [0.,1.]
        gain=_difference_interval(accuracy,linear) if linear is not None else [-1.,1.]
        phase_bounds.append(dict(phase_index=p['cell']['phase_index'],qualified=p['qualified'],original_linear_known=linear is not None,gain_interval=gain))
    full=all(s['qualified'] for s in settings.values());primary=None
    if full:
        value=next(m for m in settings['failed-near-edge']['means'] if m['window']=='k4')
        primary=dict(estimand='Full-reference-mixture minus original covariance-linear first-bit BA at4T',
            setting='failed-near-edge',phases=24,nominal_gain=value['nominal_gain'],gain_interval=value['gain_interval'],
            numerical_interval_only=True,fresh_holdout=False)
    calculations=[r for p in retained for r in p['calculations']]
    method_comparisons=[r for p in retained for r in p['method_comparisons']]
    return dict(complete=all(p['complete'] for p in retained),qualified=full,secondary_qualified=all(p['secondary_qualified'] for p in retained),
        cells=retained,settings=settings,primary=primary,
        primary_failure_bounds=dict(phases=24,phase_rows=phase_bounds,gain_interval=_mean_interval([p['gain_interval'] for p in phase_bounds]),
            qualified_phases=sum(p['qualified'] for p in phase_bounds),scope='Fixed24-phase numerical/missingness bounds, not a confidence interval'),
        counts=dict(cells=144,classifier_definitions=432,classifier_definitions_available=sum(sum(p['definitions_available'].values()) for p in retained),
            finite_class_method_rows=sum(len(p['class_rows']) for p in retained),
            primary_finite_probabilities=sum(r['family']=='finite' and r['recipe']=='primary' for r in calculations),
            secondary_finite_probabilities=sum(r['family']=='finite' and r['recipe']=='one_mode' for r in calculations),
            alternate_fine_probabilities=sum(r['family']=='alternate' for r in calculations),
            reference_model_controls=sum(r['family']=='reference' for r in calculations),total_new_probability_calculations=len(calculations),
            auxiliary_linear_reference_calculations=sum(r['family']=='reference' for r in calculations),
            mixture_plus_auxiliary_probability_calculations=len(calculations)+sum(r['family']=='reference' for r in calculations),
            auxiliary_reference_linear_cdfs_complete=sum(r['family']=='reference' and (r.get('linear_reference') or {}).get('complete') is True for r in calculations),
            auxiliary_reference_linear_cdfs_qualified=sum(r['family']=='reference' and (r.get('linear_reference') or {}).get('qualified') is True for r in calculations),
            calculations_complete=sum(r['result']['complete'] for r in calculations),calculations_qualified=sum(r['result']['qualified'] for r in calculations),
            primary_method_comparisons=sum(r['recipe']=='primary' for r in method_comparisons),
            secondary_method_comparisons=sum(r['recipe']=='one_mode' for r in method_comparisons),
            reference_comparisons=sum(len(p['reference_comparisons']) for p in retained),
            method_context_window_groups=sum(len(p['groups']) for p in retained),
            cells_complete=sum(p['complete'] for p in retained),cells_qualified=sum(p['qualified'] for p in retained)),
        scope='Outcome-motivated analysis of all24 reused phases per setting; first bit with unknown balanced second bit, fixed phase and Gaussian noise; no erasure/capacity claim')

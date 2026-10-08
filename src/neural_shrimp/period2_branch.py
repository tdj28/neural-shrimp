"""Bounded exploratory full-pair continuation; no stability/branch proof claims.

Families return (map, analytic Jacobian) for two log-rate controls. Every
accepted node carries both phases; neither angles nor neuron permutations are
used for association. Dense eigensystem residuals are backward diagnostics.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment


class PilotStop(RuntimeError):
    """A resource/deadline stop, never evidence of a bifurcation."""


@dataclass(frozen=True)
class Options:
    closure_atol: float = 1e-9
    closure_rtol: float = 1e-8
    newton_relative_correction: float = 1e-10
    divisor_separation: float = 1e-6
    candidate_distance: float = .02
    phase_link_displacement: float = .01
    phase_link_fraction_of_separation: float = .25
    phase_cost_gap_fraction_of_separation: float = .5
    pair_replay_agreement: float = 1e-7
    eigen_residual: float = 1e-10
    phase_spectrum_agreement: float = 1e-7
    stability_margin: float = 1e-7
    newton_iterations_per_corrector: int = 20
    line_search_halvings: int = 12

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError('positive finite option required: '+name)
        for name in ('newton_iterations_per_corrector', 'line_search_halvings'):
            if type(getattr(self, name)) is not int:
                raise ValueError('integer option required: '+name)


class Budget:
    def __init__(self, seconds=120., corrections=160, clock=time.monotonic):
        if not np.isfinite(seconds) or seconds <= 0 or type(corrections) is not int or corrections < 1:
            raise ValueError('invalid budget')
        self.clock, self.deadline = clock, clock()+seconds
        self.limit, self.calls = corrections, 0

    def check(self):
        if self.clock() >= self.deadline:
            raise PilotStop('deadline')

    def corrector(self):
        self.check()
        if self.calls >= self.limit:
            raise PilotStop('phase_corrector_limit')
        self.calls += 1
        return self.calls


def finite(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise FloatingPointError('nonfinite state or Jacobian')
    return x


def scale(*states):
    return max(1., *(float(np.max(np.abs(x))) for x in states))


def distance(x, y):
    return float(np.max(np.abs(x-y)))/scale(x, y)


def normalized(x, y, o):
    return float(np.max(np.abs(x-y)/(o.closure_atol+o.closure_rtol*np.maximum(np.abs(x), np.abs(y)))))


def phase_match(parent, child, o, *, replay=False):
    parent, child = finite(parent), finite(child)
    if parent.shape != child.shape or parent.ndim != 2 or parent.shape[0] != 2:
        raise ValueError('matching requires two equal full-pair shapes')
    costs = [distance(parent, child), distance(parent, child[::-1])]
    phase = int(costs[1] < costs[0])
    separation = float(np.max(np.abs(parent[0]-parent[1])))/scale(parent)
    cap = min(o.phase_link_displacement, o.phase_link_fraction_of_separation*separation)
    if replay:
        cap = min(cap, o.pair_replay_agreement)
    gap = costs[1-phase]-costs[phase]
    passed = costs[phase] <= cap and gap >= o.phase_cost_gap_fraction_of_separation*separation and separation >= o.divisor_separation
    return dict(passed=bool(passed), costs=costs, phase=phase, separation=separation,
                displacement_cap=cap, cost_gap=gap, replay=bool(replay))


def correct_phase(f, jac, seed, o, budget, emit):
    x, seed = finite(seed).copy(), finite(seed).copy()
    call = budget.corrector()
    started = budget.clock()
    out = dict(call=call, seed=seed, state=x.copy(), converged=False, reason='pending', iterations=0,
               final_relative_correction=None, candidate_distance=None)
    emit('corrector_start', out)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for iteration in range(o.newton_iterations_per_corrector):
                budget.check()
                y = finite(f(x)); z = finite(f(y)); residual = z-x
                j0, j1 = finite(jac(x)), finite(jac(y))
                step = np.linalg.solve(j1@j0-np.eye(len(x)), -residual)
                budget.check()
                correction = float(np.max(np.abs(step)))/scale(x)
                out.update(state=x.copy(), iterations=iteration+1, final_relative_correction=correction)
                emit('corrector_iteration', dict(call=call, iteration=iteration+1, state=x.copy(),
                     residual=residual, step=step, final_relative_correction=correction))
                if correction <= o.newton_relative_correction:
                    x = finite(x+step)
                    out.update(state=x.copy(), converged=True, reason='converged')
                    break
                accepted = False
                for half in range(o.line_search_halvings+1):
                    budget.check()
                    trial = finite(x+(2.**-half)*step)
                    ztrial = finite(f(finite(f(trial))))
                    accepted = bool(np.max(np.abs(ztrial-trial)) < np.max(np.abs(residual)))
                    emit('line_search', dict(call=call, iteration=iteration+1, halving=half,
                         state=trial, residual=ztrial-trial, accepted=accepted))
                    if accepted:
                        x = trial
                        break
                out['state'] = x.copy()
                if not accepted:
                    out['reason'] = 'line_search_stalled'
                    break
            else:
                out['reason'] = 'newton_iteration_limit'
            out['candidate_distance'] = distance(seed, x)
            if out['candidate_distance'] > o.candidate_distance:
                out.update(converged=False, reason='remote_corrected_phase')
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as error:
        out.update(state=x.copy(), converged=False, reason=type(error).__name__+': '+str(error))
    except PilotStop:
        out.update(state=x.copy(), reason='interrupted',elapsed_seconds=budget.clock()-started)
        emit('corrector_end', out)
        raise
    out['elapsed_seconds']=budget.clock()-started
    emit('corrector_end', out)
    return out


def state_checks(f, pair, o):
    pair = finite(pair)
    images = finite(np.stack([f(p) for p in pair]))
    second = finite(np.stack([f(p) for p in images]))
    closures = [normalized(second[i], pair[i], o) for i in range(2)]
    transitions = [normalized(images[i], pair[1-i], o) for i in range(2)]
    sep = float(np.max(np.abs(pair[0]-pair[1])))/scale(pair)
    return dict(passed=bool(max(closures+transitions) <= 1 and sep >= o.divisor_separation),
                closure=closures, transitions=transitions, separation=sep)


def eigen_residuals(matrix, values, left, right):
    n = len(matrix)
    if values.shape != (n,) or left.shape != (n,n) or right.shape != (n,n):
        raise ValueError('incomplete eigensystem')
    if not all(np.all(np.isfinite(v)) for v in (matrix,values,left,right)):
        raise ValueError('nonfinite eigensystem')
    norm = float(np.linalg.norm(matrix, 2))
    rn, ln = np.linalg.norm(right, axis=0), np.linalg.norm(left, axis=0)
    if np.any(rn == 0) or np.any(ln == 0):
        raise ValueError('zero eigenvector')
    denom = norm+np.abs(values)
    if np.any(denom == 0):
        # Exact zero matrices have exact zero eigenpairs; use unit absolute scale.
        denom = np.maximum(denom, np.finfo(float).tiny)
    rr = np.linalg.norm(matrix@right-right*values, axis=0)/(denom*rn)
    lr = np.linalg.norm(matrix.conj().T@left-left*np.conj(values), axis=0)/(denom*ln)
    overlap = np.abs(np.sum(np.conj(left)*right,axis=0))/(ln*rn)
    conditions = [float(1/v) if v > 0 and np.isfinite(1/v) else None for v in overlap]
    return dict(right_residual=rr, left_residual=lr, eigenvalue_condition=conditions,
                norm=norm)


def spectral_checks(j0, j1, o, budget=None):
    spectra = []
    for matrix in (finite(j1)@finite(j0), finite(j0)@finite(j1)):
        if budget: budget.check()
        values, left, right = eig(matrix, left=True, right=True, check_finite=True)
        residuals = eigen_residuals(matrix, values, left, right)
        if budget: budget.check()
        gaps = np.abs(values[:,None]-values[None,:]); np.fill_diagonal(gaps,np.inf)
        condition = float(np.linalg.cond(matrix-np.eye(len(matrix))))
        radius = float(np.max(np.abs(values)))
        spectra.append(dict(matrix=matrix, values=values, left=left, right=right, **residuals,
            condition_m_minus_i=condition if np.isfinite(condition) else None,
            min_spectral_gap=float(np.min(gaps)) if len(values)>1 else None,
            radius=radius, stability='attracting' if radius<1-o.stability_margin else
            'unstable' if radius>1+o.stability_margin else 'neutral_or_unresolved',
            mode_identity='not_tracked; residuals are not forward error bounds'))
    costs = np.abs(spectra[0]['values'][:,None]-spectra[1]['values'][None,:])
    rows, cols = linear_sum_assignment(costs)
    error = float(np.max(costs[rows,cols]))
    passed = error <= o.phase_spectrum_agreement and all(
        max(np.max(s['right_residual']), np.max(s['left_residual'])) <= o.eigen_residual for s in spectra)
    return dict(passed=bool(passed), phases=spectra, assignment=cols, multiset_error=error)


def qualify_pair(family, params, seeds, o, budget, emit):
    params = finite(params); f, jac = family(params)
    first = correct_phase(f,jac,seeds[0],o,budget,emit)
    if not first['converged']:
        return dict(accepted=False,reason='first_phase_corrector',corrections=[first])
    budget.check()
    second_seed = finite(f(first['state'])) if seeds[1] is None else seeds[1]
    second = correct_phase(f,jac,second_seed,o,budget,emit)
    result = dict(accepted=False,reason='second_phase_corrector',corrections=[first,second])
    if not second['converged']: return result
    pair = np.stack([first['state'],second['state']])
    # Independently corrected opposite phases must generate the same pair.
    from_first = np.stack([first['state'],finite(f(first['state']))])
    from_second = np.stack([finite(f(second['state'])),second['state']])
    independent = phase_match(from_first,from_second,o,replay=True)
    checks = state_checks(f,pair,o)
    result.update(pair=pair,independent_phase_match=independent,checks=checks,reason='state_gate')
    if not checks['passed'] or not independent['passed'] or independent['phase'] != 0: return result
    # Persist the corrected pair before potentially expensive eigensystem work.
    emit('pair_spectrum_pending', dict(params=params,pair=pair,checks=checks,corrections=[first,second]))
    budget.check()
    spectra = spectral_checks(jac(pair[0]),jac(pair[1]),o,budget)
    result.update(spectra=spectra,accepted=spectra['passed'],reason='qualified' if spectra['passed'] else 'spectrum_gate')
    return result


class Continuation:
    def __init__(self,family,o,budget,emit=lambda kind,data: None):
        self.family,self.o,self.budget,self.emit=family,o,budget,emit
        self.nodes, self.edges = {}, []

    def node(self,params,seeds,purpose):
        key = f'node-{len(self.nodes):04d}'
        record = dict(id=key,params=finite(params),rates=np.array([10.**float(v) for v in params]),
                      purpose=purpose,accepted=False,reason='pending')
        self.nodes[key] = record
        self.emit('node_pending', {**record,'seeds':[None if s is None else np.asarray(s) for s in seeds]})
        try:
            record.update(qualify_pair(self.family,params,seeds,self.o,self.budget,self.emit))
        except (FloatingPointError,np.linalg.LinAlgError,ValueError) as error:
            record.update(reason='qualification_error:'+type(error).__name__+': '+str(error))
        self.emit('node',record)
        return record

    def align(self,parent,child,replay=False):
        if not child['accepted']: return dict(passed=False,reason='child_not_qualified')
        return phase_match(parent['pair'],child['pair'],self.o,replay=replay)

    def link(self,parent,target,predictor=None,purpose='output'):
        edge = dict(id=f'edge-{len(self.edges):04d}',parent=parent['id'],target=finite(target),
                    purpose=purpose,status='pending',nodes=[],parent_phase=parent.get('carried_phase',0))
        self.edges.append(edge); self.emit('edge_pending',edge)
        def finish(reason):
            edge.update(status='accepted' if reason=='accepted' else 'rejected',reason=reason)
            self.emit('edge',edge)
            return edge
        direct = self.node(target,parent['pair'] if predictor is None else predictor,purpose+':direct')
        edge['nodes'].append(direct['id']); edge['direct']=direct['id']
        edge['parent_match']=self.align(parent,direct)
        if not edge['parent_match']['passed']: return finish('direct_or_parent_match')
        # Retain native phases; all edges carry explicit selected cyclic bits.
        midpoint = (np.asarray(parent['params'])+target)/2
        half = self.node(midpoint,parent['pair'],purpose+':half')
        edge['nodes'].append(half['id']); edge['half']=half['id']
        edge['half_parent_match']=self.align(parent,half)
        if not edge['half_parent_match']['passed']: return finish('half_or_parent_match')
        halfpair = half['pair'][::-1] if edge['half_parent_match']['phase'] else half['pair']
        endpoint = self.node(target,halfpair,purpose+':half_endpoint')
        edge['nodes'].append(endpoint['id']); edge['endpoint']=endpoint['id']
        edge['half_child_match']=self.align(half,endpoint)
        edge['direct_half_match']=self.align(direct,endpoint,replay=True)
        if not edge['half_child_match']['passed'] or not edge['direct_half_match']['passed']:
            return finish('direct_half_mismatch')
        childpair = direct['pair'][::-1] if edge['parent_match']['phase'] else direct['pair']
        reverse = self.node(parent['params'],childpair,purpose+':reverse')
        edge['nodes'].append(reverse['id']); edge['reverse']=reverse['id']
        edge['reverse_match']=self.align(parent,reverse,replay=True)
        if not edge['reverse_match']['passed']: return finish('reverse_mismatch')
        return finish('accepted')

    def segment(self,parent,target,min_step=6.25e-5,max_halvings=4,previous=None):
        """Connect a requested output endpoint or stop at an explicit gap.

        Each failed main step halves the trial step; accepted substeps remain
        capped at that reduced size for this segment. Diagnostic half steps
        may be below min_step. No rejected edge is ever marked accepted.
        """
        target = finite(target); current=parent; previous=previous
        width=abs(float(target[1]-current['params'][1])); trial_width=width
        retries=0; path=[]
        if width == 0 or target[0] != current['params'][0]: raise ValueError('output segment required')
        while current['params'][1] != target[1]:
            self.budget.check()
            remaining=target[1]-current['params'][1]
            delta=np.copysign(min(trial_width,abs(remaining)),remaining)
            next_params=np.array(target if abs(remaining)<=trial_width else [target[0],current['params'][1]+delta])
            predictor=current['pair']
            if previous is not None:
                denom=current['params'][1]-previous['params'][1]
                if denom != 0:
                    predictor=current['pair']+(next_params[1]-current['params'][1])/denom*(current['pair']-previous['pair'])
            edge=self.link(current,next_params,predictor)
            if edge['status'] != 'accepted':
                retries+=1; trial_width/=2
                if retries>max_halvings or trial_width<min_step*(1-1e-10):
                    return dict(status='unresolved_gap',parent=parent['id'],last_accepted=current['id'],target=target,path=path,
                                failed_edge=edge['id'],reason=edge['reason']),current,previous
                continue
            child=self.nodes[edge['direct']].copy()
            child['carried_phase']=edge['parent_match']['phase']
            if child['carried_phase']: child['pair']=child['pair'][::-1]
            previous,current=current,child
            path.append(edge['id'])
        return dict(status='connected',parent=parent['id'],target=target,path=path,endpoint=current['id']),current,previous

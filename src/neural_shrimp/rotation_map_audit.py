"""Exploratory rotation-map diagnostics; never establishes branch identity."""
from collections import Counter
from math import gcd
import numpy as np

DEFAULTS = dict(offsets=[0.002, 0.004, 0.008, 0.016], chaos=0.005,
                min_planarity=0.95, max_hidden_jump=0.02, max_birth_jump=0.01,
                min_den=5, max_den=32)
FIELDS = ('rho', 'status', 'period', 'ftle', 'planar_share')


def validate_map(m):
    for key in ('log10_hidden', 'log10_output'):
        axis = m[key]
        if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
            raise ValueError('invalid axis: ' + key)
        if not (np.diff(axis) > 0).all() or not np.allclose(np.diff(axis), np.diff(axis)[0]):
            raise ValueError('axis must be uniformly increasing: ' + key)
    shape = (len(m['log10_hidden']), len(m['log10_output']))
    if any(m[k].shape != shape for k in FIELDS):
        raise ValueError('map shape mismatch')
    if not np.isin(m['status'], range(6)).all():
        raise ValueError('unknown status')


def exclusion(status, period, rho, ftle, planar, cfg):
    # Exclusive reasons in this declared order. Stage-1 status remains a screen.
    if status == 0: return 'divergent_screen'
    if status == 1: return 'fixed_screen'
    if status == 2 and period <= 2: return 'period_at_most_two_screen'
    if status == 3: return 'chaotic_screen'
    if status == 5: return 'drift_screen'
    if not np.isfinite([rho, ftle, planar]).all(): return 'nonfinite_diagnostic'
    if not 0 <= rho <= 0.5: return 'rotation_out_of_range'
    if abs(ftle) >= cfg['chaos']: return 'ftle_outside_band'
    if planar < cfg['min_planarity']: return 'low_planarity'
    return 'eligible'


def finite_value(x):
    return float(x) if np.isfinite(x) else None


def pointwise(m, rows, cfg):
    h, o = m['log10_hidden'], m['log10_output']
    output = []
    for offset in cfg['offsets']:
        pairs = []
        for k, row in enumerate(rows):
            if row['type'] != 'neimark_sacker': continue
            target_h = row['log10_hidden_boundary'] + offset
            target_o = row['log10_output']
            p = dict(trace_row=k, requested_offset=offset, target_hidden=target_h,
                     target_output=target_o, birth_rotation=row['theta_over_2pi'])
            if not (h[0] <= target_h <= h[-1] and o[0] <= target_o <= o[-1]):
                p['reason'] = 'outside_map'
                pairs.append(p)
                continue
            i, j = int(np.argmin(abs(h-target_h))), int(np.argmin(abs(o-target_o)))
            status, period = int(m['status'][i,j]), int(m['period'][i,j])
            rho, ftle, planar = [float(m[f][i,j]) for f in ('rho','ftle','planar_share')]
            p.update(grid_index=[i,j], actual_hidden=float(h[i]), actual_output=float(o[j]),
                     actual_offset=float(h[i]-row['log10_hidden_boundary']),
                     output_rounding=float(o[j]-target_o), status=status, period=period,
                     rho=finite_value(rho), ftle=finite_value(ftle), planar_share=finite_value(planar),
                     reason=exclusion(status, period, rho, ftle, planar, cfg))
            if p['reason'] == 'eligible':
                p['abs_rotation_difference'] = abs(rho-row['theta_over_2pi'])
            pairs.append(p)
        eligible = [p for p in pairs if p['reason'] == 'eligible']
        diffs = [p['abs_rotation_difference'] for p in eligible]
        output.append(dict(offset=offset, ns_rows=len(pairs), eligible=len(eligible),
                           exclusions=dict(sorted(Counter(p['reason'] for p in pairs).items())),
                           eligible_status_counts=dict(sorted(Counter(str(p['status']) for p in eligible).items())),
                           median_abs_difference=float(np.median(diffs)) if diffs else None,
                           max_abs_difference=max(diffs) if diffs else None, pairs=pairs))
    return output


def geometric_edges(rows, cfg):
    """Keep row adjacency; these are geometry screens, not branch continuation."""
    outputs = np.array([r['log10_output'] for r in rows])
    if len(outputs) < 2 or not (np.diff(outputs)>0).all():
        raise ValueError('trace rows must retain strictly increasing output order')
    spacing = float(np.median(np.diff(outputs)))
    edges = []
    for k, (a,b) in enumerate(zip(rows[:-1],rows[1:])):
        e = dict(rows=[k,k+1])
        if a['type'] != 'neimark_sacker' or b['type'] != 'neimark_sacker':
            e['reason'] = 'non_ns_endpoint'
        elif b['log10_output']-a['log10_output'] > 1.5*spacing:
            e['reason'] = 'output_gap'
        else:
            e['delta_hidden'] = b['log10_hidden_boundary']-a['log10_hidden_boundary']
            e['delta_birth_rotation'] = b['theta_over_2pi']-a['theta_over_2pi']
            if abs(e['delta_hidden'])>cfg['max_hidden_jump']:
                e['reason'] = 'hidden_jump'
            elif abs(e['delta_birth_rotation'])>cfg['max_birth_jump']:
                e['reason'] = 'angle_jump'
            else: e['reason'] = 'geometrically_admissible'
        edges.append(e)
    return edges


def segment_distance_envelope(a, b, cells):
    """Float geometric bounds conditional on a root somewhere on [a,b].

    min_C dist(C,[a,b]) <= dist(root,Cset) <= min_C max(dist(C,a),dist(C,b)).
    They are not rigorous interval bounds nor uncertainty about the true curve.
    """
    v = b-a
    t = np.clip((cells-a)@v/float(v@v), 0, 1)
    lower = np.linalg.norm(cells-a-t[:,None]*v, axis=1).min()
    upper = np.maximum(np.linalg.norm(cells-a,axis=1),np.linalg.norm(cells-b,axis=1)).min()
    return [float(lower),float(upper)]


def provisional_roots(m, rows, edges, cfg, samples):
    h, o = m['log10_hidden'], m['log10_output']
    scale = np.array([h[1]-h[0],o[1]-o[0]])
    roots = []
    for edge in edges:
        if edge['reason'] != 'geometrically_admissible': continue
        a, b = [rows[k] for k in edge['rows']]
        ra, rb = a['theta_over_2pi'],b['theta_over_2pi']
        if ra == rb: continue
        av = np.array([a['log10_hidden_boundary'],a['log10_output']])
        bv = np.array([b['log10_hidden_boundary'],b['log10_output']])
        for q in range(cfg['min_den'],cfg['max_den']+1):
            for numerator in range(1,(q+1)//2):
                if gcd(numerator,q)!=1: continue
                rho = numerator/q
                if not min(ra,rb)<rho<=max(ra,rb): continue
                t = (rho-ra)/(rb-ra)
                root = av+t*(bv-av)
                mask = ((m['status']==2)&(m['period']==2*q)&np.isfinite(m['rho'])
                        &(abs(m['rho']-rho)<=1/samples+1e-12))
                ii,jj = np.nonzero(mask)
                record = dict(rows=edge['rows'], rotation=f'{numerator}/{q}', period=2*q,
                              endpoints=[av.tolist(),bv.tolist()], endpoint_rotations=[ra,rb],
                              interpolation_weight=float(t), interpolated_root=root.tolist(),
                              branch_identity='unestablished', candidate_cells=len(ii),
                              segment_length_grid_cells=float(np.linalg.norm((bv-av)/scale)),
                              within_map=bool(h[0]<=root[0]<=h[-1] and o[0]<=root[1]<=o[-1]))
                if len(ii):
                    cells = np.column_stack([h[ii],o[jj]])/scale
                    ds = np.linalg.norm(cells-root/scale,axis=1)
                    nearest = int(np.argmin(ds))
                    record.update(nearest_candidate_grid_index=[int(ii[nearest]),int(jj[nearest])],
                                  interpolated_distance_grid_cells=float(ds[nearest]),
                                  segment_distance_envelope_grid_cells=segment_distance_envelope(av/scale,bv/scale,cells))
                else:
                    record.update(interpolated_distance_grid_cells=None,segment_distance_envelope_grid_cells=None)
                roots.append(record)
    return roots


def analyze(m, families, cfg, samples):
    validate_map(m)
    report = dict(evidence_level='exploratory_float_diagnostics', config=cfg,
                  branch_identity='unestablished_between_all_trace_rows_and_map_trajectories', families={})
    for name, trace in families.items():
        rows = trace['rows']
        edges = geometric_edges(rows,cfg)
        report['families'][name] = dict(start=trace['args']['start'], total_rows=len(rows),
             row_types=dict(sorted(Counter(r['type'] for r in rows).items())), edges=edges,
             edge_counts=dict(sorted(Counter(e['reason'] for e in edges).items())),
             comparisons=pointwise(m,rows,cfg), roots=provisional_roots(m,rows,edges,cfg,samples))
    return report

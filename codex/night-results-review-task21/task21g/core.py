"""Own finite RM geometry; no producer imports, fitting, model or target I/O."""
import math
import numpy as np

BINS = (16, 24, 32, 48, 64)
QUANTILES = (0., .1, .25, .5, .75, .9, 1.)


def plain(v):
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return plain(v.tolist())
    if isinstance(v, np.generic):
        return plain(v.item())
    if isinstance(v, float) and not math.isfinite(v):
        return {'nonfinite': 'nan' if math.isnan(v) else '+inf' if v > 0 else '-inf'}
    return v


def extrema(values):
    """Exact plateau runs; interior only, leftmost representative, endpoints protected."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError('finite one-dimensional curve required')
    groups = []
    for i, value in enumerate(y):
        if not groups or value != groups[-1]['value']:
            groups.append({'left': i, 'right': i, 'value': float(value)})
        else:
            groups[-1]['right'] = i
    out = []
    for a, b, c in zip(groups, groups[1:], groups[2:]):
        if ((b['value'] > a['value'] and b['value'] > c['value'])
                or (b['value'] < a['value'] and b['value'] < c['value'])):
            out.append({**b, 'index': b['left'], 'kind': 'maximum' if b['value'] > a['value'] else 'minimum',
                        'left_excursion': abs(b['value']-a['value']),
                        'right_excursion': abs(b['value']-c['value'])})
    return {'plateaus': groups, 'interior': out}


def prominence_profile(x, y, normalization_range=None):
    """Complete finite cancellation hierarchy, not a calibrated persistence test.

    Vertices are endpoints plus interior plateau extrema. At each step take the
    adjacent pair of least vertical separation, ties by current left index.
    Remove both interior vertices, or only the interior vertex beside a protected
    endpoint. Prune newly nonextremal interior vertices; keep both endpoints.
    Effective removal threshold is nondecreasing. No selected pass threshold.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or y.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite matching curves required')
    if len(x) > 1 and not (np.diff(x) > 0).all():
        raise ValueError('curve abscissae must increase')
    ex = extrema(y)
    norm = (float(y.min()),float(y.max())) if normalization_range is None and len(y) else normalization_range
    span = None if norm is None else float(norm[1]-norm[0])
    if len(y) < 2:
        return {'initial': ex, 'events': [], 'endpoint_only': list(range(len(y)))}
    vertices = [0] + [v['index'] for v in ex['interior']] + [len(y)-1]
    events, threshold = [], 0.
    yrange = float(y.max()-y.min())
    while len(vertices) > 2:
        edge = min(range(len(vertices)-1), key=lambda i: (abs(float(y[vertices[i+1]]-y[vertices[i]])), i))
        a, b = vertices[edge:edge+2]
        height = abs(float(y[b]-y[a])); threshold = max(threshold, height)
        removed = [v for v in (a, b) if v not in (0, len(y)-1)]
        vertices = [v for v in vertices if v not in removed]
        pruned = []
        while len(vertices) > 2:
            bad = [i for i in range(1,len(vertices)-1)
                   if not ((y[vertices[i]] > y[vertices[i-1]] and y[vertices[i]] > y[vertices[i+1]])
                           or (y[vertices[i]] < y[vertices[i-1]] and y[vertices[i]] < y[vertices[i+1]]))]
            if not bad:
                break
            discard = [vertices[i] for i in bad]
            pruned.extend(discard); vertices = [v for v in vertices if v not in discard]
        events.append({'edge': [a,b], 'edge_height': height, 'effective_threshold': threshold,
                       'range_normalized_threshold': threshold/span if span is not None and math.isfinite(span) and span > 0 else None,
                       'removed': removed, 'pruned_nonextrema': pruned,
                       'remaining_interior_indices': vertices[1:-1], 'remaining_count': len(vertices)-2})
    return {'initial': ex, 'events': events, 'endpoint_only': vertices,
            'median_curve_range': yrange, 'normalization_range': norm,
            'normalization_scope': 'provided finite-pair y range' if normalization_range is not None else 'this finite curve range',
            'no_selected_threshold': True}


def describe(v):
    a = np.asarray(v, dtype=float)
    return {'count': len(a), 'quantiles': np.quantile(a, QUANTILES, method='linear') if len(a) else None}


def partition(x, y, temporal_indices, bins, mode):
    """All nonempty bins retained; equal-count ties move together by midrank.

    Width bins: right-edge convention, final endpoint included by clipping.
    Count bins: floor(B*midrank/n), exact x ties never split. Empty bin IDs break
    the median-curve run in both conventions. No minimum-count exclusion.
    """
    x, y = np.asarray(x,dtype=float), np.asarray(y,dtype=float)
    ti = np.asarray(temporal_indices)
    if x.ndim != 1 or x.shape != y.shape or ti.shape != x.shape or ti.dtype.kind not in 'iu':
        raise ValueError('pair/index dimensions')
    if bins not in BINS or mode not in ('width','count'):
        raise ValueError('undeclared partition')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('partition requires explicit finite-pair selection')
    if not len(x):
        return {'status':'undefined_empty','bins':bins,'mode':mode,'rows':[],'runs':[]}
    lo, hi = float(x.min()),float(x.max()); span = hi-lo
    if not math.isfinite(span) or span <= 0:
        return {'status':'undefined_x_range','bins':bins,'mode':mode,'x_range':[lo,hi],'rows':[],'runs':[]}
    edges = np.linspace(lo,hi,bins+1) if mode == 'width' else None
    if mode == 'width':
        ids = np.clip(np.searchsorted(edges,x,side='right')-1,0,bins-1)
    else:
        order = np.argsort(x,kind='stable'); ids = np.empty(len(x),dtype=int)
        left = 0
        while left < len(order):
            right = left+1
            while right < len(order) and x[order[right]] == x[order[left]]:
                right += 1
            bin_id = min(bins-1,int(math.floor(bins*(left+right-1)/(2*len(order)))))
            ids[order[left:right]] = bin_id; left = right
    rows=[]; runs=[]; current=[]
    for b in range(bins):
        m = ids == b; indices = np.flatnonzero(m)
        row={'bin':b,'count':int(m.sum()),'pair_indices':ti[m],
             'declared_edges':edges[b:b+2] if edges is not None else None}
        if len(indices):
            row.update(x=describe(x[m]),y=describe(y[m]),median_x=float(np.median(x[m])),median_y=float(np.median(y[m])),
                       min_x=float(x[m].min()),max_x=float(x[m].max()),singleton=len(indices)==1)
            parity=[]
            for p in (0,1):
                pm=m & (ti%2 == p)
                parity.append({'parity':p,'x':describe(x[pm]),'y':describe(y[pm])})
            row['parity']=parity
            row['odd_minus_even_median_y']=(float(parity[1]['y']['quantiles'][3]-parity[0]['y']['quantiles'][3])
                                           if all(z['y']['count'] for z in parity) else None)
            current.append(b)
        elif current:
            runs.append(current);current=[]
        rows.append(row)
    if current:
        runs.append(current)
    summaries=[]
    for run in runs:
        xx=[rows[b]['median_x'] for b in run]; yy=[rows[b]['median_y'] for b in run]
        summaries.append({'bin_ids':run,'median_x':xx,'median_y':yy,'profile':prominence_profile(xx,yy,(float(y.min()),float(y.max())))})
    return {'status':'complete','bins':bins,'mode':mode,'rows':rows,'runs':summaries,
            'empty_bins':[r['bin'] for r in rows if not r['count']], 'all_nonempty_bins_used':True,
            'parity_scope':'descriptive unadjusted conditional locations; no strand significance or single-valuedness gate'}


def literal_producer_count(x,y):
    """Own re-expression of the fixed40-width/min5 contract; all omissions retained."""
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        return {'status':'undefined_nonfinite_or_empty_input'}
    edges=np.linspace(x.min(),x.max(),41); ids=np.clip(np.digitize(x,edges)-1,0,39)
    rows=[];cy=[]
    for b in range(40):
        m=ids==b; kept=int(m.sum())>=5
        rows.append({'bin':b,'count':int(m.sum()),'kept':kept,'median_y':float(np.median(y[m])) if m.any() else None})
        if kept:
            cy.append(float(np.median(y[m])))
    signs=np.sign(np.diff(cy));nonzero=signs[signs!=0]
    count=0 if len(cy)<3 else int(np.sum(nonzero[1:]!=nonzero[:-1]))
    return {'status':'complete','count':count,'rows':rows,'slopes':signs,'nonzero_slopes':nonzero,
            'scope':'literal gaps bridged and zero slopes deleted; not adopted as a reliable shape gate'}


def analyze_sequence(sequence):
    s=np.asarray(sequence,dtype=np.float64)
    if s.ndim!=1 or len(s)<2:
        raise ValueError('one-dimensional sequence with at least two states required')
    x,y=s[:-1],s[1:];ti=np.arange(len(x));finite=np.isfinite(x)&np.isfinite(y)
    out={'states':len(s),'pairs':len(x),'finite_pairs':int(finite.sum()),'nonfinite_pair_indices':ti[~finite],
         'nonfinite_state_indices':np.flatnonzero(~np.isfinite(s)),
         'literal_producer_count':literal_producer_count(x,y),'partitions':[],
         'no_unimodality_or_dimension_pass':True}
    for label,a in (('x',x[finite]),('y',y[finite])):
        out[label+'_finite_range']=None if not len(a) else [float(a.min()),float(a.max())]
        out[label+'_extremum_pair_indices']=None if not len(a) else {'minimum':ti[finite][a==a.min()],'maximum':ti[finite][a==a.max()]}
    for mode in ('width','count'):
        for bins in BINS:
            out['partitions'].append(partition(x[finite],y[finite],ti[finite],bins,mode))
    return out

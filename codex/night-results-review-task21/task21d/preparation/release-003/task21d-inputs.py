"""Source-bound admission and saved normal-form scalar checks; no native replay."""
import math
import numpy as np


def finite(x):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):
        raise ValueError('finite numeric scalar required')
    return float(x)


def delta(saved,own):
    saved,own=finite(saved),finite(own)
    return {'saved':saved,'recomputed':own,'saved_minus_recomputed':saved-own,'absolute_difference':abs(saved-own)}


def normal_form(nf,expected):
    """Check only saved scalar arithmetic/classification and explicit input identity."""
    for key,value in expected['source'].items():
        if nf['source'][key]!=value: raise ValueError('normal-form source mismatch: '+key)
    for key,value in expected['args'].items():
        if nf['args'][key]!=value: raise ValueError('normal-form args mismatch: '+key)
    coordinate=expected['coordinate']
    if coordinate not in ('alpha','log10_temperature'): raise ValueError('unsupported parameter coordinate')
    declared='temperature' if coordinate=='log10_temperature' else 'alpha'
    if nf['args']['param']!=declared: raise ValueError('normal-form parameter coordinate mismatch')
    hs=[finite(h) for h in nf['args']['steps']]
    steps=nf['b_steps']
    if len(hs)!=len(steps) or not hs or len(set(hs))!=len(hs) or any(h<=0 for h in hs):
        raise ValueError('invalid normal-form step ladder')
    if [finite(s['h_relative']) for s in steps]!=hs: raise ValueError('normal-form step ordering mismatch')
    bs=[finite(s['b']) for s in steps]
    gaps=[abs(y-x) for x,y in zip(bs,bs[1:])]
    k=min(range(len(gaps)),key=gaps.__getitem__) if gaps else 0
    own_b=(bs[k]+bs[k+1])*.5 if gaps else bs[0]
    pair=hs[k:k+2] if gaps else hs[:1]
    a=finite(nf['a']); own_ab=a*own_b; finite(own_ab)
    if own_ab==0: raise ValueError('undefined pi/sqrt(abs(ab))')
    mu=nf['multiplier']
    if not isinstance(mu,list) or len(mu)!=2: raise ValueError('normal-form complex pair required')
    real,imag=map(finite,mu)
    residual=finite(nf['orbit_residual']); cosine=finite(nf['left_right_cosine'])
    second=None if nf['second_modulus'] is None else finite(nf['second_modulus'])
    if residual<0 or second is not None and second<0: raise ValueError('negative saved residual/modulus')
    return {'source':nf['source'],'args':nf['args'],'saved_b_steps':steps,'adjacent_absolute_b_differences':gaps,
            'selected_pair_index':k,'selected_pair':pair,'pair_matches_saved':nf['b_plateau_pair']==pair,
            'b':delta(nf['b'],own_b),'ab':delta(nf['ab'],own_ab),
            'pi_over_sqrt_abs_ab':delta(nf['pi_over_sqrt_ab'],float(np.pi/np.sqrt(abs(own_ab)))),
            'crossing_from_saved_fields':{'multiplier':mu,'imaginary_within_1e-9':abs(imag)<=1e-9,
                'real_plus_one_within_0_01':abs(real-1)<=.01},
            'orbit_residual_saved':residual,'second_modulus_saved':second,'left_right_cosine_saved':cosine,
            'residual_second_modulus_extra_enforced_gate':False,
            'coordinate':coordinate,'native_derivatives_or_orbit_recomputed':False}


def classification_diagnostic(cell,index):
    """Compare the original scalar rule without replacing the saved class."""
    values=cell['lyapunov']
    if not isinstance(values,list) or any(isinstance(x,bool) or not isinstance(x,(int,float)) for x in values):
        raise ValueError('numeric saved exponent list required')
    lead=finite(cell['lead_period'])
    if lead<0 or lead!=int(lead): raise ValueError('nonnegative integer symbolic period required')
    p=cell['period']
    positive=all(x>1e-3 for x in values)
    own=('periodic' if p>0 else 'hidden chaos' if positive and lead>=1 else
         'visible chaos' if positive else 'irregular, exponent not positive at both offsets')
    return {'input_index':index,'saved_class':cell['class'],'reconstructed_class':own,
            'matches':cell['class']==own,'period_saved':p,'period_in_source_range_0_to_64':0<=p<=64,
            'lyapunov_saved':values,'lyapunov_all_finite':all(math.isfinite(x) for x in values),
            'all_exponents_strictly_above_1e_3':positive,'lead_period_saved':cell['lead_period'],
            'saved_class_used_for_window_selection':True}


def admit_line(line,definition,nf,curve,spec):
    """Require the complete unique declared cell list and frozen command association."""
    args=line['args']
    if args!=spec['declared_line_args']: raise ValueError('line command differs from frozen original declaration')
    if args['mode']!='cells' or args.get('clamp_entropy') is not None:
        raise ValueError('expected explicit ordinary temperature cell list')
    cells=line['cells']; pairs=definition['cells']; n=spec['expected_cells']
    if len(cells)!=n or len(pairs)!=n: raise ValueError('missing or surplus line/definition cells')
    clean=[]
    for p in pairs:
        if not isinstance(p,list) or len(p)!=2: raise ValueError('declared coordinate pair required')
        clean.append([finite(p[0]),finite(p[1])])
    coords=[p[0] for p in clean]
    if len(set(coords))!=n: raise ValueError('duplicate declared coordinate')
    if any(y<=x for x,y in zip(coords,coords[1:])): raise ValueError('declared coordinates must increase')
    alphas={p[1] for p in clean}
    if len(alphas)!=1: raise ValueError('expected fixed-alpha line')
    alpha=clean[0][1]
    actual=[]
    for c,pair in zip(cells,clean):
        cp=[finite(c['log10_temperature']),finite(c['alpha'])]; actual.append(cp[0])
        if cp!=pair: raise ValueError('exact ordinal coordinate/alpha mismatch')
        if not isinstance(c['class'],str): raise ValueError('cell class missing')
        period=c['period']
        if isinstance(period,bool) or not isinstance(period,(int,float)) or not math.isfinite(period) or period<0 or int(period)!=period:
            raise ValueError('invalid detected period')
    if len(set(actual))!=n: raise ValueError('duplicate saved coordinate')
    # Exact original399b writer defaults, confirmed by prefix supplement.
    if (nf['source']['prompt'],nf['source']['opener'])!=('open','Hello!'):
        raise ValueError('resolved line/normal-form prefix mismatch')
    joint=['model','layer','vectors','concept','window']
    for key in joint:
        if args[key]!=nf['source'][key]: raise ValueError('line/normal-form identity mismatch: '+key)
    if finite(nf['source']['alpha'])!=alpha: raise ValueError('line/normal-form alpha mismatch')
    if nf['source']['period']!=spec['base_period'] or curve['period']!=spec['base_period']:
        raise ValueError('base-period association mismatch')
    original=spec['normal_form_source_curve']['original_path']
    marker='/explore/'
    if original.count(marker)!=1 or not original.endswith('/curve.json'): raise ValueError('curve source path contract')
    run=original.split(marker)[1].rsplit('/',1)[0]
    if nf['source']['run']!=run or nf['args']['run']!=run: raise ValueError('normal-form curve run mismatch')
    if nf['source']['index']!=0 or nf['args']['index']!=0 or nf['args']['nearest_fold'] is not False:
        raise ValueError('expected declared direct curve index0')
    pt=curve['points'][0]
    for key in ['log10_temperature','alpha']:
        if finite(pt[key])!=finite(nf['source'][key]): raise ValueError('normal-form curve coordinate mismatch: '+key)
    diagnostics=[classification_diagnostic(c,i) for i,c in enumerate(cells)]
    return {'classification_diagnostics':diagnostics,
            'classification_mismatch_indices':[r['input_index'] for r in diagnostics if not r['matches']],
            'source_period_range_violation_indices':[r['input_index'] for r in diagnostics if not r['period_in_source_range_0_to_64']],
            'cell_count':n,'unique_coordinates':n,'exact_declared_grid':True,'model':args['model'],'alpha':alpha,
            'first_coordinate':coords[0],'last_coordinate':coords[-1],
            'joint_keys_checked':joint,'effective_prefix_checked':{'prompt':'open','opener':'Hello!'},'classifications_inherited':True,'curve_source_point':pt,
            'no_negative_lyapunov_gate_added':True}

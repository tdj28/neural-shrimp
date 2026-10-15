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


def admit_line(line,expected,nf):
    """Require the complete unique ordinal grid and frozen command association."""
    args=line['args']
    for key,value in expected['args'].items():
        if args[key]!=value: raise ValueError('line command mismatch: '+key)
    if args['mode']!='line' or args.get('clamp_entropy') is not None:
        raise ValueError('only ordinary fixed-alpha temperature line supported')
    n=args['n']
    if isinstance(n,bool) or not isinstance(n,int) or n<1: raise ValueError('invalid line count')
    ends=args['log_temperature']
    if not isinstance(ends,list) or len(ends)!=2: raise ValueError('line endpoints required')
    low,high=map(finite,ends)
    if not low<high: raise ValueError('line must increase')
    alpha=finite(args['alpha']); cells=line['cells']
    if len(cells)!=n: raise ValueError('missing or surplus line cells')
    coords=[finite(c['log10_temperature']) for c in cells]
    if len(set(coords))!=n: raise ValueError('duplicate coordinate')
    expected_grid=np.linspace(low,high,n)
    if coords!=expected_grid.tolist(): raise ValueError('exact ordinal grid mismatch')
    for c in cells:
        if finite(c['alpha'])!=alpha: raise ValueError('cell alpha mismatch')
        if not isinstance(c['class'],str): raise ValueError('cell class missing')
        p=c['period']
        if isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or p<0 or int(p)!=p:
            raise ValueError('invalid detected period')
    for key in expected['joint_keys']:
        if args[key]!=nf['source'][key]: raise ValueError('line/normal-form identity mismatch: '+key)
    if finite(nf['source']['alpha'])!=alpha: raise ValueError('line/normal-form alpha mismatch')
    if nf['source']['period']!=expected['base_period']: raise ValueError('base-period association mismatch')
    return {'cell_count':n,'unique_coordinates':n,'exact_grid':True,'model':args['model'],'alpha':alpha,
            'first_coordinate':coords[0],'last_coordinate':coords[-1],
            'joint_keys_checked':expected['joint_keys'],'classifications_inherited':True,
            'no_negative_lyapunov_gate_added':True}

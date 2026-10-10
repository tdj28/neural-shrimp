"""Saved floating diagnostics for failed native unit-pair domains; not proofs."""
import math
import numpy as np
from .rigorous_eigenpair_locus import require


def middle_radius(v):
    require(isinstance(v,dict)and 'mid'in v and 'rad'in v,'encoded finite ball')
    m,e=v['mid'];r,f=v['rad'];require(all(type(x)is int for x in(m,e,r,f))and r>=0,'dyadic ball')
    mid=math.ldexp(float(m),e);rad=math.ldexp(float(r),f)
    require(math.isfinite(mid)and math.isfinite(rad),'finite floating ball proposal')
    return mid,rad


def positive_scaling(B,iterations=512):
    B=np.asarray(B,dtype=float);n=B.shape[0]
    require(B.shape==(n,n)and n>0 and np.isfinite(B).all()and np.all(B>=0),'finite nonnegative square matrix')
    require(type(iterations)is int and 1<=iterations<=1024,'bounded numerical iterations')
    w=np.ones(n)
    for _ in range(iterations):
        v=B@w;top=float(np.max(v))
        if top==0:return {'weights':w.tolist(),'collatz_min':0.,'collatz_max':0.,'iterations':0}
        w=np.maximum(v/top,1e-200)
    ratios=(B@w)/w
    return {'weights':w.tolist(),'collatz_min':float(np.min(ratios)),'collatz_max':float(np.max(ratios)),'iterations':iterations}


def diagnose(chart):
    require(chart['schema']=='neural-shrimp-native-unit-pair-tube-v1'and chart['shooting_dimension']==704,'full704 failed chart')
    require(chart['gates']=={'anchor_identified':True,'primitive_period':True,'realified_frame_independent':True,'uniform_normalized_unit_pair':False},'root-bound failure only')
    require(len(chart['defect'])==len(chart['preconditioner'])==704,'saved full dimensions')
    def parse(rows):
        m=np.empty((len(rows),len(rows[0])));r=np.empty_like(m)
        for i,row in enumerate(rows):
            require(len(row)==m.shape[1],'rectangular saved matrix')
            for j,v in enumerate(row):m[i,j],r[i,j]=middle_radius(v)
        return m,r
    dm,dr=parse(chart['defect']);B=np.abs(dm)+dr
    C,Cr=parse(chart['preconditioner']);v,vr=parse(chart['predictor_residual'])
    require(C.shape==(704,704)and v.shape==(704,1),'complete forcing dimensions')
    force=(np.abs(C@v)+np.abs(C)@vr+Cr@np.abs(v)+Cr@vr)[:,0]
    radii=np.array([sum(middle_radius(v))for v in chart['coordinate_radii']])
    require(np.all(radii>0)and np.isfinite(force).all(),'positive old radii and finite forcing')
    scale=positive_scaling(B);groups={}
    for name,ids in [('state',[39*k+i for k in range(18)for i in range(13)]),('q',[39*k+13+i for k in range(18)for i in range(13)]),('r',[39*k+26+i for k in range(18)for i in range(13)]),('rates',[702,703])]:
        rows=B.sum(axis=1)[ids];ratios=((force+B@radii)/radii)[ids]
        groups[name]={'max_absolute_row_sum':float(np.max(rows)),'max_image_ratio':float(np.max(ratios)),'strict_inclusions':sum(chart['coordinate_inclusions'][i]for i in ids),'coordinates':len(ids)}
    answer={'scope':'Saved binary64 proposal diagnostics only; no certified spectral radius, new domain, root, branch or stability conclusion.',
            'scaling':scale,'groups':groups,'forcing_upper_proposal':force.tolist(),'old_radii':radii.tolist(),'absolute_defect_max_row_sum':float(np.max(B.sum(axis=1)))}
    if scale['collatz_max']<.95:
        candidate=np.linalg.solve(np.eye(704)-B,force)
        answer['resolvent_proposal']={'nonnegative':bool(np.isfinite(candidate).all()and np.all(candidate>=0)),'radii':candidate.tolist(),'max_linear_residual':float(np.max(np.abs((np.eye(704)-B)@candidate-force)))}
    return answer

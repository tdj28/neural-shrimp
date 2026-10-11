"""Saved705 binary64 conditioning proposals; never a certificate."""
import numpy as np
from .rigorous_eigenpair_locus import require
from .unit_pair_conditioning import middle_radius,positive_scaling

def diagnose(chart):
    require(chart['schema']=='neural-shrimp-native-unit-pair-pal-v1'and chart['shooting_dimension']==705,'full705 connected continuation chart')
    require(chart['gates']=={'anchor_identified':False,'primitive_period':True,'realified_frame_independent':True,'uniform_full_native_unit_pair_locus':True}and len(chart['coordinate_inclusions'])==705 and all(chart['coordinate_inclusions']),'successful full705 uniform root only')
    require(len(chart['defect'])==len(chart['preconditioner'])==705,'saved full dimensions')
    def parse(rows):
        m=np.empty((len(rows),len(rows[0])));r=np.empty_like(m)
        for i,row in enumerate(rows):
            require(len(row)==m.shape[1],'rectangular saved matrix')
            for j,v in enumerate(row):m[i,j],r[i,j]=middle_radius(v)
        return m,r
    dm,dr=parse(chart['defect']);B=np.abs(dm)+dr
    C,Cr=parse(chart['preconditioner']);v,vr=parse(chart['predictor_residual'])
    require(C.shape==(705,705)and v.shape==(705,1),'complete forcing dimensions')
    force=(np.abs(C@v)+np.abs(C)@vr+Cr@np.abs(v)+Cr@vr)[:,0]
    radii=np.array([sum(middle_radius(v))for v in chart['coordinate_radii']])
    require(np.all(radii>0)and np.isfinite(force).all(),'positive old radii and finite forcing')
    scale=positive_scaling(B);groups={}
    for name,ids in [('state',[39*k+i for k in range(18)for i in range(13)]),('q',[39*k+13+i for k in range(18)for i in range(13)]),('r',[39*k+26+i for k in range(18)for i in range(13)]),('rates',[702,703]),('angle',[704])]:
        rows=B.sum(axis=1)[ids];ratios=((force+B@radii)/radii)[ids]
        groups[name]={'max_absolute_row_sum':float(np.max(rows)),'max_image_ratio':float(np.max(ratios)),'strict_inclusions':sum(chart['coordinate_inclusions'][i]for i in ids),'coordinates':len(ids)}
    answer={'scope':'Saved binary64 proposal diagnostics only; no certified spectral radius, new domain, root, branch or stability conclusion.',
            'scaling':scale,'groups':groups,'forcing_upper_proposal':force.tolist(),'old_radii':radii.tolist(),'absolute_defect_max_row_sum':float(np.max(B.sum(axis=1)))}
    if scale['collatz_max']<.95:
        candidate=np.linalg.solve(np.eye(705)-B,force)
        answer['resolvent_proposal']={'nonnegative':bool(np.isfinite(candidate).all()and np.all(candidate>=0)),'radii':candidate.tolist(),'max_linear_residual':float(np.max(np.abs((np.eye(705)-B)@candidate-force)))}
    return answer


def project_geometry(points,normal,origin):
    normal=np.asarray(normal,dtype=float);origin=np.asarray(origin,dtype=float)
    require(normal.shape==origin.shape==(705,)and np.isfinite(normal).all()and np.isfinite(origin).all(),'full finite gauge')
    rows=[]
    for point in points:
        x=np.r_[point['native_flat'],point['t']]
        require(x.shape==(705,)and np.isfinite(x).all(),'all native coordinates and angle')
        rows.append({'index':len(rows),'t':float(x[-1]),'rates':x[-3:-1].tolist(),'gauge_projection':float(normal@(x-origin)),'distance_from_origin':float(np.linalg.norm(x-origin)),'complement_max_modulus':float(point['diagnostics']['complement_max_modulus'])})
    require(len(rows)>=2,'retained numerical geometry prefix')
    return {'points':rows,'projection_reversal_proposals':[i for i in range(1,len(rows)-1)if (rows[i]['gauge_projection']-rows[i-1]['gauge_projection'])*(rows[i+1]['gauge_projection']-rows[i]['gauge_projection'])<=0],
      'scope':'Saved native floating projections only. No rigorous curve identity, gauge monotonicity, turning point, complementary spectrum or event bracket is inferred.'}

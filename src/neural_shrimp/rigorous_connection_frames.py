"""Phasewise dyadic frames and exact shared-parameter joins for shooting tubes.

Numerical Schur/QR only proposes frames. Every inverse, conjugation, root join
and stability verdict uses Arb. The historical shooting kernel is unchanged.
"""
from fractions import Fraction
import numpy as np
from scipy.linalg import schur
from flint import arb,arb_mat,ctx
from .rigorous_orbits import exact,dyadic,eye,midpoint_matrix,norm_inf_upper,ball_json,matrix_json
from .rigorous_connection import tube,restore_ball,balanced_product,power_gate,require


def restore_matrix(rows):return arb_mat([[restore_ball(v) for v in row] for row in rows])
def fraction_mid(value):
    require(value.is_finite(),'finite proposal')
    a,b=value.mid().man_exp();return Fraction(int(a))*Fraction(2)**int(b)


def propose_frames(blocks):
    points=[np.array([[float(m[i,j].mid()) for j in range(m.ncols())]
                      for i in range(m.nrows())],dtype=np.float64) for m in blocks]
    product=np.eye(points[0].shape[0])
    for m in points:product=m@product
    require(np.isfinite(product).all(),'finite Schur proposal')
    _,base=schur(product,output='real')
    frames=[base]
    for m in points[:-1]:
        q,_=np.linalg.qr(m@frames[-1]);frames.append(q)
    require(all(np.isfinite(x).all() for x in frames),'finite QR proposal')
    return [arb_mat([[exact(dyadic(float(x))) for x in row] for row in f]) for f in frames]


def framed_product(blocks,frames,tick=lambda:None):
    require(len(frames)==len(blocks)>0,'cyclic frame inventory')
    n=blocks[0].nrows();inverse=[];checks=[]
    for frame in frames:
        require(frame.nrows()==frame.ncols()==n,'square frame dimensions')
        require(all(frame[i,j].is_exact() and frame[i,j].is_finite() for i in range(n) for j in range(n)),'fixed exact dyadic frames')
        tick();inv=frame.inv();pre=midpoint_matrix(inv)
        defect=norm_inf_upper(eye(n)-pre*frame)
        require(defect.is_finite() and defect<1,'verified frame invertibility')
        inverse.append(inv);checks.append({'inverse':matrix_json(inv),'inverse_defect':ball_json(defect)})
    changed=[inverse[(j+1)%len(blocks)]*block*frames[j] for j,block in enumerate(blocks)]
    return balanced_product(changed),changed,checks


def anchored_at_parameter(result,anchor):
    if anchor is None:return False,[]
    rate=Fraction(anchor['rate']);left,right=map(Fraction,result['exact_parameter_endpoints'])
    require(left<=rate<=right,'anchor common parameter in chart')
    n=result['statement']['phase_dimension'];period=result['statement']['period']
    require(len(anchor['phase_boxes'])==period and all(len(row)==n for row in anchor['phase_boxes']),'anchor native dimensions')
    center=[restore_ball(v) for row in result['center'] for v in row]
    slope=restore_matrix(result['predictor_slope']);radius=restore_ball(result['correction_radius'])
    offset=exact(str(rate-Fraction(result['parameter_center'])))
    checks=[]
    for j,row in enumerate(anchor['phase_boxes']):
        for i,value in enumerate(row):
            index=j*n+i
            predicted=center[index]+slope[index,0]*offset
            checks.append(bool(abs(restore_ball(value)-predicted).upper()<=radius))
    return bool(all(checks)),checks


def framed_tube(model,initial,rate,*,anchor=None,tick=lambda:None,observe=lambda kind,row:None,**settings):
    result=tube(model,initial,rate,anchor=None,tick=tick,observe=observe,**settings)
    with ctx.workprec(settings.get('precision',256)):
        result['baseline_gates']=dict(result['gates'])
        blocks=[restore_matrix(m) for m in result['phase_jacobians']]
        result['framing']={'method':'real Schur of midpoint native product; phasewise QR; fixed dyadic frames; last frame equals first',
                           'proposal_only':True,'cyclic_final_frame_index':0}
        try:
            frames=propose_frames(blocks)
            monodromy,changed,inverses=framed_product(blocks,frames,tick)
            tick();attracting,power,powers=power_gate(monodromy,settings.get('max_power',4096))
            result['framing'].update(passed=True,frames=[matrix_json(m) for m in frames],
                inverses=inverses,phase_jacobians=[matrix_json(m) for m in changed],
                monodromy=matrix_json(monodromy),power_bounds=powers,attraction_power=power)
        except (ValueError,ZeroDivisionError,RuntimeError) as exc:
            attracting=False
            result['framing'].update(passed=False,error={'type':type(exc).__name__,'message':str(exc)})
        joined,checks=anchored_at_parameter(result,anchor)
        result['anchor_inclusions']=checks
        result['anchor_parameter']=None if anchor is None else anchor['rate']
        result['gates'].update(uniform_attraction=attracting,anchor_identified=joined)
        result['passed']=bool(all(result['gates'].values()))
        result['evidence_rung']=3 if result['passed'] else 0
        result['schema']='neural-shrimp-framed-shooting-tube-v1'
        result['statement']['stability']='cyclic exact-frame conjugation before native derivative product; infinity-norm power bound'
        return result


def two_charts(model,inputs,*,synthetic_profile=False,tick=lambda:None,observe=lambda kind,row:None,**settings):
    first=framed_tube(model,inputs['initial'],inputs['rate'],anchor=inputs['anchor'],tick=tick,observe=observe,**settings)
    observe('chart_completed',{'chart':0,'result':first})
    charts=[first]
    qualifies=all(first['gates'][k] for k in ('uniform_root','primitive_period','anchor_identified'))
    if qualifies or synthetic_profile:
        with ctx.workprec(settings.get('precision',256)):
            width=Fraction(2)**settings['halfwidth_exponent'];rate=Fraction(inputs['rate'])
            n=model.n;period=len(inputs['initial'])
            base=[restore_ball(v) for row in first['center'] for v in row]
            slope=restore_matrix(first['predictor_slope']);offset=exact(str(2*width))
            proposal=[str(fraction_mid(base[i]+slope[i,0]*offset)) for i in range(period*n)]
            endpoint=first['endpoint_root_enclosures'][1]
            anchor={'rate':str(rate+width),'phase_boxes':[endpoint[j*n:(j+1)*n] for j in range(period)]}
            second=framed_tube(model,[proposal[j*n:(j+1)*n] for j in range(period)],str(rate+2*width),anchor=anchor,tick=tick,observe=observe,**settings)
        observe('chart_completed',{'chart':1,'result':second});charts.append(second)
    gates={'two_charts':len(charts)==2,
           'connected_primitive_branch':len(charts)==2 and all(all(r['gates'][k] for k in ('uniform_root','primitive_period','anchor_identified')) for r in charts),
           'uniform_attraction':len(charts)==2 and all(r['gates']['uniform_attraction'] for r in charts)}
    return {'schema':'neural-shrimp-two-framed-charts-v1','passed':all(gates.values()),'gates':gates,
            'evidence_rung':3 if all(gates.values()) else 0,'research_status':'exploratory; independent outcome review pending',
            'charts':charts,'original_anchor_rate':inputs['rate'],
            'connected_parameter_endpoints':[first['exact_parameter_endpoints'][0],charts[-1]['exact_parameter_endpoints'][1]] if gates['connected_primitive_branch'] else None,
            'scope':'two exact common-parameter phase-labelled charts, only if connected_primitive_branch passes; attraction separate',
            'not_claimed':['connection to fold/flip','whole window','other rate families','practical utility']}

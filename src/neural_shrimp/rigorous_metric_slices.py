"""Restrict a proved affine shooting tube to exact closed parameter slices."""
from fractions import Fraction
from flint import arb
from .rigorous_orbits import exact,ball_json,matrix_json
from .rigorous_connection import require,restore_ball
from .rigorous_connection_frames import restore_matrix
from .rigorous_periodic_lyapunov import periodic_witness


def partition(chart,count=16):
    require(type(count) is int and count==16,'fixed sixteen slices')
    left,right=map(Fraction,chart['exact_parameter_endpoints']);step=(right-left)/count
    require(step>0,'positive chart width')
    result=[(str(left+i*step),str(left+(i+1)*step)) for i in range(count)]
    require(result[0][0]==str(left) and result[-1][1]==str(right) and all(a[1]==b[0] for a,b in zip(result,result[1:])),'closed exact coverage')
    return result


def derivative_slice(model,chart,left,right,tick=lambda:None):
    require(chart['gates']['uniform_root'],'proved parent root required')
    lo,hi=Fraction(left),Fraction(right);a,b=map(Fraction,chart['exact_parameter_endpoints'])
    require(a<=lo<=hi<=b,'slice contained in parent')
    p=chart['statement']['period'];n=model.n
    require(p==18 and n==13,'fixed native18x13')
    h0=Fraction(chart['parameter_center']);center=(lo+hi)/2;width=(hi-lo)/2
    delta=arb(exact(str(center-h0)),exact(str(width)))
    rate=arb(exact(str(center)),exact(str(width)))
    base=[restore_ball(v) for row in chart['center'] for v in row]
    slope=restore_matrix(chart['predictor_slope']);image=restore_matrix(chart['image_correction'])
    boxes=[base[i]+slope[i,0]*delta+image[i,0] for i in range(p*n)]
    blocks=[]
    for j in range(p):
        tick();_,jac,_=model.step(boxes[j*n:(j+1)*n],rate);blocks.append(jac)
    return blocks,{'exact_endpoints':[str(lo),str(hi)],'floor_only':lo==hi,'parameter_interval':ball_json(rate),
        'phase_boxes':[[ball_json(v) for v in boxes[j*n:(j+1)*n]] for j in range(p)],
        'phase_jacobians':[matrix_json(m) for m in blocks],
        'root_enclosure':'unchanged parent affine predictor restricted to this interval plus unchanged proved uniform Krawczyk image'}


def evaluate_slice(model,chart,left,right,*,tick=lambda:None,observe=lambda k,r:None):
    blocks,domain=derivative_slice(model,chart,left,right,tick)
    observe('derivative_domain',domain)
    proof=periodic_witness(blocks,precision=256,tick=tick,observe=observe)
    return {'domain':domain,'proof':proof,'passed':proof['passed']}

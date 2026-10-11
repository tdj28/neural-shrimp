"""Independent endpoint residual reconstruction before changing affine gauge.

The caller authenticates a completed independent uniform proof and its exact
known endpoint. This routine only tightens that same existing root.
"""
from fractions import Fraction as F
from flint import arb,arb_mat,ctx
import review_unit_pair_pal_math as A
import review_known_endpoint_math as L
need=A.need;R=A.R;M=A.M

def endpoint_residual(coeff,whole,h):
    n=coeff.nrows()
    need(n==whole.nrows() and coeff.ncols()==whole.ncols()==4 and h.is_finite() and h.is_exact() and h>0,'full endpoint Taylor data and positive exact halfwidth')
    return arb_mat([[coeff[i,0]+coeff[i,1]*h+coeff[i,2]*h*h+whole[i,3]*h*h*h]for i in range(n)])

def localize_right(chart,directions,paths,known,precision=384,tick=lambda:None):
    need(chart['schema']=='neural-shrimp-native-unit-pair-pal-v1'and chart['dimension']==13 and chart['period']==18 and chart['shooting_dimension']==705,'same full705 known curve')
    cfg=chart['settings'];N=705
    need(len(directions)==144 and len(paths)==108 and len(known)==N,'complete known acquisition and root')
    sc=F(chart['parameter_center']);h=F(2)**cfg['halfwidth_exponent'];rho=F(2)**cfg['radius_exponent']
    need(chart['parameter_endpoints']==[str(sc-h),str(sc+h)],'literal known endpoint parameter')
    c,s,a=[[M.point(v)for v in chart[k]]for k in ['center','slope','curvature']]
    w=list(map(F,chart['weights']));nn=list(map(F,chart['normal']));oo=list(map(F,chart['origin']))
    need(all(len(v)==N for v in [c,s,a,w,nn,oo])and all(v>0 for v in w),'full exact geometry')
    need([M.point(v)for v in chart['coordinate_radii']]==[rho*v for v in w],'same original correction box')
    for row in chart['preconditioner']:
        for v in row:M.point(v)
    with ctx.workprec(precision):
        c,s,a=[[R.exact(v)for v in row]for row in [c,s,a]];w=[R.exact(v)for v in w];nn=[R.exact(v)for v in nn];oo=[R.exact(v)for v in oo];h=R.exact(h);sc=R.exact(sc);d=arb(0,h)
        whole=[c[i]+s[i]*d+a[i]*d*d/2+arb(0,(R.exact(rho)*w[i]).upper())for i in range(N)]
        for old,new in zip(chart['whole_box'],whole,strict=True):A.U.contains(old,A.ball_json(new),'complete inherited derivative domain')
        _,J=A.rebuild(whole,directions[108:],13,18,nn,oo,sc+d);tick()
        coeff=A.path_coefficients(c,s,a,arb(0),paths[36:72],13,18,nn,oo,sc);tick()
        wc=A.path_coefficients(c,s,a,d,paths[72:],13,18,nn,oo,sc);tick()
        G=endpoint_residual(coeff,wc,h);predictor=[c[i]+s[i]*h+a[i]*h*h/2 for i in range(N)]
        answer=L.localize(R.matrix(chart['preconditioner']),J,G,predictor,known,A.vector_json(whole),w);tick()
        answer.update(parameter=chart['parameter_endpoints'][1],residual=A.vector_json([G[i,0]for i in range(N)]),scope='Independent direct full705 Jacobian and endpoint Taylor residual; weighted Neumann enclosure and three intersections of the same independently existing right-endpoint root. Changed-gauge membership remains a separate full705 check.')
        return answer

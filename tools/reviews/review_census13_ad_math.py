"""Independent scalar-loss AD and same-box orbit theorem for the13D census."""
from fractions import Fraction as F
from flint import arb,arb_mat
import review_quadratic_connection_math as R
import review_joined_period18_math as M
from neural_shrimp.rigorous_orbits import ball_json,vector_json
need=M.need

def norm(a):return max(sum((abs(a[i,j]).upper()for j in range(a.ncols())),arb(0)).upper()for i in range(a.nrows()))
def prove(inputs,proposal,step):
    p=inputs['period'];center=[R.original_ball(v)for v in proposal['center']];box=[R.original_ball(v)for v in proposal['box']];n=len(center)
    need(type(p)is int and 1<=p<=64 and 1<=n<=13 and len(box)==n,'bounded native dimensions')
    need(all(v.is_exact()for v in center)and all(M.point(c)==F(v['mid'][0])*F(2)**v['mid'][1]for c,v in zip(proposal['center'],proposal['box'],strict=True)),'same exact centered box')
    C=R.matrix(proposal['preconditioner']);need(C.nrows()==C.ncols()==n and all(v.is_exact()for row in C.tolist()for v in row),'fixed exact preconditioner')
    end=center
    for _ in range(p):end,_=step(end)
    residual=arb_mat([[end[i]-center[i]]for i in range(n)]);state=box;mon=M.ident(n);orbit=[box]
    for _ in range(p):
        state,jac=step(state);mon=jac*mon;orbit.append(state)
    defect=M.ident(n)-C*(mon-M.ident(n));error=arb_mat([[R.original_ball(dict(mid=[0,0],rad=v['rad']))]for v in proposal['box']]);correction=-C*residual+defect*error
    inclusion=[bool(box[i].contains_interior(center[i]+correction[i,0]))for i in range(n)];q=norm(defect)
    divisors=[dict(divisor=k,separating_coordinates=[i for i in range(n)if not(orbit[k][i]-box[i]).contains(0)])for k in range(1,p)if p%k==0]
    power=mon;k=1;bounds=[];attracting=False
    while k<=4096:
        b=norm(power);bounds.append(dict(power=k,bound=ball_json(b)))
        if b.is_finite()and b<1:attracting=True;break
        if not b.is_finite():break
        power=power*power;k*=2
    gates=dict(existence_and_local_uniqueness=bool(all(inclusion)and q.is_finite()and q<1),primitive_period=all(bool(v['separating_coordinates'])for v in divisors),attraction=attracting)
    return dict(passed=all(gates.values()),gates=gates,dimension=n,period=p,same_exact_original_box=True,inclusions=inclusion,contraction=ball_json(q),divisors=divisors,power_bounds=bounds,attraction_power=k if attracting else None,box=proposal['box'],scope='Fresh independent scalar-loss second-order AD with Arb, separately implemented full same-box contraction/divisor/monodromy-power theorem. Exact original map/data; shared Arb and existing independently reviewed scalar AD implementation. No census geometry or utility theorem.')

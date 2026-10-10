"""Independent conditional localization/tangent audit, using saved derivatives.

Root existence is supplied by the separately audited quadratic chart. This
module makes no model call and never infers existence from intersections.
"""
from fractions import Fraction
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import ball_json,vector_json,matrix_json
import review_joined_period18_math as M
from review_localization_math import original_ball,original_matrix,exact,eq
need=M.need


def contains(a,b,why):
    x,y=M.interval(a);u,v=M.interval(b);need(x<=u<=v<=y,why)


def decoded(record):
    # Match a public constructor's documented outward widening, never narrow.
    original_ball(record)
    return arb(arb(tuple(record['mid'])),arb(tuple(record['rad'])))


def dm(rows):return arb_mat([[decoded(v) for v in row] for row in rows])


def localization(chart,start,rows,answer):
    p,n=chart['statement']['period'],chart['statement']['phase_dimension'];N=p*n+1
    need(chart['passed'] and all(chart['gates'].values()),'known inherited root')
    ends=list(map(Fraction,start['exact_endpoints']));lo,hi=ends;sc=(lo+hi)/2;half=(hi-lo)/2
    a,b=map(Fraction,chart['exact_parameter_endpoints'])
    need(a<=lo<=hi<=b and answer['exact_endpoints']==start['exact_endpoints'],'exact closed slice')
    need(Fraction(start['center_parameter'])==sc and answer['passed'] and answer['scientific_claim'] and answer['iterations']==3 and answer['precision']==256,'complete native localization')
    need(start['normal']==chart['normal'] and start['origin']==chart['origin'] and start['preconditioner']==chart['preconditioner'],'exact same gauge/preconditioner')
    need(start['inherited_image']==chart['image_correction'],'same proved root image')
    z=list(map(M.point,chart['center']));s=[M.point(v[0]) for v in chart['predictor_slope']];t=[M.point(v[0]) for v in chart['predictor_curvature']]
    c=list(map(M.point,start['center']));shift=sc-Fraction(chart['parameter_center'])
    need(len(z)==len(s)==len(t)==len(c)==N,'full native dimension')
    for i in range(N):
        oldlo,oldhi=M.interval(chart['image_correction'][i][0]);disp=z[i]+s[i]*shift+t[i]*shift*shift/2-c[i]
        elo,ehi=M.interval(start['initial_error'][i][0]);need(elo<=oldlo+disp and oldhi+disp<=ehi,'all recentering and inherited error retained')
        slo,shi=M.interval(start['shifted_parent_center'][i]);need(slo<=z[i]+s[i]*shift+t[i]*shift*shift/2<=shi,'shifted exact predictor')
        need(M.point(start['slope'][i][0])==s[i]+t[i]*shift and M.point(start['curvature'][i][0])==t[i],'shifted full quadratic derivative')
    normal=list(map(Fraction,chart['normal']));origin=list(map(Fraction,chart['origin']))
    gauge=sum(normal[i]*(c[i]-origin[i]) for i in range(N))-sc
    gl,gh=M.interval(start['center_residual'][-1][0]);need(gl<=gauge<=gh,'actual gauge residual')
    prior=start['initial_error']
    with ctx.workprec(256):
        C=original_matrix(start['preconditioner']);G=original_matrix(start['center_residual']);S=original_matrix(start['slope']);T=original_matrix(start['curvature']);d=arb(0,exact(half));I=M.ident(N)
        need(C.nrows()==C.ncols()==N and all(v.is_exact() for row in C.tolist() for v in row),'exact full preconditioner')
        rhs=arb_mat([[int(i==N-1)] for i in range(N)])
        pred=[exact(c[i])+S[i,0]*d+T[i,0]*d*d/2 for i in range(N)];velocity=S+T*d
        need(len(rows)==3,'all localization intersections')
        for k,row in enumerate(rows):
            need(row['iteration']==k and row['prior_error']==prior,'ordered intersections')
            eq(arb_mat([[v] for v in pred]),[[v] for v in row['predictor']],'actual curved predictor')
            eq(velocity,row['path_velocity'],'actual path velocity')
            need(len(row['derivative_domain'])==N,'whole native domain')
            for i in range(N):
                u,v=M.interval(row['predictor'][i]);el,eh=M.interval(prior[i][0]);dl,dh=M.interval(row['derivative_domain'][i])
                need(dl<=u+min(0,el) and v+max(0,eh)<=dh,'convex predictor/root segment covered')
            need(M.interval(row['derivative_domain'][-1])[0]>0,'positive physical hidden rate')
            A=original_matrix(row['jacobian']);need(A.nrows()==A.ncols()==N,'native bordered matrix')
            for i in range(N):need(A[N-1,i].is_exact() and A[N-1,i]==exact(normal[i]),'fixed exact gauge row')
            # Each source-bound phase acts on its state and the physical hidden
            # rate, with -I to the next native phase; unrelated blocks are zero.
            for j in range(p):
                for i in range(n):
                    for col in range(p*n):
                        phase,component=divmod(col,n)
                        if phase!=j:
                            expect=-int(phase==(j+1)%p and component==i)
                            need(A[j*n+i,col].is_exact() and A[j*n+i,col]==expect,'native off-phase ordering')
            E=original_matrix(prior);path=A*velocity-rhs;residual=G+path*d;defect=I-C*A;image=-C*residual+defect*E
            retained=arb_mat([[E[i,0].intersection(image[i,0])] for i in range(N)])
            for value,key in [(path,'path_derivative'),(residual,'centered_residual'),(defect,'defect'),(image,'image'),(retained,'retained_error')]:eq(value,row[key],'independent '+key)
            prior=row['retained_error']
        eq(retained,answer['final_error'],'final correction')
        final=[pred[i]+retained[i,0] for i in range(N)];eq(arb_mat([[v] for v in final]),[[v] for v in answer['final_state_box']],'final physical root box')
    return {'passed':True,'exact_endpoints':start['exact_endpoints'],'coordinates':N,'intersections':3,'final_state_box':answer['final_state_box']}


def tangent(pre,whole,weights,saved,physical):
    """Independent weighted Neumann solve; no centerline/tangent assumption."""
    with ctx.workprec(256):
        C=dm(pre);A=dm(whole);N=A.nrows();need(A.ncols()==C.nrows()==C.ncols()==len(weights)==N,'full tangent system')
        w=[exact(v) for v in weights];need(all(v.is_exact() and v>0 for v in w),'positive exact weights')
        D=M.ident(N)-C*A
        q=max((sum((abs(D[i,j]).upper()*w[j] for j in range(N)),arb(0))/w[i]).upper() for i in range(N))
        need(q.is_finite() and q<1 and saved['passed'],'strict uniform implicit inverse')
        v=C*arb_mat([[int(i==N-1)] for i in range(N)]);dv=D*v
        tail=(max((abs(dv[i,0]).upper()/w[i]).upper() for i in range(N))/(1-q)).upper()
        y=arb_mat([[arb(v[i,0],w[i]*tail)] for i in range(N)])
        eq(v,saved['preconditioned_rhs'],'implicit RHS');eq(y,saved['initial_derivative_box'],'Neumann tail enclosure')
        need(ball_json(q)==saved['weighted_neumann_bound'] and ball_json(tail)==saved['weighted_error_bound'],'weighted inverse/tail identity')
        need(len(saved['refinements'])==3,'complete implicit intersections')
        for row in saved['refinements']:
            image=v+D*y;y=arb_mat([[y[i,0].intersection(image[i,0])] for i in range(N)])
            eq(image,row['image'],'implicit fixed-point image');eq(y,row['retained'],'implicit intersection')
        eq(y,saved['derivative_box'],'final uniform tangent')
        need(len(physical)==N,'full physical tangent')
        for i in range(N):contains(physical[i],saved['derivative_box'][i][0],'actual mixed derivative tangent encloses proved tangent')
        return {'passed':True,'coordinates':N,'weighted_inverse_bound':ball_json(q),'actual_tangent_retained':True}

"""Independent saved arithmetic for localization of a known shooting root.

No target map/gradient/Hessian evaluation or floating decomposition. Derivative
values are trusted only through the caller's exact input/source/domain bindings.
"""
from fractions import Fraction
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import ball_json,matrix_json
import review_joined_period18_math as M


def eq(a,record,why):
    M.need(a.nrows()==len(record) and all(a.ncols()==len(row) for row in record),why+' shape')
    for i,row in enumerate(record):
        for j,v in enumerate(row):
            got=ball_json(a[i,j]);M.need(got['mid']==v['mid'] and got['rad']==v['rad'],why+' ball identity')

def exact(q):
    q=Fraction(q);return arb(q.numerator)/q.denominator

def parent_image(chart):
    """Bind the actual smaller known-root image, not merely its radius."""
    p,n=chart['statement']['period'],chart['statement']['phase_dimension'];N=p*n
    with ctx.workprec(256):
        A=arb_mat(N,N)
        for phase,block in enumerate(chart['phase_jacobians']):
            J=M.mat(block);M.need(J.nrows()==J.ncols()==n,'native parent dimension')
            for i in range(n):
                for j in range(n):A[phase*n+i,phase*n+j]+=J[i,j]
                A[phase*n+i,((phase+1)%p)*n+i]-=1
        eq(A,chart['jacobian_tube'],'parent shooting assembly')
        C=M.mat(chart['preconditioner']);S=M.mat(chart['predictor_slope']);B=M.mat(chart['parameter_partial_tube']);H=M.mat(chart['residual_center'])
        for row in chart['preconditioner']:
            for v in row:M.point(v)
        rho=arb((1,chart['settings']['radius_exponent']));d=arb(0,arb((1,chart['settings']['halfwidth_exponent'])))
        path=A*S+B;D=M.ident(N)-C*A;image=-C*(H+path*d)+D*arb_mat([[arb(0,rho)]for _ in range(N)])
        eq(image,chart['image_correction'],'inherited parent image')
        M.need(M.norm(D)<1 and all(abs(image[i,0]).upper()<rho for i in range(N)),'valid strict parent root image')
        return {'passed':True,'native_coordinates':N,'parent_image_byte_reduction':True,'contraction':ball_json(M.norm(D))}


def audit_localization(chart,loc,ends):
    p,n=chart['statement']['period'],chart['statement']['phase_dimension'];N=p*n
    M.need(chart['gates']['uniform_root'],'known parent existence required')
    lo,hi=map(Fraction,ends);parentlo,parenthi=map(Fraction,chart['exact_parameter_endpoints']);hc=(lo+hi)/2;half=(hi-lo)/2
    M.need(parentlo<=lo<=hi<=parenthi and loc['exact_endpoints']==list(ends),'exact contained parameter interval')
    M.need(Fraction(loc['center_parameter'])==hc and loc['precision']==256 and loc['iterations']==3 and len(loc['refinements'])==3,'fixed center/precision/three refinements')
    M.need(loc['inherited_primitive_period']==bool(chart['gates']['primitive_period']),'primitive inherited unchanged')
    M.need(loc['predictor_slope']==chart['predictor_slope'] and loc['preconditioner']==chart['preconditioner'],'same exact predictor/C')
    base=[M.point(v)for row in chart['center']for v in row];center=[M.point(row[0])for row in loc['center']]
    slopes=[M.point(row[0])for row in chart['predictor_slope']]
    M.need(len(base)==len(center)==len(slopes)==N,'native coordinate count')
    known=chart['image_correction']
    # Exact rational endpoint arithmetic independently includes all recentering.
    for i in range(N):
        oldlo,oldhi=M.interval(known[i][0]);shift=base[i]+slopes[i]*(hc-Fraction(chart['parameter_center']))-center[i]
        a,b=M.interval(loc['initial_correction'][i][0]);M.need(a<=oldlo+shift and oldhi+shift<=b,'entire inherited root correction retained')
    rate_lo,rate_hi=M.interval(loc['parameter_interval']);M.need(rate_lo<=lo<=hi<=rate_hi,'rate interval covers exact endpoints')
    prior=loc['initial_correction'];summaries=[]
    with ctx.workprec(256):
        C=M.mat(loc['preconditioner']);S=M.mat(loc['predictor_slope']);H=M.mat(loc['center_residual']);d=arb(0,exact(half))
        for k,row in enumerate(loc['refinements']):
            M.need(row['iteration']==k and row['prior_correction']==prior,'complete ordered correction chain')
            M.need(len(row['derivative_phase_boxes'])==p and all(len(v)==n for v in row['derivative_phase_boxes']),'native derivative domain shape')
            for i in range(N):
                elo,ehi=M.interval(prior[i][0]);slo,shi=M.interval(row['predictor_error_segment'][i][0]);M.need(slo<=min(0,elo) and max(0,ehi)<=shi,'derivative segment includes predictor and root error')
                blo,bhi=M.interval(row['derivative_phase_boxes'][i//n][i%n]);spread=abs(slopes[i])*half
                M.need(blo<=center[i]-spread+slo and center[i]+spread+shi<=bhi,'whole convex derivative domain')
            A=M.mat(row['jacobian']);B=M.mat(row['parameter_partial']);E=M.mat(prior)
            M.need(A.nrows()==A.ncols()==N and B.nrows()==N and B.ncols()==1,'full shooting derivative dimensions')
            path=A*S+B;residual=H+path*d;D=M.ident(N)-C*A;image=-C*residual+D*E
            eq(path,row['path_derivative'],'parameter path');eq(residual,row['centered_residual'],'centered residual');eq(D,row['defect'],'defect');eq(image,row['image'],'new correction image')
            retained=arb_mat([[E[i,0].intersection(image[i,0])]for i in range(N)])
            eq(retained,row['retained_correction'],'outward rounded intersection')
            prior=row['retained_correction'];summaries.append({'iteration':k,'coordinates':N,'image_arithmetic_identical':True,'maximum_correction_radius':ball_json(max(retained[i,0].rad().upper()for i in range(N)))})
        M.need(loc['final_correction']==prior,'terminal correction identity')
        for i in range(N):
            elo,ehi=M.interval(prior[i][0]);a,b=M.interval(loc['phase_boxes'][i//n][i%n]);spread=abs(slopes[i])*half
            M.need(a<=center[i]-spread+elo and center[i]+spread+ehi<=b,'final native phase domains cover known root')
    return {'passed':True,'exact_endpoints':list(ends),'floor_only':lo==hi,'refinements':summaries,'primitive_inherited':loc['inherited_primitive_period'],'scope':'independent exact recenter/domain checks and same-precision saved arithmetic; analytic derivative acquisition remains source-bound'}

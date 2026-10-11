"""Full-error Jacobian Taylor coefficients along a finite cubic predictor.

The original reviewed gradient-over-PathJet operations and assembly are reused.
Only the complete cubic path substitution changes. Constant correction E stays
in the value of every d0 and whole-d state/mode/rate/angle, never in its path
first/second coefficients. This yields J0(E),J1(E),J2(I,E), not center-only jets.
"""
from flint import arb,arb_mat
from .rigorous_gradient_path import PathJet,GradientPathJet,entire_path,entire_first_paths
from .rigorous_eigenpair_locus import require,finite
from .rigorous_orbits import matrix_json,ball_json


def cubic_jacobian_coefficients(model,center,slope,curvature,cubic,error,d,period,normal,*,tick=lambda:None,observe=lambda k,v:None,retain_assembly=True):
    require(type(retain_assembly)is bool,'explicit assembly retention mode')
    n=model.n;m=3*n;size=m*period+3
    require(type(period)is int and 1<=period<=18 and 2<=n<=13,'native phase dimensions')
    require(all(len(v)==size for v in [center,slope,curvature,cubic,error,normal]),'all native mode/rate/angle coordinates')
    require(all(v.is_finite()and v.is_exact()for v in center+slope+curvature+cubic+normal),'exact finite predictor and gauge')
    require(d.is_finite()and all(v.is_finite()and v.contains(0)for v in error),'finite parameter and full zero-containing correction domain')
    z=[PathJet([center[i]+slope[i]*d+curvature[i]*d*d/2+cubic[i]*d*d*d+error[i],slope[i]+curvature[i]*d+3*cubic[i]*d*d,curvature[i]/2+3*cubic[i]*d,cubic[i]])for i in range(size)]
    require(z[-3].a[0]>0 and z[-2].a[0]>0,'whole actual rates positive')
    c,e,dd=entire_path(z[-1]);cp,ep,dp=entire_first_paths(z[-1])
    J=[arb_mat(size,size)for _ in range(3)]
    for phase in range(period):
        off=m*phase;after=m*((phase+1)%period);both=[]
        for name,group in [('q',n),('r',2*n)]:
            x=[GradientPathJet(PathJet(z[off+i].a,z[off+group+i].a),[int(i==j)for j in range(n+2)])for i in range(n)]
            rates=[GradientPathJet(z[size-3+i],[int(j==n+i)for j in range(n+2)])for i in range(2)]
            observe('gradient_path_input',{'phase':phase,'direction':name,'d':ball_json(d),'inputs':[v.serialized()for v in x+rates]});tick()
            ret=model.cusp_step(x,*rates)
            require(len(ret)==n and all(isinstance(v,GradientPathJet)and len(v.g)==n+2 for v in ret),'complete native gradient-path return')
            require(all(a.is_finite()for v in ret for p in [v.v,*v.g]for a in p.a+p.b),'finite gradient-path coefficients')
            observe('gradient_path_return',{'phase':phase,'direction':name,'returns':[v.serialized()for v in ret]});both.append(ret)
        qret,rret=both
        for k in range(3):
            for i in range(n):
                for j in range(n):
                    J[k][off+i,off+j]+=qret[i].g[j].a[k]
                    J[k][off+n+i,off+j]+=qret[i].g[j].b[k]
                    J[k][off+2*n+i,off+j]+=rret[i].g[j].b[k]
                    J[k][off+n+i,off+n+j]+=qret[i].g[j].a[k]
                    J[k][off+2*n+i,off+2*n+j]+=rret[i].g[j].a[k]
                for a in range(2):
                    J[k][off+i,size-3+a]+=qret[i].g[n+a].a[k]
                    J[k][off+n+i,size-3+a]+=qret[i].g[n+a].b[k]
                    J[k][off+2*n+i,size-3+a]+=rret[i].g[n+a].b[k]
                if phase<period-1:
                    for group in (0,n,2*n):J[k][off+group+i,after+group+i]-=int(k==0)
                else:
                    J[k][off+i,after+i]-=int(k==0)
                    J[k][off+n+i,after+n+i]-=c.a[k];J[k][off+n+i,after+2*n+i]+=dd.a[k]
                    J[k][off+2*n+i,after+n+i]-=e.a[k];J[k][off+2*n+i,after+2*n+i]-=c.a[k]
                    J[k][off+n+i,size-1]+=(-cp*z[after+n+i]+dp*z[after+2*n+i]).a[k]
                    J[k][off+2*n+i,size-1]+=(-ep*z[after+n+i]-cp*z[after+2*n+i]).a[k]
            if phase==0:
                for i in range(n):
                    J[k][size-3,n+i]=z[n+i].a[k]
                    J[k][size-2,n+i]=z[2*n+i].a[k];J[k][size-2,2*n+i]=z[n+i].a[k]
    for j in range(size):J[0][size-1,j]=normal[j]
    require(all(finite(v)for v in J),'full finite native Jacobian Taylor coefficients')
    summary={'d':ball_json(d),'coefficient_orders':[0,1,2],'shape':[size,size],'derived_assembly_retained':retain_assembly}
    if retain_assembly:summary['coefficients']=[matrix_json(v)for v in J]
    observe('gradient_path_assembly_return',summary)
    return J

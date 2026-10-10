"""Direct native-return paired daughters; all primitive periods checked separately."""
import numpy as np
from .clark_rnn_boundaries import jets

def pair_fixed_amplitude(z,j,g,amplitude,projection,period=6):
    """P=F^period, P(x)=y, P(y)=x, projection.(x-y)=2*amplitude.

    This positive native return uses no sign quotient. A noncollapsed pair is
    an F^(2*period) root; every proper native divisor must still be excluded.
    """
    n=len(j);projection=np.asarray(projection)
    if len(z)!=2*n+1 or projection.shape!=(n,)or not np.isfinite(projection).all():
        raise ValueError('paired state/projection dimensions')
    if not np.isfinite(amplitude)or amplitude<=0 or abs(projection@projection-1)>1e-12:
        raise ValueError('positive amplitude and normalized fixed projection')
    x,y=z[:n],z[n:2*n];d=float(z[-1])
    ex,ux,vx,cx,px=jets(x,j,g,d,period)
    ey,uy,vy,cy,py=jets(y,j,g,d,period)
    h=np.r_[ex-y,ey-x,projection@(x-y)-2*amplitude]
    b=np.zeros((2*n+1,2*n+1));b[:n,:n]=ux[:,:n];b[:n,n:2*n]=-np.eye(n);b[:n,-1]=ux[:,n+1]
    b[n:2*n,:n]=-np.eye(n);b[n:2*n,n:2*n]=uy[:,:n];b[n:2*n,-1]=uy[:,n+1]
    b[-1,:n]=projection;b[-1,n:2*n]=-projection
    return h,b,dict(theta_x=np.array(x),theta_y=np.array(y),gain=g,delta=d,amplitude=float(amplitude),projection=projection.copy(),
        Px=ex,Py=ey,first_x=ux,first_y=uy,second_x=vx,second_y=vy,return_cycle_x=cx,return_cycle_y=cy,return_phase_x=px,return_phase_y=py)

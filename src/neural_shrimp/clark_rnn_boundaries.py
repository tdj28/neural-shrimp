"""Analytic native first/second derivatives for numerical RNN cycle boundaries."""
import numpy as np
from .clark_rnn import step,jacobian,divisors

def jets(theta,j,g,delta,period):
    """Endpoint, d(endpoint)/d(x,g,delta), symmetric second derivative."""
    n=len(theta);m=n+2;z=np.array(theta,copy=True);u=np.zeros((n,m));u[:,:n]=np.eye(n);v=np.zeros((n,m,m));states=[];phase=[]
    eg=np.eye(m)[n];ed=np.eye(m)[n+1]
    for _ in range(period):
        states.append(z.copy());a=jacobian(z,j,g,delta);phase.append(a)
        t=np.tanh(z);s=1-t*t;t2=-2*t*s;ju=(j*s[None,:])@u
        vv=np.einsum('ij,jab->iab',a,v)+delta*g*np.einsum('ij,j,ja,jb->iab',j,t2,u,u)
        vg=delta*ju;vd=-u+g*ju
        vv+=vg[:,:,None]*eg[None,None,:]+vg[:,None,:]*eg[None,:,None]
        vv+=vd[:,:,None]*ed[None,None,:]+vd[:,None,:]*ed[None,:,None]
        jt=j@t;vv+=jt[:,None,None]*(eg[:,None]*ed[None,:]+ed[:,None]*eg[None,:])[None,:,:]
        uu=a@u+delta*jt[:,None]*eg[None,:]+(-z+g*jt)[:,None]*ed[None,:]
        z=step(z,j,g,delta);u=uu;v=vv
    return z,u,v,np.array(states),np.array(phase)

def augmented(z,j,g,period,sign):
    n=len(j);theta=z[:n];delta=float(z[n]);q=z[n+1:]
    if sign not in [1,-1]:raise ValueError('boundary sign')
    end,u,v,states,phase=jets(theta,j,g,delta,period);m=u[:,:n];a=m-np.eye(n);null=m-sign*np.eye(n)
    h=np.concatenate([end-theta,null@q,[(q@q-1)/2]])
    dh=np.zeros((2*n+1,2*n+1));dh[:n,:n]=a;dh[:n,n]=u[:,n+1]
    dh[n:2*n,:n]=np.einsum('ijk,j->ik',v[:,:n,:n],q);dh[n:2*n,n]=v[:,:n,n+1]@q;dh[n:2*n,n+1:]=null;dh[-1,n+1:]=q
    return h,dh,{'theta':theta.copy(),'delta':delta,'right_vector':q.copy(),'endpoint':end,'first':u,'second':v,'cycle':states,'phase_jacobians':phase,'monodromy':m}

def qualify_boundary(z,j,g,period,sign,emit=lambda kind,value:None):
    h,dh,a=augmented(z,j,g,period,sign);emit('qualification_augmented',dict(a,augmented_residual=h,augmented_jacobian=dh));n=len(j);m=a['monodromy'];q=a['right_vector'];e,vec=np.linalg.eig(m);emit('qualification_eigen',{'eigenvalues':e,'eigenvectors':vec});le,lv=np.linalg.eig(m.T);emit('qualification_left_eigen',{'eigenvalues':le,'eigenvectors':lv})
    index=int(np.argmin(np.abs(e-sign)));li=int(np.argmin(np.abs(le-sign)));w=lv[:,li]
    if np.max(np.abs(w.imag))>1e-9:raise ValueError('complex critical left vector')
    w=w.real;overlap=float(w@q)
    if abs(overlap)<1e-8:raise ValueError('degenerate left/right overlap')
    w=w/overlap;other=np.delete(e,index);scale=max(1,float(np.max(np.abs(a['cycle']))));seps={str(d):float(np.max(np.abs(a['cycle'][d]-z[:n]))/scale) for d in divisors(period)}
    bqq=np.einsum('ijk,j,k->i',a['second'][:,:n,:n],q,q);param=a['first'][:,n+1]
    fold_a=float(w@param);fold_c=float(w@bqq);moving=None;alpha=None
    if sign==-1:
        moving=np.linalg.solve(m-np.eye(n),-param);emit('branch_solve',{'matrix':m-np.eye(n),'rhs':-param,'solution':moving})
        alpha=float(w@(np.einsum('ijk,j,k->i',a['second'][:,:n,:n],q,moving)+a['second'][:,:n,n+1]@q))
    res=float(np.max(np.abs(h)));critical=float(abs(e[index]-sign));otherdist=float(np.min(np.abs(other-sign))) if len(other) else 2.;rho=float(np.max(np.abs(other))) if len(other) else 0.;leftres=float(np.max(np.abs(w@m-sign*w)))
    regular=abs(fold_a)>1e-5 and abs(fold_c)>1e-5 if sign==1 else abs(alpha)>1e-5
    good=res<1e-9 and critical<1e-7 and otherdist>1e-4 and rho<1-1e-5 and leftres<1e-7 and all(x>1e-6 for x in seps.values()) and regular
    a.update(augmented_residual=h,augmented_jacobian=dh,eigenvalues=e,eigenvectors=vec,left_eigenvalues=le,left_eigenvectors=lv,left_vector=w,bqq=bqq,parameter_derivative=param)
    if moving is not None:a['branch_derivative']=moving
    scalar={'kind':'fold' if sign==1 else 'flip','status':'numerical_boundary_candidate' if good else 'unqualified','residual':res,'critical_error':critical,'other_distance':otherdist,'complement_radius':rho,'left_residual':leftres,'proper_divisor_separations':seps,'fold_a_delta':fold_a if sign==1 else None,'fold_c':fold_c if sign==1 else None,'total_flip_alpha_delta':alpha,'scope':'numerical local boundary only; no interval, flip cubic, 2D connection or whole shrimp claim'}
    return a,scalar


def family_augmented(z,j,period,sign):
    """Nine equations in (x4,g,delta,q4), or its dimension-generic analog."""
    n=len(j);theta=z[:n];g=float(z[n]);delta=float(z[n+1]);q=z[n+2:]
    if sign not in [1,-1]:raise ValueError('boundary sign')
    end,u,v,states,phase=jets(theta,j,g,delta,period);m=u[:,:n];null=m-sign*np.eye(n)
    h=np.r_[end-theta,null@q,(q@q-1)/2]
    b=np.zeros((2*n+1,2*n+2));b[:n,:n+2]=u;b[:n,:n]-=np.eye(n)
    b[n:2*n,:n+2]=np.einsum('ijk,j->ik',v[:,:n,:],q);b[n:2*n,n+2:]=null;b[-1,n+2:]=q
    return h,b,{'theta':theta.copy(),'gain':g,'delta':delta,'right_vector':q.copy(),'endpoint':end,'first':u,'second':v,'cycle':states,'phase_jacobians':phase,'monodromy':m}


def oriented_null(b,reference=None):
    u,s,vh=np.linalg.svd(b,full_matrices=True);t=vh[-1].copy()
    if reference is None:
        if t[len(t)//2-1]<0:t=-t
    elif t@reference<0:t=-t
    residual=float(np.max(np.abs(b@t)))
    return t,{'matrix':b,'u':u,'s':s,'vh':vh,'null_vector':t,'null_residual':residual}


def third_directional(theta,direction,j,g,delta,period):
    """D^3 F^period(theta)[q,q,q] through analytic scalar tanh rules."""
    z=np.array(theta,copy=True);v=np.array(direction,copy=True);b=np.zeros_like(z);c=np.zeros_like(z);records=[]
    for k in range(period):
        t=np.tanh(z);s=1-t*t;t2=-2*t*s;t3=s*(6*t*t-2);a=jacobian(z,j,g,delta)
        newc=a@c+delta*g*(j@(3*t2*v*b+t3*v**3));newb=a@b+delta*g*(j@(t2*v*v));newv=a@v
        records.append(np.stack([z,v,b,c]));z=step(z,j,g,delta);v=newv;b=newb;c=newc
    return c,np.array(records)

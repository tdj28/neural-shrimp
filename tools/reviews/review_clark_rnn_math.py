"""Independent reductions of saved fixed-RNN arrays; no trajectory/solver API."""
import numpy as np


def need(ok,why):
    if not ok:raise ValueError(why)


def near(a,b,why,rtol=5e-11,atol=1e-12):
    a,b=np.asarray(a),np.asarray(b)
    need(a.shape==b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)),why+' finite/shape')
    need(np.all(np.abs(a-b)<=atol+rtol*np.maximum(np.abs(a),np.abs(b))),why+' arithmetic')
    return float(np.max(np.abs(a-b))) if a.size else 0.


def native(cycle,j,g,d,js,m,end=None):
    """Evaluate each existing phase independently, never advance a new state."""
    c=np.asarray(cycle);p,n=c.shape;need(j.shape==(n,n) and js.shape==(p,n,n),'native dimensions')
    targets=(1-d)*c+d*g*(np.tanh(c)@j.T)
    expected=(1-d)*np.eye(n)[None,:,:]+d*g*j[None,:,:]*(1-np.tanh(c)**2)[:,None,:]
    derivative_error=near(js,expected,'analytic phase J')
    near(targets[:-1],c[1:],'saved native transitions')
    if end is not None:near(targets[-1],end,'saved native endpoint')
    product=np.eye(n)
    for a in js:product=a@product
    product_error=near(m,product,'native ordered product')
    scale=max(1.,float(np.max(np.abs(c))))
    closure=float(np.max(np.abs(targets[-1]-c[0]))/scale)
    sep={str(k):float(np.max(np.abs(c[k]-c[0]))/scale) for k in range(1,p) if p%k==0}
    return dict(closure=closure,separations=sep,derivative_error=derivative_error,product_error=product_error)


def spectrum(m,e,v):
    n=len(m);need(e.shape==(n,) and v.shape==(n,n),'all native spectrum columns')
    norms=np.sqrt(np.sum(np.abs(v)**2,axis=0));need(np.all(norms>0),'nonzero eigenvectors')
    residual=np.sqrt(np.sum(np.abs(m@v-v*e[None,:])**2,axis=0))/np.maximum(1,np.max(np.sum(np.abs(m),axis=1))*norms)
    need(np.all(np.isfinite(residual)),'finite eigen residual')
    # All columns and two characteristic moments are checked. This remains a
    # floating diagnostic, never a rigorous complete-spectrum enclosure.
    near(np.trace(m),np.sum(e),'spectrum trace',rtol=2e-9,atol=2e-9)
    near(np.trace(m@m),np.sum(e*e),'spectrum second moment',rtol=2e-9,atol=2e-9)
    return float(np.max(residual))


def lifted_jets(cycle,j,g,d):
    """Compose full (x,g,d)->(F(x,g,d),g,d) derivatives at saved phases.

    Uses an explicit lifted local Hessian and full chain rule, independent of
    the producer's mixed-jet recurrence. No new phase is generated.
    """
    n=len(j);m=n+2;u=np.eye(m);v=np.zeros((m,m,m))
    for x in cycle:
        t=np.tanh(x);s=1-t*t;local=np.eye(m);h=np.zeros((m,m,m))
        local[:n,:n]=(1-d)*np.eye(n)+d*g*j*s[None,:]
        local[:n,n]=d*(j@t);local[:n,n+1]=-x+g*(j@t)
        for a in range(n):
            for k in range(n):
                h[a,k,k]=d*g*j[a,k]*(-2*t[k]*s[k])
                h[a,k,n]=h[a,n,k]=d*j[a,k]*s[k]
                h[a,k,n+1]=h[a,n+1,k]=-int(a==k)+g*j[a,k]*s[k]
            h[a,n,n+1]=h[a,n+1,n]=(j@t)[a]
        v=np.einsum('ij,jab->iab',local,v)+np.einsum('ijk,ja,kb->iab',h,u,u)
        u=local@u
    return u[:n],v[:n]


def augmented(a,j,g,sign):
    n=len(j);d=float(a['delta']);q=a['right_vector'];x=a['theta'];u,h=lifted_jets(a['cycle'],j,g,d)
    near(a['first'],u,'independent lifted first');near(a['second'],h,'independent lifted second',rtol=2e-10,atol=1e-10)
    native(a['cycle'],j,g,d,a['phase_jacobians'],a['monodromy'],a['endpoint'])
    m=u[:,:n];r=np.r_[a['endpoint']-x,(m-sign*np.eye(n))@q,(q@q-1)/2]
    dh=np.zeros((2*n+1,2*n+1));dh[:n,:n]=m-np.eye(n);dh[:n,n]=u[:,n+1]
    dh[n:2*n,:n]=np.einsum('ijk,j->ik',h[:,:n,:n],q)
    dh[n:2*n,n]=h[:,:n,n+1]@q;dh[n:2*n,n+1:]=m-sign*np.eye(n);dh[-1,n+1:]=q
    near(r,a.get('h',a.get('augmented_residual')),'augmented residual',rtol=2e-10,atol=2e-11)
    near(dh,a.get('dh',a.get('augmented_jacobian')),'augmented Jacobian',rtol=2e-10,atol=1e-10)
    return r


def boundary(a,meta,j):
    sign=1 if meta['kind']=='fold' else -1;n=len(j);r=augmented(a,j,meta['gain'],sign)
    m=a['monodromy'];q=a['right_vector'];w=a['left_vector'];near(w@q,1,'left right normalization')
    er=spectrum(m,a['eigenvalues'],a['eigenvectors']);lr=float(np.max(np.abs(w@m-sign*w)))
    spectrum(m.T,a['left_eigenvalues'],a['left_eigenvectors'])
    e=a['eigenvalues'];idx=int(np.argmin(np.abs(e-sign)));other=np.delete(e,idx)
    crit=float(abs(e[idx]-sign));dist=float(np.min(np.abs(other-sign)));rho=float(np.max(np.abs(other)))
    param=a['first'][:,n+1];bqq=np.einsum('ijk,j,k->i',a['second'][:,:n,:n],q,q)
    near(param,a['parameter_derivative'],'parameter derivative');near(bqq,a['bqq'],'second contraction')
    aa=float(w@param);cc=float(w@bqq);alpha=None
    if sign==-1:
        moving=a['branch_derivative'];near((m-np.eye(n))@moving,-param,'total branch derivative solve',rtol=2e-9,atol=2e-9)
        alpha=float(w@(np.einsum('ijk,j,k->i',a['second'][:,:n,:n],q,moving)+a['second'][:,:n,n+1]@q))
        near(alpha,meta['total_flip_alpha_delta'],'total flip crossing')
    else:near(aa,meta['fold_a_delta'],'fold transverse');near(cc,meta['fold_c'],'fold quadratic contraction')
    scale=max(1,float(np.max(np.abs(a['cycle']))));sep={str(k):float(np.max(np.abs(a['cycle'][k]-a['theta']))/scale)for k in range(1,len(a['cycle']))if len(a['cycle'])%k==0}
    need(set(sep)==set(meta['proper_divisor_separations']),'all proper divisors')
    for k,v in sep.items():near(v,meta['proper_divisor_separations'][k],'primitive separation')
    for k,v in [('residual',np.max(np.abs(r))),('critical_error',crit),('other_distance',dist),('complement_radius',rho),('left_residual',lr)]:near(v,meta[k],k,atol=2e-11)
    good=np.max(np.abs(r))<1e-9 and crit<1e-7 and dist>1e-4 and rho<1-1e-5 and lr<1e-7 and all(v>1e-6 for v in sep.values()) and (abs(aa)>1e-5 and abs(cc)>1e-5 if sign==1 else abs(alpha)>1e-5)
    need(good==(meta['status']=='numerical_boundary_candidate'),'unchanged boundary decision')
    return {'kind':meta['kind'],'qualified':bool(good),'gain':meta['gain'],'delta':float(a['delta']),'eigen_residual':er,'fold_a':aa if sign==1 else None,'fold_c':cc if sign==1 else None,'total_flip_alpha':alpha}


def periods(tail):
    n=tail.shape[1];out=np.zeros(n,dtype=int);res=np.full(n,np.nan);scale=np.maximum(1,np.max(np.abs(tail),axis=(0,2)))
    for p in range(1,65):
        ids=np.flatnonzero((out==0)&(np.max(np.abs(tail[-1]-tail[-1-p]),axis=1)/scale<=1e-8))
        if len(ids):
            err=np.max(np.abs(tail[-4*p:,ids]-tail[-5*p:-p,ids]),axis=(0,2))/scale[ids];good=err<=1e-8
            out[ids[good]]=p;res[ids[good]]=err[good]
    return out,res

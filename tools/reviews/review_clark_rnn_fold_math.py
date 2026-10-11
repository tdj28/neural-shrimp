"""Independent explicit native RNN mixed derivatives and augmented-fold proof.

No producer jet/step/iterate is imported. Scalar tanh is evaluated via exp(2x),
then explicit first/second derivatives are propagated by matrix recurrences.
"""
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import ball_json,vector_json,matrix_json
from review_clark_rnn_c1_math import need,exact,restore,eye,norm,strict_nested,endpoints

class Native:
    def __init__(self,inputs):
        self.n=len(inputs['j0']);self.j=[[exact(v)for v in row]for row in inputs['j0']];self.g=exact(inputs['gain'])
        need(self.n>0 and all(len(r)==self.n for r in self.j)and self.g>0,'native constants')
    def augmented(self,z,p,tick=lambda:None):
        n=self.n;need(len(z)==2*n+1 and type(p)is int and 1<=p<=64,'augmented shape')
        x=list(z[:n]);delta=z[n];q=list(z[n+1:]);g=arb_mat([[arb(int(i==k))for k in range(n+1)]for i in range(n)])
        d=list(q);mix=arb_mat(n,n+1);orbit=[list(x)];blocks=[]
        for _ in range(p):
            tick();e=[(2*v).exp()for v in x];t=[(a-1)/(a+1)for a in e];s=[4*a/((a+1)*(a+1))for a in e];h=[-2*t[k]*s[k]for k in range(n)]
            J=arb_mat([[(1-delta)*int(i==k)+delta*self.g*self.j[i][k]*s[k]for k in range(n)]for i in range(n)])
            par=[-x[i]+self.g*sum((self.j[i][k]*t[k]for k in range(n)),arb(0))for i in range(n)]
            xnew=[(1-delta)*x[i]+delta*self.g*sum((self.j[i][k]*t[k]for k in range(n)),arb(0))for i in range(n)]
            gnew=J*g
            for i in range(n):gnew[i,n]+=par[i]
            dnew=[sum((J[i,k]*d[k]for k in range(n)),arb(0))for i in range(n)]
            mnew=J*mix
            for i in range(n):
                for c in range(n+1):
                    mnew[i,c]+=delta*self.g*sum((self.j[i][k]*h[k]*g[k,c]*d[k]for k in range(n)),arb(0))
                    if c==n:mnew[i,c]+=-d[i]+self.g*sum((self.j[i][k]*s[k]*d[k]for k in range(n)),arb(0))
            x,g,d,mix=xnew,gnew,dnew,mnew;orbit.append(list(x));blocks.append(J)
        M=arb_mat([[g[i,k]for k in range(n)]for i in range(n)]);A=M-eye(n)
        residual=[x[i]-z[i]for i in range(n)]+[d[i]-q[i]for i in range(n)]+[(sum((v*v for v in q),arb(0))-1)/2]
        jac=arb_mat([[A[i,k]for k in range(n)]+[g[i,n]]+[arb(0)]*n for i in range(n)]+[[mix[i,k]for k in range(n+1)]+[A[i,k]for k in range(n)]for i in range(n)]+[[arb(0)]*(n+1)+q])
        return arb_mat([[v]for v in residual]),jac,{'M':M,'parameter':[g[i,n]for i in range(n)],'quadratic':[sum((mix[i,k]*q[k]for k in range(n)),arb(0))for i in range(n)],'orbit':orbit,'native_jacobians':blocks}

def independent_proof(inputs,proposal,precision=384,tick=lambda:None):
    with ctx.workprec(precision):
        model=Native(inputs['map_inputs']);n=model.n;p=inputs['period'];dim=2*n+1
        center=[restore(v)for v in proposal['center']];box=[restore(v)for v in proposal['box']]
        need(len(center)==dim and all(v.is_exact()for v in center)and all(c==b.mid()for c,b in zip(center,box)),'exact centered full box')
        C=arb_mat([[restore(v)for v in row]for row in proposal['preconditioner']]);need(C.nrows()==dim and C.ncols()==dim and all(C[i,j].is_exact()for i in range(dim)for j in range(dim)),'exact preconditioner')
        r,_,_=model.augmented(center,p,tick);_,J,data=model.augmented(box,p,tick)
        D=eye(dim)-C*J;E=arb_mat([[restore({'mid':[0,0],'rad':v['rad']})]for v in proposal['box']]);K=arb_mat([[v]for v in center])-C*r+D*E
        inclusions=[bool(box[i].contains_interior(K[i,0]))for i in range(dim)];contraction=norm(D)
        gates={'augmented_root':all(inclusions)and bool(contraction<1),'open_delta':bool(0<box[n]<1)}
        out={'passed':False,'gates':gates,'precision':precision,'box':proposal['box'],'inclusions':inclusions,'contraction':ball_json(contraction),'krawczyk':matrix_json(K),'residual':matrix_json(r),'jacobian':matrix_json(J),'defect':matrix_json(D),'orbit':[vector_json(x)for x in data['orbit']],'native_jacobians':[matrix_json(x)for x in data['native_jacobians']]}
        if not all(gates.values()):return out
        M=data['M'];A=M-eye(n);q=box[n+1:];B=arb_mat([[A[j,i]for j in range(n)]+[q[i]]for i in range(n)]+[q+[arb(0)]])
        inv=B.inv();left=inv*arb_mat([[arb(0)]]*n+[[arb(1)]]);w=[left[i,0]for i in range(n)];finite=all(inv[i,j].is_finite()for i in range(n+1)for j in range(n+1))
        a=sum((w[i]*data['parameter'][i]for i in range(n)),arb(0));c=sum((w[i]*data['quadratic'][i]for i in range(n)),arb(0))
        Q=M-arb_mat([[q[i]*w[j]for j in range(n)]for i in range(n)]);rows=[];power=Q;exponent=1;attracting=False
        while exponent<=512:
            tick();upper=norm(power);rows.append({'power':exponent,'norm_upper':ball_json(upper)})
            if upper<1:attracting=True;break
            power=power*power;exponent*=2
        divisors=[{'divisor':d,'separating_coordinates':[i for i in range(n)if not(data['orbit'][d][i]-box[i]).contains(0)]}for d in range(1,p)if p%d==0]
        gates.update(simple_plus_one=finite,a_nonzero=bool(a.is_finite()and not a.contains(0)),c_nonzero=bool(c.is_finite()and not c.contains(0)),primitive=all(bool(d['separating_coordinates'])for d in divisors),attracting_complement=attracting)
        out.update(passed=all(gates.values()),border=matrix_json(B),inverse=matrix_json(inv),left_augmented=matrix_json(left),a=ball_json(a),c=ball_json(c),quadratic_coefficient=ball_json(c/2),delta_second=ball_json(-c/a),monodromy=matrix_json(M),parameter=vector_json(data['parameter']),second_direction=vector_json(data['quadratic']),deflated=matrix_json(Q),power_bounds=rows,divisors=divisors,scope='Independent exp-form native derivative recurrence and fold proof on supplied nested full box; same Arb library, no window/branch theorem')
        return out

"""Exploratory PCP1 trajectory, primitive recurrence and Floquet diagnostics."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
from explore_llm_soft_loop import SoftLoop


def recurrence_errors(states, max_period=64, positions=128):
    assert len(states) >= positions + max_period
    end = states[-positions:]
    return [float(np.max(np.linalg.norm(end-states[-positions-p:-p], axis=-1)/np.linalg.norm(end, axis=-1))) for p in range(1,max_period+1)]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--out',type=Path,required=True);p.add_argument('--source',required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();deadline=start+3600
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def record(n,v):(a.out/n).write_text(json.dumps(v,indent=2)+'\n')
    try:
        setup=json.loads((a.input/'setup.json').read_text())
        loop=SoftLoop('qwen0.5b',12,2,'open','Hello!',1,'float64','cpu'); t=loop.torch;t.set_num_threads(2)
        assert loop.sm.engine.revision==setup['model_revision']
        P0=loop.E[t.tensor([setup['ids0']])];P1=loop.E[t.tensor([setup['ids1']])]
        record('setup.json',dict(source=a.source,parent_setup_sha256=sha(a.input/'setup.json'),parent_setup=setup,settings=[.75,.875,1.],settling=512,measured=256,period_max=64,recurrence_positions=128,tolerance=1e-10,wall_limit_seconds=3600,threads=2,qualification='Exploratory, fresh common initial state at each prompt. Numerical cycles, not interval-certified.'))
        def guard():
            if time.monotonic()>deadline:raise TimeoutError('Attractor diagnostic CPU limit')
        def F(x):guard();return loop.step(x,[2.])[0]
        def J(x,v):
            guard()
            with t.no_grad():return loop.step_jvp(x,[2.],v)[2]
        rows=[]
        for index,s in enumerate([.75,.875,1.]):
            loop.prefix_embeds=(1-s)*P0+s*P1;x=loop.prepare(np.zeros((1,896)));history=[]
            for step in range(768):
                x=F(x);history.append(x.numpy().reshape(-1).copy())
                if step%64==63:
                    np.save(a.out/f'trajectory-{index}.npy',np.array(history))
                    record('progress.json',dict(s=s,updates=step+1))
            states=np.array(history);assert np.isfinite(states).all()
            errors=recurrence_errors(states[-256:]);period=next((i+1 for i,e in enumerate(errors) if e<1e-10),None)
            row=dict(s=s,recurrence_errors=errors,period_candidate=period)
            if period:
                cycle=states[-period:];x=t.tensor(cycle[0],dtype=t.float64).reshape(1,1,-1)
                def compose(y):
                    for _ in range(period):y=F(y)
                    return y
                def mv(v):
                    y=x.clone();w=t.tensor(v,dtype=t.float64).reshape(1,1,-1)
                    for _ in range(period):w=J(y,w);y=F(y)
                    return w.numpy().reshape(-1)
                closure=float((compose(x)-x).norm()/x.norm());assert closure<1e-10
                divisors={str(d):errors[d-1] for d in range(1,period) if period%d==0}
                assert all(v>1e-8 for v in divisors.values()),'Primitive period ambiguous'
                direction=np.random.default_rng(19).normal(size=896);direction/=np.linalg.norm(direction);v=x.new_tensor(direction).reshape_as(x);exact=mv(direction);fd=[]
                for h in [1e-3,5e-4,2.5e-4,1.25e-4]:
                    numerical=((compose(x+h*v)-compose(x-h*v))/(2*h)).numpy().reshape(-1)
                    fd.append(float(np.linalg.norm(numerical-exact)/np.linalg.norm(exact)))
                assert fd[-1]<1e-5 and fd[-1]<fd[0]/4,('Composed derivative gate',fd)
                op=LinearOperator((896,896),matvec=mv,dtype=np.float64);spectra=[]
                for seed in [13,29]:
                    vals,vec=eigs(op,k=3,ncv=16,tol=1e-8,maxiter=100,v0=np.random.default_rng(seed).normal(size=896));order=np.argsort(-abs(vals));vals=vals[order];vec=vec[:,order];resids=[]
                    for k in range(3):resids.append(float(np.linalg.norm(mv(vec[:,k].real)+1j*mv(vec[:,k].imag)-vals[k]*vec[:,k])))
                    assert max(resids)<1e-6
                    spectra.append(dict(eigenvalues=[[float(z.real),float(z.imag)] for z in vals],residuals=resids,radius=float(max(abs(vals)))))
                assert abs(spectra[0]['radius']-spectra[1]['radius'])<1e-6
                row.update(closure=closure,proper_divisors=divisors,composed_fd_errors=fd,spectra=spectra,classification='numerically stable primitive cycle' if spectra[0]['radius']<1 else 'unstable recurrent cycle')
                np.save(a.out/f'cycle-{index}.npy',cycle)
            else:
                # Bounded finite-window diagnostic: twin sizes share an unperturbed base.
                base=t.tensor(states[-1],dtype=t.float64).reshape(1,1,-1);u=np.random.default_rng(19).normal(size=896);u/=np.linalg.norm(u);v=base.new_tensor(u).reshape_as(base)
                offsets=[1e-5,3e-6];twins=[base+eps*base.norm()*v for eps in offsets];growth=[];bases=[]
                for step in range(512):
                    nxt=F(base);new_twins=[];g=[]
                    for eps,tw in zip(offsets,twins):
                        delta=F(tw)-nxt;dn=delta.norm();assert float(dn)>0
                        g.append(float(t.log(dn/(eps*base.norm()))));new_twins.append(nxt+eps*nxt.norm()*delta/dn)
                    base=nxt;twins=new_twins;growth.append(g);bases.append(base.numpy().reshape(-1).copy())
                    if step%64==63:np.savez(a.out/f'growth-{index}.npz',growth=growth,states=bases)
                g=np.array(growth);assert np.isfinite(g).all()
                row.update(classification='unresolved finite-window dynamics',offsets=offsets,growth_means=g.mean(0).tolist(),growth_halves=[g[:256].mean(0).tolist(),g[256:].mean(0).tolist()],later_recurrence_errors=recurrence_errors(np.array(bases)))
            rows.append(row);record('partial.json',rows);print(json.dumps(row),flush=True)
        record('result.json',dict(success=True,rows=rows,seconds=time.monotonic()-start))
    except BaseException as e:record('failure.json',dict(error=repr(e),seconds=time.monotonic()-start));raise
    finally:record('manifest.json',{f.name:sha(f) for f in a.out.iterdir() if f.is_file() and f.name!='manifest.json'})
if __name__=='__main__':main()

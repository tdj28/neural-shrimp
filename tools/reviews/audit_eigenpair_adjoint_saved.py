"""Independent saved bordered-adjoint and scalar-functional reductions.

No producer numerical imports, inverse, solve, derivative or decomposition.
Authenticated whole-box derivative and known eigenpair-root existence are
inherited premises; all subsequent arithmetic is independently reconstructed.
"""
from audit_norm_cocycle_saved import *
import audit_norm_cocycle_saved as A
import math,sys
BASE=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype')
PINS=[('attempt-008-wider-march','8914b8547ac17ccc2c2ec28589a71c17884420d48b3e0c89827fc7530e3c8e06','c700c7b393e1565e2a8577a19336dd14b819eb649a915e1f4ddd76433fcbb795','d42d15298ec6fa70259a3e079f74e3108c941040',0),
('attempt-001-shooting-fold-target','35661bb57aa3afd53d3cec77f3429a10cff166e4387337a00e74915c5ecbf7ee','a5c578cf6f437d662654bf1a159b8a903d1104e8f7f6a5eec94abdba34a7fd3a','501f356e65d6dfb5e88d748c8c74c70f304c3f16',2),
('attempt-001-adjoint-refinement','d16ad5278f7bbbfafa7b6e890ae7e2a675ee2fef0fca53b93d6fdc824ea11c3a','919baf17f247261fa32aa17bd30e5301956f54132a98a7e6d1206f310045ffc2','85aa258fd0029be811c65b4915abf9b14bf227b8',2),
('attempt-001-adjoint-functionals','3f620248f5f8d2fbdcbd56287bab19ea85bb0db509618086342af3d5560c0b93','135afb9dda860d759747da28d37bbedf874c69e6116e2a06e13057b340c981bd','1843ddcafef651a74af609fe45be1f2d007e95b6',0)]
AUDIT_OUT=None

def restore(v):
    """Recover the exact serialized dyadic ball; validate both stored fields."""
    need(set(v)=={'mid','rad','display'} and len(v['mid'])==len(v['rad'])==2,'saved ball schema')
    need(all(type(k)is int for k in v['mid']+v['rad']),'integer dyadics')
    r,e=v['rad'];need(r>=0 and r.bit_length()<=30,'saved magnitude')
    m=arb(tuple(v['mid']))
    # Public radius constructors round up; undo only their extra guard ulp,
    # then require identical serialized endpoints before using the ball.
    b=arb(m,arb((r,e))-arb((1,e+r.bit_length()-32))) if r else m
    need(enc(b)==clean(v),'exact dyadic ball restoration');return b

def matrix(x):return arb_mat([[restore(v) for v in row] for row in x])
def vector(x):return [restore(v) for v in x]
def vs(x):return [enc(v) for v in x]
def same(a,b,label):need(a==clean(b),label)
def events(root,n):
    paths=sorted((root/'events').glob('*.json.gz'));need(len(paths)==n,'complete event count')
    need([int(p.name.split('-')[0]) for p in paths]==list(range(n)),'consecutive events')
    return [(p.name.split('-',1)[1].removesuffix('.json.gz'),read(p)) for p in paths]

def bind(row):
    name,outer,inner,commit,exitcode=row;root=BASE/name
    inv=[inventory(root,outer),inventory(root/'run',inner)]
    rt=read(root/'run/runtime.json');la=read(root/'launcher-result.json');out=read(root/'run/outcome.json')
    need(rt['commit']==la['commit']==commit and rt['mode']==la['mode']=='target','execution/mode')
    need(la['exit_code']==exitcode and la['stop_reason'] is None and not la['owned_live_pids'],'retained launch status')
    need(rt['nice']>=10 and set(rt['threads'].values())=={'1'} and rt['precision']==256,'precision/thread/nice')
    need(type(out['elapsed_seconds'])in(float,int) and math.isfinite(out['elapsed_seconds']) and 0<out['elapsed_seconds']<rt['seconds_cap'],'worker deadline')
    need(0<out['peak_rss_kib']<rt['rss_cap_kib'],'worker RSS')
    for name,item in rt['sources'].items():
        need(sha(subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT))==item['sha256'],'historical source '+name)
        need(sha((root/'run/source'/name).read_bytes())==item['sha256'],'archived source '+name)
    return root/'run',rt,out,inv

def blocks(jac,x,n,p):
    size=2*n*p+2;k=n*p;need(jac.nrows()==jac.ncols()==len(x)==size,'native derivative dimensions')
    indices=[2*n*j+i for j in range(p) for i in range(n)]
    q=[x[i+n] for i in indices];aa=arb_mat([[jac[i,j] for j in indices] for i in indices])
    b=arb_mat(k+1,k+1)
    for i in range(k):
        for j in range(k):b[i,j]=aa[j,i]
        b[i,k]=q[i]
        b[k,i]=q[i-(k-n)] if i>=k-n else arb(0)
    rates=arb_mat([[jac[i,size-2+h] for h in range(2)] for i in indices])
    c=[sum((jac[2*n*j+n+i,2*n*j+h]*q[j*n+h] for h in range(n)),arb(0)) for j in range(p) for i in range(n)]
    need(sum((v*v for v in q[:n]),arb(0))>0,'nonzero normalization premise compatible')
    return b,rates,c

def border_reduce(b,c):
    k=b.nrows();need(b.ncols()==c.nrows()==c.ncols()==k and finite(b) and all(v.is_exact() for row in c.tolist() for v in row),'fixed exact border proposal')
    d=eye(k)-c*b
    weights=[[arb(1)]*k,[sum((abs(c[i,j]).upper() for j in range(k)),arb(0)).upper() for i in range(k)]]
    need(all(v>0 for w in weights for v in w),'positive weights')
    bounds=[max((sum((abs(d[i,j]).upper()*w[j] for j in range(k)),arb(0))/w[i]).upper() for i in range(k)) for w in weights]
    index=0 if bounds[0]<=bounds[1] else 1;w=weights[index];q=bounds[index];need(q<1,'uniform invertibility')
    rhs=arb_mat([[int(i==k-1)] for i in range(k)]);y0=arb_mat([[c[i,k-1]] for i in range(k)]);res=rhs-b*y0;f=c*res
    beta=max((abs(f[i,0]).upper()/w[i]).upper() for i in range(k));tail=(beta/(1-q)).upper()
    y=[y0[i,0]+arb(0,(w[i]*tail).upper()) for i in range(k)]
    need(y[-1].contains(0),'exact-root alpha compatibility')
    return d,weights,bounds,index,rhs,y0,res,f,beta,tail,y

def refine(y,y0,f,d):
    k=len(y);need(y[-1].contains(0),'root alpha premise')
    e=arb_mat([[y[i]-y0[i,0]] for i in range(k)]);e[k-1,0]=-y0[k-1,0];initial=e;trace=[]
    for it in range(5):
        im=f+d*e;new=arb_mat([[e[i,0].intersection(im[i,0])] for i in range(k)])
        need(new[k-1,0].contains(-y0[k-1,0]),'root alpha intersection');new[k-1,0]=-y0[k-1,0]
        trace.append((e,im,new));e=new
    out=[y0[i,0]+e[i,0] for i in range(k)];out[-1]=arb(0)
    return initial,trace,out

def scalar(b,c,y,v):
    k=len(y);need(len(v)==k and y[-1].is_exact() and y[-1]==0 and v[-1].is_exact() and v[-1]==0,'root-only zero auxiliary')
    u=mid(c.transpose()*arb_mat([[a.mid()] for a in v]));r=arb_mat([[a] for a in v])-b.transpose()*u
    direct=sum((v[i]*y[i] for i in range(k)),arb(0));dual=u[k-1,0]+sum((r[i,0]*y[i] for i in range(k)),arb(0));intersection=direct.intersection(dual)
    return u,r,direct,dual,intersection

def audit():
    bound=[bind(v) for v in PINS];roots=[r[0] for r in bound];inv=[v for z in bound for v in z[3]];parent,original,refined,fun=roots
    need(bound[0][1]['unit']==8,'fixed parent unit')
    need(all(z[1]['native_calls']==z[2]['native_calls']==0 for z in bound[1:]),'saved-only coefficient producers')
    po=bound[0][2];need(po['passed'] and po['status']=='four_joined_charts' and len(po['progress']['accepted'])==4 and po['progress']['active_call'] is None,'complete inherited parent')
    last=po['progress']['last_accepted'];need(last==po['progress']['accepted'][-1] and last['right']=='151/128','fixed selected last chart')
    pp=parent/last['full_chart_file'];need(pp.resolve().is_relative_to(parent.resolve()),'safe chart path');chart=read(pp)
    need(chart['passed'] and all(chart['gates'].values()) and len(chart['coordinate_inclusions'])==470 and all(chart['coordinate_inclusions']),'inherited full root gates')
    need((chart['period'],chart['dimension'],chart['sigma'])==(18,13,1) and chart['parameter_endpoints']==['147/128','151/128'],'native same chart')
    need([v['divisor'] for v in chart['divisors']]==[1,2,3,6,9] and all(v['excluded'] for v in chart['divisors']) and chart['anchor_check']['passed'],'inherited primitive/join premise')
    ib=read(original/'input-binding.json');need(ib['execution']==PINS[0][3] and ib['outer']['manifest']['sha256']==PINS[0][1] and ib['full_identity']['sha256']==sha(pp.read_bytes()),'original parent identity')
    b,rates,second=blocks(matrix(chart['whole_jacobian']),vector(chart['whole_box']),13,18);del chart
    ev=events(original,8);need([k for k,v in ev]==['coefficient_inputs','border','preconditioner_return','defect_return','weight_bounds','solve_residual_return','solution_return','qualification_return'],'original event grammar')
    same(serial(b),ev[1][1]['matrix'],'native transposed border');same(serial(rates),ev[0][1]['rates'],'actual rate columns');same(vs(second),ev[0][1]['second_contractions'],'native Hessian contractions')
    c=matrix(ev[2][1]['matrix']);d,w,qs,index,rhs,y0,res,f,beta,tail,y=border_reduce(b,c)
    same(serial(d),ev[3][1]['matrix'],'uniform defect');same([vs(v)for v in w],ev[4][1]['weights'],'fixed two weights');same(vs(qs),ev[4][1]['contractions'],'strict contraction bounds')
    for k,a in [('rhs',rhs),('proposal',y0),('residual',res),('forcing',f)]:same(serial(a),ev[5][1][k],'linear '+k)
    sol=ev[6][1];need(sol['passed'] and sol['selected_weight']==index,'same accepted border');same(enc(beta),sol['beta'],'beta');same(enc(tail),sol['tail'],'tail');same(vs(y),sol['solution'],'Neumann enclosure')
    original_result=read(original/'result.json');need(clean(original_result)==clean(ev[7][1]) and original_result['gates']['simple_native_plus_one'],'original qualification binding')
    same(vs(y[:-1]),original_result['left_shooting'],'original left enclosure');same(enc(y[-1]),original_result['alpha'],'original auxiliary')
    original_aa=[sum((y[i]*rates[i,j] for i in range(234)),arb(0)) for j in range(2)];original_c=sum((y[i]*second[i] for i in range(234)),arb(0))
    same(vs(original_aa),original_result['actual_rate_coefficients'],'original rate reductions');same(enc(original_c),original_result['second_derivative_contraction'],'original c reduction')
    need(not original_result['passed'] and not bound[1][2]['passed'],'preserved original coefficient failure')
    put(AUDIT_OUT/'border.json',{'passed':True,'contractions':vs(qs),'selected':index,'source_parent':PINS[0][3],'conditional_simple_plus_one':True})
    # Later stages restore the saved balls exactly before computing.
    yr=vector(sol['solution']);y0r=matrix(ev[5][1]['proposal']);fr=matrix(ev[5][1]['forcing']);dr=matrix(ev[3][1]['matrix'])
    er=events(refined,12);need([k for k,v in er]==['conditional_start']+['image_return','intersection_return']*5+['refined_solution_return'],'refinement grammar')
    ini,trace,yy=refine(yr,y0r,fr,dr);st=er[0][1]
    for k,a in [('proposal',y0r),('forcing',fr),('defect',dr),('error',ini)]:same(serial(a),st[k],'initial '+k)
    same(vs(yr),st['initial_solution'],'same initial solution')
    for i,(old,im,new) in enumerate(trace):
        a,z=er[1+2*i][1],er[2+2*i][1];need(a['iteration']==z['iteration']==i,'five fixed iterations');same(serial(old),a['prior_error'],'prior error');same(serial(im),a['image'],'image');same(serial(new),z['error'],'intersection')
    same(vs(yy),er[-1][1]['solution'],'final root-only solution');rr=read(refined/'result.json');same(vs(yy[:-1]),rr['left_shooting'],'stored refined left')
    rre=matrix(ev[0][1]['rates']);sse=vector(ev[0][1]['second_contractions'])
    refined_aa=[sum((yy[i]*rre[i,j] for i in range(234)),arb(0)) for j in range(2)];refined_c=sum((yy[i]*sse[i] for i in range(234)),arb(0))
    same(vs(refined_aa),rr['actual_rate_coefficients'],'refined direct rates');same(enc(refined_c),rr['second_derivative_contraction'],'refined direct c');need(not rr['passed'] and not bound[2][2]['passed'],'preserved refinement failure')
    rb=read(refined/'bindings.json');need(rb['execution']==PINS[1][3] and rb['outer']['manifest']['sha256']==PINS[1][1] and rb['run']['manifest']['sha256']==PINS[1][2],'refinement original join')
    put(AUDIT_OUT/'refinement.json',{'passed':True,'intersections':5,'known_root_auxiliary_zero':True})
    fy=vector(rr['left_shooting'])+[arb(0)];fb=matrix(ev[1][1]['matrix']);fc=matrix(ev[2][1]['matrix']);frates=matrix(ev[0][1]['rates']);fsecond=vector(ev[0][1]['second_contractions'])
    fe=events(fun,15);need([k for k,v in fe]==['functional_start','dual_proposal_return','dual_residual_return','functional_bounds_return','functional_intersection_return']*3,'functional event grammar')
    result=read(fun/'result.json');labels=['actual_hidden_rate','actual_output_rate','second_derivative_contraction'];vsin=[[frates[i,j]for i in range(234)]+[arb(0)]for j in range(2)]+[fsecond+[arb(0)]];cases=[]
    for j,(label,v)in enumerate(zip(labels,vsin)):
        start,prop,residual,bounds,inter=[v for _,v in fe[5*j:5*j+5]];need(start['label']==label,'fixed functional label');u,r,di,du,iv=scalar(fb,fc,fy,v)
        same(vs(v),prop['b'],'same b');same(serial(u),prop['proposal'],'exact dual proposal');same(serial(r),residual['residual'],'whole dual residual');same(enc(di),bounds['direct'],'direct scalar');same(enc(du),bounds['dual'],'dual scalar');same(enc(iv),inter['intersection'],'intersection')
        same(clean(inter),{k:v for k,v in result['functionals'][j].items()if k!='label'},'result functional binding');need(iv<0 and inter['excludes_zero'],'independently negative scalar')
        case={'label':label,'intersection':enc(iv),'strictly_negative':True};put(AUDIT_OUT/f'functional-{j}.json',case);cases.append(case)
    binding=read(fun/'bindings.json');need(binding['execution']==PINS[1][3] and binding['outer']['manifest']['sha256']==PINS[1][1] and binding['refinement']['outer']['manifest']['sha256']==PINS[2][1],'functional parents')
    need(result['passed'] and all(result['gates'].values()) and bound[3][2]['native_calls']==bound[3][2]['new_inverses']==0,'completed functional scope')
    same([v['intersection']for v in cases[:2]],result['actual_rate_coefficients'],'final rate coefficients');same(cases[2]['intersection'],result['second_derivative_contraction'],'final c');same(enc(restore(result['second_derivative_contraction'])/2),result['quadratic_coefficient'],'c over two')
    return {'passed':True,'inventories':inv,'exact_parameter_interval':['147/128','151/128'],'contractions':vs(qs),'selected_weight':index,'cases':cases,'intersections':5,'scope':'Independent saved interval border/simple+1 and functional nondegeneracy arithmetic, conditional on authenticated parent normalized primitive eigenpair-root/whole-derivative proof. No new root/AD/inverse, no other-unit-circle exclusion, attraction, cusp or whole shrimp theorem; shared Arb.'}

def main(out):
    global AUDIT_OUT
    AUDIT_OUT=out;need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean frozen source');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();status={'passed':False};code=1
    put(out/'runtime-start.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'checker_sha256':sha(Path(__file__).read_bytes()),'helper_sha256':sha(Path(A.__file__).read_bytes()),'pid':os.getpid(),'ppid':os.getppid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'internal_seconds':60,'post_outcome':True})
    def stop(*_):raise TimeoutError('fixed60s saved audit')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,60)
    try:
        with ctx.workprec(256):status=audit()
        code=0
    except BaseException as e:status.update(error={'type':type(e).__name__,'message':str(e)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);status.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code)
        if status['peak_rss_kib']>=2*1024**2:status.update(passed=False,exit_code=1,error={'type':'MemoryError','message':'RSS cap'});code=1
        put(out/'result.json',status);files={str(p.relative_to(out)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}for p in out.rglob('*')if p.is_file()};need(sum(v['bytes']for v in files.values())<8*1024**2,'8MiB auditcap');put(out/'manifest.json',{'files':files})
    print(json.dumps(status));return code
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);raise SystemExit(main(p.parse_args().out))

"""Fixed retrospective reversal decomposition on four consumed LM references."""
import os,resource
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREADS:os.environ[key]='1'
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
import argparse,hashlib,json,platform,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import explore_tiny_lm_horizon as H
from neural_shrimp.tiny_lm_ordering import HORIZONS,reverse_product,decompose,reciprocal,profile_forecast
from neural_shrimp.tiny_lm_response import measurements
B=H.B;need=B.need
BASE=Path('/data2/neural-shrimp-data/explore/tiny-lm-ordering')
EXTRA=['src/neural_shrimp/tiny_lm_ordering.py','tools/explore_tiny_lm_ordering.py','tools/launch_tiny_lm_ordering.py','tests/test_tiny_lm_ordering.py','tests/test_tiny_lm_ordering_driver.py','docs/plans/tiny-lm-reversal-decomposition.md']
FILES=H.FILES+EXTRA

class Work(H.Work):
    def __init__(self,out,mode):
        super().__init__(out,mode);self.cap=dict(hvp=4 if mode=='profile'else 160);self.attempted=dict.fromkeys(self.cap,0);self.returned=dict.fromkeys(self.cap,0);self.times={k:[]for k in self.cap}


def source(w):
    identity=H.source(w)
    for n in EXTRA:
        e=B.entry(ROOT/n);data=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==e['sha256'],'exact committed caller source '+n);w.write('source/'+n,data);identity[n]=e
    return identity


def qualification(folder,reference,gradients,directions,tangent,frozen):
    old=B.load(folder/'outcome.json');eps=old['epsilon_values'];need(eps==[e*max(1.,float(np.linalg.norm(reference[0])))for e in(1e-4,5e-5)],'original exact amplitudes')
    branches={}
    for i in range(2):
        for k in range(2):
            row={}
            for sign in('plus','minus'):
                row[sign]=H.finite_array(folder/f'direction-{i}-epsilon-{k}-{sign}.npy',(17,15776))
                values=H.finite_array(folder/f'direction-{i}-epsilon-{k}-{sign}-ce.npy',(3,2));need(np.array_equal(values[:,0],[1.,4.,16.]),'fixed original CE readouts')
                row['ce_'+sign]={int(h):float(v)for h,v in values}
                need(row['ce_'+sign]=={int(h):float(v)for h,v in old['branch_training_ce'][f'{i}:{k}']['ce_'+sign].items()},'original raw CE and serialized outcome match')
            branches[i,k]=row
    readout={h:gradients[h]-.001*reference[h]for h in(1,4,16)}
    need(np.array_equal(np.asarray(list(readout.values())),H.finite_array(folder/'readout-gradients.npy',(3,15776))),'actual regularized-gradient to CE conversion')
    recomputed=measurements(reference,[dict(transported=tangent[i],frozen=frozen[i])for i in range(2)],branches,readout,eps)
    need(recomputed==old['rows'],'all original response qualifications recomputed from retained arrays')
    return {(i,h):next(row for row in recomputed if row['direction']==('gradient'if i==0 else'random')and row['horizon']==h)for i in range(2)for h in HORIZONS}


def target(w,parent):
    rows=[];reciprocity=[]
    for seed in(28161,28162):
        for arm in(8,256):
            theta,x,y,eta,binding=B.select(parent,'target',seed,arm,600);label=f'seed-{seed}-arm-{arm}';folder=H.RESPONSE/'run'/label
            reference=H.finite_array(folder/'reference.npy',(17,15776));gradients=H.finite_array(folder/'reference-gradients.npy',(17,15776));directions=H.finite_array(folder/'directions.npy',(2,15776))
            need(np.array_equal(reference[0],theta)and np.array_equal(reference[1:],reference[:-1]-eta*gradients[:-1]),'same promoted state and complete physical reference recurrence')
            need(binding==B.load(folder/'inputs.json'),'original exact tokens/rate/state binding');need(np.array_equal(directions[0],gradients[0]/np.linalg.norm(gradients[0])),'original gradient direction')
            tangent=[H.finite_array(folder/f'direction-{i}-transported.npy',(17,15776))for i in range(2)];frozen=[H.finite_array(folder/f'direction-{i}-frozen.npy',(17,15776))for i in range(2)]
            need(all(np.array_equal(tangent[i][0],directions[i])and np.array_equal(frozen[i][0],directions[i])for i in range(2)),'same unit initial vectors')
            qualified=qualification(folder,reference,gradients,directions,tangent,frozen)
            w.json(label+'/inputs.json',binding);w.json(label+'/qualification.json',{f'{i}:{h}':v for(i,h),v in qualified.items()});problem=H.LMProblem(H.TinyTransformer(66),torch.tensor(x),torch.tensor(y),ridge=.001)
            reversed_={}
            for h in HORIZONS:
                for i in range(2):
                    rowlabel=label+f'/direction-{i}-horizon-{h}';start=w.attempted['hvp']
                    def save(trace,indices):
                        w.array(rowlabel+'/reverse-trace',trace);w.json(rowlabel+'/state-order.json',dict(indices=indices,operation_first=start+1,operation_last=w.attempted['hvp']))
                    reverse=reverse_product(reference,directions[i],eta,h,lambda state,vector:w.call('hvp',problem,state,vector),save)
                    reversed_[i,h]=reverse;g=gradients[h]-.001*reference[h];result,a,b=decompose(directions[i],tangent[i][h],reverse,frozen[i][h],g)
                    w.array(rowlabel+'/odd',a);w.array(rowlabel+'/even-remainder',b);w.array(rowlabel+'/ce-gradient',g)
                    row=dict(seed=seed,arm=arm,direction=i,horizon=h,label=rowlabel,operation_first=start+1,operation_last=w.attempted['hvp'],decomposition=result,response=qualified[i,h]);w.json(rowlabel+'/outcome.json',row);rows.append(row)
                check=reciprocal(directions,np.asarray([tangent[i][h]for i in range(2)]),np.asarray([reversed_[i,h]for i in range(2)]));reciprocity.append(dict(seed=seed,arm=arm,horizon=h,**check));w.json(label+f'/reciprocity-{h}.json',check)
    need(len(rows)==16 and len(reciprocity)==8,'complete fixed sixteen directional rows')
    need(all(row['decomposition']['algebra_qualified']for row in rows)and all(r['passed']for r in reciprocity),'floating algebra/skew/reciprocal controls')
    selected=[r for r in rows if r['arm']==256 and r['direction']==0 and r['horizon']==16];need(len(selected)==2,'both original expanded gradient directions')
    summary=dict(material_threshold=.5,material_reversal_odd_in_both_expanded_gradients=all(r['response']['mechanism_qualified']and r['decomposition']['odd_to_error']>=.5 for r in selected),selected_rows=[dict(seed=r['seed'],qualified=r['response']['mechanism_qualified'],odd_to_error=r['decomposition']['odd_to_error'])for r in selected],scope='Retrospective consumed-development decomposition; even remainder can depend on order; ratios are not percentages; reversal preserves singular values. No training, prediction, chaos, geometry or population-utility claim.')
    return rows,reciprocity,summary


def main(args):
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);need(args.out==BASE/f'attempt001-{args.mode}'/'run','fixed exclusive output')
    need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    w=Work(args.out,args.mode);result=dict(passed=False,status='incomplete',mode=args.mode,protected_outcomes_evaluated=False)
    def stop(*_):raise TimeoutError('fixed ordering owned deadline')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,w.limit)
    try:
        identity=source(w);runtime=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=identity,python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,pid=os.getpid(),dtype='float64',device='cpu',threads=dict.fromkeys(THREADS,'1'),operation_caps=w.cap)
        w.json('runtime.json',runtime);parent,pin=B.PARENTS[args.mode];w.json('parent-binding.json',H.immutable(parent,pin))
        if args.mode=='profile':
            costs=[]
            for arm in(8,256):
                theta,x,y,eta,binding=B.select(parent,'profile',28160,arm,50);w.json(f'arm-{arm}-inputs.json',binding)
                problem=H.LMProblem(H.TinyTransformer(66),torch.tensor(x),torch.tensor(y),ridge=.001);direction=np.random.default_rng(28166).standard_normal(15776)
                times=[]
                for _ in range(2):w.call('hvp',problem,theta,direction);times.append(w.times['hvp'][-1])
                costs.append(dict(arm=arm,times=times))
            estimate=profile_forecast(costs);need(estimate<180,'fixed measured180s target admission');result.update(passed=True,status='profile_complete',profile_costs=costs,forecast_seconds=estimate)
        else:
            profile=BASE/'attempt001-profile';need(args.profile_outer and args.profile_worker,'both completed profile manifests');H.immutable(profile,args.profile_outer);H.immutable(profile/'run',args.profile_worker)
            prior=B.load(profile/'run/result.json');pr=B.load(profile/'run/runtime.json');la=B.load(profile/'launcher-result.json')
            need(prior['passed']and prior['status']=='profile_complete'and la['exit_code']==0 and not la['owned_live_pids']and la['worker_pid_absent'],'complete owned measured profile')
            need(pr['sources']==identity and all(pr[k]==runtime[k]for k in('python','numpy','torch','dtype','device','threads')),'same source and runtime profile')
            need(prior['operation_attempted']==prior['operation_returned']==dict(hvp=4)and profile_forecast(prior['profile_costs'])==prior['forecast_seconds']<180,'recomputed full160HVP cost')
            for pid in(pr['pid'],la['launcher_pid'],la['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'all owned profile processes absent')
            w.json('profile-binding.json',dict(root=str(profile),outer=args.profile_outer,worker=args.profile_worker,forecast_seconds=prior['forecast_seconds']));H.bind_response(w,identity)
            rows,reciprocity,summary=target(w,parent);result.update(passed=True,status='ordering_complete',rows=rows,reciprocity=reciprocity,summary=summary)
        need(w.attempted==w.returned==w.cap,'exact fixed operation counts');w.check()
    except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,operation_caps=w.cap,operation_attempted=w.attempted,operation_returned=w.returned)
        w.json('events.json',w.events,True);w.json('result.json',result,True);w.json('manifest.json',dict(files={str(p.relative_to(w.out)):B.entry(p)for p in w.out.rglob('*')if p.is_file()}),True)
    print(json.dumps({k:v for k,v in result.items()if k not in('rows','reciprocity')}));return 0 if result['passed']else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile-outer');p.add_argument('--profile-worker');raise SystemExit(main(p.parse_args()))

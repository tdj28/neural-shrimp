"""Independent saved-state S1R audit; no correction iterations or trajectories."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
SOURCE='db10bd1e1b25b0e24b1f0f8f97cb84035742c776'
RECEIPT='4e50f004ad6f5db80be0dea01fdf404eec71bd1b'
RAW=Path('/data2/neural-shrimp-data/explore/s1r-branch-association/attempt-001')
MANIFEST='45c35991b7ec56f7d9888b6dcf28434396ff94f65de7bf90ba9822408c5f3c8b'
REPRESENTATIVES=('node-0000','node-0004','node-0005','node-0009','node-0013','node-0017')


def require(value,message):
    if not value:raise ValueError(message)


def digest(data):return hashlib.sha256(data).hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def entry(path):
    data=Path(path).read_bytes();return dict(bytes=len(data),sha256=digest(data))


def inventory(folder,manifest_sha):
    require(digest((folder/'manifest.json').read_bytes())==manifest_sha,'manifest hash')
    m=read_json(folder/'manifest.json')
    require(set(p.name for p in folder.iterdir())==set(m['files'])|{'manifest.json'},'manifest inventory')
    for name,e in m['files'].items():
        p=folder/name
        require(Path(name).name==name and not p.is_symlink() and entry(p)==e,'artifact hash '+name)
    return m


def record(folder,name):
    obj=read_json(folder/name);arrays={}
    if 'arrays_file' in obj:
        p=folder/obj['arrays_file'];require(Path(obj['arrays_file']).name==obj['arrays_file'],'array path')
        require(digest(p.read_bytes())==obj['arrays_sha256'],'array archive hash')
        with np.load(p,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    used=set()
    def decode(value):
        if isinstance(value,dict) and 'array' in value:
            a=arrays[value['array']];used.add(value['array'])
            require(a.dtype.str==value['dtype'] and list(a.shape)==value['shape'] and value['order']=='C','array metadata')
            require(digest(np.ascontiguousarray(a).tobytes())==value['sha256'],'array bytes')
            return a
        if isinstance(value,dict):return {k:decode(v) for k,v in value.items()}
        if isinstance(value,list):return [decode(v) for v in value]
        return value
    value=decode(obj['payload']);require(used==set(arrays),'unreferenced array');return obj['kind'],value


def loss_gradient_hessian(theta,x,y,model):
    """Direct sample prediction Jacobian plus residual curvature, no HVP reuse."""
    d,w=model['d'],model['width'];theta=np.asarray(theta)
    require(model['activation']=='tanh' and not model['bias'],'supported independent model')
    require(theta.shape==(d*w+w,) and np.isfinite(theta).all(),'independent state')
    s_in,s_out=model['in_scale'],model['out_scale'];lam=model['ridge'];scale=2*model['loss_scale']/len(x)
    W=theta[:d*w].reshape(d,w);v=theta[d*w:]
    h=np.tanh(s_in*x@W);s=1-h*h;s2=-2*h*s;error=s_out*h@v-y
    pred_j=np.column_stack(((s_in*x[:,:,None]*(s_out*s*v)[:,None,:]).reshape(len(x),d*w),s_out*h))
    gradient=scale*(pred_j.T@error)+lam*theta
    hessian=scale*(pred_j.T@pred_j)+lam*np.eye(len(theta))
    for unit in range(w):
        ids=np.arange(d)*w+unit;out=d*w+unit
        hessian[np.ix_(ids,ids)]+=scale*s_in*s_in*s_out*v[unit]*(x.T@((error*s2[:,unit])[:,None]*x))
        cross=scale*s_in*s_out*(x.T@(error*s[:,unit]))
        hessian[ids,out]+=cross;hessian[out,ids]+=cross
    loss=model['loss_scale']*np.mean(error*error)+lam*(theta@theta)/2
    return loss,gradient,hessian


def pair_match(parent,child,replay=False):
    scale=max(1.,float(np.max(np.abs(parent))),float(np.max(np.abs(child))))
    costs=[float(np.max(np.abs(parent-child)))/scale,float(np.max(np.abs(parent-child[::-1])))/scale]
    sep=float(np.max(np.abs(parent[0]-parent[1])))/max(1.,float(np.max(np.abs(parent))))
    phase=int(costs[1]<costs[0]);cap=min(.01,.25*sep,1e-7 if replay else float('inf'))
    gap=costs[1-phase]-costs[phase]
    return dict(costs=costs,phase=phase,separation=sep,displacement_cap=cap,cost_gap=gap,
                passed=costs[phase]<=cap and gap>=.5*sep and sep>=1e-6,replay=replay)


def compare_match(actual,saved):
    require(actual['passed'] and actual['passed']==saved['passed'] and actual['phase']==saved['phase'],'phase decision')
    for key in ('costs','separation','displacement_cap','cost_gap'):
        require(np.allclose(actual[key],saved[key],atol=1e-14,rtol=1e-13),'phase diagnostic '+key)


def audit():
    started=time.monotonic();m=inventory(RAW,MANIFEST);runtime=read_json(RAW/'runtime-start.json');result=read_json(RAW/'result.json')
    require(runtime['source_commit']==SOURCE,'execution source')
    for name,e in runtime['sources'].items():
        data=subprocess.check_output(['git','show',SOURCE+':'+name],cwd=ROOT)
        require(dict(bytes=len(data),sha256=digest(data))==e and entry(ROOT/name)==e,'source binding '+name)
    for e in runtime['inputs'].values():require(entry(e['path'])=={k:e[k] for k in ('bytes','sha256')},'input binding')
    receipt=json.loads(subprocess.check_output(['git','show',RECEIPT+':docs/s1r-branch-association-receipt.json'],cwd=ROOT))
    require(receipt['execution_source_commit']==SOURCE and receipt['source_review_commit']=='cdf2ad51034db515fab99846fbd88e61a4528a46','receipt source/review')
    for name,e in receipt['raw_files'].items():require(entry(RAW/name)==e,'receipt raw file '+name)
    require(receipt['raw_file_count']==len(m['files'])+1 and receipt['raw_total_bytes']==sum(p.stat().st_size for p in RAW.iterdir()),'receipt raw inventory')
    require(receipt['input_inventory']==runtime['inputs'] and receipt['source_inventory']==runtime['sources'],'receipt input/source inventory')
    require(receipt['runtime']['phase_corrector_calls']==result['phase_corrector_calls'],'receipt operation count')
    require(receipt['raw_files']['manifest.json']==entry(RAW/'manifest.json'),'receipt manifest')
    for key in ('checker','launcher','preparation_failure'):
        e=receipt[key];require(entry(e['path'])=={k:e[k] for k in ('bytes','sha256')},'auxiliary receipt '+key)
    import neural_shrimp.period2_branch as producer_core
    def forbidden(*args,**kwargs):raise AssertionError('audit must not invoke a Newton corrector')
    producer_core.correct_phase=forbidden
    import check_s1r_branch
    replay=check_s1r_branch.check(RAW)
    events=[json.loads(line) for line in (RAW/'journal.jsonl').read_text().splitlines()]
    nodes={};edges={};pending={};inputs=None;corrector_ends=[]
    for event in events:
        kind,p=record(RAW,event['file'])
        if kind=='node':nodes[p['id']]=p
        if kind=='edge':edges[p['id']]=p
        if kind=='node_pending':pending[p['id']]=p
        if kind=='inputs':inputs=p
        if kind=='corrector_end':corrector_ends.append(p)
    require(len(nodes)==21 and len(edges)==4 and len(corrector_ends)==42,'fixed attempt counts')
    require(all(n['accepted'] for n in nodes.values()) and all(e['status']=='accepted' for e in edges.values()),'unexpected rejected node/edge')
    require(all(c['converged'] for c in corrector_ends),'incomplete phase correction')
    phase_bits=[]
    def seeds(node,expected):require(np.array_equal(np.stack(pending[node]['seeds']),expected),'seed/predictor binding '+node)
    for link in result['hidden_links']:
        pair=nodes[link['parent']]['pair'][::(-1 if link['parent_phase'] else 1)]
        actual=pair_match(pair,nodes[link['child']]['pair']);compare_match(actual,link['match']);seeds(link['child'],pair);phase_bits.append(actual['phase'])
    previous=None;last_direction=None
    for segment in result['segments']:
        if segment['direction']!=last_direction:previous=None;last_direction=segment['direction']
        require(len(segment['path'])==1,'unexpected retry/subdivision')
        e=edges[segment['path'][0]];parent=nodes[e['parent']];pair=parent['pair'][::(-1 if e['parent_phase'] else 1)]
        direct,half,end,reverse=[nodes[e[k]] for k in ('direct','half','endpoint','reverse')]
        predictor=pair if previous is None else pair+(direct['params'][1]-parent['params'][1])/(parent['params'][1]-previous['params'][1])*(pair-previous['pair'])
        seeds(direct['id'],predictor);seeds(half['id'],pair)
        tests=[('parent_match',pair,direct['pair'],False),('half_parent_match',pair,half['pair'],False),
               ('half_child_match',half['pair'],end['pair'],False),('direct_half_match',direct['pair'],end['pair'],True),('reverse_match',pair,reverse['pair'],True)]
        for key,a,b,replay_flag in tests:
            actual=pair_match(a,b,replay_flag);compare_match(actual,e[key]);phase_bits.append(actual['phase'])
        seeds(end['id'],half['pair'][::(-1 if e['half_parent_match']['phase'] else 1)])
        seeds(reverse['id'],direct['pair'][::(-1 if e['parent_match']['phase'] else 1)])
        previous=dict(params=parent['params'],pair=pair)
    model=inputs['model'];require(model==dict(activation='tanh',bias=False,d=8,width=8,in_scale=.5,out_scale=.125,loss_scale=1.,ridge=.001),'independent model identity')
    x,y=inputs['x'],inputs['y'];representatives=[]
    for name in REPRESENTATIVES:
        n=nodes[name];pair=n['pair'];rates=np.r_[np.full(64,n['rates'][0]),np.full(8,n['rates'][1])]
        def f(t):return t-rates*loss_gradient_hessian(t,x,y,model)[1]
        jac=[np.eye(72)-rates[:,None]*loss_gradient_hessian(t,x,y,model)[2] for t in pair]
        images=np.stack([f(t) for t in pair]);second=np.stack([f(t) for t in images])
        closure=float(np.max(np.abs(second-pair)/(1e-9+1e-8*np.maximum(np.abs(second),np.abs(pair)))))
        transition=float(np.max(np.abs(images-pair[::-1])/(1e-9+1e-8*np.maximum(np.abs(images),np.abs(pair[::-1])))))
        separation=float(np.max(np.abs(pair[0]-pair[1])))/max(1.,float(np.max(np.abs(pair))))
        require(max(closure,transition)<=1 and separation>=1e-6,'independent full-state gate')
        phases=[]
        for i,matrix in enumerate((jac[1]@jac[0],jac[0]@jac[1])):
            saved=n['spectra']['phases'][i];matrix_error=float(np.max(np.abs(matrix-saved['matrix'])))
            require(matrix_error<=2e-11,'independent monodromy mismatch')
            values=np.linalg.eigvals(matrix);cost=np.abs(values[:,None]-saved['values'][None,:]);rows,cols=linear_sum_assignment(cost)
            spectrum_error=float(np.max(cost[rows,cols]));require(spectrum_error<=1e-8,'independent spectrum mismatch')
            radius=float(np.max(np.abs(values)));stability='attracting' if radius<1-1e-7 else 'unstable' if radius>1+1e-7 else 'neutral_or_unresolved'
            require(stability==saved['stability'],'independent stability label')
            norm=float(np.linalg.norm(matrix,2));sv=saved['values'];rv=saved['right'];lv=saved['left']
            right=float(np.max(np.linalg.norm(matrix@rv-rv*sv,axis=0)/((norm+np.abs(sv))*np.linalg.norm(rv,axis=0))))
            left=float(np.max(np.linalg.norm(matrix.T.conj()@lv-lv*sv.conj(),axis=0)/((norm+np.abs(sv))*np.linalg.norm(lv,axis=0))))
            require(max(right,left)<=1e-10,'independent eigenpair residual')
            phases.append(dict(radius=radius,stability=stability,matrix_max_abs_error=matrix_error,spectrum_match_error=spectrum_error,right_residual=right,left_residual=left))
        representatives.append(dict(node=name,params=n['params'].tolist(),closure=closure,transition=transition,separation=separation,phases=phases))
    return dict(status='passed',evidence='independent numerical saved-state and branch-witness consistency; no uniqueness or historical association',source_commit=SOURCE,source_review_commit=receipt['source_review_commit'],producer_receipt_commit=RECEIPT,raw_manifest_sha256=MANIFEST,raw_file_count=len(m['files'])+1,raw_bytes=sum(p.stat().st_size for p in RAW.iterdir()),all_raw_read_only=all(not p.stat().st_mode&0o222 for p in RAW.iterdir()),producer_saved_checker=replay,qualified_pairs=len(nodes),phase_corrector_records=len(corrector_ends),accepted_output_edges=len(edges),phase_bits=phase_bits,predictor_seed_checks='all four hidden links and all direct/half/reverse output witnesses',representatives=representatives,historical_family_association=result['historical_family_association'],elapsed_seconds=time.monotonic()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    for name in ('tools/audit_s1r_branch_outcome.py','tests/test_s1r_branch_outcome_review.py','docs/reviews/s1r-branch-outcome-card.md'):
        require((ROOT/name).read_bytes()==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'uncommitted audit source')
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(status='started',audit_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),source_sha256=digest(Path(__file__).read_bytes()),cpu_threads=1,gpu=False,paid=False)
    def stopped(signum,frame):raise TimeoutError('bounded audit time limit')
    signal.signal(signal.SIGALRM,stopped);signal.signal(signal.SIGTERM,stopped);signal.alarm(45)
    try:report.update(audit())
    except BaseException as e:report.update(status='failed',error=type(e).__name__+': '+str(e));raise
    finally:
        signal.alarm(0)
        with (args.out/'audit.json').open('x') as handle:json.dump(report,handle,indent=2,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    print(json.dumps(dict(status=report['status'],elapsed_seconds=report['elapsed_seconds'],representatives=len(report['representatives'])),indent=2))

if __name__=='__main__':main()

"""Explicit saved-state branch pilot; never replaces a seed with a fixed-start transient.

Prepared caller only. Execution requires reviewed source/input/resource card.
"""
import argparse,hashlib,json,sys,time,os
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate(c):
    if c['period']!=5 or c['dtype']!='float64' or c['log_temperature']!=-.1:
        raise ValueError('unreviewed family')
    if c['free'] not in ('alpha','alpha2') or c['direction'] not in (-1,1):
        raise ValueError('invalid continuation direction')
    if type(c['steps']) is not int or not 1<=c['steps']<=24:
        raise ValueError('bounded24point pilot required')
    if type(c['seed_index']) is not int or c['seed_index']<0:
        raise ValueError('explicit seed index required')
    if c['tol']!=1e-10 or c['alpha_scale']!=10 or c['chunk']!=8:
        raise ValueError('unreviewed numerical settings')
    if not 0<c['ds_min']<=c['ds']<=c['ds_max']<=.002:
        raise ValueError('unreviewed step bounds')
    if len(c['bounds'])!=2 or not c['bounds'][0]<c['bounds'][1]:
        raise ValueError('finite window required')
    if not c['bounds'][0]<=c[c['free']]<=c['bounds'][1]:
        raise ValueError('seed outside window')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();c=json.loads(Path(a.card).read_text());validate(c)
    for e in c['inputs']:
        if sha(e['path'])!=e['sha256']:raise ValueError('input hash mismatch')
    if sha(c['seed_path'])!=c['seed_sha256']:raise ValueError('seed hash mismatch')
    states=np.load(c['seed_path'],allow_pickle=False)
    if states.ndim!=2 or states.shape[1]!=896 or states.dtype!=np.float64 or not np.isfinite(states).all() or c['seed_index']>=len(states):
        raise ValueError('invalid seed array')
    os.environ['NEURAL_SHRIMP_DATA']=c['data_root'];os.environ['HF_HUB_OFFLINE']='1'
    root=Path(c['repository']);sys.path.insert(0,str(root/'tools'))
    import explore_llm_soft_loop as esl
    import explore_llm_phase_map as elpm
    import bifurcation_curves as bc
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from branch_trace import trace_branch
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def write(name,value):(out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    start=time.monotonic()
    loop=esl.SoftLoop('qwen0.5b',4,c['threads'],'open','Hello!',1,'float64',c['device'])
    loop.sm.add_steer_layer(16)
    v=elpm.load_vectors(c['vectors'])['gratitude'];w=elpm.load_vectors(c['vectors2'])['hostility']
    loop.extra=(16,np.array([c['alpha2']*w]))
    cell=esl.SoftOrbit(loop,c['alpha']*v,10.0**c['log_temperature'])
    X=loop.torch.as_tensor(states[c['seed_index']].reshape(1,1,896),dtype=loop.state_dtype,device=loop.device)
    returns=cell.return_distances(X,5)
    write('seed-check.json',{'return_distances':returns,'seed_index':c['seed_index'],'seed_sha256':c['seed_sha256']})
    if not returns[4]<=1e-10 or not returns[0]>1e-7:
        raise ValueError('seed residual or primitive-period gate failed')
    fam=esl.soft_family(loop,v,5,10,c['chunk'],reference=loop,exact_jvp=True,second=(16,w),log_temperature=-.1)
    def on_point(rec):
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
    def on_attempt(rec):
        # Infinity is an explicit failed-residual marker, never a JSON NaN/Infinity token.
        if not np.isfinite(rec['relative_residual']):rec['relative_residual']='nonfinite'
        with (out/'attempts.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
    def on_state(index,state):np.save(out/f'accepted-state-{index:03d}.npy',state)
    def on_matrix(index,matrix):np.save(out/f'monodromy-{index:03d}.npy',matrix)
    records,reason,ys=trace_branch(fam,fam.to_np(X),[10*c['alpha'],10*c['alpha2']],free=0 if c['free']=='alpha' else 1,direction=c['direction'],steps=c['steps'],ds=c['ds'],ds_min=c['ds_min'],ds_max=c['ds_max'],tol=1e-10,max_newton=10,beyond=.005,bounds=tuple(10*b for b in c['bounds']),on_point=on_point,k=896,refine_k=4,on_attempt=on_attempt,on_state=on_state,on_matrix=on_matrix)
    for row in records:
        row['lambda']/=10
        if 'turn_lambda_estimate' in row:row['turn_lambda_estimate']/=10
    np.save(out/'states.npy',np.array(ys).reshape((len(ys),1,896)))
    # Explicit producer-compatible metadata for the existing curve reader.
    args=dict(model='qwen0.5b',layer=4,threads=c['threads'],dtype='float64',reference_dtype='none',device=c['device'],prompt='open',opener='Hello!',window=1,vectors=c['vectors'],concept='gratitude',log_temperature=-.1,alpha=c['alpha'],alpha2=c['alpha2'],block2=16,vectors2=c['vectors2'],concept2='hostility')
    write('branch.json',{'args':args,'free':c['free'],'period':5,'seed_origin':{'path':c['seed_path'],'sha256':c['seed_sha256'],'index':c['seed_index']},'seed_return_distances':returns,'points':records,'crossings':bc.classify_crossings(records),'stop':reason,'seconds':time.monotonic()-start,'crossroad_claim':False})
    for e in c['inputs']:
        if sha(e['path'])!=e['sha256']:raise ValueError('postrun input mismatch')

if __name__=='__main__':main()

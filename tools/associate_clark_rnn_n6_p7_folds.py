#!/usr/bin/env python3
"""Fixed two-sign quadratic proposals for three saved native p7 fold roots."""
import argparse,importlib.util,json,os,resource,signal,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p7base',ROOT/'tools/explore_clark_rnn_n6_p7_boundaries.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B);S=B.S
RAW=B.R.RAW/'n6-p7-boundaries-001'
PIN='a24758997c8ca85d4df17e31784c06978949ea9ee9fb0434041a2a396edf8178'
TAGS=['g16-dir-1-fold','g16-dir1-fold','g25-dir1-fold']
EXTRA=B.EXTRA+['tools/associate_clark_rnn_n6_p7_folds.py','tools/launch_clark_rnn_n6_p7_fold_association.py','tests/test_clark_rnn_n6_p7_fold_association.py','docs/plans/clark-rnn-n6-p7-fold-association.md']

def predictors(theta,q,a,c,change,scale):
    vals=np.r_[theta,q,a,c,change,scale]
    if not np.all(np.isfinite(vals)) or scale<1 or abs(a)<=1e-5 or abs(c)<=1e-5:raise ValueError('finite nondegenerate predictor inputs')
    squared=-2*a*change/c
    if not np.isfinite(squared) or squared<=0:raise ValueError('no real quadratic branch proposal')
    amplitude=float(np.sqrt(squared))
    if np.max(np.abs(amplitude*q))>.1*scale:raise ValueError('predictor step guard')
    return amplitude,[theta-amplitude*q,theta+amplitude*q]

def associated(point,parent):
    if point is None:return {'passed':False,'error':None,'native_match':None}
    scale=max(1,float(np.max(np.abs(parent))))
    error=float(np.max(np.abs(point['cycle']-parent))/scale)
    match=B.phase_match(parent,point['cycle'])
    return {'passed':bool(np.isfinite(error) and error<=1e-7 and match['passed']),'error':error,'native_match':match}

def original_identity(meta,identity):
    for match in (meta.get('native_identity',{}),identity):
        if match.get('passed') is not True or match.get('best_sign')!=1 or match.get('best_shift')!=0 or not np.isfinite(match.get('native_error',np.nan)) or match['native_error']>=.05:
            raise ValueError('original native phase/sign gate must already pass')
    keys=('passed','best_error','best_sign','best_shift','native_error')
    if any(meta['native_identity'][k]!=identity[k] for k in keys):raise ValueError('original identity record join')

class Worker(B.Worker):
    def check(self):
        if self.interrupted:raise InterruptedError('SIGTERM')
        if time.monotonic()-self.start>20:raise TimeoutError('20s internal')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('worker448MiB')
        if self.calls>=120:raise RuntimeError('120 native/augmented return cap')

def verify(root,pin):
    if S.sha(root/'manifest.json')!=pin:raise ValueError('fixed manifest pin')
    inv=json.loads((root/'manifest.json').read_text())['files']
    if set(inv)!={str(p.relative_to(root))for p in root.rglob('*')if p.is_file() and p!=root/'manifest.json'}:raise ValueError('complete exact inventory')
    for name,r in inv.items():
        p=root/name
        if Path(name).is_absolute()or'..'in Path(name).parts or p.is_symlink()or p.stat().st_size!=r['bytes']or S.sha(p)!=r['sha256']:raise ValueError('input file hash')

def decode(path):
    meta=json.loads(path.read_text());f=path.parent/meta['array_file']
    if S.sha(f)!=meta['array_sha256']:raise ValueError('record array hash')
    with np.load(f,allow_pickle=False)as data:arrays={k:data[k].copy()for k in data.files}
    return meta,arrays

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p)for p in EXTRA})
    if os.getpriority(os.PRIO_PROCESS,0)<10 or any(os.environ.get(k)!='1'for k in B.R.M.THREADS):raise ValueError('CPU1 nice10')
    args.out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT=args.out;S.RAW_CAP=16*1024**2;w=Worker(args.out);signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True))
    booked=[0]
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists()else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite raw cap')
        booked[0]=new
    S.reserve=reserve
    S.save(args.out/'runtime-start.json',{'source':identity,'pid':os.getpid(),'started_unix':time.time(),'nice':os.getpriority(os.PRIO_PROCESS,0),'python':sys.version,'numpy':np.__version__,'threads':{k:os.environ.get(k)for k in B.R.M.THREADS},'internal_seconds':20,'call_cap':120,'science_cap':S.RAW_CAP})
    result={'status':'incomplete','folds':[]}
    try:
        if int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024<1024**3:raise MemoryError('less than1GiB headroom')
        j=np.eye(6)*.2+np.ones((6,6))*.1;x=np.arange(1,7)/10;z=np.r_[x,.5,[1.,0,0,0,0,0]];dur=[]
        for _ in range(10):dur.append(w.it(x,j,2.,.5,'synthetic_profile')[4]);dur.append(w.aug(z,j,2.,1,'synthetic_profile')[3])
        projection=120*max(dur)+5;S.save(args.out/'profile.json',{'durations':dur,'projection':projection,'gate':projection<=20,'target_inputs_unopened_before_profile':True})
        if not np.isfinite(projection)or projection>20:raise RuntimeError('resource profile failed')
        verify(RAW,PIN);ref=B.R.RAW/'n6-p7-refinement-001';verify(ref,B.PIN)
        with np.load(ref/'run/inputs.npz',allow_pickle=False)as f:j=f['j'].copy()
        records={}
        for p in sorted((RAW/'run').glob('*-boundary_result.json')):
            meta=json.loads(p.read_text())
            if meta['tag']in TAGS:records[meta['tag']]=(p,*decode(p))
        if set(records)!=set(TAGS):raise ValueError('exact three saved fold records')
        old=json.loads((RAW/'run/result.json').read_text());directions={d['boundary']['tag']:d for s in old['slices']for d in s['directions']if d.get('boundary')}
        identity_records={}
        for p in (RAW/'run').glob('*-boundary_identity.json'):
            meta=json.loads(p.read_text())
            if meta['tag']in TAGS:identity_records[meta['tag']]=(p,*decode(p))
        S.save(args.out/'input-binding.json',{'parent_outer_manifest':PIN,'refinement_outer_manifest':B.PIN,'refinement_inputs_sha256':S.sha(ref/'run/inputs.npz'),'fixed_tags':TAGS,'definition':'two signed quadratic fold proposals at each unchanged last_closed_delta; no new fold root localization'})
        for tag in TAGS:
            p,meta,data=records[tag];ip,im,ia=identity_records[tag]
            if meta['kind']!='boundary_result' or meta['status']!='boundary_identity_unresolved' or meta['reverse_status']!='step_guard':raise ValueError('fixed failed association provenance')
            original_identity(meta,im)
            g=meta['gain'];delta=float(meta['delta']);target=float(directions[tag]['last_closed_delta']);parent=ia['parent_cycle'];theta=data['theta'];q=data['right_vector']
            if parent.shape!=(7,6)or not np.array_equal(ia['child_cycle'],data['cycle'])or abs(target-delta)>.0011:raise ValueError('native input identity')
            w.check();w.calls+=1
            arrays,qualification=B.qualify_boundary(np.r_[theta,delta,q],j,g,7,1,lambda kind,value:w.emit(kind,dict(value,tag=tag)))
            report={'tag':tag,'gain':g,'fold_delta':delta,'parent_delta':target,'qualification':qualification,'attempts':[]};result['folds'].append(report)
            if qualification['status']!='numerical_boundary_candidate':report['status']='saved_fold_requalification_failed';continue
            if not np.all(np.isfinite(arrays['monodromy']))or not np.all(np.isfinite(arrays['eigenvectors'])):raise ValueError('nonfinite boundary eigensystem')
            parent_point,parent_status=w.correct(parent[0],j,g,target,tag+'-parent-replay');parent_check=associated(parent_point,parent)
            report.update(parent_status=parent_status,parent_check=parent_check)
            if not parent_check['passed']:report['status']='parent_replay_failed';continue
            scale=max(1,float(np.max(np.abs(parent))))
            try:amplitude,proposals=predictors(theta,q,qualification['fold_a_delta'],qualification['fold_c'],target-delta,scale)
            except ValueError as exc:report.update(status='predictor_rejected',error=str(exc));continue
            for sign,proposal in zip([-1,1],proposals):
                w.emit('quadratic_predictor',{'tag':tag,'sign':sign,'gain':g,'fold_delta':delta,'target_delta':target,'theta_fold':theta,'right_vector':q,'amplitude':amplitude,'proposal':proposal,'parent_cycle':parent,'fold_a':qualification['fold_a_delta'],'fold_c':qualification['fold_c']})
                point,status=w.correct(proposal,j,g,target,tag+f'-predictor{sign}');check=associated(point,parent)
                report['attempts'].append({'sign':sign,'corrector_status':status,'association':check,'radius':None if point is None else point['radius'],'closure':None if point is None else point['closure']})
                w.emit('association_return',{'tag':tag,'sign':sign,'status':status,'association':check,'cycle':np.full((7,6),np.nan)if point is None else point['cycle']})
            report['status']='numerically_associated_fold'if any(a['association']['passed']for a in report['attempts'])else'fold_association_unresolved'
            S.save(args.out/(tag+'-result.json'),report)
        w.check();result['status']='completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,calls=w.calls,scope='numerical reverse association via disclosed two-sign quadratic fold proposals only; no new branch trace, interval or whole shrimp claim')
        S.save(args.out/'result.json',result,final=True);S.manifest(args.out)
    return 0 if result['status']=='completed'else 2
if __name__=='__main__':sys.exit(main())

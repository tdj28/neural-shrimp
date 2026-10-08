"""Independent saved-Q/array audit only: no QR, model, derivative or trajectory."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREADS:os.environ[key]='1'
import argparse,hashlib,json,math,platform,signal,subprocess,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/data2/neural-shrimp-data/explore/cifar/input-span')
REFERENCE=Path('/data2/neural-shrimp-data/explore/cifar/device-benchmark/reference-001')
LINEAR=Path('/data2/neural-shrimp-data/explore/cifar/linear-diagnostic/attempt-001')
ORIGINAL_PINS=('414e8f2b1491a7f28cd1e1522376f7b008d7d3ac57bd29f69f0d5f4b953add38','d47983793f58b033389fca5fc4aedc9ad78798b38cef06b2f3a4cf0f731534a3')
REVIEWED_SOURCE='17453ecc08b1d0b08c159f506b9447b1b6061968'
SOURCE_FILES=('src/neural_shrimp/cifar_input_span.py','tools/diagnose_cifar_input_span.py','tests/test_cifar_input_span.py','src/neural_shrimp/cifar.py','docs/plans/cifar-input-span-diagnostic.md','docs/plans/cifar-input-span-execution.md')
AUDIT_FILES=('tools/reviews/audit_cifar_input_span_outcome.py','tools/reviews/launch_cifar_input_span_audit.py','tests/reviews/test_cifar_span_outcome_audit.py','docs/reviews/cifar-input-span-outcome-audit-card.md','docs/reviews/cifar-input-span-outcome-audit-pins.json')
PIN_KEYS={'profile_manifest_sha256','target_manifest_sha256','profile_launcher_manifest_sha256','target_launcher_manifest_sha256','profile_source_commit','target_source_commit','source_review_commit','wrapper_sha256'}
LABELS=('original','trial-0','trial-1','trial-2','trial-3')
VECTOR_NAMES=tuple(label+'-'+kind for label in LABELS for kind in ('state','residual'))+('p','ap','e')
DATA_CAP=3584*1024

def require(v,m):
    if not v:raise ValueError(m)
def canonical(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def digest(b):return hashlib.sha256(b).hexdigest()
def entry(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return dict(bytes=path.stat().st_size,sha256=h.hexdigest())
def read(p):return json.loads(p.read_bytes())
def inventory(folder,pin,expected=None):
    require(folder.is_dir() and not folder.is_symlink(),'real inventory folder')
    require((folder/'manifest.json').is_file() and not (folder/'manifest.json').is_symlink(),'real manifest')
    require(entry(folder/'manifest.json')['sha256']==pin,'pinned manifest '+str(folder))
    rows=read(folder/'manifest.json');require(isinstance(rows,dict),'flat inventory')
    require(expected is None or set(rows)==set(expected),'expected complete inventory')
    require(set(p.name for p in folder.iterdir())==set(rows)|{'manifest.json'},'exact directory contents')
    for name,e in rows.items():
        p=folder/name
        require(Path(name).name==name and name not in ('.','..') and not p.is_symlink() and p.is_file(),'safe inventory member')
        require(entry(p)==e,'bound inventory member '+name)
    return rows

def finite(a,shape=None):
    a=np.asarray(a);require(a.dtype==np.float64 and np.isfinite(a).all() and (shape is None or a.shape==shape),'finite float64 shape')
    return a

def nrm(a):
    a=finite(a)
    with np.errstate(over='raise',invalid='raise'):v=float(np.sqrt(np.sum(a*a,dtype=np.float64)))
    require(math.isfinite(v),'finite norm');return v

def close(a,b):
    require(type(a) in (float,int) and type(b) in (float,int) and math.isfinite(a) and math.isfinite(b),'finite scalar comparison')
    require(math.isclose(a,b,rel_tol=5e-12,abs_tol=2e-14),'saved scalar mismatch')

def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def info(a):return dict(shape=list(a.shape),dtype=a.dtype.str,order='C',sha256=digest(np.ascontiguousarray(a).tobytes()))
def closeness(a,b):
    difference=nrm(finite(a)-finite(b,a.shape));limit=2e-12*max(1.,nrm(a),nrm(b))
    require(difference<=limit,'saved-array reconstruction mismatch');return difference

def gate(actual,predicted,scale,saved):
    error=nrm(actual-predicted);limit=1e-12+1e-8*scale;passed=error<=limit
    close(error,saved['error']);close(limit,saved['limit']);close(scale,saved['scale'])
    require(saved['passed'] is passed,'saved gate decision');return passed

def basis(x,q,r,saved):
    n,d=x.shape;finite(q,(d,n));finite(r,(n,n))
    gram=q.T@q;pq=q@gram
    values=dict(orthogonality_frobenius=nrm(gram-np.eye(n)),
      representation_relative=nrm(x-(x@q)@q.T)/max(1.,nrm(x)),
      implicit_idempotence_on_q=nrm(q@(q.T@pq)-pq)/max(1.,nrm(pq)),
      qr_reconstruction_relative=nrm(x.T-q@r)/max(1.,nrm(x)))
    values['x_i_minus_p_relative']=values['representation_relative']
    for k,v in values.items():close(v,saved[k])
    require(saved['retained_columns']==n and saved['rank_truncation'] is False,'all thin columns retained')
    passed=(values['orthogonality_frobenius']<=1e-10 and values['representation_relative']<=1e-12 and
      values['implicit_idempotence_on_q']<=1e-10 and values['qr_reconstruction_relative']<=1e-12)
    require(saved['passed'] is passed,'basis gate decision');return values,passed

def decompose(v,active,perp,x,q,width,saved):
    d=x.shape[1];cut=d*width;w=v[:cut].reshape(d,width);nw=perp[:cut].reshape(d,width)
    reconstructed=v.copy();reconstructed[:cut]=(q@(q.T@w)).reshape(-1)
    difference=closeness(active,reconstructed);closeness(perp,v-reconstructed)
    require(same(active[cut:],v[cut:]) and same(perp[cut:],np.zeros_like(v[cut:])),'non-first-layer packing')
    pp=active.copy();pp[:cut]=(q@(q.T@active[:cut].reshape(d,width))).reshape(-1)
    norm,an,pn=nrm(v),nrm(active),nrm(perp)
    checks=saved['checks'];passed=True
    for key,a,b,scale in [('reconstruction',active+perp,v,norm),('implicit_projector_idempotence',pp,active,norm),
       ('complement_orthogonality',q.T@nw,np.zeros((q.shape[1],width)),norm),
       ('x_complement_preactivation',x@nw,np.zeros((len(x),width)),nrm(x@w))]:
        passed=gate(a,b,scale,checks[key]) and passed
    pythagoras=abs(norm*norm-an*an-pn*pn);limit=1e-12+1e-8*norm*norm
    values=dict(norm=norm,active_norm=an,complement_norm=pn,pythagorean_error=pythagoras,pythagorean_limit=limit,
        complementary_squared_fraction=pn*pn/(norm*norm) if norm else None,active_squared_fraction=an*an/(norm*norm) if norm else None)
    for key,vv in values.items():
        if vv is None:require(saved[key] is None,'zero vector fraction')
        elif key=='pythagorean_error':
            require(abs(vv-saved[key])<=256*np.finfo(np.float64).eps*max(1.,norm*norm),'squared-norm cancellation agreement')
        else:close(vv,saved[key])
    return dict(**values,reconstruction_difference=difference),bool(passed and pythagoras<=limit)

def decay_audit(eta,lam,decay):
    require(type(eta) is float and type(lam) is float and math.isfinite(eta) and math.isfinite(lam),'finite binary64 coefficients')
    exact_q=1-Fraction(eta)*Fraction(lam);factor=float((1.-eta*lam)**2-1.)
    require(decay['eta']==eta and decay['lam']==lam and decay['period']==2,'fixed decay parameters')
    require(Fraction(decay['eta_exact'])==Fraction(eta) and Fraction(decay['lam_exact'])==Fraction(lam),'exact rational coefficients')
    require(Fraction(decay['exact_q'])==exact_q and Fraction(decay['exact_factor'])==exact_q**2-1,'exact rational decay algebra')
    require(decay['eta_hex']==eta.hex() and decay['lam_hex']==lam.hex() and decay['rounded_q_hex']==float(1.-eta*lam).hex() and decay['rounded_factor_hex']==factor.hex(),'rounded factor bit identities')
    require(decay['rounded_q']==float(1.-eta*lam) and decay['rounded_factor']==factor and decay['elimination_valid'] is (exact_q**2!=1) and decay['omitted_directions_attracting'] is (abs(exact_q**2)<1),'decay hypotheses')
    return exact_q,factor

def decision(all_pass,fraction):
    require(type(all_pass) is bool,'boolean gates')
    require(fraction is None or (type(fraction) is float and math.isfinite(fraction) and fraction>=0),'valid fraction')
    return bool(all_pass and fraction is not None and fraction>=.5)

def source_record(record,commit):
    require(record['source']['commit']==commit and set(record['source']['sha256'])==set(SOURCE_FILES),'execution source inventory')
    for name,sha in record['source']['sha256'].items():
        require(digest(subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT))==sha,'exact historical source '+name)
        require(digest(subprocess.check_output(['git','show',REVIEWED_SOURCE+':'+name],cwd=ROOT))==sha,'reviewed six source bytes '+name)
    require(record['cpu_threads']==1 and record['nice']>=10 and record['gpu'] is False and record['paid'] is False and record['dynamics_evaluations']==0,'execution resource scope')
    require(all(record['environment']['thread_env'][k]=='1' for k in THREADS),'execution thread limits')

def launcher(mode,pins,raw_entries):
    label='profile-001' if mode=='profile' else 'attempt-001';folder=BASE/(label+'-launcher')
    key='profile' if mode=='profile' else 'target'
    rows=inventory(folder,pins[key+'_launcher_manifest_sha256'],{'launcher-source.py','runtime-start.json','process.json','stdout.log','receipt.json'})
    require(rows['launcher-source.py']['sha256']==pins['wrapper_sha256'],'exact orchestration source')
    runtime=read(folder/'runtime-start.json');receipt=read(folder/'receipt.json');process=read(folder/'process.json')
    require(runtime['mode']==mode and runtime['source_commit']==pins[key+'_source_commit'] and runtime['review_commit']==pins['source_review_commit'],'launcher source/review')
    expected=['timeout','--signal=TERM','--kill-after='+('2s' if mode=='profile' else '5s'),'8s' if mode=='profile' else '45s',
      'nice','-n','10','/data2/neural-shrimp/.venv/bin/python','tools/diagnose_cifar_input_span.py',mode,'--out',str(BASE/label),'--review-commit',pins['source_review_commit']]
    if mode=='collect':expected+=['--profile-manifest-sha256',pins['profile_manifest_sha256']]
    require(runtime['argv']==expected,'exact launched command')
    require(runtime['cwd']=='/home/d7082791602/.codex/worktrees/post-edge-next-steps/neural-shrimp' and runtime['out']==str(BASE/label),'launcher paths')
    require(type(receipt['elapsed_seconds']) in (int,float) and math.isfinite(receipt['elapsed_seconds']) and receipt['elapsed_seconds']>0,'finite launcher elapsed')
    require(all(runtime['thread_env'][k]=='1' for k in THREADS),'launcher thread limits')
    require(runtime['log_limit']==128*1024 and runtime['launcher_byte_limit']==256*1024 and runtime['supervisor_seconds']==(10 if mode=='profile' else 50),'launcher fixed caps')
    require(process['timeout_pid']==receipt['timeout_pid'] and type(process['timeout_pid']) is int and process['timeout_pid']>0,'owned process PID')
    require(receipt['raw_files']==raw_entries,'launcher/raw inventory mirror')
    require(receipt['raw_bytes']==sum(v['bytes'] for v in raw_entries.values()),'launcher raw byte count')
    require(rows['stdout.log']['bytes']<=128*1024,'log cap')
    size=sum(v['bytes'] for v in rows.values())+(folder/'manifest.json').stat().st_size;require(size<=256*1024,'launcher cap')
    return dict(bytes=size,receipt=receipt,manifest_sha256=pins[key+'_launcher_manifest_sha256'])

def audit(pins,emit):
    profile_dir=BASE/'profile-001';target=BASE/'attempt-001'
    pi=inventory(profile_dir,pins['profile_manifest_sha256'],{'runtime-start.json','profile.json','result.json'})
    ti=inventory(target,pins['target_manifest_sha256'])
    profile=read(profile_dir/'result.json');pruntime=read(profile_dir/'runtime-start.json');p=read(profile_dir/'profile.json')
    result=read(target/'result.json');runtime=read(target/'runtime-start.json')
    source_record(profile,pins['profile_source_commit']);source_record(pruntime,pins['profile_source_commit'])
    source_record(result,pins['target_source_commit']);source_record(runtime,pins['target_source_commit'])
    require(profile['source']['sha256']==result['source']['sha256'],'same profiled source bytes')
    require(profile['environment']==result['environment']==runtime['environment']==pruntime['environment'],'same measured environment')
    require(all(v['review_commit']==pins['source_review_commit'] for v in (profile,pruntime,result,runtime)),'exact source review')
    require(runtime['approved_profile_manifest_sha256']==result['approved_profile_manifest_sha256']==pins['profile_manifest_sha256'],'external profile pin carried to target')
    require(pruntime['approved_profile_manifest_sha256'] is None and profile['approved_profile_manifest_sha256'] is None,'profile had no profile input')
    require(profile['mode']==pruntime['mode']=='profile' and result['mode']==runtime['mode']=='collect','fixed actions')
    require(profile['status']==p['status']=='profile_passed' and profile['profile']==p,'completed profile mirror')
    require(type(p['qr_seconds']) in (int,float) and 0<p['qr_seconds']<=profile['elapsed_seconds']<=5,'actual profile timing')
    require(p['projected_target_seconds']==2*p['qr_seconds']+5<=30,'unchanged cost gate')
    for a_name,shape in [('x',[512,3072]),('q',[3072,512]),('r',[512,512])]:
        a=p[a_name];require(a['shape']==shape and a['dtype']=='<f8' and a['order']=='C' and len(a['sha256'])==64 and all(c in '0123456789abcdef' for c in a['sha256']),'profile array hash metadata')
    bc=p['basis_checks'];require(bc['passed'] is True and bc['retained_columns']==512 and bc['rank_truncation'] is False,'profile complete basis gate')
    for metric,limit in [('orthogonality_frobenius',1e-10),('representation_relative',1e-12),('x_i_minus_p_relative',1e-12),('implicit_idempotence_on_q',1e-10),('qr_reconstruction_relative',1e-12)]:
        v=bc[metric];require(type(v) in (int,float) and math.isfinite(v) and 0<=v<=limit,'profile basis numeric gate')
    require(bc['representation_relative']==bc['x_i_minus_p_relative'],'profile basis mirror')
    for final,start in ((profile,pruntime),(result,runtime)):
        require(final['pid']==start['pid'] and type(final['pid']) is int and final['pid']>0 and final['started_utc']==start['started_utc'] and start['status']=='running','execution PID/time consistency')
    require(p['shape']==[3072,512] and p['retained_columns']==512 and p['seed']==28240,'same-shaped fixed synthetic profile')
    require(profile['settings']==result['settings']==runtime['settings']==pruntime['settings'],'same source settings')
    for key,value in dict(d=3072,n=512,width=32,k=10,precision='float64',period=2,retained_columns=512,orthogonality_limit=1e-10,
      representation_limit=1e-12,identity_atol=1e-12,identity_rtol=1e-8,triage_fraction=.5,profile_seconds=5,target_seconds=30,raw_cap=64*1024**2,reserve=1024**2,external_reserve=512*1024,synthetic_seed=28240).items():
        require(profile['settings'][key]==value,'fixed scientific/resource setting '+key)
    launchers={}
    for mode,folder,rows in [('profile',profile_dir,pi),('collect',target,ti)]:
        launchers[mode]=launcher(mode,pins,{**rows,'manifest.json':entry(folder/'manifest.json')})
    size=sum(sum(v['bytes'] for v in rows.values())+(folder/'manifest.json').stat().st_size for folder,rows in [(profile_dir,pi),(target,ti)])+sum(l['bytes'] for l in launchers.values())
    require(size<=64*1024**2,'aggregate profile/target/launcher cap')
    emit('provenance.json',dict(pins=pins,aggregate_producer_bytes=size,launchers=launchers,profile_projection_seconds=p['projected_target_seconds']))
    require(launchers['profile']['receipt']['exit_code']==0 and launchers['profile']['receipt']['error'] is None and not launchers['profile']['receipt']['log_truncated'],'successful profile process')
    if result['status']=='failed_or_interrupted':
        return dict(verdict='producer_incomplete_no_triage',status=result['status'],error=result.get('error'),aggregate_producer_bytes=size)
    require(result['status'] in ('diagnostic_passed','identity_check_failed'),'declared completed target result')
    lr=launchers['collect']['receipt'];require(lr['error'] is None and lr['log_truncated'] is False,'target process not interrupted')
    require(lr['exit_code']==(0 if result['status']=='diagnostic_passed' else 2),'target exit/result consistency')
    profile_binding=read(target/'profile-binding.json')
    require(result['profile_binding']==profile_binding==dict(path=str(profile_dir),manifest=entry(profile_dir/'manifest.json'),source_commit=pins['profile_source_commit'],projection_seconds=p['projected_target_seconds'],retained_bytes=sum(v['bytes'] for v in pi.values())+(profile_dir/'manifest.json').stat().st_size),'target profile binding mirror')
    original=[inventory(REFERENCE,ORIGINAL_PINS[0],{'reference.npz','result.json'}),inventory(LINEAR,ORIGINAL_PINS[1],{'applied-direction.npy','direction.npy','events.jsonl','initial-map1.npy','initial-map2.npy','initial-residual.npy','initial-theta.npy','initial.json','input-identity.json','linear.json','parity.json','result.json','returned-direction.json','runtime-start.json','true-linear-residual.npy'}|{f'trial-{i}-{s}.npy' for i in range(4) for s in ('map1','map2','residual','theta')})]
    bound=read(target/'input-binding.json');ref=read(REFERENCE/'result.json');old=read(LINEAR/'result.json');old_runtime=read(LINEAR/'runtime-start.json');old_id=read(LINEAR/'input-identity.json')
    require(bound['state_order']==result['state_order']==list(LABELS),'five fixed state order')
    require(set(bound['array_identities'])==set(VECTOR_NAMES),'thirteen vector identities')
    require(old_id['bundle']==str(REFERENCE) and old_id['bundle_manifest_sha256']==ORIGINAL_PINS[0],'reference path binding')
    require(bound['reference_inventory']==original[0] and bound['linear_inventory']==original[1],'complete original inventory mirror')
    require(bound['reference_manifest']==ORIGINAL_PINS[0] and bound['linear_manifest']==ORIGINAL_PINS[1],'original manifest binding')
    require(ref['status']=='complete' and old['status']=='diagnostic_complete' and old['source_sha256']==old_runtime['source_sha256'],'completed original diagnostics')
    require(ref['source_commit']==bound['reference_source']=='a3debab72195d5ac58bee5562cb59afd8b6466e8' and old['source_commit']==old_runtime['source_commit']==bound['linear_source']=='a9830e714d259597a53cee6ef7be41a9f07bb205','original frozen sources')
    for record in (ref,old_runtime):
        for name,sha in record['source_sha256'].items():require(digest(subprocess.check_output(['git','show',record['source_commit']+':'+name],cwd=ROOT))==sha,'original source bytes')
    require(old_id==old['input_identity'] and old_id['data_identity']==ref['data_identity']==bound['data_identity'],'original exact data identity')
    require(bound['model']==ref['model']==dict(d=3072,width=32,k=10) and bound['eta']==ref['eta']==old_id['eta']==.00827753 and bound['lam']==ref['lam']==old_id['lam']==.01,'original packing/rate/ridge')
    with np.load(REFERENCE/'reference.npz',allow_pickle=False) as z:
        require(set(z.files)=={'x','y','theta','directions','gradient','hvp','j2vp'},'reference archive inventory')
        x=finite(z['x'],(512,3072));y=finite(z['y'],(512,10));theta=finite(z['theta'],(98666,))
    require(info(x)==bound['x']==ref['data_identity']['x'] and info(y)==bound['y']==ref['data_identity']['y'],'exact subset-A arrays')
    q=finite(np.load(target/'basis-q.npy',allow_pickle=False),(3072,512));r=finite(np.load(target/'basis-r.npy',allow_pickle=False),(512,512))
    saved_basis=read(target/'basis-checks.json');require(result['basis_checks']=={k:v for k,v in saved_basis.items() if k!='qr_seconds'},'target basis mirror')
    basis_values,all_pass=basis(x,q,r,saved_basis);emit('basis-audit.json',dict(values=basis_values,passed=all_pass))
    decay=read(target/'decay-factor.json');require(result['decay']==decay,'decay mirror')
    exact_q,factor=decay_audit(bound['eta'],bound['lam'],decay)
    vectors={};parts={};rows={v['name']:v for v in result['vectors']};require(len(result['vectors'])==13 and list(rows)==list(VECTOR_NAMES),'full13 vector order')
    metrics=[];expected={'runtime-start.json','profile-binding.json','input-binding.json','basis-q.npy','basis-r.npy','basis-checks.json','decay-factor.json','linear-identities.json','ap-predicted-complement.npy','e-predicted-complement.npy','result.json'}
    for name in VECTOR_NAMES:
        if name in ('p','ap','e'):filename={'p':'direction.npy','ap':'applied-direction.npy','e':'true-linear-residual.npy'}[name]
        else:
            label,what=name.rsplit('-',1);prefix='initial' if label=='original' else label;filename=prefix+'-'+('theta' if what=='state' else 'residual')+'.npy'
        v=finite(np.load(LINEAR/filename,allow_pickle=False),(98666,));vectors[name]=v
        require(info(v)==bound['array_identities'][name],'input vector byte identity')
        active=finite(np.load(target/(name+'-active.npy'),allow_pickle=False),v.shape);perp=finite(np.load(target/(name+'-complement.npy'),allow_pickle=False),v.shape);parts[name]=(active,perp)
        saved=read(target/(name+'-checks.json'));require(saved==rows[name],'per-vector result mirror')
        values,passed=decompose(v,active,perp,x,q,32,saved)
        expected.update(name+suffix for suffix in ('-active.npy','-complement.npy','-checks.json'))
        if name.endswith('-residual'):
            label=name.removesuffix('-residual');state=vectors[label+'-state']
            map2=finite(np.load(LINEAR/(('initial' if label=='original' else label)+'-map2.npy'),allow_pickle=False),v.shape)
            require(same(v,map2-state),'saved residual bitwise relation')
            predicted=factor*parts[label+'-state'][1];saved_prediction=finite(np.load(target/(name+'-predicted-complement.npy'),allow_pickle=False),v.shape)
            closeness(saved_prediction,predicted);passed=gate(perp,predicted,nrm(v),saved['period2_complement_identity']) and passed
            expected.add(name+'-predicted-complement.npy')
        require(saved['passed'] is passed,'per-vector overall decision');all_pass=all_pass and passed;metrics.append(dict(name=name,**values,passed=passed))
    require(set(ti)==expected,'exact completed target output inventory')
    require(same(theta,vectors['original-state']),'original theta bytes')
    require(len(old['trials'])==4,'exact four trials')
    for i,alpha in enumerate((1.,.5,.25,.125)):
        require(old['trials'][i]['index']==i and old['trials'][i]['alpha']==alpha,'saved damping order')
        require(same(vectors[f'trial-{i}-state'],theta+alpha*vectors['p']),'trial base/direction bytes')
    linear=read(target/'linear-identities.json');require(result['linear_identities']==linear,'linear result mirror')
    pp,ap,ep,rp=(parts[name][1] for name in ('p','ap','e','original-residual'));ap_expected=factor*pp;e_expected=factor*(pp+parts['original-state'][1])
    closeness(np.load(target/'ap-predicted-complement.npy',allow_pickle=False),ap_expected);closeness(np.load(target/'e-predicted-complement.npy',allow_pickle=False),e_expected)
    values=[('ap_complement_identity',ap,ap_expected,nrm(vectors['ap'])),('e_complement_identity',ep,e_expected,nrm(vectors['e'])),
      ('e_equals_ap_plus_r',vectors['e'],vectors['ap']+vectors['original-residual'],nrm(vectors['e'])),
      ('e_active_equals_ap_active_plus_r_active',parts['e'][0],parts['ap'][0]+parts['original-residual'][0],nrm(vectors['e'])),
      ('e_complement_equals_ap_complement_plus_r_complement',ep,ap+rp,nrm(vectors['e']))]
    linear_pass=all([gate(a,b,s,linear['checks'][key]) for key,a,b,s in values])
    exact_sum=same(vectors['e'],vectors['ap']+vectors['original-residual']);require(linear['checks']['saved_sum_bitwise_equal'] is exact_sum,'saved linear bytes')
    linear_pass=linear_pass and exact_sum;require(linear['passed'] is linear_pass,'linear decision');all_pass=all_pass and linear_pass
    fraction=next(row['complementary_squared_fraction'] for row in metrics if row['name']=='original-residual')
    if fraction is None:require(result['original_complementary_squared_residual_fraction'] is None,'zero original residual fraction')
    else:close(fraction,result['original_complementary_squared_residual_fraction'])
    triage=decision(all_pass,fraction);require(result['prioritize_reduced_solver'] is triage and result['all_numerical_gates_passed'] is all_pass,'unchanged triage decision')
    require(result['dimensions']==dict(full=98666,active=16746,complement=81920),'fixed dimensions')
    require(result['status']==('diagnostic_passed' if all_pass else 'identity_check_failed'),'overall status')
    emit('vector-audit.json',dict(vectors=metrics,linear_checks_passed=linear_pass,exact_q=str(exact_q),exact_factor=str(exact_q**2-1),rounded_factor=factor))
    return dict(verdict='saved_outcome_audit_pass',producer_status=result['status'],all_numerical_gates_passed=all_pass,
      original_complementary_squared_residual_fraction=fraction,prioritize_reduced_solver=triage,aggregate_producer_bytes=size,
      scope='saved-Q numerical consistency and fixed triage only; no QR, model, derivative, solver, trajectory, orbit or causal GMRES claim')

def run(out,pins_path,pins_sha,source):
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');require(not out.exists(),'exclusive output')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==source,'audit execution source')
    hashes={}
    for name in AUDIT_FILES:
        raw=(ROOT/name).read_bytes();require(raw==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed audit source');hashes[name]=digest(raw)
    require(entry(pins_path)['sha256']==pins_sha,'externally approved metadata pins')
    pins=read(pins_path);require(set(pins)==PIN_KEYS,'complete metadata pins')
    for key,value in pins.items():
        size=40 if key.endswith('commit') else 64
        require(isinstance(value,str) and len(value)==size and all(c in '0123456789abcdef' for c in value),'full metadata identity '+key)
    out.mkdir(parents=True,exist_ok=False)
    def emit(name,value,final=False):
        data=canonical(value);require(sum(p.stat().st_size for p in out.iterdir())+len(data)<=DATA_CAP-(0 if final else 512*1024),'audit data cap')
        with (out/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    started=time.monotonic();record=dict(status='running',audit_source=source,audit_sources=hashes,pins=pins,pins_file=entry(pins_path),pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),cpu_threads=1,started_utc=datetime.now(timezone.utc).isoformat(),load_average=os.getloadavg(),environment=dict(python=platform.python_version(),numpy=np.__version__,thread_env={k:os.environ[k] for k in THREADS}),gpu=False,paid=False,dynamics_evaluations=0)
    emit('runtime-start.json',record)
    def stop(*_):raise TimeoutError('saved-outcome audit15s cap')
    old_alarm=signal.signal(signal.SIGALRM,stop);old_term=signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,15)
    try:record.update(audit(pins,emit),status='finished')
    except BaseException as e:record.update(status='failed',verdict='audit_failed',error=type(e).__name__+': '+str(e))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_alarm);signal.signal(signal.SIGTERM,old_term)
        record['elapsed_seconds']=time.monotonic()-started;record['finished_utc']=datetime.now(timezone.utc).isoformat();emit('result.json',record,final=True)
        emit('manifest.json',{p.name:entry(p) for p in sorted(out.iterdir())},final=True)
        for p in out.iterdir():p.chmod(0o444)
        out.chmod(0o555)
    print(json.dumps(record,sort_keys=True));return 0 if record.get('verdict') in ('saved_outcome_audit_pass','producer_incomplete_no_triage') else 2
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--pins',type=Path,required=True);p.add_argument('--pins-sha256',required=True);p.add_argument('--source-commit',required=True);a=p.parse_args()
    raise SystemExit(run(a.out,a.pins,a.pins_sha256,a.source_commit))

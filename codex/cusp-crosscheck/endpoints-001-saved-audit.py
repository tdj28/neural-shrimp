"""Two fixed endpoints: saved arrays/field arithmetic only, never model or AD.

No producer imports, eigensolve, inverse, dense acquisition, bootstrap or tuning.
The original full eigensystem's unretained eigenvectors remain source-dependent;
its saved residual vector is checked, and retained left/right pairs are replayed.
"""
from pathlib import Path
import argparse,collections,hashlib,json,math,os,re,resource,sys,time,traceback

RAW=Path('/data2/neural-shrimp-data/codex/llm-cusp-crosscheck')
ATTEMPT=RAW/'endpoints-001'
SOURCE='925b73fa1f274b60b5a1c74cc584e1de9dc13cea'
RECEIPT='0cf633c0484e51f0a7a96c750c653cf972d02f35880f5cc9ebf13b37b61d2fc1'
PLAN='8ba8b6f322f01c37ba98c1389f19a42eb8b22d70869b9078fda3e6ae36532894'
T=dict(closure=1e-10,primitive=1e-6,phase_composition=1e-12,critical=1e-8,imaginary=1e-10,
       unit_circle_gap=1e-3,isolation=1e-3,eigen=1e-8,dense_jvp=1e-10,finite_difference=1e-3,
       finite_difference_floor=1e-5,finite_difference_reduction=.3,homological=1e-8,near_mu_coefficient=1e-3,nonzero_a=1e-10)
STEPS=[1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5];ALPHA_STEPS=[7e-6,2e-6,7e-7]
PARAMS={13:(.1893809804983369,.06661104169386665),14:(.1893794755174379,.06660969640358283)}
SCALAR_ABS=5e-12;SCALAR_REL=5e-10  # Saved reduction comparison only; never a scientific threshold.

def require(condition,label):
    if not condition:raise ValueError(label)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def pinned(path,digest):
    require(Path(path).is_file() and not Path(path).is_symlink() and not Path(path).stat().st_mode&0o222,'immutable regular metadata')
    require(sha(path)==digest,'metadata pin: '+str(path));return read(path)
def available():return next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
def norm(x):return math.sqrt(math.fsum(float(abs(v))**2 for v in x.flat))
def dot(a,b):return math.fsum(float(x)*float(y) for x,y in zip(a,b))
def rel(a,b,floor=1e-30):return norm(a-b)/max(floor,norm(b))
def scalar_error(a,b):return abs(float(a)-float(b))/max(abs(float(b)),1.)
def ladder(errors):
    return len(errors)>=3 and all(math.isfinite(e) for e in errors) and max(errors[-2:])<=T['finite_difference'] and (
        max(errors[-2:])<=T['finite_difference_floor'] or max(errors[-2:])<=T['finite_difference_reduction']*max(errors[0],1e-30))
def signs(reference,estimates):
    values=[reference,*estimates[-2:]]
    return len(estimates)>=2 and all(math.isfinite(v) and abs(v)>T['nonzero_a'] for v in values) and (all(v>0 for v in values) or all(v<0 for v in values))
def cubic(left,third,mixed):
    direct=dot(left,third)/6;center=dot(left,mixed)/2;total=direct+center
    return dict(c=total,direct=direct,center_manifold=center,cancellation_ratio=(abs(direct)+abs(center))/max(abs(total),sys.float_info.min))
def homological_saved(M,r,l,B,h,border,mu):
    b=.5*dot(l,B);rhs=B-2*b*r;res=mu**2*h-M@h-rhs;orth=dot(l,h)
    denominator=max(1.,norm(rhs))
    return dict(b=b,border=border,multiplier_used=mu,h2_norm=norm(h),orthogonality=orth,
        bordered_relative_residual=math.sqrt(norm(res+border*r)**2+orth**2)/denominator,
        unbordered_relative_residual=norm(res)/denominator)
def homological_gate(row):return max(row['bordered_relative_residual'],row['unbordered_relative_residual'],abs(row['orthogonality'])/max(1.,row['h2_norm']))<=T['homological']

class StopAtGate(Exception):
    def __init__(self,stage,gates):self.stage,self.gates=stage,gates

class Audit:
    def __init__(self):self.began=time.monotonic();self.checks=0;self.deltas=[];self.exact_arrays=0;self.inventory={};self.array_references=set()
    def guard(self):
        require(time.monotonic()-self.began<115,'audit cooperative deadline')
        require(available()>=16*1024**3,'audit shared RAM floor')
    def number(self,label,saved,computed):
        require(math.isfinite(float(saved)) and math.isfinite(float(computed)),label+' finite')
        delta=float(computed)-float(saved);self.checks+=1
        limit=SCALAR_ABS+SCALAR_REL*abs(float(saved))
        if delta:self.deltas.append(dict(path=label,saved=float(saved),independent=float(computed),difference=delta,comparison_tolerance=limit))
        require(abs(delta)<=limit,label+' saved scalar differs')
    def fields(self,label,saved,computed):
        for key,value in computed.items():
            if isinstance(value,dict):self.fields(label+'/'+key,saved[key],value)
            elif isinstance(value,(bool,str)) or value is None:require(type(saved[key]) is type(value) and saved[key]==value,label+'/'+key)
            else:self.number(label+'/'+key,saved[key],value)
    def exact(self,label,a,b):
        require(a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b),label+' exact array association');self.exact_arrays+=1
    def decode(self,value,directory):
        if isinstance(value,dict) and set(value)=={'path','shape','dtype'}:
            path=directory/value['path'];key=str(path.relative_to(ATTEMPT))
            require(path.parent==directory and key in self.inventory,'inventoried direct native array')
            arr=np.load(path,allow_pickle=False,mmap_mode='r')
            require(list(arr.shape)==value['shape'] and str(arr.dtype)==value['dtype'] and arr.dtype in (np.dtype('float64'),np.dtype('complex128')),'native array shape/dtype')
            require(np.isfinite(arr).all(),'finite complete saved array');self.array_references.add(key);return arr
        if isinstance(value,dict):return {k:self.decode(v,directory) for k,v in value.items()}
        if isinstance(value,list):return [self.decode(v,directory) for v in value]
        if isinstance(value,float):require(math.isfinite(value),'finite JSON diagnostic')
        return value
    def inventory_check(self,receipt):
        rows=receipt['artifacts'];require(len(rows)==1191,'literal original artifact count')
        inventory={}
        for row in rows:
            self.guard();name=row['path'];p=ATTEMPT/name
            require(not Path(name).is_absolute() and '..' not in Path(name).parts and name not in inventory and name!='receipt.json','unique relative artifact')
            require(p.is_file() and not p.is_symlink() and p.resolve()==p and not p.stat().st_mode&0o222,'immutable original artifact')
            require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'artifact hash '+name);inventory[name]=row
        observed={str(p.relative_to(ATTEMPT)) for p in ATTEMPT.rglob('*') if p.is_file()}
        require(observed==set(inventory)|{'receipt.json'},'complete original inventory');self.inventory=inventory

class Events:
    def __init__(self,a,index):
        self.a,self.index=a,index;self.directory=ATTEMPT/'run'/str(index);self.rows=[];self.position=0;self.counts=collections.Counter();self.gates=[];self.calls=[]
        for p in sorted(self.directory.glob('*.json')):
            match=re.fullmatch(r'(\d{4})-(.+)\.json',p.name);require(match is not None,'native event name')
            require(int(match[1])==len(self.rows),'ordered complete native event indices')
            self.rows.append((match[2],a.decode(read(p),self.directory)))
    def take(self,name):
        self.a.guard();require(self.position<len(self.rows) and self.rows[self.position][0]==name,'ordered native event '+name)
        row=self.rows[self.position][1];self.position+=1;return row
    def upcoming(self,name):return next(row for tag,row in self.rows[self.position:] if tag==name)
    def gate(self,stage,computed):
        saved=self.take(stage+'-gates');require(saved==computed,stage+' exact scientific gates')
        self.gates.append(dict(stage=stage,saved=saved,independent=computed))
        if not all(computed.values()):raise StopAtGate(stage,computed)
    def call(self,name,state,**extra):
        self.counts[name]+=1;tag=name+'-'+str(self.counts[name]).zfill(3)
        row=self.take(tag+'-input');temperature,alpha=PARAMS[self.index]
        require(row['alpha']==alpha and row['log10_temperature']==temperature and row['period']==7,'actual native map/rate/period')
        self.a.exact(tag+'/state',row['state'],state)
        for key,value in extra.items():
            if isinstance(value,np.ndarray):self.a.exact(tag+'/'+key,row[key],value)
            elif isinstance(value,list):
                require(len(row[key])==len(value),tag+'/'+key+' count')
                for j,v in enumerate(value):self.a.exact(tag+'/'+key+'/'+str(j),row[key][j],v)
            else:require(row[key]==value,tag+'/'+key)
        out=self.take(tag+'-return');require(math.isfinite(out['seconds']) and out['seconds']>=0,'native elapsed')
        self.calls.append(dict(operation=name,seconds=out['seconds']))
        return out['value']

def endpoint(a,index,state,vector,profile_matrix_hash):
    e=Events(a,index);original=e.take('original-input');temperature,alpha=PARAMS[index]
    require(original['index']==index and original['alpha']==alpha and original['log10_temperature']==temperature,'exact original point association')
    a.exact('original state',original['state'],state);a.exact('original vector',original['vector'],vector)
    summary=read(ATTEMPT/'run'/('endpoint-'+str(index)+'.json'));details=dict(index=index,gates=e.gates)
    try:
        pre=e.take('prefix-input');require(pre==dict(alpha=alpha,log10_temperature=temperature,period=7),'prefix actual parameter')
        pre=e.take('prefix-return');require(float(pre['alpha'])==alpha and len(pre['layers'])==24 and all(len(pair)==2 for pair in pre['layers']),'full prefix/alpha')
        e.counts['prefix']=1;current=state;first=None
        for i in range(7):current=e.call('step',current);first=current if i==0 else first
        value=e.call('composed',state);scale=max(1.,norm(state));orbit=dict(closure=norm(value-state)/scale,composition_relative=rel(current,value),proper_divisor_separations={'1':norm(first-state)/scale})
        a.fields('orbit/'+str(index),e.take('orbit-diagnostics'),orbit);details['orbit']=orbit
        e.gate('orbit',dict(closure=orbit['closure']<=T['closure'],phase_composition=orbit['composition_relative']<=T['phase_composition'],primitive=orbit['proper_divisor_separations']['1']>T['primitive']))
        batches=[]
        if index==14:
            for start in range(0,896,16):
                basis=np.zeros((16,896));basis[np.arange(16),start+np.arange(16)]=1
                batches.append(e.call('dense-batch',np.broadcast_to(state,(16,896)),directions=basis,column_start=start))
            assembled=e.take('assembled-monodromy');require(assembled['columns']==896,'complete new matrix')
        mon=e.take('monodromy-source');M=mon['matrix'];require(M.shape==(896,896) and mon['reused']==(index==13),'original matrix/reuse role')
        if index==14:
            a.exact('assembled matrix',assembled['matrix'],M)
            for j,output in enumerate(batches):a.exact('retained dense columns',M[:,16*j:16*j+16],output.T)
        else:
            # Reuse its already audited original matrix; do not repeat profile columns.
            record=next(p for p in e.directory.glob('*-monodromy-source-data-matrix.npy'))
            require(sha(record)==profile_matrix_hash,'exact audited profile matrix reuse')
        raw=e.take('raw-full-spectrum')['eigenvalues'];require(raw.shape==(896,) and np.isfinite(raw).all(),'complete finite raw spectrum')
        if e.rows[e.position][0]=='classification-refusal':
            refusal=e.take('classification-refusal');details['classification_refusal']=refusal
            raise StopAtGate('fold-classification',{'real_isolated_hyperbolic_plus_one':False})
        basis=e.take('classified-fold-basis');r,l=basis['right'],basis['left'];values=basis['eigenvalues']
        require(r.shape==l.shape==(896,) and values.shape==(896,) and basis['all_eigenpair_residuals'].shape==(896,),'full basis/spectrum records')
        k=int(np.argmin(abs(values-1)));mu=float(values[k].real);other=np.delete(values,k)
        require(abs(values[k].imag)<=T['imaginary'] and abs(mu-1)<=T['critical'],'actual real +1 classification')
        require(min(abs(abs(other)-1))>T['unit_circle_gap'] and min(abs(other-values[k]))>T['isolation'],'simple +1/hyperbolic complement')
        require(np.min(basis['all_eigenpair_residuals'])>=0 and np.max(basis['all_eigenpair_residuals'])<T['eigen'],'saved complete eigensystem residuals')
        metrics=dict(multiplier=mu,multiplier_minus_one=mu-1,next_modulus=float(max(abs(other))),nearest_other_unit_circle=float(min(abs(abs(other)-1))),
            complement_attracting=bool(max(abs(other))<1),isolation=float(min(abs(other-values[k]))),norm_right=norm(r),pairing=dot(l,r),
            right_residual=norm(M@r-mu*r),left_residual=norm(M.T@l-mu*l)/norm(l),orientation='largest-absolute right coordinate positive')
        a.fields('basis/'+str(index),basis,metrics)
        require(max(metrics['right_residual'],metrics['left_residual'])<T['eigen'] and abs(metrics['norm_right']-1)<=1e-10 and abs(metrics['pairing']-1)<=1e-10 and r[int(np.argmax(abs(r)))]>0,'retained mode/gauge equations')
        details['classification']={key:v for key,v in metrics.items() if key not in ('orientation',)}
        # The raw eigvals call and retained classified eig call may use different order.
        require(np.allclose(np.sort_complex(raw),np.sort_complex(values),rtol=SCALAR_REL,atol=SCALAR_ABS),'two saved spectrum calls agree')
        j0=e.call('composed-jvp',state,direction=r)
        probe_input=e.rows[e.position][1];probe=probe_input['direction'];expected=np.arange(1,897,dtype=np.float64);expected/=norm(expected)
        require(np.allclose(probe,expected,rtol=0,atol=2e-16),'literal deterministic full-state probe')
        jp=e.call('composed-jvp',state,direction=probe)
        directional=e.take('dense-directional');dense_errors={}
        for name,v,action in [('critical',r,j0),('probe',probe,jp)]:
            a.exact(name+' retained direction',directional[name]['vector'],v);a.exact(name+' retained native action',directional[name]['action'],action)
            err=rel(M@v,action);a.number(name+' dense agreement',directional[name]['dense_relative_error'],err);dense_errors[name]=err
        e.gate('dense-composed',{key:v<=T['dense_jvp'] for key,v in dense_errors.items()});details['dense_errors']=dense_errors
        B=e.call('state-ad-2',state,directions=[r,r]);C=e.call('state-ad-3',state,directions=[r,r,r]);A=e.call('alpha-ad',state)
        solves=e.take('homological-solves');computed={}
        for name,mult in [('fold',1.),('near_mu',mu)]:
            row=solves[name];require(row['h2'].shape==(896,),'full retained homological vector')
            computed[name]=homological_saved(M,r,l,B,row['h2'],row['border'],mult);a.fields('homological/'+str(index)+'/'+name,row,computed[name])
        e.gate('homological',{key:homological_gate(v) for key,v in computed.items()});details['homological']=computed
        H,Hmu=solves['fold']['h2'],solves['near_mu']['h2']
        BH=e.call('state-ad-2',state,directions=[r,H]);BHmu=e.call('state-ad-2',state,directions=[r,Hmu])
        coeff=cubic(l,C,BH);near=cubic(l,C,BHmu);aa=dot(l,A);bb=computed['fold']['b'];near_relative=scalar_error(near['c'],coeff['c'])
        rawcoef=dict(a_physical_alpha=aa,b=bb,ab=aa*bb,b_sign_a=bb*math.copysign(1.,aa) if abs(aa)>T['nonzero_a'] else None,
            coefficient=coeff,near_mu_coefficient=near,near_mu_relative=near_relative,h2_relative_change=rel(Hmu,H),multiplier_minus_one=mu-1,
            normalization=dict(norm_right=norm(r),left_right=dot(l,r)),definition='x(u)=x*+r*u+h2*u^2/2+..., ||r||2=1,l^Tr=1; clock one full period',orientation=basis['orientation'])
        originalcoef=e.take('raw-ad-coefficients');a.fields('coefficients/'+str(index),originalcoef,rawcoef);details['coefficients']=rawcoef
        fd=[]
        for h in STEPS:
            saved=e.upcoming('finite-difference-row');require(saved['h_relative']==h,'fixed full FD ladder')
            s=saved['h_absolute'];a.number('FD absolute offset',s,h*(norm(state) or 1.));xp,xm=state+s*r,state-s*r
            plus=e.call('composed-jvp',xp,direction=r);minus=e.call('composed-jvp',xm,direction=r)
            v=np.stack((H,Hmu));hp=e.call('composed-jvp',np.broadcast_to(xp,(2,896)),direction=v);hm=e.call('composed-jvp',np.broadcast_to(xm,(2,896)),direction=v)
            bfd=(plus-minus)/(2*s);cfd=(plus-2*j0+minus)/(s*s);mixed=(hp-hm)/(2*s)
            cp=cubic(l,cfd,mixed[0]);np_=cubic(l,cfd,mixed[1]);mapplus=e.call('composed',xp);mapminus=e.call('composed',xm)
            row=dict(h_relative=h,h_absolute=s,b=.5*dot(l,bfd),coefficient=cp,near_mu_coefficient=np_,errors=dict(b_projected=scalar_error(.5*dot(l,bfd),bb),Brr=rel(bfd,B,1.),Crrr=rel(cfd,C,1.),Brh2=rel(mixed[0],BH,1.),Brh2_near=rel(mixed[1],BHmu,1.),critical_map_jvp=rel((mapplus-mapminus)/(2*s),j0,1.),c_direct=scalar_error(cp['direct'],coeff['direct']),c_center=scalar_error(cp['center_manifold'],coeff['center_manifold']),c_total=scalar_error(cp['c'],coeff['c'])))
            a.fields('FD/'+str(index)+'/'+str(h),e.take('finite-difference-row'),row);fd.append(row)
        probe_rows=[]
        for h in STEPS:
            saved=e.upcoming('probe-finite-difference-row');require(saved['h_relative']==h,'fixed probe ladder');s=saved['h_absolute'];a.number('probe absolute offset',s,h*(norm(state) or 1.))
            plus=e.call('composed',state+s*probe);minus=e.call('composed',state-s*probe)
            row=dict(h_relative=h,h_absolute=s,error=rel((plus-minus)/(2*s),jp,1.));a.fields('probeFD/'+str(index)+'/'+str(h),e.take('probe-finite-difference-row'),row);probe_rows.append(row)
        alpha_rows=[]
        for h in ALPHA_STEPS:
            plus=e.call('alpha-offset-map',state,actual_alpha=alpha+h,offset=h);minus=e.call('alpha-offset-map',state,actual_alpha=alpha-h,offset=-h);estimate=(plus-minus)/(2*h)
            row=dict(h_absolute=h,a_physical_alpha=dot(l,estimate),error=rel(estimate,A,1.),projected_error=scalar_error(dot(l,estimate),aa));a.fields('alphaFD/'+str(index)+'/'+str(h),e.take('alpha-finite-difference-row'),row);alpha_rows.append(row)
        details.update(fd_rows=fd,probe_fd_rows=probe_rows,alpha_fd_rows=alpha_rows)
        gates={key:ladder([row['errors'][key] for row in fd]) for key in fd[0]['errors']}
        gates.update(physical_alpha_projected=ladder([r['projected_error'] for r in alpha_rows]),probe_jvp=ladder([r['error'] for r in probe_rows]),physical_alpha=ladder([r['error'] for r in alpha_rows]),near_mu=math.isfinite(near_relative) and near_relative<=T['near_mu_coefficient'])
        e.gate('coefficient-precision',gates)
        resolved=dict(a=signs(aa,[r['a_physical_alpha'] for r in alpha_rows]),b=signs(bb,[r['b'] for r in fd]),c=signs(coeff['c'],[r['coefficient']['c'] for r in fd]))
        require(e.take('numerical-sign-resolution')['signs']==resolved,'fixed finest-row sign decision')
        require(summary['status']=='numerical_endpoint_qualified' and summary['numerical_qualified'] is True,'qualified row status')
        require(summary['precision_gates']==gates and summary['numerical_sign_resolution']==resolved and summary['orientation_bracket_ready']==(resolved['a'] and resolved['b']) and summary['cubic_nonzero_supported']==resolved['c'],'saved qualification/sign gates')
        # Summary copies of every retained row are compared to original events,
        # independently reconstructed values above remain separately reported.
        require(summary['coefficients']==originalcoef,'summary coefficient copy')
        a.fields('summary/'+str(index),summary,dict(closure=orbit['closure'],proper_divisor_separations=orbit['proper_divisor_separations'],critical_multiplier=mu,next_modulus=metrics['next_modulus'],other_unit_circle_distance=metrics['nearest_other_unit_circle'],complement_attracting=metrics['complement_attracting']))
        for key,tag in [('fd_rows','finite-difference-row'),('probe_fd_rows','probe-finite-difference-row'),('alpha_fd_rows','alpha-finite-difference-row')]:require(summary[key]==[row for name,row in e.rows if name==tag],'summary '+key)
        details.update(numerical_qualified=True,numerical_sign_resolution=resolved,orientation_bracket_ready=resolved['a'] and resolved['b'],cubic_nonzero_supported=resolved['c'])
    except StopAtGate as stop:
        require(summary['status']=='numerical_endpoint_unqualified' and summary['numerical_qualified'] is False and summary['stage']==stop.stage and summary['gates']==stop.gates,'preserved stopped/unqualified outcome')
        details.update(numerical_qualified=False,failed_stage=stop.stage,failed_gates=stop.gates)
    require(e.take('endpoint-result')==summary and e.position==len(e.rows),'terminal exact endpoint summary/no later calls')
    require(summary['index']==index and summary['certificate_passed'] is False and summary['cusp_qualified'] is False,'unchanged endpoint claim scope')
    require(summary['operation_counts']==dict(e.counts),'complete actual operation counts')
    if 'timings' in summary:
        require(len(summary['timings'])==len(e.calls),'complete timing count')
        for saved,call in zip(summary['timings'],e.calls):
            require(saved['operation']==call['operation'] and math.isfinite(saved['seconds']) and saved['seconds']>=call['seconds'],'ordered timings include retained return')
    details.update(operation_counts=dict(e.counts),event_count=len(e.rows),original_summary=summary,certificate_passed=False,cusp_qualified=False)
    return details

def closed_now(closure,manifest):
    identities=[closure['original_worker'],closure['original_launcher']]
    require(closure['original_session']==44092 and closure['session_reaped_exit_code']==0 and closure['all_original_pids_groups_sessions_absent'] is True and closure['receipt_sha256']==RECEIPT,'original closure authority')
    for row,key in zip(identities,('worker_identity','launcher_identity')):
        for field in ('pid','start','pgid','sid','uid'):require(row[field]==manifest[key][field],'original process identity')
        require(not (Path('/proc')/str(row['pid'])).exists(),'original PID remains')
    groups={x['pgid'] for x in identities};sessions={x['sid'] for x in identities}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        except (FileNotFoundError,ProcessLookupError):continue
        require(int(fields[2]) not in groups and int(fields[3]) not in sessions,'original group/session remains')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--card',type=Path,required=True);parser.add_argument('--card-sha256',required=True);args=parser.parse_args()
    require(sha(args.card)==args.card_sha256,'exact review card');card=read(args.card)
    require(card['schema']=='codex-cusp-endpoint-saved-audit-v1' and card['admission']=='reviewed' and card['source_sha256']==sha(Path(__file__)),'reviewed exact audit source')
    require(card['comparison']==dict(absolute_tolerance=SCALAR_ABS,relative_tolerance=SCALAR_REL,all_nonzero_differences_retained=True),'frozen saved-reduction comparison')
    require(card['envelope']==dict(cpu_threads=1,nice=10,memory_gib=1,timeout_seconds=120,maximum_output_mib=8,minimum_start_available_gib=20,minimum_runtime_available_gib=16),'fixed audit envelope')
    require(os.getpriority(os.PRIO_PROCESS,0)==10 and available()>=20*1024**3,'audit CPU priority/startup headroom')
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CPU,(115,115))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    global np
    import numpy as np
    require(card['backend']==dict(python=sys.version,numpy=np.__version__),'literal audit Python/NumPy backend')
    out=RAW/'endpoints-001-independent-review-001';out.mkdir(exist_ok=False);(out/'executed-audit.py').write_bytes(Path(__file__).read_bytes());(out/'card.json').write_bytes(args.card.read_bytes())
    a=Audit();report=dict(passed=False,scope=__doc__,source_sha256=sha(Path(__file__)),card_sha256=args.card_sha256,rows=[])
    try:
        rec=pinned(ATTEMPT/'receipt.json',RECEIPT);require(rec['source_commit']==SOURCE and rec['plan_sha256']==PLAN and rec['exit_code']==0 and rec['timed_out'] is False and rec['worker_termination_verified'] is True and not rec['uncertain_ownership'],'complete original endpoint attempt')
        a.inventory_check(rec);manifest=read(ATTEMPT/'manifest.json');require(manifest['source_commit']==SOURCE and manifest['plan_sha256']==PLAN,'original source and plan')
        expected_command=['/data2/neural-shrimp/.venv/bin/python','-I','-B','-u',str(ATTEMPT/'source/codex/cusp-crosscheck/endpoint_runtime.py'),'--plan',str(ATTEMPT/'source/codex/cusp-crosscheck/endpoints-001-plan.json'),'--output',str(ATTEMPT/'run')]
        require(manifest['command']==expected_command,'original exact worker command')
        auth=pinned(Path(card['authority']['root'])/'report.json',card['authority']['report_sha256']);closure=pinned(Path(card['authority']['root'])/'closure.json',card['authority']['closure_sha256'])
        am=pinned(Path(card['authority']['root'])/'manifest.json',card['authority']['manifest_sha256'])
        authority_files={row['path']:row['sha256'] for row in am['files']}
        require(authority_files['report.json']==card['authority']['report_sha256'] and authority_files['closure.json']==card['authority']['closure_sha256'] and authority_files['original-receipt.json']==RECEIPT,'sealed authority archive association')
        require(closure['original_worker']['command']==expected_command,'closure original command')
        require(auth['passed'] is True and auth['receipt_sha256']==RECEIPT and auth['source_commit']==SOURCE and auth['artifacts_checked']==1191 and auth['source_git_blobs_checked']==11,'sealed source/closure audit')
        closed_now(closure,manifest)
        source_names={r['path'] for r in manifest['source']}
        authority_sources={r['path']:r['sha256'] for r in auth['source_files']}
        require(len(manifest['source'])==len(source_names)==len(auth['source_files'])==len(authority_sources)==11 and source_names==set(authority_sources),'complete captured and previously Git-verified source sets')
        for row in manifest['source']:
            a.guard();key='source/'+row['path'];path=ATTEMPT/key
            require(key in a.inventory and sha(path)==row['sha256']==a.inventory[key]['sha256']==authority_sources[row['path']],'exact captured source bytes and inherited Git authority '+row['path'])
        planpath=ATTEMPT/'run/original-plan.json';require(sha(planpath)==PLAN,'original reviewed plan');plan=read(planpath)
        require(plan['indices']==[13,14] and plan['tolerances']==T and plan['fd_steps']==STEPS and plan['parameter_steps']==ALPHA_STEPS,'unchanged gates/steps')
        require(all(plan['source_sha256'].get(row['path'],row['sha256'])==row['sha256'] for row in manifest['source']),'original plan/source agreement')
        binding=read(ATTEMPT/'run/input-binding.json');require(binding['plan']==plan and binding['source_commit']==SOURCE and binding['plan_sha256']==PLAN,'original input binding')
        pins=card['inputs'];candidate=pinned(Path(pins['candidate_root'])/'manifest.json',pins['candidate_manifest_sha256']);ancestry=pinned(Path(pins['ancestry_root'])/'manifest.json',pins['ancestry_manifest_sha256'])
        require(binding['authority_sha256']==pins['candidate_manifest_sha256'] and binding['ancestry_manifest_sha256']==pins['ancestry_manifest_sha256'],'original authority identities')
        require(set(binding['inputs'])=={'states.npy','curve.json'} and set(binding['selected'])=={'13','14'},'exact candidate and row inventory')
        for index,(temperature,alpha) in PARAMS.items():
            selected=binding['selected'][str(index)]
            require(selected['log10_temperature']==temperature and selected['alpha']==alpha and selected['theta'][0]==temperature and selected['theta'][1]/10==alpha,'literal endpoint parameter units')
        for name,expected in binding['inputs'].items():
            path=Path(pins['candidate_root'])/name;entry=next(r for r in candidate['files'] if r['name']==name)
            require(sha(path)==entry['sha256']==expected and path.stat().st_size==entry['bytes'] and not path.stat().st_mode&0o222,'original candidate source')
        for entry in ancestry['files']:
            path=Path(pins['ancestry_root'])/entry['name'];require(sha(path)==entry['sha256'] and path.stat().st_size==entry['bytes'] and not path.stat().st_mode&0o222,'original map ancestry file')
        require(binding['ancestry']['declared_map']==dict(model='qwen0.5b',layer=12,prompt='open',opener='Hello!',window=1,vectors='llm-phase-map/vectors-l12-p1b',concept='hostility-vs-kindness') and len(binding['ancestry']['prefix_ids'])==37,'declared map and prefix convention')
        original_states=np.load(Path(pins['candidate_root'])/'states.npy',allow_pickle=False,mmap_mode='r');require(original_states.shape==(40,896) and original_states.dtype==np.float64,'original candidate container')
        require(sha(pins['vector_path'])==pins['vector_sha256']==binding['ancestry']['vector_sha256'],'steering vector input')
        with np.load(pins['vector_path'],allow_pickle=False) as z:vector=np.asarray(z['hostility-vs-kindness'],dtype=np.float64).copy()
        require(vector.shape==(896,) and vector.dtype==np.float64 and np.isfinite(vector).all(),'full finite vector')
        profile=pinned(Path(plan['profile']['root'])/'receipt.json',plan['profile']['receipt_sha256'])
        require(profile['source_commit']==plan['profile']['source_commit'] and profile['plan_sha256']==plan['profile']['plan_sha256'] and profile['exit_code']==0 and profile['timed_out'] is False and profile['worker_termination_verified'] is True and not profile['uncertain_ownership'],'completed original profile only')
        pa=pinned(Path(card['profile_audit']['path']),card['profile_audit']['sha256']);require(pa['status']=='saved_profile_evidence_passed' and pa['source_commit']==plan['profile']['source_commit'] and pa['plan_sha256']==plan['profile']['plan_sha256'] and pa['receipt_sha256']==plan['profile']['receipt_sha256'] and pa['complete_coordinate_directions']==896 and pa['matrix_exactly_reassembles_native_columns'] is True and pa['original_state_exact_match'] is True and pa['former_groups_and_sessions_absent'] is True and pa['tool_session_69207_reaped_exit0_by_coordinator'] is True and pa['cusp_qualified'] is False,'accepted original matrix authority')
        require(read(ATTEMPT/'run/backend.json')==pa['backend'],'same profile/evaluator backend')
        reuse=read(ATTEMPT/'run/profile-reuse.json');require(reuse['receipt_sha256']==pa['receipt_sha256'] and reuse['matrix_reused'] is True and reuse['all_896_column_inputs_returns_verified'] is True,'actual recorded profile reuse')
        for index in (13,14):report['rows'].append(endpoint(a,index,original_states[index],vector,pa['dense_matrix_sha256']))
        require(a.array_references=={name for name in a.inventory if name.endswith('.npy')},'every original native array bound')
        status=read(ATTEMPT/'run/status.json');require(status['status']=='endpoint_check_completed' and status['computational_completed'] is True and status['source_commit']==SOURCE,'completed worker status')
        require(status['rows']==[r['original_summary'] for r in report['rows']] and status['all_endpoints_qualified']==all(r['numerical_qualified'] for r in report['rows']) and status['cusp_qualified'] is False and status['certificate_passed'] is False,'complete original row/claim status')
        a.inventory_check(rec);closed_now(closure,manifest);a.guard()
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=1024**3,'final audit RSS')
        report.update(passed=True,receipt_sha256=RECEIPT,source_commit=SOURCE,plan_sha256=PLAN,artifacts_checked=len(a.inventory),captured_source_hashes_verified=11,fresh_git_queries=0,
            inherited_git_blob_verification=dict(report_path=str(Path(card['authority']['root'])/'report.json'),report_sha256=card['authority']['report_sha256'],source_commit=SOURCE,source_git_blobs_checked=11,source_files=auth['source_files']),
            arrays_checked=len(a.array_references),exact_array_associations=a.exact_arrays,scalar_checks=a.checks,nonzero_scalar_differences=a.deltas,
            original_worker_status=status,original_receipt_status={k:v for k,v in rec.items() if k!='artifacts'},closure=closure,
            interpretation='Two original numerical endpoints and any supported coefficient-sign bracket only; no located/nondegenerate cusp or interval certificate.',
            inherited_limitation='Full dense eigenvector matrix was not retained; its saved all-eigenpair residuals are source-bound diagnostics. Retained left/right eigenpair equations and homological solutions were replayed without solving.',
            execution_attribution="Inference's original execution report and reaped-session evidence, relayed to Root; this audit only rechecks saved authority and current absence.",
            inherited_input_limitations=binding['ancestry']['limitations'],
            audit_backend=dict(python=sys.version,numpy=np.__version__,nice=os.getpriority(os.PRIO_PROCESS,0),thread_environment={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}),
            model_calls=0,ad_or_jvp_calls=0,eigensolves=0,inverses=0,profile_arrays_reaudited=False,certificate_passed=False,cusp_qualified=False)
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc(),scalar_checks=a.checks,nonzero_scalar_differences=a.deltas);raise
    finally:
        report.update(seconds=time.monotonic()-a.began,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        data=(json.dumps(report,indent=2,allow_nan=False)+'\n').encode();require(len(data)<7*1024**2,'audit output reserve')
        (out/'report.json').write_bytes(data)
        for path in out.iterdir():path.chmod(0o444)
        print(json.dumps(dict(passed=report['passed'],path=str(out/'report.json'),sha256=sha(out/'report.json'),seconds=report['seconds'],max_rss_kib=report['max_rss_kib'])))

if __name__=='__main__':main()

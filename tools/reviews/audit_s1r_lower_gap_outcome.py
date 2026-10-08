"""Independent saved-only S1R lower-gap arithmetic audit. No model or decomposition calls."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS:os.environ[name]='1'
import argparse,hashlib,json,math,platform,re,signal,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RAW=Path('/data2/neural-shrimp-data/explore/s1r-lower-gap/attempt-001')
LAUNCH=RAW.with_name('attempt-001-launcher')
PRODUCER='5cecd5f0785f119a2631ae13ac375f9d044b2f26'
PLAN='docs/plans/s1r-lower-gap-inputs.json'
PRODUCER_FILES=('src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','src/neural_shrimp/period2_branch.py','src/neural_shrimp/period2_branch_io.py','tools/audit_s1r_branch_outcome.py','src/neural_shrimp/s1r_gap_diagnostic.py','tools/diagnose_s1r_lower_gap.py','tools/launch_s1r_lower_gap.py',PLAN,'docs/plans/s1r-lower-gap-next.md','docs/plans/s1r-lower-gap-implementation.md')
AUDIT_FILES=('tools/reviews/audit_s1r_lower_gap_outcome.py','tools/reviews/launch_s1r_lower_gap_audit.py','tests/reviews/test_s1r_lower_gap_outcome_audit.py','docs/reviews/s1r-lower-gap-outcome-audit-card.md')
ROLES=('predecessor_native_phase_0','predecessor_native_phase_1','last_native_phase_0','last_native_phase_1','nearest_failure_seed','nearest_failure_state')
COORDS=('raw_A','raw_B','scaled_A','scaled_B')
LIMITS=dict(fixed_states=6,maps=12,jacobians=12,svds=24,internal_seconds=15.,external_seconds=20,kill_grace_seconds=5,store_bytes=7*1024**2,launcher_bytes=1024**2)
TOL=dict(agreement_absolute=2e-11,closure_absolute=1e-9,closure_relative=1e-8,primitive_separation=1e-6,rank_low=1e-12,rank_high=1e-10,svd_residual=1e-10,tangent_residual=1e-10,soft_isolation_ratio=.1,orientation_cosine_min=1e-8,nearzero_residual=1e-12,material_forcing=1e-6,backward_error=1e-12,h_scale=.000125)
MIRROR_RTOL=2e-11
MIRROR_ATOL=5e-14


def require(value,message):
    if not value:raise ValueError(message)
def canonical(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def digest(data):return hashlib.sha256(data).hexdigest()
def entry(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return dict(bytes=path.stat().st_size,sha256=h.hexdigest())
def read(path):return json.loads(path.read_bytes())
def exact(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes(order='C')==b.tobytes(order='C')
def array(a,shape=None,finite=True):
    a=np.asarray(a);require(a.dtype==np.float64 and (shape is None or a.shape==shape),'float64 array shape')
    if finite:require(np.isfinite(a).all(),'finite array')
    return a
def norm(a):
    a=array(a);value=math.sqrt(math.fsum(float(x)*float(x) for x in a.flat));require(math.isfinite(value),'finite Euclidean/Frobenius norm');return value
def dot(a,b):
    a,b=array(a),array(b);require(a.shape==b.shape,'dot shapes');v=math.fsum(float(x)*float(y) for x,y in zip(a.flat,b.flat));require(math.isfinite(v),'finite dot');return v
def scalar(actual,expected,label='scalar'):
    if expected is None:require(actual is None,label+' abstention');return
    require(type(actual) in (float,int) and type(expected) in (float,int) and math.isfinite(actual) and math.isfinite(expected),'finite '+label)
    require(math.isclose(actual,expected,rel_tol=MIRROR_RTOL,abs_tol=MIRROR_ATOL),label+' disagreement')
def close_array(actual,expected,label='array'):
    a,b=array(actual),array(expected);require(a.shape==b.shape,label+' shape')
    require(bool(np.all(np.abs(a-b)<=MIRROR_ATOL+MIRROR_RTOL*np.abs(b))),label+' disagreement')
def compare(actual,expected,label='value'):
    if isinstance(expected,np.ndarray):close_array(actual,expected,label)
    elif isinstance(expected,dict):
        require(isinstance(actual,dict) and set(expected)<=set(actual),label+' keys')
        for k,v in expected.items():compare(actual[k],v,label+'.'+k)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),label+' length')
        for i,v in enumerate(expected):compare(actual[i],v,label+'.'+str(i))
    elif type(expected) in (float,int):scalar(actual,expected,label)
    else:require(type(actual)==type(expected) and actual==expected,label+' exact decision')


def directory(folder):
    require(folder.is_dir() and not folder.is_symlink(),'regular folder')
    require(all(p.is_file() and not p.is_symlink() for p in folder.iterdir()),'regular complete files')
    return {p.name:entry(p) for p in sorted(folder.iterdir())}
def inventory(folder,pin):
    files=directory(folder);require(files['manifest.json']['sha256']==pin,'external manifest pin')
    saved=read(folder/'manifest.json');members=saved['files']
    require(members=={k:v for k,v in files.items() if k!='manifest.json'},'complete inventory')
    return files

def decode_record(folder,name):
    require(Path(name).name==name and name.endswith('.json'),'safe record name')
    record=read(folder/name);archives={};used=set()
    if 'arrays_file' in record:
        af=record['arrays_file'];require(Path(af).name==af and af==name[:-5]+'.npz','safe matching record archive')
        require(entry(folder/af)['sha256']==record['arrays_sha256'],'record archive hash')
        with np.load(folder/af,allow_pickle=False) as z:
            require(len(z.files)==len(set(z.files)),'unique archive members');archives={k:z[k] for k in z.files}
    def decode(value):
        if isinstance(value,dict) and 'array' in value:
            require(set(value)=={'array','dtype','shape','order','sha256'},'complete array descriptor')
            key=value['array'];require(key in archives,'referenced archive member');used.add(key);a=archives[key]
            require(a.dtype.str==value['dtype'] and list(a.shape)==value['shape'] and value['order']=='C' and digest(np.ascontiguousarray(a).tobytes())==value['sha256'],'array byte identity')
            return a
        if isinstance(value,dict):return {k:decode(v) for k,v in value.items()}
        if isinstance(value,list):return [decode(v) for v in value]
        return value
    payload=decode(record['payload']);require(used==set(archives),'exact referenced archive inventory')
    return record,payload

def expected_sequence():
    kinds=['bound_inputs','saved_line_search_reduction']
    for role in ROLES:
        for phase in ('first','second'):kinds+=['operation_started',role+'_'+phase+'_phase_analytic_return',role+'_'+phase+'_phase']
        kinds += [role+'_assembly',role+'_agreement']
        if role==ROLES[-1]:kinds+=['stalled_residual_agreement','saved_linear_equation']
        kinds.append(role+'_scales')
        for coord in COORDS:kinds+=['operation_started',role+'_'+coord+'_svd',role+'_'+coord+'_svd_checks',role+'_'+coord+'_summary']
        kinds.append(role+'_complete')
    return kinds+['phase_tangent_transport']

def records(folder,files,complete,result):
    journal=[] if 'journal.jsonl' not in files else [json.loads(line) for line in (folder/'journal.jsonl').read_text().splitlines()]
    expected=expected_sequence();require(len(journal)<=len(expected),'record count cap')
    values={};used=set();attempted=dict(maps=0,jacobians=0,svds=0);returned=attempted.copy();pending=None
    for i,row in enumerate(journal):
        kind=row['kind'];name=f'{i:05d}-{kind}.json'
        require(row==dict(sequence=i,kind=expected[i],file=name),'exact causal journal prefix')
        record,payload=decode_record(folder,name);require(record['kind']==kind and record['sequence']==i,'record journal mirror')
        used.add(name)
        if 'arrays_file' in record:used.add(record['arrays_file'])
        require(kind=='operation_started' or kind not in values,'unique stage')
        if kind=='operation_started':
            require(pending is None,'one pending operation')
            upcoming=expected[i+1];kinds=['svds'] if upcoming.endswith('_svd') else ['maps','jacobians']
            require(payload['kinds']==kinds,'exact operation type')
            for key in kinds:attempted[key]+=1;require(attempted[key]<=LIMITS[key],'attempt cap')
            require(payload['attempted']==attempted,'attempt counters')
            pending=kinds
        elif kind.endswith('_phase') or kind.endswith('_svd'):
            require(pending==(['svds'] if kind.endswith('_svd') else ['maps','jacobians']),'returned operation type')
            for key in pending:returned[key]+=1
            pending=None
        if kind!='operation_started':values[kind]=payload
    require(result['operation_attempted']==attempted,'attempted mirror')
    # Returned counters increment immediately before durable emit; an interrupted emit
    # may leave one return not journaled, but never more than the pending operation.
    claimed=result['operation_returned'];require(set(claimed)==set(returned),'return counter keys')
    delta={k:claimed[k]-returned[k] for k in returned}
    require(all(type(claimed[k]) is int and 0<=claimed[k]<=attempted[k] for k in claimed),'returned caps')
    allowed={k:int(pending is not None and k in pending) for k in returned}
    require(delta==dict.fromkeys(returned,0) or (pending is not None and delta==allowed),'durable/returned discrepancy boundary')
    extras=set(files)-used-{'runtime-start.json','input-plan.json','journal.jsonl','checkpoint.json','result.json','manifest.json'}
    require(all(re.fullmatch(r'\d{5}-[a-zA-Z0-9_]+\.(json|npz)(\.tmp)?',name) or name=='checkpoint.json.tmp' for name in extras),'only declared interrupted record artifacts')
    if complete:
        require(len(journal)==len(expected) and not extras and pending is None and attempted==returned==dict(maps=12,jacobians=12,svds=24),'complete record/counter inventory')
    if 'checkpoint.json' in files:
        check=read(folder/'checkpoint.json');last=None if not journal else journal[-1]
        require(check==dict(status=result['status'],latest=last),'terminal checkpoint')
    return values,dict(attempted=attempted,durably_returned=returned,claimed_returned=claimed,pending_operation=pending,uncommitted_record_files=sorted(extras),journal_records=len(journal))


def phase_check(theta,rates,analytic,phase,hidden=64):
    theta=array(theta);n=len(theta);rates=array(rates,(n,));g=array(analytic['gradient'],(n,));H=array(analytic['hessian'],(n,n))
    scalar(phase['loss'],analytic['loss'],'phase loss')
    require(exact(phase['theta'],theta),'phase input bytes')
    image=theta-rates*g;J=np.eye(n)-rates[:,None]*H
    partial=-np.r_[math.log(10.)*rates[:hidden],np.zeros(n-hidden)]*g
    for key,want in [('image',image),('jacobian',J),('parameter',partial)]:close_array(phase[key],want,key)
    return image,J,partial

def assembly_check(theta,rates,first_analytic,first,second_analytic,second,saved,hidden=64):
    f,J0,c0=phase_check(theta,rates,first_analytic,first,hidden)
    ff,J1,c1=phase_check(f,rates,second_analytic,second,hidden)
    M=J1@J0;values=dict(state=theta,map1=f,map2=ff,partial_h_F=c0,R=ff-theta,M=M,b=J1@c0+c1,A=M-np.eye(len(theta)))
    compare(saved,values,'assembly')
    # Store's assembly excludes J0/J1; retain references locally only, no fresh model call.
    return dict(**values,J0=J0,J1=J1)

def rank(s):
    s=array(s);require(s.ndim==1 and len(s)>1 and bool(np.all(s>=0)) and bool(np.all(s[:-1]>=s[1:])),'ordered nonnegative spectrum')
    high=1e-10*max(1.,float(s[0]));low=1e-12*max(1.,float(s[0]));middle=any(low<float(v)<=high for v in s)
    status='ambiguous_rank' if middle else 'full_row_rank_resolved' if float(s[-1])>high else 'numerically_rank_deficient'
    ratio=float(s[0])/float(s[-1]) if s[-1]>0 else None
    return dict(status=status,low=low,high=high,small_count=sum(float(x)<=low for x in s),resolved_count=sum(float(x)>high for x in s),sigma_max=float(s[0]),sigma_min=float(s[-1]),condition=ratio if ratio is not None and math.isfinite(ratio) else None,isolated_soft_direction=bool(s[-2]>high and s[-1]<=.1*s[-2]))

def svd_check(matrix,saved,checks):
    m,n=matrix.shape;require(n in (m,m+1),'square or once-augmented matrix')
    compare(saved['matrix'],matrix,'decomposition matrix')
    U=array(saved['U'],(m,m));s=array(saved['singular_values'],(m,));Vh=array(saved['Vh'],(n,n));V=Vh[:m].T
    scale=max(1.,norm(matrix));right=matrix@V-U*s;left=matrix.T@U-V*s
    rr=np.array([norm(right[:,j])/scale for j in range(m)]);lr=np.array([norm(left[:,j])/scale for j in range(m)])
    reconstructed=(U*s)@Vh[:m];reconstruction=norm(matrix-reconstructed)/scale
    orthogonal=max(norm(U.T@U-np.eye(m)),norm(Vh@Vh.T-np.eye(n)))
    values=dict(right_residual=rr,left_residual=lr,reconstruction=reconstruction,orthogonality=orthogonal,rank=rank(s))
    compare(checks,values,'SVD checks');require(max(float(rr.max()),float(lr.max()),reconstruction,orthogonal)<=1e-10,'fixed SVD residual/orthogonality gate')
    return values

def tangent(matrix,Vh,rank_info,columns,secant,raw):
    status=rank_info['status']
    if status!='full_row_rank_resolved':return dict(status='abstain_'+status,tangent=None)
    v=Vh[-1].copy();converted=columns*v;converted/=norm(converted)
    sr=norm(matrix@v)/max(1.,norm(matrix));rr=norm(raw@converted)/max(1.,norm(raw))
    if max(sr,rr)>1e-10:return dict(status='abstain_null_residual',tangent=None,raw_residual=rr,scaled_residual=sr)
    oriented='unoriented_no_secant';cosine=None
    if secant is not None:
        transformed=secant/columns;length=norm(transformed)
        if length==0:return dict(status='abstain_zero_secant',tangent=None)
        cosine=dot(v,transformed)/length
        if abs(cosine)<=1e-8:return dict(status='abstain_ambiguous_orientation',tangent=None,orientation_cosine=cosine)
        if cosine<0:v=-v;converted=-converted;cosine=-cosine
        oriented='toward_saved_predecessor_to_last_secant'
    return dict(status='local_null_tangent',tangent=converted,scaled_unit_tangent=v,orientation=oriented,secant_cosine=cosine,raw_residual=rr,scaled_residual=sr,h_component=float(converted[-1]),scope='kernel of local residual derivative; branch tangent only at a qualified root')

def geometry_check(assembly,d,secant,values,role):
    A,b,R=assembly['A'],assembly['b'],assembly['R'];n=len(b);d=array(d,(n,));require(np.all(d>=1),'fixed state scales')
    raw=np.column_stack((A,b));columns=np.r_[d,.000125]
    compare(values[role+'_scales'],dict(row_scale=1/d,state_column_scale=d,augmented_column_scale=columns,scaled_residual=R/d),'scales')
    matrices=dict(raw_A=A,raw_B=raw,scaled_A=(A*d[None,:])/d[:,None],scaled_B=(raw*columns[None,:])/d[:,None]);out={}
    for coord,matrix in matrices.items():
        name=role+'_'+coord;saved=values[name+'_svd'];checks=svd_check(matrix,saved,values[name+'_svd_checks']);summary=dict(rank=checks['rank'])
        if coord.endswith('_B'):
            col=np.ones(n+1) if coord=='raw_B' else columns
            summary['null_tangent']=tangent(matrix,saved['Vh'],checks['rank'],col,secant,raw)
        else:
            bcoord=b if coord=='raw_A' else b*.000125/d;coupling=dot(saved['U'][:,-1],bcoord);length=norm(bcoord)
            summary.update(smallest_left_parameter_coupling=coupling,normalized_left_parameter_coupling=abs(coupling)/length if length>0 else None,coupling_qualified=checks['rank']['isolated_soft_direction'])
        compare(values[name+'_summary'],summary,'geometry summary');out[coord]=summary
    return out


def secants(parent,child,parent_h,child_h,exchange):
    require(type(exchange) is int and exchange in (0,1),'phase exchange bit');out={}
    for phase in (0,1):
        target=phase^exchange;delta=np.r_[child[target]-parent[phase],child_h-parent_h]
        out['parent',phase]=delta;out['child',target]=delta
    return out

def agreement(assembly,state,node,phase):
    if phase is None:return dict(scope='nonclosed fixed-state residual geometry',qualified_endpoint=False)
    def scaled(a,b):return float(np.max(np.abs(a-b)/(1e-9+1e-8*np.maximum(np.abs(a),np.abs(b)))))
    closure=scaled(assembly['map2'],state);transition=scaled(assembly['map1'],node['pair'][1-phase]);pair=node['pair']
    sep=float(np.max(np.abs(pair[0]-pair[1])))/max(1.,float(np.max(np.abs(pair))))
    error=float(np.max(np.abs(assembly['M']-node['spectra']['phases'][phase]['matrix'])))
    return dict(qualified_endpoint=bool(closure<=1 and transition<=1 and sep>=1e-6 and error<=2e-11),closure_normalized=closure,transition_normalized=transition,primitive_separation=sep,saved_monodromy_max_absolute_error=error)

def linear_check(A,R,theta,step,saved_R):
    error=A@step+R;old=A@step+saved_R;length=norm(R);threshold=1e-12*max(1.,norm(theta));absolute=norm(error)
    denominator=norm(A)*norm(step)+length;difference=norm(R-saved_R)
    q=absolute/length if length>threshold else None;backward=absolute/denominator if denominator>0 else None
    status='abstain_nearzero_residual' if q is None else 'arithmetic_resolution_limited' if difference>1e-6*length else 'linear_equation_resolved' if q<=1e-6 and backward is not None and backward<=1e-12 else 'material_linear_equation_mismatch'
    return dict(error=error,error_using_saved_R=old,absolute_linear_residual=absolute,true_relative_forcing=q,nearzero_threshold=threshold,residual_norm=length,residual_reconstruction_difference=difference,normwise_backward_error_frobenius=backward,verdict=status)

def trial_check(iteration,trials):
    require(iteration['call']==473 and iteration['iteration']==3 and len(trials)==13,'fixed saved trial scope')
    theta,R,p=(array(iteration[k]) for k in ('state','residual','step'));initial=float(np.max(np.abs(R)));out=[]
    for i,row in enumerate(trials):
        require(row['call']==473 and row['iteration']==3 and row['halving']==i,'thirteen trial order')
        require(exact(row['state'],theta+2.**-i*p),'exact saved trial state')
        value=float(np.max(np.abs(array(row['residual'],R.shape))));accepted=value<initial
        require(type(row['accepted']) is bool and row['accepted']==accepted,'strict infinity-merit rule')
        out.append(dict(halving=i,alpha=2.**-i,residual_inf=value,ratio=value/initial if initial>0 else None,accepted=accepted))
    return out


def exact_tree(a,b,label='binding'):
    if isinstance(b,np.ndarray):require(exact(a,b),label+' array bytes')
    elif isinstance(b,dict):
        require(isinstance(a,dict) and set(a)==set(b),label+' dictionary')
        for k,v in b.items():exact_tree(a[k],v,label+'.'+str(k))
    elif isinstance(b,list):
        require(isinstance(a,list) and len(a)==len(b),label+' list')
        for i,v in enumerate(b):exact_tree(a[i],v,label+'.'+str(i))
    else:require(type(a)==type(b) and a==b,label+' scalar')

def source_check(files,commit):
    require(set(files)==set(PRODUCER_FILES),'complete producer import/card inventory')
    for name,identity in files.items():
        for revision in (PRODUCER,commit):
            data=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
            require(identity==dict(bytes=len(data),sha256=digest(data)),'frozen producer bytes '+name)


def original_inputs(plan):
    old=Path(plan['collection_root']);files=inventory(old,plan['collection_manifest']['sha256'])
    require(files['manifest.json']==plan['collection_manifest'],'original manifest identity')
    for name,pin in plan['collection_metadata'].items():require(files[name]==pin,'original metadata pin')
    runtime=read(old/'runtime-start.json');require(runtime['source_commit']==plan['producer_execution_commit'],'original execution source')
    for name,identity in runtime['sources'].items():
        for revision in (plan['producer_execution_commit'],plan['producer_reviewed_source']):
            data=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
            require(identity==dict(bytes=len(data),sha256=digest(data)),'original source bytes')
    decoded={};encoded={}
    for name,pin in plan['selected_records'].items():
        require(files[name]==pin['json'] and files[pin['arrays_file']]==pin['arrays'],'selected historical files')
        record,payload=decode_record(old,name);decoded[name]=payload;encoded[name]=record['payload']
    binding=decoded['00000-bound_inputs.json']
    require(binding['model']==dict(activation='tanh',bias=False,d=8,in_scale=.5,loss_scale=1.,out_scale=.125,ridge=.001,width=8),'S1R model identity')
    array(binding['x'],(72,8));array(binding['y'],(72,));slots=[]
    for spec,role in zip(plan['six_future_fixed_slots'],ROLES,strict=True):
        require(spec['role']==role,'fixed role order');node=decoded[spec['record']];raw=encoded[spec['record']]
        require(raw['params']==spec['params'] and raw['rates']==spec['rates'],'exact control descriptors')
        phase=spec.get('native_phase')
        if phase is not None:
            require(type(phase) is int and phase in (0,1) and node['accepted'] is True and raw['pair']==spec['pair'],'qualified native pair')
            state=node['pair'][phase]
        else:
            key='seed' if role.endswith('_seed') else 'state';require(raw['corrections'][0][key]==spec['state'],'failed-state descriptor');state=node['corrections'][0][key]
        slots.append(dict(role=role,state=array(state,(72,)),params=array(node['params'],(2,)),rates=array(node['rates'],(2,)),native_phase=phase,node=node))
    parent,last=slots[0]['node'],slots[2]['node'];edge=decoded['05002-edge.json'];failure=decoded['05267-edge.json']
    require(edge['status']=='accepted' and edge['parent']==parent['id'] and edge['direct']==last['id'],'saved accepted edge')
    require(failure['parent']==last['id'] and failure['direct']==slots[4]['node']['id'],'saved failed edge')
    pphase=edge['parent_phase'];lphase=edge['parent_match']['phase'];carried=failure['parent_phase']
    require(type(pphase) is int and type(lphase) is int and pphase in (0,1) and lphase in (0,1) and carried==lphase,'native carried phase binding')
    exchange=pphase^lphase;sec=secants(parent['pair'],last['pair'],float(parent['params'][0]),float(last['params'][0]),exchange)
    for i,slot in enumerate(slots):slot['secant']=sec[('parent' if i<2 else 'child'),slot['native_phase'] if i<4 else carried]
    equation=plan['saved_last_linear_equation'];iteration=decoded[equation['record']]
    require(iteration['call']==473 and iteration['iteration']==3 and exact(iteration['state'],slots[-1]['state']),'saved Newton state')
    for k in ('state','residual','step'):require(encoded[equation['record']][k]==equation[k],'saved equation descriptor')
    trials=[decoded[f'{i:05d}-line_search.json'] for i in range(5252,5265)]
    require(all(s['params'][1]==parent['params'][1] for s in slots),'fixed output control')
    return dict(slots=slots,iteration=iteration,trials=trials,model=binding['model'],identity=binding['identity'],x=binding['x'],y=binding['y'],scale=np.maximum(1.,np.abs(last['pair'][0])),phase_binding=dict(parent_phase=pphase,exchange=exchange,last_carried_phase=carried)),files


def validate_arithmetic(values,original,complete,result,plan_identity):
    summaries={};assemblies={};geometry={};failures=[]
    if 'bound_inputs' not in values:return dict(validated_roles=[],partial_validation='bound inputs not durably present')
    bound=values['bound_inputs'];require(bound['plan_file']=='input-plan.json' and bound['plan_identity']==plan_identity,'saved plan reference')
    for key in ('identity','model','x','y','scale','phase_binding'):exact_tree(bound[key],original[key],'bound '+key)
    expected_slots=[{k:s[k] for k in ('role','state','params','rates','native_phase','secant')} for s in original['slots']]
    exact_tree(bound['slots'],expected_slots,'six original slots');exact_tree(bound['saved_iteration'],original['iteration'],'saved iteration');exact_tree(bound['saved_trials'],original['trials'],'saved trials')
    trial_values=trial_check(original['iteration'],original['trials'])
    if 'saved_line_search_reduction' in values:compare(values['saved_line_search_reduction'],trial_values,'thirteen trial reduction');compare(result['saved_line_search'],trial_values,'result trial reduction')
    for slot in original['slots']:
        role=slot['role'];n=len(slot['state']);rates=np.r_[np.full(64,slot['rates'][0]),np.full(8,slot['rates'][1])]
        needed=[role+'_first_phase_analytic_return',role+'_first_phase',role+'_second_phase_analytic_return',role+'_second_phase',role+'_assembly']
        try:
            if role+'_first_phase' in values:phase_check(slot['state'],rates,values[needed[0]],values[needed[1]])
            if not all(key in values for key in needed):break
            assembly=assembly_check(slot['state'],rates,*[values[k] for k in needed[:4]],values[needed[4]])
            assemblies[role]=assembly;row=dict(assembly_checked=True)
            if role+'_agreement' in values:
                qualified=agreement(assembly,slot['state'],slot['node'],slot['native_phase']);compare(values[role+'_agreement'],qualified,'endpoint agreement');row['agreement']=qualified
                if slot['native_phase'] is not None:require(qualified['qualified_endpoint'],'fixed endpoint agreement gate')
            if role==ROLES[-1]:
                delta=float(np.max(np.abs(assembly['R']-original['iteration']['residual'])))
                if 'stalled_residual_agreement' in values:compare(values['stalled_residual_agreement'],dict(max_absolute_error=delta,passed=delta<=2e-11),'stalled residual agreement');require(delta<=2e-11,'stalled residual fixed gate')
                if 'saved_linear_equation' in values:
                    linear=linear_check(assembly['A'],assembly['R'],slot['state'],original['iteration']['step'],original['iteration']['residual'])
                    compare(values['saved_linear_equation'],linear,'saved Newton equation');compare(result['saved_linear_equation'],{k:v for k,v in linear.items() if not isinstance(v,np.ndarray)},'compact Newton equation');row['linear_equation']=linear
            completed_svd=[]
            # Validate every durably saved decomposition, including a partial role.
            d=original['scale'];raw=np.column_stack((assembly['A'],assembly['b']));col=np.r_[d,.000125]
            matrices=dict(raw_A=assembly['A'],raw_B=raw,scaled_A=assembly['A']*d[None,:]/d[:,None],scaled_B=raw*col[None,:]/d[:,None])
            for coord in COORDS:
                name=role+'_'+coord
                if name+'_svd_checks' in values:
                    if not all(role+'_'+c+'_summary' in values for c in COORDS):svd_check(matrices[coord],values[name+'_svd'],values[name+'_svd_checks'])
                    completed_svd.append(coord)
            row['checked_decompositions']=completed_svd
            if all(role+'_'+coord+'_summary' in values for coord in COORDS):
                geom=geometry_check(assembly,d,slot['secant'],values,role);geometry[role]=geom;row['geometry']=geom
            if role+'_complete' in values:
                require(role in geometry,'complete role all decompositions')
                expected=dict(coordinate_statuses={k:v['rank']['status'] for k,v in geometry[role].items()},tangent_scope='numerical qualified-endpoint branch tangent' if slot['native_phase'] is not None else 'local residual level-set tangent at nonclosed state',accepted_orbit=False,fold_verified=False)
                compare(values[role+'_complete'],expected,'role completion')
            summaries[role]=row
        except (ValueError,ArithmeticError,KeyError) as error:
            if complete:raise
            failures.append(dict(role=role,error=type(error).__name__+': '+str(error),scope='partial record unvalidated; no arithmetic consistency claim for this role'));break
    actual_completed=[role for role in ROLES if role+'_complete' in values]
    require(result['completed_roles']==actual_completed or (not complete and result['completed_roles']==actual_completed[:-1]),'completed-role result mirror/durable boundary')
    transport=[]
    if 'phase_tangent_transport' in values:
        for prefix in ('predecessor','last'):
            zero,one=prefix+'_native_phase_0',prefix+'_native_phase_1'
            for coord in ('raw_B','scaled_B'):
                t0=geometry[zero][coord]['null_tangent'];t1=geometry[one][coord]['null_tangent']
                row=dict(endpoint=prefix,coordinates=coord)
                if t0['tangent'] is None or t1['tangent'] is None:row['status']='abstain_unqualified_tangent'
                else:
                    a=assemblies[zero];v=t0['tangent'];prop=np.r_[a['J0']@v[:-1]+a['partial_h_F']*v[-1],v[-1]];length=norm(prop)
                    row.update(status='descriptive_phase_transport',propagated=prop,oriented_cosine=dot(prop,t1['tangent'])/length if length>0 else None)
                transport.append(row)
        compare(values['phase_tangent_transport'],transport,'native phase transport')
    if complete:require(len(summaries)==6 and len(geometry)==6 and len(transport)==4 and not failures,'full arithmetic scope')
    return dict(roles=summaries,validated_roles=list(summaries),phase_transport=transport,saved_trials=trial_values,unvalidated_partial_roles=failures,shared_primitive='Saved analytic gradient/Hessian values are reused; no independent model or derivative regeneration.')


def producer_provenance(pins,emit):
    launch_files=inventory(LAUNCH,pins['launcher_manifest_sha256']);ls=read(LAUNCH/'runtime-start.json');lr=read(LAUNCH/'receipt.json')
    require(set(launch_files)=={'launcher-source.py','runtime-start.json','stdout.log','receipt.json','manifest.json'},'launcher exact file names')
    for value in (ls,lr):require(value['source_commit']==pins['execution_commit'] and value['review_commit']==pins['source_review_commit'],'execution/review mirrors')
    source_check(ls['sources'],pins['execution_commit']);require(launch_files['launcher-source.py']==ls['sources']['tools/launch_s1r_lower_gap.py'],'archived producer launcher')
    require(ls['out']==str(RAW) and ls['launcher']==str(LAUNCH),'fixed producer paths')
    require(ls['command']==['timeout','--signal=TERM','--kill-after=5s','20s','nice','-n','10','/data2/neural-shrimp/.venv/bin/python','tools/diagnose_s1r_lower_gap.py','--out',str(RAW),'--review-commit',pins['source_review_commit']],'fixed producer command')
    require(ls['nice']>=10 and all(ls['thread_env'][k]=='1' for k in THREADS) and ls['gpu_used'] is False,'producer resources')
    for key,value in dict(external_seconds=20,kill_grace_seconds=5,supervisor_seconds=25.,rss_cap_bytes=2*1024**3,store_bytes=7*1024**2,launcher_bytes=1024**2,log_bytes=128*1024).items():require(ls[key]==value,'configured cap '+key)
    require(ls['group_pid']==lr['group_pid'] and (ls['group_pid'] is None or type(ls['group_pid']) is int and ls['group_pid']>0),'owned PID mirror')
    require(not lr['owned_live_pids_remaining'],'owned producer processes exited')
    require(type(lr['elapsed_seconds']) in (float,int) and math.isfinite(lr['elapsed_seconds']) and 0<=lr['elapsed_seconds']<=26.1,'actual wrapper duration')
    require(type(lr['max_group_rss_bytes']) is int and lr['max_group_rss_bytes']>=0,'actual group RSS')
    require(datetime.fromisoformat(ls['started_utc'])<=datetime.fromisoformat(lr['finished_utc']),'producer wrapper time order')
    raw_files=directory(RAW) if RAW.exists() else {};require(lr['raw_inventory']==raw_files and lr['raw_file_count']==len(raw_files),'launcher raw inventory mirror')
    size=sum(v['bytes'] for v in raw_files.values());lsize=sum(v['bytes'] for v in launch_files.values())
    require(size==lr['raw_bytes']<=7*1024**2 and lsize<=1024**2 and launch_files['stdout.log']['bytes']<=128*1024 and lr['within_store_cap'] is True,'actual storage bounds')
    rawpin=pins['raw_manifest_sha256']
    if rawpin is None:require('manifest.json' not in raw_files,'null pin only for absent manifest')
    else:require(raw_files['manifest.json']['sha256']==rawpin,'raw manifest external pin')
    if lr['raw_manifest_verified'] is not True:
        require(lr['successful'] is False,'partial cannot be successful')
        if rawpin is not None:
            try:valid=read(RAW/'manifest.json')['files']=={k:v for k,v in raw_files.items() if k!='manifest.json'}
            except (ValueError,KeyError,TypeError):valid=False
            require(not valid,'manifest rejection contradicts complete manifest')
        emit('provenance.json',dict(pins=pins,raw_inventory=raw_files,launcher_inventory=launch_files,launcher=lr))
        return None,None,None,dict(verdict='incomplete_inventory_only',producer_status='missing_or_malformed_scientific_manifest',arithmetic_qualified=False)
    require(inventory(RAW,rawpin)==raw_files,'raw complete inventory')
    rt=read(RAW/'runtime-start.json');result=read(RAW/'result.json')
    source_check(rt['sources'],pins['execution_commit']);require(rt['sources']==ls['sources'],'science launcher source inventory')
    for key in ('source_commit','review_commit','schema','sources','limits','tolerances','accepted_orbit','applied_correction','fold_verified'):require(result[key]==rt[key],'runtime/result '+key)
    require(rt['source_commit']==pins['execution_commit'] and rt['review_commit']==pins['source_review_commit'] and rt['limits']==LIMITS and rt['tolerances']==TOL,'fixed science settings')
    require(rt['status']=='running' and rt['completed_roles']==[] and rt['numpy']=='2.5.3' and rt['scipy']=='1.18.1','runtime start/libraries')
    require(rt['nice']>=10 and all(rt['thread_env'][k]=='1' for k in THREADS) and rt['gpu'] is False and rt['paid'] is False,'science resource scope')
    require(all(result[k] is False for k in ('accepted_orbit','applied_correction','fold_verified')),'no orbit/correction/fold acceptance')
    require(result['status'] in ('diagnostic_complete','failed_or_interrupted'),'recognized producer status');complete=result['status']=='diagnostic_complete'
    require(lr['scientific_complete'] is complete,'scientific completion mirror')
    duration=result['elapsed_seconds'];require(type(duration) in (float,int) and math.isfinite(duration) and 0<=duration<=lr['elapsed_seconds'],'actual worker duration')
    times=[datetime.fromisoformat(t) for t in (ls['started_utc'],rt['started_utc'],result['finished_utc'],lr['finished_utc'])];require(all(t.tzinfo is not None for t in times) and times==sorted(times),'ordered execution times')
    require(type(rt['pid']) is int and rt['pid']>0 and rt['pid']!=ls['group_pid'],'worker PID')
    seen=lr['observed_owned_pids'];require(seen==sorted(set(seen)) and all(type(pid) is int and pid>0 for pid in seen),'owned PID set')
    if complete:require(lr['successful'] is True and lr['process_exit']==0 and lr['failure'] is None and not lr['log_truncated'] and rt['pid'] in seen and lr['max_group_rss_bytes']<=2*1024**3,'completed producer execution')
    else:require(lr['successful'] is False,'incomplete producer cannot succeed')
    emit('provenance.json',dict(pins=pins,raw_inventory=raw_files,launcher_inventory=launch_files,launcher=lr,producer_runtime=rt,worker_seconds=duration,record_seconds_over_cooperative_limit=max(0.,duration-15.)))
    return raw_files,result,complete,None


def audit(pins,emit):
    raw_files,result,complete,early=producer_provenance(pins,emit)
    if early is not None:return early
    values,counters=records(RAW,raw_files,complete,result);emit('record-audit.json',counters)
    plan=json.loads(subprocess.check_output(['git','show',PRODUCER+':'+PLAN],cwd=ROOT))
    if 'input-plan.json' not in raw_files:
        require(not complete and not values,'no records before saved input plan')
        return dict(verdict='incomplete_provenance_only',producer_status=result['status'],arithmetic_qualified=False,producer_error=result.get('error'))
    require(read(RAW/'input-plan.json')==plan,'exact frozen input plan')
    original,files=original_inputs(plan);emit('historical-inventory.json',dict(root=plan['collection_root'],manifest_sha256=plan['collection_manifest']['sha256'],files=files))
    checks=validate_arithmetic(values,original,complete,result,entry(RAW/'input-plan.json'));emit('arithmetic-audit.json',checks)
    return dict(verdict='saved_outcome_audit_pass' if complete else 'incomplete_saved_prefix_audited',producer_status=result['status'],arithmetic_qualified=complete,producer_error=result.get('error'),completed_roles=result['completed_roles'],operation_counts=counters,checks=checks,scope='Saved algebra and provenance only. No derivative/model/JVP/SVD/solve/correction; no fold, ancestry or utility conclusion.')


def serializable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serializable(v) for v in value]
    return value

def run(out,pins_path,pins_sha,source):
    require(os.getpriority(os.PRIO_PROCESS,0)>=10 and not out.exists(),'nice10 exclusive output')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==source,'exact audit source')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean audit checkout')
    sources={}
    for name in AUDIT_FILES:
        b=(ROOT/name).read_bytes();require(b==subprocess.check_output(['git','show',source+':'+name],cwd=ROOT),'committed audit file');sources[name]=dict(bytes=len(b),sha256=digest(b))
    require(entry(pins_path)['sha256']==pins_sha,'external metadata hash');pins=read(pins_path)
    require(set(pins)=={'execution_commit','source_review_commit','raw_manifest_sha256','launcher_manifest_sha256'},'complete pin schema')
    for key,value in pins.items():
        if key=='raw_manifest_sha256' and value is None:continue
        require(isinstance(value,str) and re.fullmatch('[0-9a-f]{'+('40' if key.endswith('commit') else '64')+'}',value),'full metadata pin')
    for key in ('execution_commit','source_review_commit'):subprocess.run(['git','cat-file','-e',pins[key]+'^{commit}'],cwd=ROOT,check=True)
    out.mkdir(parents=True,exist_ok=False)
    def emit(name,value,final=False):
        require(Path(name).name==name,'safe output name');data=canonical(serializable(value));limit=(7 if final else 6)*1024**2
        require(sum(p.stat().st_size for p in out.iterdir())+len(data)<=limit,'audit storage limit')
        with (out/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        fd=os.open(out,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    started=time.monotonic();record=dict(status='running',audit_source_commit=source,audit_source_inventory=sources,pins=pins,pins_file=entry(pins_path),pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),thread_env={k:os.environ[k] for k in THREADS},numpy=np.__version__,python=platform.python_version(),started_utc=datetime.now(timezone.utc).isoformat(),gpu=False,paid=False,new_model_evaluations=0,new_derivatives=0,new_svds=0,new_solves=0)
    emit('runtime-start.json',record)
    def stop(*_):raise TimeoutError('15-second saved lower-gap audit cap')
    old_alarm=signal.signal(signal.SIGALRM,stop);old_term=signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,15.)
    try:record.update(audit(pins,emit),status='finished')
    except BaseException as error:record.update(status='failed',verdict='audit_failed',error=type(error).__name__+': '+str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_alarm);signal.signal(signal.SIGTERM,old_term)
        record.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        emit('result.json',record,True);emit('manifest.json',dict(files={p.name:entry(p) for p in sorted(out.iterdir())}),True)
        for p in out.iterdir():p.chmod(0o444)
        out.chmod(0o555)
    print(json.dumps({k:record.get(k) for k in ('status','verdict','error','elapsed_seconds','producer_status','arithmetic_qualified')},sort_keys=True))
    return 0 if record['status']=='finished' else 2

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--pins',type=Path,required=True);parser.add_argument('--pins-sha256',required=True);parser.add_argument('--source-commit',required=True);args=parser.parse_args();raise SystemExit(run(args.out,args.pins,args.pins_sha256,args.source_commit))

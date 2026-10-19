"""Task34 cascade-P3 conditional assembly core. No certificate release.

The enclosing reviewed run card must pin this source, the input plan and all
external review/closure receipts. This module performs no model evaluation.
"""
import json,hashlib,sys,shlex,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[1]/'codex/certify-llm')]
import numpy as np
from flint import ctx,arb,arb_mat
import point_bridge as points
import derivative_bridge as derivatives
import return_operational as returns
import phase_domain as domain
import derivative_coverage as coverage
import shooting_gates as gates
import shooting_certificate as certificate
from cache_io import unpack

def same(a,b,label):
    if domain.canonical_hash(a)!=domain.canonical_hash(b):raise ValueError(label+' differs')
def pinned(e):
    p=Path(e['path'])
    if p.is_symlink() or not p.is_file() or points.sha(p)!=e['sha256']:raise ValueError('pinned original bytes')
    return p

def original_pin(e):
    """Streaming original-byte pin, with real path and optional exact size."""
    p=Path(e['path'])
    if p.is_symlink() or not p.is_file() or p.resolve()!=p or returns.sha(p)!=e['sha256']:
        raise ValueError('original closure bytes differ')
    if 'bytes'in e and (type(e['bytes'])is not int or p.stat().st_size!=e['bytes']):
        raise ValueError('original closure byte count differs')
    return p


def authenticate_returns(plan,repository):
    """Authenticate the closed receiver output, not a producer summary."""
    ref=plan['operational_return']
    if Path(ref['path'])!=returns.RECEIVER/'operational.json':
        raise ValueError('exact Task34 receiver002 output required')
    report=returns.read(original_pin(ref))
    for key,value in [('schema','codex-task34-original-return-admission-v1'),
        ('status','all_42_original_returns_operationally_admitted'),('certificate_passed',False),
        ('packet_sha256',returns.PACKET_SHA),('scientific_source',returns.SCI),
        ('unique_phase_column_assignments',2688),('complete_signed_endpoint_coverage',False)]:
        same(report[key],value,'closed operational receiver '+key)
    packet_path=Path(repository)/returns.PACKET_REL
    if returns.sha(packet_path)!=returns.PACKET_SHA:raise ValueError('independent dispatch packet')
    packet=returns.read(packet_path);assigned=returns.assignments(packet)
    rows=report['attempts']
    if type(rows)is not list or [r['attempt']for r in rows]!=[r['attempt']for r in assigned]:
        raise ValueError('complete ordered receiver attempt allocation')
    for row,expected in zip(rows,assigned):
        for key,value in [('period',3),('phase_index',expected['phase']),
            ('start_column',expected['start_column']),('source_commit',returns.SCI)]:
            same(row[key],value,'receiver assignment '+key)
        same(row['operational']['source_commit'],returns.SCI,'receiver original source')
        same(row['original_archive_receipt_sha256'],row['operational']['receipt_sha256'],'receiver original receipt hash')
    pods=report['pods']
    if type(pods)is not list or [p['pod']for p in pods]!=[c['pod']for c in returns.CONFIGS]:
        raise ValueError('complete original pod allocation')
    for i,pod in enumerate(pods):
        same(pod['attempts'],[r['attempt']for r in assigned[5*i:5*i+5]],'receiver pod assignment')
    same(plan['campaign_source'],returns.SCI,'independent campaign source')
    return dict(report=report,packet=packet,rows={r['attempt']:r for r in rows},
        pods={p['pod']:p for p in pods},checked_pods={})


def authenticate_pod(pod_name,context):
    """Join each original result/outer archive/termination/preflight once."""
    if pod_name in context['checked_pods']:return context['checked_pods'][pod_name]
    cfgs=[c for c in returns.CONFIGS if c['pod']==pod_name]
    if len(cfgs)!=1:raise ValueError('unknown original pod')
    cfg=cfgs[0];pod=context['pods'][pod_name];base=returns.DATA/'runpod/soft64'/cfg['stamp']
    folder=returns.RECEIVER/pod_name/'explore/llm-soft-loop/pod64/certpack'/pod_name
    for key,path in [('archive',base/'results.tgz'),('result',base/'result.json'),
        ('orchestrator_log',returns.DATA/'runpod'/('T34C-P3-'+pod_name[-1].upper()+'-orchestrator.log')),
        ('preflight',folder/'preflight.json'),('prepare',folder/'prepare.json')]:
        if Path(pod[key]['path'])!=path:raise ValueError('original pod path assignment '+key)
    same(pod['archive']['sha256'],cfg['sha256'],'original outer archive pin')
    original_pin(pod['archive'])
    result=returns.read(original_pin(pod['result']))
    pod_id=returns.check_result(result,cfg,pod['attempts'],base)
    same(pod['pod_id'],pod_id,'original pod identity');same(pod['outer_source'],cfg['outer_commit'],'original outer source')
    same(pod['scientific_source'],returns.SCI,'original scientific source')
    termination=returns.check_termination(original_pin(pod['orchestrator_log']).read_bytes(),pod_id)
    same(pod['termination'],termination,'original matched termination line')
    preflight=returns.check_preflight(returns.read(original_pin(pod['preflight'])),pod['attempts'],context['packet'])
    same(pod['preflight_admission'],preflight,'original effective-capacity admission')
    preparation=returns.read(original_pin(pod['prepare']))
    if (preparation.get('step')!='prepare' or preparation.get('ok')is not True
        or preparation.get('hash_problems')!=[] or preparation.get('copied_upstream')!=len(context['packet']['upstream_files'])):
        raise ValueError('original preparation failed')
    answer=dict(pod=pod,pod_id=pod_id,folder=folder,termination=termination)
    context['checked_pods'][pod_name]=answer
    return answer


def authenticate_closure(ref,context):
    """Original Task34 lifecycle/archive/receipt join, with no fabricated summary."""
    path=Path(ref['attempt']);name=path.name
    if path!=returns.OWNED/name or path.resolve()!=path:raise ValueError('original owned shard path')
    if name not in context['rows']:raise ValueError('shard absent from authenticated receiver')
    row=context['rows'][name]
    for key,expected in [('receipt_sha256',row['original_archive_receipt_sha256']),
        ('source_commit',returns.SCI),('phase_index',row['phase_index']),('start_column',row['start_column'])]:
        same(ref[key],expected,'assembly original assignment '+key)
    candidates=[p['pod']for p in context['pods'].values()if name in p['attempts']]
    if len(candidates)!=1:raise ValueError('unique original pod assignment')
    authority=authenticate_pod(candidates[0],context);pod=authority['pod'];folder=authority['folder']
    same(row['pod_id'],authority['pod_id'],'receiver successful pod')
    same(row['whole_archive_sha256'],pod['archive']['sha256'],'receiver outer archive')
    same(row['result'],pod['result'],'receiver original result')
    same(row['orchestrator_log'],pod['orchestrator_log'],'receiver original log')
    same(row['termination_record'],authority['termination'],'receiver termination join')
    same(row['termination_record_line'],authority['termination']['line_number'],'receiver termination line')
    for key,nested in [('outer_closure','outer_closure'),('pack','pack'),('result','result'),('termination_record','orchestrator_log')]:
        same(row[key+'_path'],row[nested]['path'],'receiver flat path '+key)
        same(row[key+'_sha256'],row[nested]['sha256'],'receiver flat hash '+key)
    if Path(row['outer_closure']['path'])!=folder/name/'run.json' or Path(row['pack']['path'])!=folder/('pack-'+name+'.json'):
        raise ValueError('original run/pack path assignment')
    closure=returns.check_run(returns.read(original_pin(row['outer_closure'])),name)
    same(row['closure_admission'],closure,'receiver original closure')
    pack=returns.read(original_pin(row['pack']));native=folder/(name+'.tar')
    if native.is_symlink() or not native.is_file() or native.resolve()!=native:raise ValueError('original native archive required')
    native_hash=returns.check_pack(pack,name,native)
    same(native_hash,row['archive_sha256'],'receiver original native archive')
    prefix=returns.PurePosixPath('codex/certificate-ladder')/name
    with tarfile.open(native,'r:')as t:
        members,_=returns.scan_tar(t,prefix=prefix)
        receipts=[m for m,p in members if p==prefix/'receipt.json']
        if len(receipts)!=1 or not receipts[0].isfile() or receipts[0].size>16*1024**2:
            raise ValueError('original unique native receipt member')
        receipt_bytes=t.extractfile(receipts[0]).read()
    same(hashlib.sha256(receipt_bytes).hexdigest(),ref['receipt_sha256'],'original tar receipt bytes')
    native_receipt=returns.parse(receipt_bytes)
    if (native_receipt.get('source_commit')!=returns.SCI or type(native_receipt.get('exit_code'))is not int
        or native_receipt['exit_code']!=0 or native_receipt.get('timed_out')is not False):
        raise ValueError('original native receipt not successful')
    original_pin(dict(path=str(path/'receipt.json'),sha256=ref['receipt_sha256']))
    return dict(attempt=name,source_commit=returns.SCI,receipt_sha256=ref['receipt_sha256'],
        original_native_archive_sha256=native_hash,original_outer_archive_sha256=pod['archive']['sha256'],
        outer_closure_sha256=row['outer_closure']['sha256'],pack_sha256=row['pack']['sha256'],
        result_sha256=pod['result']['sha256'],termination_log_sha256=pod['orchestrator_log']['sha256'],
        termination_line=authority['termination'],producer_termination_reported=True,
        original_native_receipt_bytes_joined=True,certificate_passed=False)

REQUIRED_SOURCES=set(derivatives.producer.SOURCE_PATHS)|{
    'codex/certificate-ladder-task34/assembly_core.py',
    'codex/certificate-ladder-task34/point_bridge.py',
    'codex/certificate-ladder-task34/derivative_bridge.py',
    'codex/certificate-ladder-task34/return_operational.py',
    'codex/certificate-ladder-task34/returned_shard.py',
    'codex/certify-llm/shooting_certificate.py'}

def authenticate_sources(plan,repository):
    if set(plan['assembly_source_sha256'])!=REQUIRED_SOURCES:
        raise ValueError('complete assembly source set')
    for name,h in plan['assembly_source_sha256'].items():
        if points.sha(Path(repository)/name)!=h:raise ValueError('assembly source pin')
    if set(plan['reviews'])!={'point_evaluator','derivative_evaluator','map_regularity','gate_mathematics'}:
        raise ValueError('review scope')
    for ref in plan['reviews'].values():pinned(ref)
    binding=json.loads(pinned(plan['review_binding']).read_text())
    pinned(binding['historical_evaluator_review'])
    for key in ('map','derivative_source_sha256','reviews','cell','campaign_source','operational_return'):
        same(binding[key],plan[key],'review binding '+key)
    same(binding['assembly_source_sha256'],plan['assembly_source_sha256'],'reviewed assembly source')
    same(binding['point_plan'],plan['point_plan'],'reviewed point authority')
    if binding['status']!='source_reviewed_conditional_enclosures':
        raise ValueError('source review disposition')
    return binding

def exact_matrix(a):
    if a.shape!=(896,896) or not np.isfinite(a).all():raise ValueError('full finite proposal')
    values=[]
    for value in a.flat:
        num,den=float(value).as_integer_ratio();power=den.bit_length()-1
        if den!=1<<power:raise ValueError('nonbinary proposal')
        values.append(arb((num,-power)))
    return arb_mat(896,896,values)

def assemble(plan,out,repository):
    if (ctx.prec,ctx.threads)!=(1024,1):raise ValueError('fixed assembly context')
    p=plan['period']
    if type(p)is not int or p!=3 or plan.get('cell')!='cascade-p3':raise ValueError('exact cascade-p3 period3 allocation')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        with (out/name).open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    authenticate_sources(plan,repository)
    import torch
    same(derivatives.producer.backend_definition(torch.__version__),plan['derivative_backend'],'assembly backend')
    definition=json.loads(pinned(plan['definition']).read_text());point_plan=json.loads(pinned(plan['point_plan']).read_text())
    same(definition['period'],p,'definition period');same(point_plan['period'],p,'point period')
    same(point_plan['cell'],plan['cell'],'same point named cell')
    same(point_plan['definition'],plan['definition'],'same frozen point domain')
    same(definition['map'],plan['map'],'exact map')
    images,point_report=points.authenticate(point_plan,repository=repository,unpack=unpack,pack=domain.pack_ball,precision=1024)
    write('point-admission.json',point_report)
    refs=plan['shards'];expected=[(i,j)for i in range(p)for j in range(0,896,64)]
    if [(r['phase_index'],r['start_column'])for r in refs]!=expected:raise ValueError('exact ordered coverage assignments')
    operational=authenticate_returns(plan,repository);closures=[]
    for ref in refs:
        same(ref['source_commit'],plan['campaign_source'],'campaign source')
        closures.append(authenticate_closure(ref,operational)) # Original records only; no remote queries.
    write('closure-admission.json',dict(operational_return=plan['operational_return'],original_closures=closures,certificate_passed=False))
    jacobians,dr=derivatives.authenticate_and_collect(plan['definition']['path'],plan['definition']['sha256'],expected_map=plan['map'],expected_backend=plan['derivative_backend'],expected_source_sha256=plan['derivative_source_sha256'],expected_prefix=plan['derivative_prefix'],attempts=[{k:r[k]for k in ('attempt','receipt_sha256','source_commit')}for r in refs])
    write('derivative-admission.json',dr)
    if len(dr['authenticated_attempts'])!=len(refs):raise ValueError('full authenticated coverage')
    for ref,audit in zip(refs,dr['authenticated_attempts']):
        same(audit['phase_index'],ref['phase_index'],'phase assignment');same(audit['direction_indices'],list(range(ref['start_column'],ref['start_column']+64)),'column assignment')
    centers=[[unpack(v)for v in row]for row in definition['centers']];radii=[unpack(v)for v in definition['radii']]
    same([[domain.pack_ball(v)for v in row]for row in centers],definition['centers'],'exact native centres')
    same([domain.pack_ball(v)for v in radii],definition['radii'],'exact native radii')
    _,_,mixed,shooting=coverage._nominal(definition['centers'],definition['radii'],p,896)
    cov=dr['coverage'];same(cov['phase_domain_sha256'],mixed,'phase domain');same(cov['shooting_domain_sha256'],shooting,'shooting domain')
    same(cov['phase_matrix_sha256'],[gates._matrix_sha256(m)for m in jacobians],'actual native matrices')
    for i in range(p):
        for j in range(896):
            lo,hi=points.endpoints(point_report['original_images'][i][j]);v=cov['original_primal_intersections'][i][j]
            if max(lo,points.dy(v['lower']))>min(hi,points.dy(v['upper'])):raise ValueError('original point/box primal contradiction')
    proposal=pinned(plan['preconditioner'])
    # Tie proposal bytes to the already-authenticated original refinement card.
    card=json.loads(pinned(point_plan['card']).read_text())
    same(str(proposal),card['preconditioner'],'original proposal path')
    if not any(e['path']==str(proposal)and e['sha256']==plan['preconditioner']['sha256']for e in card['inputs']):raise ValueError('proposal input authority')
    with np.load(proposal,allow_pickle=False)as z:
        if z['A'].shape!=(p,896,896)or not np.array_equal(z['parameters'],[.0650,-.14798]):raise ValueError('proposal family')
        A=[exact_matrix(v)for v in z['A']];B=exact_matrix(z['B'])
    bindings=dict(domain_sha256=gates.domain_sha256(centers,radii),enclosures_sha256=gates.enclosures_sha256(images,jacobians),proposals_sha256=gates.proposals_sha256(A,B))
    same(bindings['domain_sha256'],shooting,'native gate domain')
    admission=dict(certificate_passed=False,plan_sha256=domain.canonical_hash(plan),bindings=bindings,point_report_sha256=points.sha(out/'point-admission.json'),derivative_report_sha256=points.sha(out/'derivative-admission.json'))
    write('admission.json',admission)
    contract=gates.ShootingEnclosureContract(domain.canonical_hash(plan['map']),domain.canonical_hash({'point_plan':plan['point_plan'],'derivative_source':plan['derivative_source_sha256'],'review_binding':plan['review_binding']}),points.sha(out/'admission.json'),p,bindings['domain_sha256'],bindings['enclosures_sha256'],bindings['proposals_sha256'],gates.SHOOTING_DECLARATION)
    result=certificate.check_shooting_certificate(centers,radii,images,jacobians,A,B,contract=contract,bound_method='factored',powers=(1,2,4,8))
    if result['certificate_passed']is not False:raise ValueError('separate release required')
    if (ctx.prec,ctx.threads)!=(1024,1):raise ValueError('assembly context changed')
    for key,value in bindings.items():same(result[key],value,'actual gate input '+key)
    write('conditional-gates.json',result)
    for name,h in plan['assembly_source_sha256'].items():
        if points.sha(Path(repository)/name)!=h:raise ValueError('assembly source changed')
    return dict(certificate_passed=False,conditional_gates=result['conditional_gates'],independent_witness_review_pending=True)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();ctx.prec=1024;ctx.threads=1
    card=json.loads(Path(args.card).read_text())
    answer=assemble(card,args.out,HERE.parents[1])
    with (Path(args.out)/'summary.json').open('x')as f:json.dump(answer,f,indent=2);f.write('\n')
    print(json.dumps(answer),flush=True)

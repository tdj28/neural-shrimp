"""Prepared Task33 conditional assembly core. No CLI or certificate release.

The enclosing reviewed run card must pin this source, the input plan and all
external review/closure receipts. This module performs no model evaluation.
"""
import json,hashlib,sys,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[1]/'codex/certify-llm')]
import numpy as np
from flint import ctx,arb,arb_mat
import point_bridge as points
import derivative_bridge as derivatives
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

def authenticate_closure(ref):
    """Bind original wrapper closure and producer termination to this attempt."""
    run=json.loads(pinned(ref['outer_closure']).read_text())
    name=Path(ref['attempt']).name
    if (run.get('step')!='run' or run.get('success') is not True
        or run.get('closure_verified') is not True
        or type(run.get('exit_code')) is not int or run['exit_code']!=0
        or run.get('watchdog_stop') is not None
        or run.get('killed_by_watchdog') is not None
        or run.get('alive_after_exit')!=[] or run.get('alive_after_cleanup')!=[]):
        raise ValueError('original wrapper not successfully closed')
    command=shlex.split(run['command'])
    for option,value in (('--attempt',name),('--driver','derivative_shard'),
                         ('--plan',name+'-plan.json'),('--threads','1'),('--seconds','3600')):
        if command.count(option)!=1 or command.index(option)+1>=len(command) or command[command.index(option)+1]!=value:
            raise ValueError('outer command assignment')
    if 'codex/coexistence-task33/launch.py' not in command:
        raise ValueError('outer launcher')
    summary=json.loads(pinned(ref['termination']).read_text())
    termination=summary.get('terminated',{})
    if termination.get('terminated') is not True or termination.get('in_inventory') is not False:
        raise ValueError('producer termination incomplete')
    rows=[row for row in summary['attempts'] if row['attempt']==name]
    if len(rows)!=1 or rows[0]['receipt_sha256']!=ref['receipt_sha256']:
        raise ValueError('termination receipt assignment')
    return {'outer_closure_sha256':ref['outer_closure']['sha256'],
            'termination_sha256':ref['termination']['sha256'],
            'attempt':name,'producer_termination_reported':True}

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
    if type(p)is not int or p not in (2,3):raise ValueError('period2/3')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    def write(name,value):(out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    for name,h in plan['assembly_source_sha256'].items():
        if points.sha(Path(repository)/name)!=h:raise ValueError('assembly source pin')
    if set(plan['reviews'])!={'point_evaluator','derivative_evaluator','map_regularity','gate_mathematics'}:raise ValueError('review scope')
    for ref in plan['reviews'].values():pinned(ref)
    definition=json.loads(pinned(plan['definition']).read_text());point_plan=json.loads(pinned(plan['point_plan']).read_text())
    same(definition['period'],p,'definition period');same(point_plan['period'],p,'point period')
    same(point_plan['definition'],plan['definition'],'same frozen point domain')
    same(definition['map'],plan['map'],'exact map')
    images,point_report=points.authenticate(point_plan,repository=repository,unpack=unpack,pack=domain.pack_ball,precision=1024)
    write('point-admission.json',point_report)
    refs=plan['shards'];expected=[(i,j)for i in range(p)for j in range(0,896,64)]
    if [(r['phase_index'],r['start_column'])for r in refs]!=expected:raise ValueError('exact ordered coverage assignments')
    for ref in refs:
        same(ref['source_commit'],plan['campaign_source'],'campaign source')
        authenticate_closure(ref) # Original records only; no remote process queries.
    jacobians,dr=derivatives.authenticate_and_collect(plan['definition']['path'],plan['definition']['sha256'],expected_map=plan['map'],expected_backend=plan['derivative_backend'],expected_source_sha256=plan['derivative_source_sha256'],expected_prefix=plan['derivative_prefix'],attempts=[{k:r[k]for k in ('attempt','receipt_sha256','source_commit')}for r in refs])
    write('derivative-admission.json',dr)
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
        if z['A'].shape!=(p,896,896)or not np.array_equal(z['parameters'],[.06468,-.148]):raise ValueError('proposal family')
        A=[exact_matrix(v)for v in z['A']];B=exact_matrix(z['B'])
    bindings=dict(domain_sha256=gates.domain_sha256(centers,radii),enclosures_sha256=gates.enclosures_sha256(images,jacobians),proposals_sha256=gates.proposals_sha256(A,B))
    same(bindings['domain_sha256'],shooting,'native gate domain')
    admission=dict(certificate_passed=False,plan_sha256=domain.canonical_hash(plan),bindings=bindings,point_report_sha256=points.sha(out/'point-admission.json'),derivative_report_sha256=points.sha(out/'derivative-admission.json'))
    write('admission.json',admission)
    contract=gates.ShootingEnclosureContract(domain.canonical_hash(plan['map']),domain.canonical_hash(plan['derivative_source_sha256']),points.sha(out/'admission.json'),p,bindings['domain_sha256'],bindings['enclosures_sha256'],bindings['proposals_sha256'],gates.SHOOTING_DECLARATION)
    result=certificate.check_shooting_certificate(centers,radii,images,jacobians,A,B,contract=contract,bound_method='factored',powers=(1,2))
    if result['certificate_passed']is not False:raise ValueError('separate release required')
    write('conditional-gates.json',result)
    for name,h in plan['assembly_source_sha256'].items():
        if points.sha(Path(repository)/name)!=h:raise ValueError('assembly source changed')
    return dict(certificate_passed=False,conditional_gates=result['conditional_gates'],independent_witness_review_pending=True)

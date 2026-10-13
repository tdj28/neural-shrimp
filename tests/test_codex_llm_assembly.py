"""Real tiny complete point/shard authentication into unchanged rigorous gates.

Only fixture dimensions, input metadata, owned paths and synthetic producer
outputs differ from the target. Successful admission functions and gates are
not mocked. No checkpoint or actual target attempt is read.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codex/certify-llm'))
import assembly as app
import point_evidence as point
import test_codex_llm_point_evidence as ptest
import test_codex_llm_capsule_admission as ctest
import test_codex_llm_derivative_evidence as dtest
from ball_jet import Jet
from streamed_head import argmax_diagnostic
producer=point.derivative
write,read=ctest.write,ctest.saved.read_json


def cycle_point_writer(candidate,request):
    """Synthetic local maps have F(c_i)=c_(i+1), DF=1/2 on disjoint boxes."""
    ORIGINAL_POINT_WRITER(candidate,request)
    report=read(candidate/'parallel-pass.json')
    for i in range(8):
        out=candidate/f'phase-{i:02d}'
        payload=read(out/'point.json');payload['value']=request['centers'][(i+1)%8]
        write(out/'point.json',payload)
        ctest.mutate(out/'status.json',lambda v:v.update(point_sha256=app.sha(out/'point.json')))
        rp=candidate/'parallel-workers'/f'phase-{i:02d}.receipt.json'
        receipt=read(rp);receipt['artifacts']=ctest.scheduling.inventory(out);write(rp,receipt)
        report['phase_receipts'][i]['receipt_sha256']=app.sha(rp)
    write(candidate/'parallel-pass.json',report)

ORIGINAL_POINT_WRITER=ctest.save_synthetic_worker_pass


class LocalCycleMap:
    numerator=1
    denominator=2
    primal_offset=0
    def __init__(self,center,successor):self.center,self.successor=center,successor
    def forward(self,state,directions):
        live=[Jet(x,tuple(directions[i,k]for k in range(producer.COLUMNS)))for i,x in enumerate(state)]
        for layer in range(producer.LAYERS):
            self.progress('step_start',layer,[live]);self.progress('step_end',layer,[live])
        # All unlisted synthetic logits equal zero; token 1 has exact logit 1.
        self.last_head_diagnostics=argmax_diagnostic([arb(0),arb(1)])
        self.last_head_diagnostics.update(vocabulary_count=producer.VOCABULARY,competitor_count=producer.VOCABULARY-1)
        for kind in('head_logits','head_derivatives'):
            for first in range(0,producer.VOCABULARY,1024):self.head_progress(kind,first,min(first+1024,producer.VOCABULARY))
        slope=arb(self.numerator)/self.denominator
        return [self.successor[i]+(x-self.center[i])*slope+self.primal_offset for i,x in enumerate(state)],directions*slope


def build_shard(owned,phase,definition_path,definition,prefix,backend,pins):
    attempt=owned/f'final-shard-{phase}';out=attempt/'run';out.mkdir(parents=True)
    dtest.snapshot(attempt,producer.SOURCE_PATHS)
    bundle=dict(schema='codex-derivative-evaluator-source-bundle-v1',source_sha256=pins)
    plan=dict(schema=producer.SCHEMA,definition_path=str(definition_path),definition_sha256=app.sha(definition_path),
        map_sha256=point.domain.canonical_hash(definition['map']),backend_sha256=point.domain.canonical_hash(backend),source_sha256=pins,
        evaluator_sha256=point.domain.canonical_hash(bundle),prefix_attempt=prefix['attempt'],prefix_receipt_sha256=prefix['receipt_sha256'],
        prefix_map_source_sha256=prefix['map_source_sha256'],prefix_cache_sha256=prefix['cache_sha256'],phase_index=phase,start_column=0,
        precision=1024,token_chunk=1024,envelope=dict(producer.ENVELOPE))
    path=attempt/'source/codex/certify-llm/test-final-shard-plan.json';write(path,plan)
    declared,binding,raw=producer.admit_definition(plan,ptest.ASSETS,ptest.IDS,ptest.TV)
    (out/'candidate-definition.json').write_bytes(raw)
    write(out/'inputs.json',dict(plan=plan,plan_sha256=app.sha(path),source_commit='4'*40,
        driver_sha256=pins['codex/certify-llm/derivative_shard.py'],binding=binding,precision=1024,threads=1,certificate_passed=False))
    artifact,state,directions,records=producer.construct_inputs(declared,phase,0)
    write(out/'domain-construction.json',artifact)
    write(out/'direction-inputs.json',dict(direction_indices=[0],directions=records,directions_sha256=point.domain.canonical_hash(records)))
    write(out/'prefix-binding.json',dict(source_commit='3'*40,receipt_sha256=prefix['receipt_sha256'],graph_sha256=prefix['map_source_sha256'],
        prefix_sha256=prefix['cache_sha256'],precision=1024,receiver_construction_checked=True,steering_layer=12,hashes_to_arrays_authenticated_by_this_loader=False))
    centers=[[point.coverage._decode(v)for v in row]for row in definition['centers']]
    model=LocalCycleMap(centers[phase],centers[(phase+1)%8]);events=[]
    shard=producer.evaluate_shard(model,declared,binding,plan,state,directions,records,out,lambda kind,**kw:events.append(dict(event=kind,**kw)))
    events.append(dict(event='completed'));dtest.events(out/'events.jsonl',events)
    write(out/'status.json',dict(status='final_derivative_shard_completed',certificate_passed=False,phase_index=phase,direction_indices=[0],
        columns_completed=1,full_state_dimension=1,full_vocabulary=producer.VOCABULARY,precision=1024,
        shard_sha256=app.sha(out/'derivative-shard.json'),actual_inputs_sha256=shard['actual_inputs_sha256'],
        forward_seconds=shard['forward_seconds'],elapsed_seconds=max(1,shard['forward_seconds'])))
    write(attempt/'manifest.json',dict(source_commit='4'*40,threads=1,memory_gib=18,timeout_seconds=3600,
        command=[sys.executable,'-I','-B','-u',str(attempt/'source/codex/certify-llm/derivative_shard.py'),'--output',str(out),'--plan',str(path)]))
    reference=dtest.seal(attempt)
    return dict(**reference,phase_index=phase,start_column=0)


@pytest.fixture
def chain(tmp_path,monkeypatch,request):
    variant=getattr(request,'param','stable')
    if variant=='unstable':monkeypatch.setattr(LocalCycleMap,'numerator',4)
    elif variant=='disjoint':monkeypatch.setattr(LocalCycleMap,'primal_offset',100)
    monkeypatch.setattr(ctest,'save_synthetic_worker_pass',cycle_point_writer)
    generator=ptest.full.__wrapped__(tmp_path,monkeypatch);args,parent_fixture=next(generator)
    monkeypatch.setattr(app,'PERIOD',8);monkeypatch.setattr(app,'DIMENSION',1);monkeypatch.setattr(app,'COLUMNS',1)
    monkeypatch.setattr(app,'OWNED',tmp_path);monkeypatch.setattr(producer,'COLUMNS',1)
    repo=point.continuation.ROOT
    monkeypatch.setattr(app,'ROOT',repo)
    for name in app.sources():
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    definition=read(args['definition_path'])
    # Large enough for rounding in synthetic F but still disjoint; no target box.
    definition['radii']=[dtest.pack(__import__('fractions').Fraction(1,16))for _ in range(8)]
    write(args['definition_path'],definition);args['definition_sha256']=app.sha(args['definition_path'])
    pins={n:app.sha(ROOT/n)for n in app.sources()};backend=dict(point.worker.backend(ptest.TV),precision=1024,threads=1)
    reviews=[]
    for scope in app.REVIEW_SCOPES:
        name='codex/certify-llm/synthetic-'+scope+'-review.md';(repo/name).write_text('Synthetic external hypothesis review: '+scope+'\n')
        reviews.append(dict(scope=scope,path=name,sha256=app.sha(repo/name)))
    ctx.prec=1024
    refs=[build_shard(tmp_path,i,args['definition_path'],definition,args['expected_derivative_prefix'],backend,
        {n:pins[n]for n in producer.SOURCE_PATHS})for i in range(8)]
    bp=parent_fixture.plan['base_plan']
    plan=dict(schema=app.SCHEMA,period=8,dimension=1,precision=1024,
        definition=dict(path=str(args['definition_path']),sha256=args['definition_sha256']),
        point_admission=dict(path=str(args['admission_plan_path']),sha256=args['admission_plan_sha256']),source_sha256=pins,
        map=definition['map'],map_sha256=definition['map_sha256'],point_backend=dict(backend,precision=2048),
        derivative_backend=backend,assembly_backend=backend,derivative_prefix=args['expected_derivative_prefix'],
        seed=dict(attempt=bp['center_seed_attempt'],receipt_sha256=bp['center_seed_receipt_sha256'],source_commit='2'*40,
            preconditioner_sha256=bp['preconditioner_sha256'],driver_sha256=bp['center_seed_driver_sha256'],map_source_sha256=bp['center_seed_map_source_sha256']),
        shards=refs,reviews=reviews,gate=dict(bound_method='factored',powers=[1,2],stability_weights=None),envelope=dict(app.ENVELOPE),
        enclosure_declaration=app.modules()[3].SHOOTING_DECLARATION)
    path=tmp_path/'assembly-plan.json';write(path,plan)
    out=tmp_path/'assembly-attempt/run';out.mkdir(parents=True)
    events=[]
    fixture=dict(plan=plan,path=path,out=out,events=events,parent=parent_fixture)
    yield fixture
    try:next(generator)
    except StopIteration:pass


def run(fixture,*,event=None):
    plan,path,out=fixture['plan'],fixture['path'],fixture['out'];write(path,plan)
    return app.assemble(plan,out,event or(lambda name,**kw:fixture['events'].append(dict(event=name,**kw))),
        frozen_plan_path=path,frozen_plan_sha256=app.sha(path),expected_assets=ptest.ASSETS,prefix_ids=ptest.IDS,torch_version=ptest.TV)


def test_complete_real_adapters_into_unchanged_all_conditional_gates(chain):
    result=run(chain)
    assert result['status']=='conditional_assembly_completed' and result['conditional_gates_passed']
    assert all(result['conditional_gates'].values()) and result['certificate_passed']is False
    assert result['independent_final_result_review_pending'] and result['model_evaluations']==result['optimizer_actions']==0
    admission=read(chain['out']/'admission-manifest.json');contract=read(chain['out']/'contract.json')
    assert admission['original_plan_sha256']==app.sha(chain['path'])
    conversion=read(chain['out']/'proposal-admission.json')['conversion']
    assert conversion['native_conversion_exact'] and conversion['original_binary64_values_compared']==9
    assert admission['certificate_passed']is False and contract['evidence_sha256']==app.sha(chain['out']/'admission-manifest.json')
    summary=read(chain['out']/'gate-summary.json');report=read(chain['out']/'conditional-gates.json')
    assert summary['conditional_gates']==report['conditional_result']['conditional_gates']
    assert report['conditional_result']['monodromy']['order']=='J_(p-1) ... J_1 J_0'
    assert report['certificate_passed']is False and result['report_sha256']==app.sha(chain['out']/'conditional-gates.json')
    assert contract['domain_sha256']==admission['bindings']['shooting_domain_sha256']
    assert admission['bindings']['shooting_domain_sha256']!=admission['bindings']['phase_domain_sha256']
    assert not(chain['out']/'conditional-gates.json.partial').exists()


@pytest.mark.parametrize('fault',['missing','duplicate','assignment','bool_assignment','foreign_source','missing_source','gate_powers','weights',
    'memory_alias','precision','seed_hash','seed_source','map','point_backend','derivative_backend','review_bytes','no_review','definition_bytes',
    'prototype','failed_shard','narrow_original','native_change','proposal_change','image_change','point_low_bit'])
def test_frozen_complete_evidence_contract_refuses_faults(chain,fault,monkeypatch):
    plan=chain['plan'];point_path=Path(plan['point_admission']['path'])
    if fault=='missing':plan['shards'].pop()
    elif fault=='duplicate':plan['shards'][1]['attempt']=plan['shards'][0]['attempt']
    elif fault=='assignment':plan['shards'][0]['start_column']=1
    elif fault=='bool_assignment':plan['shards'][0]['phase_index']=False
    elif fault=='foreign_source':plan['source_sha256']['codex/certify-llm/assembly.py']='0'*64
    elif fault=='missing_source':plan['source_sha256'].pop('codex/certify-llm/secant_parent.py')
    elif fault=='gate_powers':plan['gate']['powers']=[1,2,4]
    elif fault=='weights':plan['gate']['stability_weights']=[1]
    elif fault=='memory_alias':plan['envelope']['memory_gib']=18.0
    elif fault=='precision':plan['precision']=2048
    elif fault=='seed_hash':plan['seed']['preconditioner_sha256']='0'*64
    elif fault=='seed_source':plan['seed']['source_commit']='0'*40
    elif fault=='map':plan['map']['alpha_ratio']=[1,2];plan['map_sha256']=app.canonical(plan['map'])
    elif fault=='point_backend':plan['point_backend']['precision']=1024
    elif fault=='derivative_backend':plan['derivative_backend']['precision']=2048
    elif fault=='review_bytes':(app.ROOT/plan['reviews'][0]['path']).write_text('Changed external hypothesis review.')
    elif fault=='no_review':plan['reviews'].pop()
    elif fault=='definition_bytes':Path(plan['definition']['path']).write_bytes(Path(plan['definition']['path']).read_bytes()+b' ')
    elif fault in('prototype','point_low_bit'):
        path=Path(plan['definition']['path']);definition=read(path)
        if fault=='prototype':definition['role']='prototype_box'
        else:
            m,r=point.domain._record_parts(definition['centers'][0][0]);definition['centers'][0][0]=dtest.pack(m+__import__('fractions').Fraction(1,2**2400))
        write(path,definition);plan['definition']['sha256']=app.sha(path)
    elif fault in('failed_shard','narrow_original'):
        ref=plan['shards'][0];attempt=Path(ref['attempt'])
        if fault=='failed_shard':
            ctest.mutate(attempt/'run/status.json',lambda v:v.update(status='final_derivative_shard_failed'));ref['receipt_sha256']=dtest.seal(attempt)['receipt_sha256']
        else:
            ctest.mutate(attempt/'run/actual-inputs.json',lambda v:v['actual_inputs'][0].update(rad=[0,0]));ref['receipt_sha256']=dtest.refresh_shard(attempt)['receipt_sha256']
    elif fault in('native_change','proposal_change','image_change'):
        real=app.cross_bind
        def changed(*args):
            if fault=='image_change':args[6][0][0]=arb(123)
            else:args[7 if fault=='native_change'else 8][0][0,0]=arb(123)
            return real(*args)
        monkeypatch.setattr(app,'cross_bind',changed)
    with pytest.raises((ValueError,KeyError)):run(chain)
    assert not(chain['out']/'gate-summary.json').exists()
    assert ctx.prec==1024 and ctx.threads==1


def test_admission_evidence_precedes_gate_and_interruption_is_retained(chain):
    def stop(name,**kw):
        if name=='conditional_gates_started':
            assert (chain['out']/'point-admission.json').is_file()
            assert (chain['out']/'derivative-admission.json').is_file()
            assert (chain['out']/'proposal-admission.json').is_file()
            assert (chain['out']/'admission-manifest.json').is_file()
            assert (chain['out']/'contract.json').is_file()
            raise TimeoutError('synthetic gate interruption')
    with pytest.raises(TimeoutError):run(chain,event=stop)
    assert not(chain['out']/'gate-summary.json').exists()
    assert read(chain['out']/'admission-manifest.json')['certificate_passed']is False


def test_compact_summary_precedes_report_and_failed_stream_keeps_partial(chain,monkeypatch):
    real=app.write_json
    def interrupted(path,value):
        if Path(path).name=='conditional-gates.json':
            assert read(chain['out']/'gate-summary.json')['certificate_passed']is False
            with Path(path).with_suffix('.json.partial').open('w')as f:f.write('{"certificate_passed":false,')
            raise OSError('synthetic interrupted report')
        return real(path,value)
    monkeypatch.setattr(app,'write_json',interrupted)
    with pytest.raises(OSError):run(chain)
    assert (chain['out']/'conditional-gates.json.partial').exists()and not(chain['out']/'conditional-gates.json').exists()
    assert read(chain['out']/'gate-summary.json')['independent_final_result_review_pending']


def test_original_assembly_plan_bytes_are_required(chain):
    plan=chain['plan'];path=chain['path'];pin=app.sha(path);path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='assembly plan bytes'):
        app.assemble(plan,chain['out'],lambda *a,**kw:None,frozen_plan_path=path,frozen_plan_sha256=pin,
            expected_assets=ptest.ASSETS,prefix_ids=ptest.IDS,torch_version=ptest.TV)


def test_isolated_help_without_model_import():
    result=subprocess.run([sys.executable,'-I','-B',str(ROOT/'codex/certify-llm/assembly.py'),'--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--plan' in result.stdout


def test_stream_writer_preserves_partial_on_nonfinite_failure(tmp_path):
    path=tmp_path/'result.json'
    with pytest.raises(ValueError):app.write_json(path,dict(certificate_passed=False,late=float('nan')))
    assert not path.exists()and path.with_suffix('.json.partial').exists()


@pytest.mark.parametrize('chain',['unstable'],indirect=True)
def test_failed_mathematical_gates_still_retain_all_results(chain):
    result=run(chain)
    assert result['status']=='conditional_assembly_completed'
    assert not result['conditional_gates_passed'] and not result['certificate_passed']
    assert not result['conditional_gates']['local_attraction']
    assert result['conditional_gates']['primitive_period']
    report=read(chain['out']/'conditional-gates.json')['conditional_result']
    assert {'root','primitive','attraction'}<=set(report)
    assert len(report['root']['phases'])==8


@pytest.mark.parametrize('chain',['disjoint'],indirect=True)
def test_original_point_and_box_primal_contradiction_refused(chain):
    with pytest.raises(ValueError,match='point and whole-box primal'):
        run(chain)
    assert (chain['out']/'point-admission.json').exists()
    assert (chain['out']/'derivative-admission.json').exists()
    assert not(chain['out']/'gate-summary.json').exists()


@pytest.mark.parametrize('stale',[False,True])
def test_cli_failure_retains_explicit_noncertificate_status_without_model(tmp_path,stale):
    owned=tmp_path/'owned';owned.mkdir();out=owned/'attempt/run'
    if stale:out.mkdir(parents=True)
    harness=tmp_path/'probe.py'
    harness.write_text("import os,sys\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(ROOT/'codex/certify-llm'))+")\nimport assembly as m\nm.OWNED=Path("+repr(str(owned))+")\nos.environ['CERT_SOURCE_COMMIT']='a'*40\nsys.argv=['assembly','--output',"+repr(str(out))+",'--plan','/tmp/noncaptured-plan.json']\nm.main()\n")
    completed=subprocess.run([sys.executable,'-I','-B',str(harness)],capture_output=True,text=True)
    assert completed.returncode!=0
    if stale:assert not(out/'status.json').exists()
    else:
        status=read(out/'status.json')
        assert status['status']=='conditional_assembly_failed' and status['certificate_passed']is False
        assert status['independent_final_result_review_pending']


def test_duplicate_original_plan_key_refused_even_with_matching_byte_pin(chain):
    path=chain['path'];raw=path.read_bytes();path.write_bytes(b'{"schema":"ignored-first-alias",'+raw.lstrip()[1:])
    with pytest.raises(ValueError,match='duplicate original JSON key'):
        app.assemble(chain['plan'],chain['out'],lambda *a,**kw:None,frozen_plan_path=path,frozen_plan_sha256=app.sha(path),
            expected_assets=ptest.ASSETS,prefix_ids=ptest.IDS,torch_version=ptest.TV)


def test_literal_asset_and_prefix_reader_does_not_import_torch(tmp_path):
    harness=tmp_path/'literal_probe.py'
    harness.write_text("import sys\nsys.path.insert(0,"+repr(str(ROOT/'codex/certify-llm'))+")\nimport assembly\nassert 'torch' not in sys.modules\nassets,ids,version=assembly.expected_constants()\nassert set(assets)=={'model.safetensors','config.json','vectors.npz'} and len(ids)==37\nassert version and 'torch' not in sys.modules\nprint('literal-only metadata reader passed')\n")
    run=subprocess.run([sys.executable,'-I','-B',str(harness)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert 'literal-only metadata' in run.stdout


def test_different_but_exact_converter_cannot_claim_original_seed_provenance(chain,monkeypatch):
    original=point.base.exact_matrix
    def wrong(array):
        result=original(array);result[0,0]+=1
        assert result[0,0].is_exact()
        return result
    monkeypatch.setattr(point.base,'exact_matrix',wrong)
    with pytest.raises(ValueError,match='original binary64 exact value'):
        run(chain)
    assert (chain['out']/'derivative-admission.json').is_file()
    assert not(chain['out']/'proposal-admission.json').exists()
    assert not(chain['out']/'gate-summary.json').exists()


def test_torch_version_reader_keeps_local_suffix_without_executing_file(tmp_path,monkeypatch):
    version=tmp_path/'version.py';version.write_text("__version__ = '2.1.0+testbuild'\nraise RuntimeError('must never execute')\n")
    class Distribution:
        version='2.1.0'
        def locate_file(self,name):
            assert name=='torch/version.py'
            return version
    monkeypatch.setattr(app.importlib.metadata,'distribution',lambda name:Distribution())
    record=app.torch_version_metadata()
    assert record['distribution_version']=='2.1.0' and record['producer_version']=='2.1.0+testbuild'
    assert record['version_file_sha256']==app.sha(version)
    assert not record['torch_imported_by_reader'] and not record['metadata_file_executed']
    version.write_text("__version__='2.1'\n__version__='2.2'\n")
    with pytest.raises(ValueError,match='unambiguous'):app.torch_version_metadata()

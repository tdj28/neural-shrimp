"""Complete synthetic attempts only: no checkpoint or pretrained-model calls."""
import copy
from fractions import Fraction
import gzip
import json
from pathlib import Path
import sys

from flint import arb,ctx
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codex/certify-llm'))
import derivative_evidence as ev
import derivative_shard as producer
from ball_jet import Jet
from streamed_head import argmax_diagnostic

def pack(m=0,r=0):return dict(mid=ev.domain._pair(Fraction(m)),rad=ev.domain._pair(Fraction(r)))
def write(path,value):producer.write_json(path,value)
def read(path):return producer.read_json(path)
def events(path,rows):path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

class SyntheticMap:
    """A true exact linear map, with the producer's full callback contract."""
    def forward(self,state,directions):
        live=[Jet(x,tuple(directions[i,k]for k in range(producer.COLUMNS)))for i,x in enumerate(state)]
        for layer in range(producer.LAYERS):
            self.progress('step_start',layer,[live]);self.progress('step_end',layer,[live])
        self.last_head_diagnostics=argmax_diagnostic([arb(i)for i in range(producer.VOCABULARY)])
        for kind in ('head_logits','head_derivatives'):
            for first in range(0,producer.VOCABULARY,1024):self.head_progress(kind,first,min(first+1024,producer.VOCABULARY))
        return [2*x for x in state],directions*2


def seal(attempt,*,exit_code=0,timed_out=False):
    manifest=read(attempt/'manifest.json')
    manifest['source']=[dict(path=str(p.relative_to(attempt/'source')),bytes=p.stat().st_size,sha256=ev.digest(p))
                        for p in sorted((attempt/'source').rglob('*'))if p.is_file()]
    write(attempt/'manifest.json',manifest)
    receipt=dict(source_commit=manifest['source_commit'],exit_code=exit_code,timed_out=timed_out,
        elapsed_seconds=3.0,artifacts=[dict(path=str(p.relative_to(attempt)),bytes=p.stat().st_size,sha256=ev.digest(p))
            for p in sorted(attempt.rglob('*'))if p.is_file()and p.name!='receipt.json'])
    write(attempt/'receipt.json',receipt)
    return dict(attempt=str(attempt),receipt_sha256=ev.digest(attempt/'receipt.json'))


def snapshot(attempt,names):
    for name in names:
        path=attempt/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())


@pytest.fixture
def full(tmp_path,monkeypatch):
    for name,value in [('PERIOD',2),('DIMENSION',4),('COLUMNS',2),('VOCABULARY',2051),('LAYERS',2)]:
        monkeypatch.setattr(producer,name,value)
    owned=tmp_path/'owned';owned.mkdir()
    monkeypatch.setattr(producer,'OWNED',owned);monkeypatch.setattr(ev,'OWNED',owned)
    prior_threads=ctx.threads;ctx.threads=1
    with ctx.workprec(1024):
        source='1'*40;assets={'synthetic':'a'*64};ids=[1,2,3]
        mapping=producer.map_definition(assets,ids);backend=producer.backend_definition('synthetic-torch')
        prefix=owned/'prefix';(prefix/'run').mkdir(parents=True)
        snapshot(prefix,['codex/certify-llm/ball_qwen.py'])
        prefix_inputs=dict(source_commit=source,versions={k:backend[k]for k in ('python','python_flint','flint','numpy','torch')},
            arb_context=dict(precision=1024,threads=1),precision=1024,
            **{k:mapping[k]for k in ('pinned_assets','prefix_ids','alpha_ratio','log10_temperature_ratio','temperature','rotary')})
        write(prefix/'run/inputs.json',prefix_inputs)
        values=[[[[[pack(i+j+k)]*2]for k in ids]for j in range(2)]for i in range(2)]
        with gzip.open(prefix/'run/prefix-cache.json.gz','wt')as f:json.dump(values,f)
        cache=ev.digest(prefix/'run/prefix-cache.json.gz')
        events(prefix/'run/events.jsonl',[dict(event=kind,layer=i)for i in range(2)for kind in ('prefix_start','prefix_end')]+[dict(event='prefix_saved',sha256=cache)])
        write(prefix/'run/status.json',dict(status='setup_failed',phase='rigorous_box_jvp',certificate_passed=False))
        write(prefix/'manifest.json',dict(source_commit=source))
        ref=seal(prefix,exit_code=1)
        prefix_anchor=dict(**ref,map_source_sha256=ev.digest(ROOT/'codex/certify-llm/ball_qwen.py'),cache_sha256=cache,
            completion='retained_prefix_before_box_failure')
        definition=dict(schema=producer.DEFINITION_SCHEMA,role=ev.coverage.ROLE,period=2,dimension=4,
            centers=[[pack(i+j+Fraction(1,2**1700))for j in range(4)]for i in range(2)],
            radii=[pack(Fraction(1,2**1800))]*2,map=mapping,map_sha256=ev.canonical(mapping),
            selection_provenance={'synthetic_proposal_only':True})
        definition_path=owned/'candidate.json';write(definition_path,definition)
        pins=producer.source_bundle()['source_sha256'];attempts=[]
        for phase in range(2):
            for start in (0,2):
                attempt=owned/f'shard-{phase}-{start}';out=attempt/'run';out.mkdir(parents=True)
                snapshot(attempt,producer.SOURCE_PATHS)
                plan=dict(schema=producer.SCHEMA,definition_path=str(definition_path),definition_sha256=ev.digest(definition_path),
                    map_sha256=ev.canonical(mapping),backend_sha256=ev.canonical(backend),source_sha256=pins,
                    evaluator_sha256=ev.canonical(producer.source_bundle()),prefix_attempt=str(prefix),
                    prefix_receipt_sha256=ref['receipt_sha256'],prefix_map_source_sha256=prefix_anchor['map_source_sha256'],
                    prefix_cache_sha256=cache,phase_index=phase,start_column=start,precision=1024,token_chunk=1024,envelope=dict(producer.ENVELOPE))
                path=attempt/'source/codex/certify-llm/test-plan.json';write(path,plan)
                declared,binding,raw=producer.admit_definition(plan,assets,ids,'synthetic-torch')
                (out/'candidate-definition.json').write_bytes(raw)
                write(out/'inputs.json',dict(plan=plan,plan_sha256=ev.digest(path),source_commit=source,
                    driver_sha256=pins['codex/certify-llm/derivative_shard.py'],binding=binding,precision=1024,threads=1,certificate_passed=False))
                artifact,state,directions,records=producer.construct_inputs(declared,phase,start)
                write(out/'domain-construction.json',artifact)
                write(out/'direction-inputs.json',dict(direction_indices=list(range(start,start+2)),directions=records,directions_sha256=ev.canonical(records)))
                write(out/'prefix-binding.json',dict(source_commit=source,receipt_sha256=ref['receipt_sha256'],
                    graph_sha256=prefix_anchor['map_source_sha256'],prefix_sha256=cache,precision=1024,
                    receiver_construction_checked=True,steering_layer=12,hashes_to_arrays_authenticated_by_this_loader=False))
                rows=[]
                shard=producer.evaluate_shard(SyntheticMap(),declared,binding,plan,state,directions,records,out,
                    lambda kind,**kw:rows.append(dict(event=kind,**kw)))
                rows.append(dict(event='completed'));events(out/'events.jsonl',rows)
                write(out/'status.json',dict(status='final_derivative_shard_completed',certificate_passed=False,phase_index=phase,
                    direction_indices=shard['direction_indices'],columns_completed=2,full_state_dimension=4,
                    full_vocabulary=producer.VOCABULARY,precision=1024,shard_sha256=ev.digest(out/'derivative-shard.json'),
                    actual_inputs_sha256=shard['actual_inputs_sha256'],forward_seconds=shard['forward_seconds'],elapsed_seconds=1.0))
                write(attempt/'manifest.json',dict(source_commit=source,threads=1,memory_gib=18,timeout_seconds=3600,
                    command=[sys.executable,'-I','-B','-u',str(attempt/'source/codex/certify-llm/derivative_shard.py'),
                        '--output',str(out),'--plan',str(path)]))
                attempts.append(seal(attempt))
        fixture=dict(definition_path=definition_path,definition_sha256=ev.digest(definition_path),expected_map=mapping,
            expected_backend=backend,expected_source_sha256=pins,expected_prefix=prefix_anchor,attempts=attempts)
        yield fixture
    ctx.threads=prior_threads


def collect(fixture):return ev.authenticate_and_collect(**fixture)


def refresh_shard(attempt):
    """Reseal cross-file hashes after semantic corruptions, to reach real gates."""
    out=attempt/'run';s=read(out/'derivative-shard.json');r=read(out/'actual-inputs.json');d=read(out/'direction-inputs.json')
    r['actual_inputs_sha256']=ev.canonical(r['actual_inputs']);write(out/'actual-inputs.json',r)
    d['directions_sha256']=ev.canonical(d['directions']);write(out/'direction-inputs.json',d)
    s['actual_inputs']=r['actual_inputs'];s['actual_inputs_sha256']=r['actual_inputs_sha256']
    s['actual_inputs_file_sha256']=ev.digest(out/'actual-inputs.json');s['direction_inputs_file_sha256']=ev.digest(out/'direction-inputs.json')
    head=read(out/'head-diagnostics.json');head['actual_inputs_sha256']=r['actual_inputs_sha256'];write(out/'head-diagnostics.json',head)
    s['head_diagnostics_file_sha256']=ev.digest(out/'head-diagnostics.json');write(out/'derivative-shard.json',s)
    status=read(out/'status.json');status['shard_sha256']=ev.digest(out/'derivative-shard.json');status['actual_inputs_sha256']=r['actual_inputs_sha256'];write(out/'status.json',status)
    rows=[json.loads(x)for x in (out/'events.jsonl').read_text().splitlines()]
    for row in rows:
        if row['event']=='live_input_saved':row['sha256']=ev.digest(out/'actual-inputs.json')
    events(out/'events.jsonl',rows)
    return seal(attempt)


def test_complete_original_stream_all_columns_and_failed_prefix_stage(full):
    with ctx.workprec(2048):
        matrices,report=collect(full)
        assert ctx.prec==2048
    assert report['artifact_authentication_completed'] and not report['certificate_passed']
    assert report['assembly_precision']==2048 and report['source_precision']==1024
    assert report['coverage']['coverage_complete'] and len(report['authenticated_attempts'])==4
    assert report['trusted_prefix']['overall_attempt_succeeded'] is False
    assert report['trusted_prefix']['original_cache_balls_checked']==24
    assert report['trusted_prefix']['model_constructed'] is False
    assert not report['evaluator_correctness_independently_proved_by_adapter']
    assert report['git_commit_blob_verification_performed'] is False
    for matrix in matrices:
        for i in range(4):
            for j in range(4):assert matrix[i,j]==2*int(i==j)


def test_generator_is_consumed_one_original_shard_at_a_time(full,monkeypatch):
    real=ev.authenticate_shard;seen=[]
    def authenticated(reference,*args):
        shard,audit=real(reference,*args);seen.append(reference['attempt']);return shard,audit
    monkeypatch.setattr(ev,'authenticate_shard',authenticated)
    refs=full['attempts']
    def references():
        for i,ref in enumerate(refs):
            assert len(seen)==i
            yield ref
    full['attempts']=references();_,report=collect(full)
    assert len(seen)==len(report['authenticated_attempts'])==4


@pytest.mark.parametrize('fault',['missing','duplicate','too_many','receipt_hash','unlisted','bytes','source_resealed',
    'source_manifest','failed_receipt','timed_out','failed_status','prototype','same_hash_foreign_map','backend',
    'plan_source','plan_bool','command','envelope','missing_primal','nonfinite','derivative_shape','basis',
    'narrow_receiver','witness','input_hash','direction_hash','domain_hash','domain_artifact','missing_layer',
    'reordered_head','missing_head','bool_head','live_late','completed','label','label_extrema','prefix_binding'])
def test_success_is_never_inferred_from_self_reported_metadata(full,fault):
    if fault=='missing':full['attempts'].pop()
    elif fault=='duplicate':full['attempts'][1]=full['attempts'][0]
    elif fault=='too_many':full['attempts']+=full['attempts'][:1]
    elif fault=='receipt_hash':full['attempts'][0]['receipt_sha256']='0'*64
    else:
        attempt=Path(full['attempts'][0]['attempt']);out=attempt/'run'
        if fault=='unlisted':(out/'unlisted').write_text('extra')
        elif fault=='bytes':(out/'inputs.json').write_text('{}')
        else:
            file=None
            if fault=='source_resealed':
                (attempt/'source'/producer.SOURCE_PATHS[1]).write_text('# altered source\n')
            elif fault=='source_manifest':
                manifest=read(attempt/'manifest.json');manifest['source'].pop();write(attempt/'manifest.json',manifest)
                receipt=read(attempt/'receipt.json')
                for row in receipt['artifacts']:
                    if row['path']=='manifest.json':row.update(bytes=(attempt/'manifest.json').stat().st_size,sha256=ev.digest(attempt/'manifest.json'))
                write(attempt/'receipt.json',receipt);full['attempts'][0]['receipt_sha256']=ev.digest(attempt/'receipt.json')
            elif fault in ('failed_receipt','timed_out'):
                full['attempts'][0]=seal(attempt,exit_code=1 if fault=='failed_receipt'else 0,timed_out=fault=='timed_out')
            elif fault in ('command','envelope'):
                file=attempt/'manifest.json';v=read(file)
                if fault=='command':v['command'][1]='-B'
                else:v['threads']=True
            elif fault in ('failed_status',):file=out/'status.json';v=read(file);v['status']='final_derivative_shard_failed'
            elif fault in ('same_hash_foreign_map','backend','plan_source','plan_bool'):
                file=out/'inputs.json';v=read(file)
                if fault=='same_hash_foreign_map':v['binding']['map']['alpha_ratio']=[1,2]
                if fault=='backend':v['binding']['backend']['precision']=2048
                if fault=='plan_source':v['plan']['source_sha256'][producer.SOURCE_PATHS[0]]='0'*64
                if fault=='plan_bool':v['plan']['phase_index']=False
            elif fault in ('prototype','missing_primal','nonfinite','derivative_shape','basis','domain_hash'):
                file=out/'derivative-shard.json';v=read(file)
                if fault=='prototype':v['role']='prototype_box'
                if fault=='missing_primal':v['primal'].pop()
                if fault=='nonfinite':v['derivatives'][0][0]['mid']=['nan',0]
                if fault=='derivative_shape':v['derivatives'][0].pop()
                if fault=='basis':v['directions'][0][0]=pack(-1)
                if fault=='domain_hash':v['shooting_domain_sha256']='0'*64
            elif fault in ('narrow_receiver','witness'):
                file=out/'actual-inputs.json';v=read(file)
                if fault=='narrow_receiver':v['actual_inputs'][0]['rad']=[0,0]
                else:v['selected_phase_endpoint_witnesses'][0]['actual_radius']=[0,0]
            elif fault=='input_hash':file=out/'derivative-shard.json';v=read(file);v['actual_inputs_file_sha256']='0'*64
            elif fault=='direction_hash':file=out/'direction-inputs.json';v=read(file);v['directions_sha256']='0'*64
            elif fault=='domain_artifact':file=out/'domain-construction.json';v=read(file);v['helper_source_sha256']='0'*64
            elif fault=='prefix_binding':file=out/'prefix-binding.json';v=read(file);v['receiver_construction_checked']=False
            elif fault in ('label','label_extrema'):
                file=out/'head-diagnostics.json';v=read(file)
                if fault=='label':v['diagnostic']['strict_margin_established']=False
                else:v['diagnostic']['candidate_lower']=pack(100)
            else:
                rows=[json.loads(x)for x in (out/'events.jsonl').read_text().splitlines()]
                if fault=='missing_layer':rows=[r for r in rows if not(r['event']=='step_end'and r['layer']==1)]
                if fault=='missing_head':rows=[r for r in rows if not(r['event']=='head_derivatives'and r['row_start']==1024)]
                if fault=='reordered_head':
                    idx=[i for i,r in enumerate(rows)if r['event']=='head_logits'];rows[idx[0]],rows[idx[1]]=rows[idx[1]],rows[idx[0]]
                if fault=='bool_head':next(r for r in rows if r['event']=='head_logits')['row_start']=False
                if fault=='live_late':r=rows.pop(0);rows.insert(2,r)
                if fault=='completed':rows.pop()
                events(out/'events.jsonl',rows)
            if file:write(file,v)
            if fault not in ('source_manifest','failed_receipt','timed_out'):
                if fault in ('input_hash','direction_hash'):full['attempts'][0]=seal(attempt)
                else:full['attempts'][0]=refresh_shard(attempt)
    with pytest.raises(ValueError):collect(full)


@pytest.mark.parametrize('fault',['missing_source','wrong_external_source','definition_hash','prototype_definition','wrong_map',
    'foreign_backend','prefix_hash','prefix_2048','prefix_assets','prefix_failed_early','prefix_timeout','prefix_stage',
    'prefix_cache','prefix_graph','prefix_shape','prefix_nonfinite'])
def test_external_candidate_source_backend_and_prefix_anchors(full,fault):
    if fault=='missing_source':full['expected_source_sha256'].pop(producer.SOURCE_PATHS[0])
    elif fault=='wrong_external_source':full['expected_source_sha256'][producer.SOURCE_PATHS[0]]='0'*64
    elif fault=='definition_hash':full['definition_sha256']='0'*64
    elif fault=='prototype_definition':
        v=read(full['definition_path']);v['role']='prototype_box';write(full['definition_path'],v);full['definition_sha256']=ev.digest(full['definition_path'])
    elif fault=='wrong_map':full['expected_map']['alpha_ratio']=[1,2]
    elif fault=='foreign_backend':full['expected_backend']['precision']=2048
    elif fault in ('prefix_hash','prefix_graph','prefix_cache'):
        key={'prefix_hash':'receipt_sha256','prefix_graph':'map_source_sha256','prefix_cache':'cache_sha256'}[fault]
        full['expected_prefix'][key]='0'*64
    else:
        attempt=Path(full['expected_prefix']['attempt']);out=attempt/'run'
        if fault in ('prefix_2048','prefix_assets'):
            v=read(out/'inputs.json')
            if fault=='prefix_2048':v['precision']=2048
            else:v['pinned_assets']['synthetic']='0'*64
            write(out/'inputs.json',v)
        if fault=='prefix_failed_early':v=read(out/'status.json');v['phase']='prefix';write(out/'status.json',v)
        if fault=='prefix_stage':rows=[json.loads(x)for x in (out/'events.jsonl').read_text().splitlines()];rows.pop(1);events(out/'events.jsonl',rows)
        if fault in ('prefix_shape','prefix_nonfinite'):
            with gzip.open(out/'prefix-cache.json.gz','rt')as f:v=json.load(f)
            if fault=='prefix_shape':v.pop()
            else:v[0][0][0][0][0]['mid']=['nan',0]
            with gzip.open(out/'prefix-cache.json.gz','wt')as f:json.dump(v,f)
            full['expected_prefix']['cache_sha256']=ev.digest(out/'prefix-cache.json.gz')
            rows=[json.loads(x)for x in (out/'events.jsonl').read_text().splitlines()];rows[-1]['sha256']=full['expected_prefix']['cache_sha256'];events(out/'events.jsonl',rows)
        ref=seal(attempt,exit_code=1,timed_out=fault=='prefix_timeout');full['expected_prefix']['receipt_sha256']=ref['receipt_sha256']
    with pytest.raises(ValueError):collect(full)

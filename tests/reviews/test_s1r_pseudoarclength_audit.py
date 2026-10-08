"""Saved synthetic controls; no producer/model/decomposition calls."""
import base64,copy,gzip,importlib.util,json
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('pseudoarc_audit',ROOT/'tools/reviews/audit_s1r_pseudoarclength_outcome.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
FIXTURE=ROOT/'tests/reviews/fixtures/pseudoarclength-synthetic-two-edges.json.gz'

def fixture():
    def decode(v):
        if isinstance(v,dict)and'synthetic_array_bytes_b64'in v:return np.frombuffer(base64.b64decode(v['synthetic_array_bytes_b64']),dtype=v['dtype']).reshape(v['shape']).copy()
        if isinstance(v,dict):return{k:decode(x)for k,x in v.items()}
        if isinstance(v,list):return[decode(x)for x in v]
        return v
    return decode(json.loads(gzip.decompress(FIXTURE.read_bytes())))

def replay(f):
    r=A.Replay(f['initial'],f['hidden'])
    for kind,p in f['events']:r.consume(kind,p)
    return r

def test_full_saved_two_edges_without_numerical_calls(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('fresh decomposition/solve forbidden')
    for name in('svd','eig','eigvals','eigh','eigvalsh','solve','lstsq','inv','pinv'):monkeypatch.setattr(np.linalg,name,forbidden)
    f=fixture();r=replay(f);s=r.summary();assert s['accepted_edges']==[0,1]and s['last_witnessed']=='node-0005'
    assert len(s['qualified_nodes'])==9 and len(s['qualified_unwitnessed_nodes'])==6 and s['returned']==f['returned']

@pytest.mark.parametrize('kind,field', [('assembly','R'),('assembly','M'),('assembly','A'),('assembly','b'),('operation_return','gradient')])
def test_retained_arithmetic_corruption_rejected(kind,field):
    f=fixture()
    for k,p in f['events']:
        if k==kind and(field in p or field in p.get('value',{})):
            v=p if field in p else p['value'];v[field].flat[0]+=.01;break
    with pytest.raises(ValueError):replay(f)

@pytest.mark.parametrize('field',['G','J','step'])
def test_bordered_inputs_and_return_bound(field):
    f=fixture()
    for k,p in f['events']:
        if p.get('kind')=='solve'and k==('operation_return'if field=='step'else'operation_start'):
            v=p['value']if field=='step'else p['inputs'];v[field].flat[0]+=.01;break
    with pytest.raises(ValueError):replay(f)

def test_map_cache_is_byte_exact_including_signed_zero():
    f=fixture()
    for k,p in f['events']:
        if k=='cache_hit'and p['kind']=='map':p['h']=-0.;break
    with pytest.raises(ValueError):replay(f)

def test_common_control_endpoint_substitution_rejected():
    f=fixture()
    for k,p in f['events']:
        if k=='common_end'and p['passed']:p['candidate_original_h']+=1e-9;break
    with pytest.raises(ValueError):replay(f)

def test_spectrum_assignment_must_be_bijection():
    f=fixture()
    for k,p in f['events']:
        if k=='qualifier_end'and p['accepted']:p['spectrum_assignment_cols'][:]=0;break
    with pytest.raises(ValueError):replay(f)

def test_actual_armijo_slope_is_reconstructed():
    f=fixture()
    for k,p in f['events']:
        if k=='armijo_trial'and'G'in p:p['slope']-=1.;break
    with pytest.raises(ValueError):replay(f)

def test_step_halving_after_success_rejected():
    f=fixture();r=replay(f)
    with pytest.raises(ValueError):r.consume('halve',dict(macro=2,new_s=r.next_s/2,parent=r.witnessed))

def test_no_third_edge_after_two_witnesses():
    f=fixture();r=replay(f)
    with pytest.raises(ValueError):r.consume('edge_start',dict(id=2,parent=r.witnessed,s=r.next_s,nodes=[],accepted=False))

def test_failed_svd_gate_is_recomputed_then_only_unwinds():
    f=fixture();r=A.Replay(f['initial'],1)
    # Perturb a saved SVD enough to fail its declared gate, preserving its
    # recomputed diagnostic values. This is a numerical rejection, not acceptance.
    for kind,p in f['events']:
        if kind=='svd_checks':
            saved=r.svds[p['id']];saved['U'][0,0]+=.001;m=saved['matrix'];U=saved['U'];s=saved['singular_values'];Vh=saved['Vh'];V=Vh[:len(s)].T;sc=max(1.,A.norm(m));d=A.M.svd_residuals(m,saved)
            p.update(right=A.norm(m@V-U*s)/sc,left=A.norm(m.T@U-V*s)/sc,reconstruction=d['reconstruction'],orthogonality=d['orthogonality'],rank=d['rank']);r.consume(kind,p);assert r.blocking is not None
            with pytest.raises(ValueError):r.consume('cache_hit',dict(kind='map',id=0,theta=np.zeros(2)))
            return
        r.consume(kind,p)
    assert False

def headers(bound):
    screen=2*(1.1480005020275712*max(256/12,102/24)+38.103352084057406*(16/473))+5
    return [['bound_inputs',bound],['cost_gate',dict(projected_seconds=screen,limit_seconds=60.,passed=True,scope='historical heuristic only')]]

def write_records(folder,events):
    folder.mkdir();journal=[]
    for index,(kind,payload)in enumerate(events):
        arrays={}
        def encode(v):
            if isinstance(v,np.ndarray):
                key=f'a{len(arrays):04d}';arrays[key]=v;return dict(array=key,dtype=v.dtype.str,shape=list(v.shape),order='C',sha256=A.G.digest(np.ascontiguousarray(v).tobytes()))
            if isinstance(v,np.generic):return v.item()
            if isinstance(v,dict):return{k:encode(x)for k,x in v.items()}
            if isinstance(v,list):return[encode(x)for x in v]
            return v
        name=f'{index:05d}-{kind}.json';record=dict(kind=kind,sequence=index,payload=encode(payload))
        if arrays:
            af=name[:-5]+'.npz';np.savez(folder/af,**arrays);record.update(arrays_file=af,arrays_sha256=A.G.entry(folder/af)['sha256'])
        (folder/name).write_bytes(A.G.canonical(record));journal.append(dict(sequence=index,kind=kind,file=name))
    (folder/'journal.jsonl').write_text(''.join(json.dumps(v)+'\n'for v in journal));return A.G.directory(folder)

def test_full_store_schema_join_and_counter_reconstruction(tmp_path):
    f=fixture();bound={'synthetic':True};events=headers(bound)+f['events'];files=write_records(tmp_path/'store',events)
    result=dict(**f['result'],operation_attempted=f['attempted'],operation_returned=f['returned'],stage_counts=f['counts'])
    d=A.records(tmp_path/'store',files,result,f['initial'],bound,hidden=1);assert d['arithmetic_complete']and d['accepted_edges']==[0,1]

def test_complete_claim_rejects_orphan_arrays(tmp_path):
    f=fixture();files=write_records(tmp_path/'store',headers({})+f['events']);files['99999-unused.npz']=dict(bytes=0,sha256='0'*64)
    result=dict(**f['result'],operation_attempted=f['attempted'],operation_returned=f['returned'],stage_counts=f['counts'])
    with pytest.raises(ValueError):A.records(tmp_path/'store',files,result,f['initial'],{},hidden=1)


@pytest.mark.parametrize('header_index',[0,1])
def test_mandatory_headers_cannot_be_removed_and_reindexed(tmp_path,header_index):
    f=fixture();events=headers({})+f['events'];events.pop(header_index);files=write_records(tmp_path/'store',events)
    result=dict(**f['result'],operation_attempted=f['attempted'],operation_returned=f['returned'],stage_counts=f['counts'])
    with pytest.raises(ValueError):A.records(tmp_path/'store',files,result,f['initial'],{},hidden=1)

def test_pre_engine_failure_remains_inventory_only(tmp_path):
    f=fixture();files=write_records(tmp_path/'store',[['bound_inputs',{}]])
    result=dict(status='failed_or_interrupted',operation_attempted={},operation_returned={},stage_counts={})
    d=A.records(tmp_path/'store',files,result,f['initial'],{},hidden=1);assert d['uninitialized_engine']and not d['arithmetic_complete']

def test_exact_frozen_source_inventory_rejects_omission():
    freeze=A.G.read(ROOT/A.FROZEN);A.source_check(freeze['files'],freeze['commit'])
    files=copy.deepcopy(freeze['files']);files.pop(next(iter(files)))
    with pytest.raises(ValueError):A.source_check(files,freeze['commit'])


def test_stage_increment_before_emit_stays_partial(tmp_path):
    f=fixture();files=write_records(tmp_path/'store',headers({}))
    result=dict(status='failed_or_interrupted',operation_attempted=dict.fromkeys(A.CAPS,0),operation_returned=dict.fromkeys(A.CAPS,0),stage_counts=dict(correctors=0,qualifiers=1,common=0,edges=0,assemblies=0),accepted_edges=[],last_witnessed=None)
    d=A.records(tmp_path/'store',files,result,f['initial'],{},hidden=1);assert not d['arithmetic_complete']and d['qualifiers']==0

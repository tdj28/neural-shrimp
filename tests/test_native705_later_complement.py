import copy,json,shutil,subprocess,sys,gzip
from pathlib import Path
from fractions import Fraction as F
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native705_later_complement as P
import launch_native705_later_complement as L

def card():return json.loads((ROOT/P.CARD).read_text())
def example():
    c=card();r=c['charts'][0];g=r['producer']['settings'];v=P.cell(g['parameter'],str(F(2)**g['halfwidth_exponent']),r['initial_depth'],0);return c,L.make_assignment(1,r,v,'source')

def test_all_eight_initial_partitions_exact_and_fixed():
    c,job=example();assert L.BASE==P.BASE and c['limits']==L.LIMITS
    total=0
    for ordinal in range(31,39):
        r=P.M.select(c,ordinal);g=r['producer']['settings'];h=str(F(2)**g['halfwidth_exponent']);cells=[P.cell(g['parameter'],h,r['initial_depth'],i)for i in range(2**r['initial_depth'])]
        assert L.coverage(g['parameter'],h,cells)['complete'];assert all(F(v['halfwidth'])==F(1,128)for v in cells);total+=len(cells)
    assert total==24;P.assignment(c,job,1,'source')
    for key,v in [('ordinal',30),('source_commit','bad'),('attempt',2),('root','/tmp/wrong')]:
        with pytest.raises(ValueError):P.assignment(c,dict(job,**{key:v}),1,'source')
    bad=copy.deepcopy(job);bad['cell']['depth']=0
    with pytest.raises(ValueError):P.assignment(c,bad,1,'source')

def test_actual_eight_proof_and_provenance_metadata():
    # No model, derivative, interval matrix reduction or target acquisition.
    for row in card()['charts']:
        i=row['independent'];p=row['producer'];ir=Path(i['root'])/'run';run=Path(p['run']);proof=P.read(ir/i['proof']['path'])
        P.M.proof_gates(proof,row);P.M.declared_producer(row,i['card'])
        assert P.O.entry(ir/i['proof']['path'])==i['proof']['identity']
        assert P.read(ir/'input-binding.json')['card']==i['card']
        assert P.read(run/'inputs.json')['settings']==p['settings']
        assert P.O.entry(run/p['domain']['path'])==p['domain']['identity']
        for key in ['passed']:
            q=copy.deepcopy(proof);q[key]=False
            with pytest.raises(ValueError):P.M.proof_gates(q,row)
        q=copy.deepcopy(proof);q['root']['coordinate_inclusions'][704]=False
        with pytest.raises(ValueError):P.M.proof_gates(q,row)
        q=copy.deepcopy(proof);q['chart']['parameter_center']='0'
        with pytest.raises(ValueError):P.M.proof_gates(q,row)

def test_independent_column_outward_restore_and_corruption(tmp_path):
    p=tmp_path/'coeff.json.gz';value=dict(encoding='mid-rad-dyadic-pairs',rows=[[[[3,-7],[7,-80]]]])
    p.write_bytes(gzip.compress(json.dumps(value).encode()))
    with ctx.workprec(384):
        m=P.M.coefficients(tmp_path,[dict(path=p.name)],n=1);assert m.nrows()==m.ncols()==1
        assert m[0,0].contains(arb((3,-7))+arb((7,-80)))and m[0,0].contains(arb((3,-7))-arb((7,-80)))
    value['rows'][0][0][1][0]=-7;p.write_bytes(gzip.compress(json.dumps(value).encode()))
    with pytest.raises(ValueError):P.M.coefficients(tmp_path,[dict(path=p.name)],n=1)

def test_only_complete_metric_failure_splits():
    c,j=example();done=dict(assignment=j,unit=1,passed=False,status='completed_correlated_bound_unresolved',metric_passed=False,complement_dimension=11,native_calls=360,profile_native_calls=180,target_native_calls=180)
    assert L.classify(done,j,dict(exit_code=2))=='split'
    assert L.classify(dict(done,native_calls=359),j,dict(exit_code=2))=='stop'
    assert L.classify(dict(done,status='failed_or_interrupted'),j,dict(exit_code=2))=='stop'

def test_independent_full_parameter_domain_and_join_corruptions():
    row=card()['charts'][4];proof=P.read(Path(row['independent']['root'])/'run'/row['independent']['proof']['path'])
    for mutate in [lambda q:q['membership'].__setitem__('parameter_inside',False),lambda q:q['membership']['coordinate_inclusions'].__setitem__(704,False),lambda q:q['proper_divisors'].__setitem__('9',[])]:
        q=copy.deepcopy(proof);mutate(q)
        with pytest.raises(ValueError):P.M.proof_gates(q,row)

def test_reserved_failure_receipts_and_complete_source_archive(tmp_path):
    used=[380*1024**2-4]
    with pytest.raises(ValueError):P.save_bound(tmp_path,'reject.json',dict(message='larger than reserve'),used)
    P.save_bound(tmp_path,'outcome.json',dict(passed=False),used,True)
    for n in P.FILES:
        d=tmp_path/'archive'/n;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,d)
    out=tmp_path/'archive';code="import sys,pathlib;sys.path.insert(0,'tools');import probe_native705_later_complement as p;r=pathlib.Path.cwd();loaded={str(pathlib.Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and pathlib.Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(p.FILES),loaded-set(p.FILES);assert p.ROOT==r"
    subprocess.run([sys.executable,'-I','-B','-c',code],cwd=out,check=True,capture_output=True)

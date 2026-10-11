import copy,gzip,json,importlib.util
from pathlib import Path
import pytest
p=Path(__file__).parents[1]/'tools/diagnose_eigenpair_partial.py';s=importlib.util.spec_from_file_location('partial',p);D=importlib.util.module_from_spec(s);s.loader.exec_module(D)


def fixture():
    gates=dict.fromkeys(('uniform_normalized_eigenpair_locus','primitive_period','anchor_identified'),True)
    q=dict(passed=True,gates=gates,period=18,dimension=13,sigma=1,shooting_dimension=470,normal=['n'],origin=['o'],parameter_endpoints=['215/128','219/128'],settings=dict(precision=256,newton_steps=2,halfwidth_exponent=-6))
    failed=copy.deepcopy(q);failed.update(passed=False,parameter_endpoints=['219/128','223/128'],coordinate_inclusions=[True]*470);failed['gates']['uniform_normalized_eigenpair_locus']=False;failed['coordinate_inclusions'][364]=False
    last=dict(call=1,index=0,right='219/128',full_chart_file=D.QUALIFIED_FILE,accepted_file=D.ACCEPTED_FILE)
    out=dict(passed=False,status='qualified_prefix_then_unresolved',calls=4,repairs=2,progress=dict(accepted=[last],last_accepted=last,active_call=3,full_chart_file=D.FAILED_FILE))
    attempts=[dict(call=i,passed=i==1)for i in range(4)];result=dict(completed=False,calls=4,repairs=2,accepted=[attempts[1]],attempts=attempts,last_chart=copy.deepcopy(q))
    accepted=dict(call=1,index=0,chart=copy.deepcopy(q));anchor=copy.deepcopy(q);anchor['parameter_endpoints']=['211/128','215/128']
    return out,result,accepted,q,failed,anchor


def test_exact_partial_prefix_allowed_but_failed_tail_not_parent():
    args=fixture();assert D.validate_prefix(*args)==args[3]


@pytest.mark.parametrize('case',['complete','wrong_stop','failed_prefix','wrong_call','missing_return','wrong_tail','second_prefix','wrong_gauge','flip_mode','width','different_coordinate','wrong_accepted_summary'])
def test_failclosed_prefix(case):
    o,r,a,q,f,anchor=fixture()
    if case=='complete':o['passed']=True
    if case=='wrong_stop':o['status']='failed_or_interrupted_with_retained_prefix'
    if case=='failed_prefix':q['passed']=False
    if case=='wrong_call':a['call']=3
    if case=='missing_return':o['progress']['full_chart_file']=None
    if case=='wrong_tail':f['parameter_endpoints']=['223/128','227/128']
    if case=='second_prefix':o['progress']['accepted'].append(copy.deepcopy(o['progress']['accepted'][0]))
    if case=='wrong_gauge':f['normal']=['different']
    if case=='flip_mode':f['sigma']=-1
    if case=='width':f['settings']['halfwidth_exponent']=-7
    if case=='different_coordinate':f['coordinate_inclusions'][364]=True;f['coordinate_inclusions'][363]=False
    if case=='wrong_accepted_summary':r['accepted'][0]=r['attempts'][3]
    with pytest.raises(ValueError):D.validate_prefix(o,r,a,q,f,anchor)


def test_gzip_record_decoder(tmp_path):
    p=tmp_path/'record.json.gz';p.write_bytes(gzip.compress(json.dumps({'passed':False}).encode()))
    assert D.read_gzip(p)=={'passed':False}


def test_complete_transitive_inventory():
    assert set(D.R.SOURCES)<=set(D.SOURCES)
    assert all((Path(__file__).parents[1]/n).is_file()for n in D.SOURCES)

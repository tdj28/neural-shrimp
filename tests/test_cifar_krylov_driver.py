import copy,importlib.util,json,shutil,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import diagnose_cifar_krylov as D

def events():
    rows=[];inv={}
    def item(name):
        value=dict(bytes=1,sha256=name);inv['run/'+name]=value;return dict(file=name,identity=value)
    for op,count in [('direct_gradient',23),('hvp',1030)]:
        for i in range(1,count+1):
            r=dict(kind='native_input',operation=op,index=i,state=item('state-000.npy'))
            if op=='hvp':r['direction']=item(f'native-hvp-{i:04d}-vector.npy')
            rows.extend([r,dict(kind='operation_started',operation=op,index=i),dict(kind='operation_returned',operation=op,index=i,seconds=.01),dict(kind='native_output',operation=op,result=item(f'native-{op}-{i:04d}-return.npy'))])
    return rows,inv

def test_exact_complete_event_indices_order_and_inventory():
    e,inv=events();r=D.events_index(e,inv);assert len(r['hvp'])==1030
    for edit in ['missing','reorder','hash','index','filename']:
        bad=copy.deepcopy(e)
        if edit=='missing':bad.pop()
        if edit=='reorder':bad[0],bad[1]=bad[1],bad[0]
        if edit=='hash':bad[0]['state']['identity']['sha256']='wrong'
        if edit=='index':bad[0]['index']=2
        if edit=='filename':bad[3]['result']=bad[7]['result']
        with pytest.raises(ValueError):D.events_index(bad,inv)

def test_literal_fixed_card_states_and_no_native_import(tmp_path):
    c=json.loads((ROOT/D.CARD).read_text());assert c['limits']==D.LIMITS and c['settings']==D.SETTINGS
    assert [r['state']for r in c['operators']]==['state-000.npy','state-010.npy','state-014.npy','state-018.npy']
    assert c['producer']['source']=='59764281d2aaeeb7ed4f18da2044dc796afb4923'
    for name in D.FILES:
        out=tmp_path/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,out)
    code='import sys,pathlib;sys.path.insert(0,'+repr(str(tmp_path/'tools'))+');import diagnose_cifar_krylov as d;loaded={str(pathlib.Path(m.__file__).resolve().relative_to(d.ROOT))for m in list(sys.modules.values())if getattr(m,"__file__",None)and pathlib.Path(m.__file__).resolve().is_relative_to(d.ROOT)};assert loaded<=set(d.FILES),loaded-set(d.FILES);assert not any(k in sys.modules for k in ["torch","scipy","explore_cifar_stationary_continuation"]);assert all((d.ROOT/n).is_file()for n in d.FILES)'
    subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)

def test_driver_and_launcher_exact_owned_envelope():
    import launch_cifar_krylov as L
    assert D.OUT.parent==L.OUT
    text=(ROOT/'tools/launch_cifar_krylov.py').read_text();assert '180' in text and '195' in text and '66*1024**2' in text and '6*1024**3' in text

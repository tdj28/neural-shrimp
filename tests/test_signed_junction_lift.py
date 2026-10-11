import sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from flint import arb,ctx
from neural_shrimp.rigorous_orbits import ball_json
from neural_shrimp.rigorous_signed_junction_lift import saved_flip_lift
import pytest

def fixture():
    b=lambda v:ball_json(arb(v));n=3;p=3;x=[b(0)]*n;q=[b(0),b(1),b(0)];r=[b(0),b(0),b(1)];w=[b(1),b(0),b(0)];rates=[b(1),b(2)];root=x+rates+q+r+[b('0.125')]+w;unit=(x+q+r)*p+rates+[b('0.125')];rows=[]
    for k in range(p):
        # Three native DF factors transport w to minus itself. Other mode
        # identities are external premises and not claimed by this fixture.
        B=[[b(-1 if i==j==0 else int(i==j))for j in range(6)]for i in range(6)]
        for name,v in [('q',q),('r',r)]:rows.append(dict(phase=k,direction=name,input=x+v+rates,value=[[v]for v in x+v],block=B))
    return root,unit,rows

def test_literal_three_phase_negative_transport_and_complete_coordinates():
    with ctx.workprec(512):
        root,unit,rows=fixture();seen=[];box,proof=saved_flip_lift(root,unit,rows,n=3,period=3,observe=lambda k,v:seen.append((k,v)))
        assert len(box)==20 and all(proof['sanity'].values())and proof['native_calls']==0 and len(seen)==6
        assert box[3]==1 and box[9]==-1 and box[15]==1

def test_same_phase_domain_and_direction_independence_reject_corruption():
    with ctx.workprec(512):
        for which in ('state','rate','mode','df'):
            root,unit,rows=fixture()
            if which=='state':rows[2]['input'][0]=ball_json(arb(1))
            elif which=='rate':rows[2]['input'][-1]=ball_json(arb(3))
            elif which=='mode':rows[2]['direction']='r'
            else:rows[1]['block']=copy.deepcopy(rows[1]['block']);rows[1]['block'][0][0]=ball_json(arb(2))
            with pytest.raises(ValueError):saved_flip_lift(root,unit,rows,n=3,period=3)

def test_wrong_sign_or_normalization_is_sanity_failure_never_root_claim():
    with ctx.workprec(512):
        root,unit,rows=fixture()
        for record in rows:record['block']=copy.deepcopy(record['block']);record['block'][0][0]=ball_json(arb(1))
        _,p=saved_flip_lift(root,unit,rows,n=3,period=3);assert not p['sanity']['mode_closed']
        root[-3]=ball_json(arb(2));_,p=saved_flip_lift(root,unit,rows,n=3,period=3);assert not p['sanity']['normalized']

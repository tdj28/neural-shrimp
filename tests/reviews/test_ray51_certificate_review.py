"""Independent synthetic identities and saved-ball controls; no target proof."""
import copy,importlib.util,math
from fractions import Fraction
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
producer=module('ray51_producer_independent','tools/certify_s2_ray51.py')
review=module('ray51_reviewer','tools/reviews/review_ray51_certificates.py')

def fixture():
 parent={'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh'},'x':[['1/2','1/4']],'y':['1/3']}
 center=[-2.,-3.];scale=[.1,.2];direction=[1/3,-.625];ray={'type':'flip','direction':direction};window={'center':center,'scale':scale,'rays':[copy.deepcopy(ray) for _ in range(52)]}
 f={'schema':'neural-shrimp-exact-log-ray-v1','ell':[str(Fraction(x)) for x in center],'ell_float':center,'direction':[str(Fraction(a*b)) for a,b in zip(scale,direction)],'direction_float':[a*b for a,b in zip(scale,direction)],'historical_scale':scale,'historical_unit_direction':direction}
 state=[i/16 for i in range(13)];q=[1.]+[0.]*12;s=.1875;row={'map_id':'S2-28011-zoom','component':587,'period':18,'certificate':{'orbit':[state[:] for _ in range(18)]}}
 point={'qualified':True,'state':state,'q':q,'s':s};num={'row_index':111,'ray_index':51,'period':18,'ray':ray,'historical_orbit':row['certificate']['orbit'],'consumed_map':parent,'family':f}
 seed={'schema':'neural-shrimp-flip-proof-seed-v1','period':18,'ray_index':51,'consumed_map':parent,'family':f,'theta':[str(Fraction(x)) for x in state],'parameter':str(Fraction(s)),'right_vector':[str(Fraction(x)) for x in q]}
 rates=[10.**float(e+s*d) for e,d in zip(center,f['direction_float'])]
 branch={'qualified':True,'qualification':{'stability':'attracting'},'opposite_qualification':{'stability':'attracting'},'state':state,'s':s,'data':{'period':36,'theta':state,'orbit':[state[:] for _ in range(37)],'s':s,'rates':rates,'rates_exact':[str(Fraction(x)) for x in rates]}}
 branches=[]
 for a in (.001,.002,.004):b=copy.deepcopy(branch);b['amplitude_requested']=a;branches.append(b)
 return parent,row,window,num,point,seed,branches

def test_independent_reconstruction_of_both_distinct_inputs():
 args=fixture()
 for kind in ('flip','period36'):
  inp,selection=producer.assemble(kind,*args);detail=review.input_identity(kind,inp,*args);assert detail['map_scope']
 assert Fraction(.1*(1/3))!=Fraction(.1)*Fraction(1/3)
 assert review.exact_family(args[2])['direction'][0]==str(Fraction(.1*(1/3)))

@pytest.mark.parametrize('change',['unrounded_product','swap_rates','perturb_rate','phase','omit_bias_model','other_slot','collapse_to18','opposite_unstable'])
def test_specific_adapter_mutations_fail(change):
 args=list(copy.deepcopy(fixture()))
 if change=='unrounded_product':args[5]['family']['direction'][0]=str(Fraction(.1)*Fraction(1/3))
 elif change=='swap_rates':args[-1][2]['data']['rates_exact'].reverse()
 elif change=='perturb_rate':args[-1][2]['data']['rates_exact'][0]=str(Fraction(math.nextafter(args[-1][2]['data']['rates'][0],math.inf)))
 elif change=='phase':args[-1][2]['data']['orbit'][0][0]+=.01
 elif change=='omit_bias_model':args[3]['consumed_map']=dict(args[3]['consumed_map'],network={'activation':'relu'})
 elif change=='other_slot':args[-1][0]['amplitude_requested']=.1
 elif change=='collapse_to18':args[-1][2]['data']['period']=18
 else:args[-1][2]['opposite_qualification']['stability']='unstable'
 with pytest.raises(ValueError):producer.assemble('period36',*args)

def test_review_rejects_silent_exponential_substitution():
 args=fixture();inp,_=producer.assemble('period36',*args);inp['lr_hidden']=str(Fraction(inp['lr_hidden'])+Fraction(1,2**120))
 with pytest.raises(ValueError):review.input_identity('period36',inp,*args)

def ball(mid=0,exp=0,rad=0,rexp=0):return {'mid':[mid,exp],'rad':[rad,rexp]}
def scalar_matrix(value,n=13):return [[ball(value if i==j else 0) for j in range(n)] for i in range(n)]
def stored_orbit():
 box=[ball(rad=1,rexp=-20) for _ in range(13)];center=[ball() for _ in range(13)];ds=[1,2,3,4,6,9,12,18];orbit=[center[:] for _ in range(37)]
 for d in ds:orbit[d]=[ball(1)]+center[1:]
 m=[[ball(1,-1) if i==j else ball() for j in range(13)] for i in range(13)]
 return {'passed':True,'evidence_rung':3,'exact_statement':{'primitive_period':36,'dimension':13},'divisors':[{'divisor':d} for d in ds],'box':box,'center':center,'krawczyk_image':center,'defect':scalar_matrix(0),'monodromy':m,'preconditioner':scalar_matrix(-2),'residual':[[ball()] for _ in range(13)],'orbit_enclosures':orbit,'attraction_witness_power':1,'attraction_witness':m}

def test_independent_saved36_reduction():
 result=review.orbit_math(stored_orbit());assert result['divisors']==[1,2,3,4,6,9,12,18] and result['power']==1

@pytest.mark.parametrize('change',['omitted_divisor','unseparated18','false_attraction','false_contraction'])
def test_saved36_false_claims_rejected(change):
 result=stored_orbit()
 if change=='omitted_divisor':result['divisors'].pop()
 elif change=='unseparated18':result['orbit_enclosures'][18]=result['center']
 elif change=='false_attraction':result['attraction_witness']=scalar_matrix(2)
 else:result['defect']=scalar_matrix(2)
 with pytest.raises(ValueError):review.orbit_math(result)

@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf'),True])
def test_resource_values_fail_closed(value):
 with pytest.raises(ValueError):review.finite_time(value,60)

def test_nested_box_strict_and_proximity_uses_radius():
 assert review.nested([ball(rad=1,rexp=-30)],[ball(rad=1,rexp=-20)])
 assert not review.nested([ball(rad=1,rexp=-20)],[ball(rad=1,rexp=-20)])
 with pytest.raises(ValueError):review.proximity({'approximation':['0']},{'box':[ball(rad=1,rexp=-10)]})

def test_inventory_extra_and_manifest_pin_fail(tmp_path):
 (tmp_path/'a').write_text('x');m={'files':review.inventory(tmp_path),'total_bytes':1};(tmp_path/'manifest.json').write_text(__import__('json').dumps(m));pin=review.sha(tmp_path/'manifest.json');review.verify(tmp_path,pin)
 (tmp_path/'extra').write_text('y')
 with pytest.raises(ValueError):review.verify(tmp_path,pin)

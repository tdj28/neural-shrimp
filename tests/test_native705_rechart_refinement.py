import importlib,sys,json,copy
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import refine_native705_rechart_endpoint as W
import launch_native705_rechart_refinement as L

def test_actual_retained_metadata_and_literal_paths():
 c=W.read(ROOT/W.CARD);r=Path(c['pilot']['root']);o=W.read(r/'outcome.json')
 assert c['pilot']['source']==o['source']and W.entry(r/'manifest.json')['sha256']==c['pilot']['outer']and o['connected_count']==3
 assert[(x['attempt'],x['root_passed'],x['joined'])for x in o['attempts']]==[(1,True,True),(2,True,True),(3,False,False),(4,True,True),(5,True,False)]
 old=W.read(r/'attempt004/input-card.json');new=W.read(r/'attempt005/input-card.json')
 assert old['settings']['parameter']=='55/512'and new['settings']['parameter']=='31/2048'and new['mode']=='rotated_gauge'and'exact_parameter'not in new['anchor']
 assert W.OUT.parent==L.C.OUT.parent and L.BASE==W.BASE
 for name in W.SOURCES:assert(ROOT/name).is_file()

def test_source_closure_and_inclusive_operation_caps():
 loaded={str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)}
 assert {n for n in loaded if not n.startswith('tests/')}<=set(W.SOURCES)
 w=W.Work();w.native=72
 with pytest.raises(ValueError):w.tick()
 w=W.Work();w.saved=144
 with pytest.raises(ValueError):w.replay_tick()
 assert 'LIMIT=255'in Path(L.__file__).read_text()and '514*1024**2'in Path(L.__file__).read_text()

def test_actual_all72_saved_pairs_metadata_only():
 r=Path(W.read(ROOT/W.CARD)['pilot']['root'])/'attempt004/run'
 # File names and immutable event ordering only; no derivative values decoded.
 a=sorted((r/'events').glob('*-gradient_path_input.json.gz'));b=sorted((r/'events').glob('*-gradient_path_return.json.gz'))
 assert len(a)==len(b)==72 and all(int(y.name.split('-')[0])==int(x.name.split('-')[0])+1 for x,y in zip(a,b,strict=True))

def test_exact_geometry_rejects_changed_original_error_domain():
 with ctx.workprec(480):
  from neural_shrimp.rigorous_orbits import vector_json
  z=vector_json([arb(0)]*705);domain={k:z for k in('center','slope','curvature','cubic')};domain.update(parameter='1/8',halfwidth_exponent=-6,radius_exponent=-10,error=vector_json([arb(0,arb((1,-10)))]*705))
  c=dict(domain,shooting_dimension=705,period=18,dimension=13,parameter_center='1/8',normal=['0']*704+['1'],origin=['0']*705,weights=['1']*705,settings=dict(halfwidth_exponent=-6,radius_exponent=-10))
  assert W.geometry(c,domain)[-1]==arb(1)/64
  domain['error'][1]={'mid':[0,0],'rad':[1,-20]}
  with pytest.raises(ValueError):W.geometry(c,domain)

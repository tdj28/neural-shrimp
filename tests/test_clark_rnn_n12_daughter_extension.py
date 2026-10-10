import importlib.util
from pathlib import Path

def module():
 p=Path(__file__).parents[1]/'tools/extend_clark_rnn_n12_daughters.py';s=importlib.util.spec_from_file_location('extension',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def row(radius,sign):
 return {'maximum_native_radius':radius,'spectra':[{'leading_real_mode':None if sign is None else {'sign':sign}}for _ in range(2)]}

def test_actual_native_crossings_and_no_crossing():
 m=module()
 assert m.crossing(row(.9,-1),row(1.1,-1))['kind']=='actual_native6_minus_bracket'
 assert m.crossing(row(1.1,1),row(.9,1))['direction']=='gain'
 assert m.crossing(row(1.1,1),row(.9,1))['kind']=='actual_native6_plus_bracket'
 assert m.crossing(row(.9,1),row(1.1,None))['kind']=='other_native6_stability_crossing'
 assert m.crossing(row(.8,1),row(.9,1))is None
 assert m.crossing(row(1.1,-1),row(.9,1))['kind']=='other_native6_stability_crossing'
 bad=row(1.1,1);bad['spectra'][1]['leading_real_mode']['sign']=-1
 assert m.crossing(row(.9,1),bad)['kind']=='other_native6_stability_crossing'

def test_fixed_extension_and_same_resources():
 m=module();assert len(m.AMPLITUDES)==32 and m.AMPLITUDES[0]==m.START+.1 and m.AMPLITUDES[-1]==m.START+3.2
 assert issubclass(m.Work,m.D.Work)and m.Work.qualify is m.D.Work.qualify and m.Work.correct_pair is m.D.Work.correct_pair

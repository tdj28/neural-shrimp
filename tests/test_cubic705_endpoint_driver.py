import importlib.util,json,gzip
from pathlib import Path
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('saved_endpoint',ROOT/'tools/localize_cubic705_endpoint.py');W=importlib.util.module_from_spec(spec);spec.loader.exec_module(W)

def test_actual_fixed_paths_and_complete_source_closure():
 assert str(W.TARGET).endswith('/cubic-continuation001/endpoint-localization001/attempt001/run')
 assert W.NEW_SOURCE=='c73978cb1bb7f747972cdf669a7f687a807b5b24'and W.AUDIT_SOURCE=='f2f0d768542d3f2bafc93f445a1c928f3d019f81'
 for n in W.SOURCES:assert(ROOT/n).is_file()
 launcher=(ROOT/'tools/launch_cubic705_endpoint.py').read_text();assert 'localize_cubic705_endpoint.py'in launcher and 'LIMIT=135'in launcher and '6*1024**2'in launcher

def test_exact_encoded_matrix_enclosure_and_rejection(tmp_path):
 p=tmp_path/'x.json.gz';row=dict(encoding='mid-rad-dyadic-pairs',rows=[[[[1,-10],[536870913,-60]]]])
 p.write_bytes(gzip.compress(json.dumps(row).encode()))
 with ctx.workprec(480):
  M=W.matrix_file(p,1,1);assert M[0,0].contains(arb((1,-10)))and M[0,0].rad()>=arb((536870913,-60))
 with pytest.raises(ValueError):W.matrix_file(p,2,1)
 row['rows'][0][0][1][0]=-1;p.write_bytes(gzip.compress(json.dumps(row).encode()))
 with pytest.raises(ValueError):W.matrix_file(p,1,1)

def test_full_independent_authority_and_endpoint_identity():
 chart=dict(parameter_endpoints=['83/512','115/512']);ep=dict(chart=chart,endpoint=['right']);proof=dict(passed=True,gates={'root':True},root={'coordinate_inclusions':[True]*705},chart=chart,endpoints=[['left'],['right']]);out=dict(passed=True,connected=True,connected_count=3,connected_right='115/512',native_calls=0,charts=[dict(passed=True,strict_inclusions=705,anchor_coordinates=705)]*3)
 assert W.admission(out,proof,ep)
 proof['root']['coordinate_inclusions'].pop()
 with pytest.raises(ValueError):W.admission(out,proof,ep)

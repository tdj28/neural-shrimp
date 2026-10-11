import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/reviews'))
import diagnose_native704_domains as D

def test_exact_dyadic_endpoints_without_rounding():
 old=[{'mid':[1,0],'rad':[1,-20]}]*704;new=[{'mid':[1,0],'rad':[1,-21]}]*704
 assert D.compare(old,new)['contained']
 new[702]={'mid':[1048578,-20],'rad':[1,-20]};r=D.compare(old,new)
 assert r['failed_coordinates']==1 and r['rows'][0]['kind']=='rate'and r['max_relative_excess']==1

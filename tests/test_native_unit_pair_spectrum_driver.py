import copy,importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).parents[1]
def load(name):
 s=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
D=load('probe_native_unit_pair_spectrum');L=load('launch_native_unit_pair_spectrum')
def b(i):return {'mid':[i,0],'rad':[0,0]}
def fixture():
 n=3;p=2;size=3*n*p+2;flat=[b(i+1)for i in range(size)];J=[[b(0)for _ in range(size)]for _ in range(size)];rows=[]
 for k in range(p):
  off=3*n*k;block=[[b((i+1)*(j+2)+k)for j in range(2*n)]for i in range(2*n)]
  for i in range(n):
   for j in range(n):J[off+i][off+j]=block[i][j]
  rows.extend([{'phase':k,'direction':'q','input':flat[off:off+2*n]+flat[-2:],'block':copy.deepcopy(block)},
               {'phase':k,'direction':'r','input':flat[off:off+n]+flat[off+2*n:off+3*n]+flat[-2:],'block':copy.deepcopy(block)}])
 return {'dimension':n,'period':p,'whole_box':flat,'whole_jacobian':J},rows

def test_all_native_domains_and_full_jacobian_correspond():
 p,rows=fixture();A=D.native_blocks(p,rows);assert len(A)==2 and len(A[0])==3
 assert D.BASE==L.BASE
@pytest.mark.parametrize('mode',['state','q','r','rate','phase','jacobian','count'])
def test_wrong_actual_whole_domain_or_derivative_cannot_supply_quotient(mode):
 p,rows=fixture()
 if mode=='count':rows.pop()
 elif mode=='phase':rows[0]['phase']=1
 elif mode=='jacobian':p['whole_jacobian'][0][0]=b(99)
 else:
  row=1 if mode=='r'else 0;i={'state':0,'q':3,'r':3,'rate':-1}[mode];rows[row]['input'][i]=b(99)
 with pytest.raises(ValueError):D.native_blocks(p,rows)

def test_failed_target_never_reaches_conditional_spectrum():
 with pytest.raises(ValueError):D.admit({}, {'passed':False})

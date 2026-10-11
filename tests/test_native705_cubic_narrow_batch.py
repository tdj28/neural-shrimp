import copy,sys,subprocess
from fractions import Fraction
from pathlib import Path
import pytest
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native705_cubic_batch as W
import probe_native705_cubic_narrow_batch as N
import launch_native705_cubic_narrow_batch as L

def test_only_fixed_width_adapter_changes_domains_and_paths():
 old=(W.BASE,W.HALFWIDTH,W.WORKER,W.SOURCES)
 try:
  N.configure();right=Fraction(79,2048)
  for i in range(1,5):
   cfg=W.settings('79/2048',i);assert cfg['halfwidth_exponent']==-5 and cfg['radius_exponent']==-10 and cfg['newton_steps']==2
   c=Fraction(cfg['parameter']);assert c-Fraction(1,32)<right<c+Fraction(1,32);right=c+Fraction(1,32)
  assert right==Fraction(587,2048)and W.BASE==W.P.BASE/'narrow001'and W.WORKER.endswith('cubic_narrow_batch.py')
  assert all((ROOT/n).is_file()for n in W.SOURCES)
 finally:W.BASE,W.HALFWIDTH,W.WORKER,W.SOURCES=old

def test_actual_narrow_entrypoints_and_default_unchanged():
 assert W.HALFWIDTH==-4
 for name in ['probe_native705_cubic_narrow_batch.py','launch_native705_cubic_narrow_batch.py']:
  r=subprocess.run([sys.executable,str(ROOT/'tools'/name),'--help'],text=True,capture_output=True);assert r.returncode==0 and 'usage:'in r.stdout
 assert W.HALFWIDTH==-4

import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prompt_control_probe import newton_fixed_point

def test_continues_unstable_fixed_point():
 torch.set_num_threads(1)
 x=torch.ones((1,1,896),dtype=torch.float64)
 # Fixed-point iteration diverges, but Newton must recover this branch.
 F=lambda x:-1.2*x+.5
 J=lambda x,v:-1.2*v
 root,n,res=newton_fixed_point(x,F,J)
 torch.testing.assert_close(root,torch.full_like(x,.5/2.2),rtol=1e-10,atol=1e-12)
 assert n<=3 and res<1e-11

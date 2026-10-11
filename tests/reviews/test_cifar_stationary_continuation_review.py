import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'tools/reviews'))
import review_cifar_stationary_continuation as A

def test_explicit_original_origin_and_descent():
 o=np.ones(7);x=o+.02;g=np.ones(7)*.01;p=-g;hp=-g
 z,s=A.arithmetic(o,x,g,p,hp,[(x+p,g*.5)])
 assert z['accepted_index']==0 and s['relative_origin_distance']==pytest.approx(.01) and s['merit_ratio']==pytest.approx(.25)

def test_global_trust_rejects_recentering():
 o=np.ones(4);x=o+.04;g=np.ones(4)*.01;p=np.ones(4)*.02;hp=-g
 with pytest.raises(ValueError):A.arithmetic(o,x,g,p,hp,[(x+p,g*.5)])

def test_wrong_trial_coordinates_rejected():
 x=np.ones(3);g=x*.01;p=-g
 with pytest.raises(ValueError):A.arithmetic(x,x,g,p,-g,[(x+p+1e-5,g*.5)])

def test_wrong_forcing_and_extra_trial_rejected():
 x=np.ones(3);g=x*.01;p=-g
 with pytest.raises(ValueError):A.arithmetic(x,x,g,p,g,[(x+p,g*.5)])
 with pytest.raises(ValueError):A.arithmetic(x,x,g,p,-g,[(x+p,g*.5),(x+p/2,g*.25)])

def test_event_order_rejects_orphan():
 with pytest.raises(ValueError):A.bind_events(Path('/tmp'),[dict(kind='native_output')],{},lambda:None)

def test_fixed_caps_and_paths():
 import launch_cifar_stationary_continuation_review as L
 assert L.OUT==A.OUT.parent and len(A.PINS)==2 and len(A.EXEC)==40

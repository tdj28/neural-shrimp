import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import baby_shrimp_jones_finish as finish


def test_branch_word_retains_unsupported_domain_and_critical_ambiguity():
    crit=[{"coordinate":.1},{"coordinate":.6}]
    assert finish.projected_word([.1,.05,.3,.8,1.01],crit)==["turn","0","1","2","outside"]


def test_precision_checks_recompute_observed_orders():
    steps=np.array([.01,.003,.001]);error=(steps**2)[:,None]*np.ones((1,3))
    orders=np.log(error[:-1]/error[1:])/np.log(steps[:-1]/steps[1:])[:,None]
    row={"finite_differences":[{"h":h,"relative_errors":e.tolist()} for h,e in zip(steps,error)],
         "orders":orders.tolist(),"plain_vs_ad_primal_error":0,"derivative_qualified":True}
    assert finish.precision(row)

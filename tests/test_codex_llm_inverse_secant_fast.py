import sys
from pathlib import Path
import time

from flint import arb,arb_mat,ctx
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'codex/certify-llm'))
import inverse_secant as original
import inverse_secant_fast as fast


@pytest.mark.parametrize('precision',[128,256,2048])
@pytest.mark.parametrize('scenario',['linear','nonnormal','collinear','zero','noisy','update_cap','step_cap'])
def test_entire_decision_and_proposal_history_matches_original(precision,scenario):
    with ctx.workprec(precision):
        A=[arb_mat([[arb(1)/2,2 if scenario=='nonnormal' else 0],[0,arb(1)/4]]) for _ in range(2)]
        B=arb_mat([[1,0],[0,1]])
        left=original.InverseSecant(A,B);right=fast.InverseSecant(A,B)
        def centers(i):
            if scenario=='zero':return [[arb(1),arb(0)],[arb(2),arb(0)]]
            scale=100000 if scenario=='update_cap' else 10 if scenario=='step_cap' else 1
            return [[arb(i*scale),arb(i if scenario=='collinear' else i*i)],
                    [arb(2*i*scale),arb(3*i if scenario=='collinear' else i*i+i)]]
        def residual(i):
            noise=arb(0,1) if scenario=='noisy' else arb(0)
            return [[arb(i)/4+noise,arb(i)/2+noise],[arb(i)+noise,arb(i)/8+noise]]
        for i in range(7):
            evidence={'synthetic_scenario':scenario,'index':i}
            a=left.update(centers(i),residual(i),centers(i+1),residual(i+1),evidence=evidence)
            b=right.update(centers(i),residual(i),centers(i+1),residual(i+1),evidence=evidence)
            assert a==b and left.history==right.history
            a=left.propose(centers(i+1),residual(i+1),evidence=evidence)
            b=right.propose(centers(i+1),residual(i+1),evidence=evidence)
            assert a['record']==b['record'] and a['centers']==b['centers'] and left.history==right.history
        assert left.decisions==right.decisions


def test_shared_dot_is_reused_without_changing_any_saved_witness(monkeypatch):
    with ctx.workprec(256):
        p,n=8,32;A=[arb_mat([[arb(1)/2 if i==j else 0 for j in range(n)] for i in range(n)]) for _ in range(p)]
        B=arb_mat([[int(i==j) for j in range(n)] for i in range(n)])
        old=[[arb(0)]*n for _ in range(p)];new=[[arb(i+j+1) for j in range(n)] for i in range(p)]
        counts={'original':0,'fast':0};outputs={}
        for name,module in [('original',original),('fast',fast)]:
            dot=module._dot
            def counted(a,b,*,dot=dot,name=name):counts[name]+=1;return dot(a,b)
            monkeypatch.setattr(module,'_dot',counted)
            kernel=module.InverseSecant(A,B)
            outputs[name]=kernel.update(old,old,new,new,evidence={'synthetic':True})
        assert outputs['original']==outputs['fast'] and outputs['fast']['accepted']
        assert counts=={'original':p*n+1,'fast':2}

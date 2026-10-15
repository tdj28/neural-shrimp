"""Manufactured component controls only. Prepared, not yet executed."""
import importlib.util
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location('own21i', Path(__file__).with_name('statistics.py'))
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)

class Choices:
    def __init__(self, rows): self.rows = iter(rows)
    def choice(self, x, n): return np.array(next(self.rows), dtype=float)
class Permutations:
    def __init__(self, rows): self.rows = iter(rows)
    def permutation(self, x): return np.array(next(self.rows), dtype=float)

def controls():
    passed=[]
    assert s.arithmetic_score('Value: -1,234; remainder 2', -1234)
    assert not s.arithmetic_score('First 2, then 5', 5)
    assert s.nq_score('THE, Eiffel Tower!', ['an Eiffel tower'])
    passed.append('literal_text_normalization_and_first_integer')
    x=s.text_score_record('12', True, 'arithmetic', {'answer':12}, 2)
    assert x['equal'] and not x['full_text_available_from_length']
    assert s.text_score_record('print(1)', False, 'mbpp', {}, 400)['own_score'] is None
    passed.append('truncation_and_no_generated_code_execution')
    x=s.sign_test([1,2,3,4,0]); assert x['sign_test_p']==.125 and x['positive']==4 and x['nonzero']==4
    assert s.sign_test([0,0])['sign_test_p']==1
    assert not s.sign_test([float('nan'),1])['finite_qualified']
    passed.append('exact_two_sided_nonzero_sign_test_and_invalid_flag')
    x=s.paired_bootstrap([0,2], Choices([[0,2],[2,2],[0,0],[2,0]]),4)
    assert x['mean']==1 and np.allclose(x['bootstrap_95'],[.075,1.925],rtol=0,atol=1e-15)
    assert x['finite_draws']==4 and x['draws_consumed']==4
    passed.append('paired_resampling_linear_percentiles')
    r=np.random.default_rng(7); expected=np.random.default_rng(7)
    s.paired_bootstrap([0,1,2],r,3)
    for _ in range(3): expected.choice(np.array([0.,1.,2.]),3)
    assert r.bit_generator.state==expected.bit_generator.state
    passed.append('paired_generator_consumption_matches_declared_sequence')
    r=np.random.default_rng(1); old=r.bit_generator.state
    assert s.paired_bootstrap([],r,4)['draws_consumed']==0 and r.bit_generator.state==old
    assert s.prompt_permutation([.5],[],r,4)['draws_consumed']==0 and r.bit_generator.state==old
    passed.append('empty_groups_explicit_without_rng_consumption')
    x=s.prompt_permutation([1],[0],Permutations([[1,0],[0,1]]),2)
    assert x['difference']==1 and x['exceedances']==2 and x['permutation_p']==1
    x=s.prompt_permutation([.5],[.5],Permutations([[.5,.5]]),1)
    assert x['exceedances']==1 and x['permutation_p']==1
    passed.append('prompt_permutation_inclusive_two_sided_ties')
    rows=[{'task':t,'label':g,'acc':a} for t,g,arr in [
        ('arithmetic','chaotic',[.5,.5,.5]),('arithmetic','periodic',[.75]),
        ('nq','chaotic',[.0]),('nq','periodic',[.25,.25,.25])] for a in arr]
    x=s.task_composition(rows)
    assert x['raw_difference']==0 and x['decomposition']['within_task']==-.25
    assert x['decomposition']['composition']==.25 and x['decomposition']['closure_difference']==0
    passed.append('task_composition_identity_with_unequal_mix')
    x=s.task_composition([r for r in rows if not(r['task']=='nq' and r['label']=='chaotic')])
    assert not x['decomposition']['defined'] and x['common_support_tasks']==['arithmetic']
    passed.append('missing_task_support_not_imputed')
    for args in [([float('nan')],[1]),([1.1],[0])]:
        try: s.prompt_permutation(*args,np.random.default_rng(0),2)
        except ValueError: pass
        else: raise AssertionError('invalid accuracy admitted')
    passed.append('nonfinite_out_of_range_prompt_accuracy_refusal')
    return passed

if __name__ == '__main__':
    import json
    print(json.dumps({'passed':controls(),'failed':0}))

import numpy as np
from neural_shrimp import tiny_lm_common_subspace as H


def fixture(operators):
    n = len(operators[0]); eta = .2
    u = np.zeros(n); u[:2] = 1/np.sqrt(2)
    mean = H.product_response(np.repeat(operators.mean(0)[None], 16, axis=0), u)
    full = H.product_response(operators, u)
    actions = np.asarray([[(np.eye(n)-a) @ v/eta for a in operators] for v in mean[:16]])
    return mean, actions, full, eta


def sequence():
    return np.asarray([np.diag([.65+.08*np.sin(t), 1.08+.05*np.cos(t)]) for t in range(16)])


def test_commuting_temporal_variation_is_not_ordering():
    saved = {}; result = H.analyze(*fixture(sequence()), save=lambda k,v:saved.__setitem__(k,v.copy()))
    assert result['passed'] and result['rank'] == 2
    assert np.linalg.norm(saved['ordered']-saved['commuting']) < 1e-11
    assert np.linalg.norm(saved['ordered']-saved['reversed']) < 1e-11
    assert np.linalg.norm(saved['temporal_gain_term']) > .01
    assert np.linalg.norm(saved['changed_mean_term']) < 1e-11


def test_rotating_basis_and_complete_three_term_identity():
    operators = []
    for t in range(16):
        a = .025*t; c,s = np.cos(a),np.sin(a); q=np.array([[c,-s],[s,c]])
        operators.append(q @ np.diag([.65,1.08]) @ q.T)
    saved={}; out=H.analyze(*fixture(np.asarray(operators)),save=lambda k,v:saved.__setitem__(k,v.copy()))
    assert out['passed']
    assert np.linalg.norm(saved['basis_alignment_term']) > .01
    assert np.linalg.norm(saved['changed_mean_term']) > .001
    assert np.linalg.norm(saved['temporal_gain_term']) < 1e-11
    assert np.allclose(saved['ordered']-saved['mean'],saved['basis_alignment_term']+saved['temporal_gain_term']+saved['changed_mean_term'])


def test_subspace_leakage_abstains():
    operators=[]
    for t,a in enumerate(sequence()):
        z=np.eye(3)*.9;z[:2,:2]=a;z[0,2]=z[2,0]=.1*(-1)**t;operators.append(z)
    out=H.analyze(*fixture(np.asarray(operators)))
    assert not out['passed'] and any('leakage' in x for x in out['reasons'])


def test_inconsistent_recorded_action_abstains():
    mean,actions,full,eta=fixture(sequence());actions[7,2,0]+=.1
    out=H.analyze(mean,actions,full,eta)
    assert not out['passed'] and any('action closure' in x for x in out['reasons'])


def test_near_degenerate_and_ambiguous_matching_abstain():
    operators=sequence();operators[3]=np.eye(2)
    out=H.analyze(*fixture(operators))
    assert not out['passed'] and 'no near-degenerate reduced eigenvalues' in out['reasons']
    q=np.array([[1.,-1.],[1.,1.]])/np.sqrt(2)
    operators[0]=np.diag([.5,1.2]);operators[1]=q @ operators[0] @ q.T
    _,_,rows=H.match_eigenvalues(operators)
    assert not rows[1]['matching_pass']


def test_crossing_rank_labels_follow_vectors_not_sorted_indices():
    operators=np.asarray([np.diag([.5+.04*t,1.13-.01*t]) for t in range(16)])
    values,_,rows=H.match_eigenvalues(operators)
    assert np.allclose(values[:,0],[.5+.04*t for t in range(16)])
    assert all(x['matching_pass'] and x['gap_pass'] for x in rows)


def test_full_ordered_fidelity_and_rank_are_mandatory():
    m,a,o,e=fixture(sequence());o[16,0]+=.5
    out=H.analyze(m,a,o,e)
    assert not out['passed'] and 'full ordered fidelity h=16' in out['reasons']
    m,a,o,e=fixture(np.repeat(np.eye(2)[None]*.9,16,axis=0))
    out=H.analyze(m,a,o,e)
    assert not out['passed'] and out['rank']==1


def test_all_saved_inputs_remain_unchanged_and_no_n_by_n_matrix():
    data=fixture(sequence());before=[x.copy() if isinstance(x,np.ndarray) else x for x in data]
    saved={};H.analyze(*data,save=lambda k,v:saved.__setitem__(k,v.copy()))
    assert all(np.array_equal(a,b) for a,b in zip(data,before))
    assert saved['coefficient_map'].shape==(16,2)

import numpy as np
from flint import arb
from neural_shrimp.rigorous_orbits import vector_json
from neural_shrimp.signed_correlated_conditioning import polynomial,budget,reconstruct

def fixed(a):a=np.asarray(a,float);return a,np.zeros_like(a)
def test_noncommuting_convolution_and_fifth_forcing_coefficient():
 C=[fixed([[1,2],[0,1]]),fixed([[0,1],[2,0]])];J=[fixed([[1,0],[2,1]]),fixed([[.1,0],[0,.2]]),fixed([[0,.1],[.3,0]])];R=[fixed([[i],[2*i]])for i in range(1,5)];D,F,_,_,_=polynomial(C,J,R,.1)
 for t in(-.1,0,.1):
  c=C[0][0]+C[1][0]*t;j=sum(x[0]*t**i for i,x in enumerate(J));r=sum(x[0]*t**i for i,x in enumerate(R));assert np.allclose(sum(x[0]*t**i for i,x in enumerate(D)),np.eye(2)-c@j);assert np.allclose(sum(x[0]*t**i for i,x in enumerate(F)),-c@r)
 assert np.array_equal(F[4][0],-C[1][0]@R[3][0])

def test_hidden_coordinate_scale_failure_is_descriptive_only():
 C=[fixed(np.eye(2)),fixed(np.zeros((2,2)))];J=[fixed([[1,-.01],[0,1]]),fixed(np.zeros((2,2))),fixed(np.zeros((2,2)))];R=[fixed(np.zeros((2,1)))for _ in range(4)];r=budget(C,J,R,[2**-20,1],2**-12,.1);assert r['original_max_weighted_row']>10000 and r['dominant_rows'][0]['coordinate']==0;assert 'not proofs'in r['scope']

def test_literal_negative_closure_and_full_mode_partials():
 def path(a,b):return [vector_json([arb(v)for v in a]),vector_json([arb(v)for v in b])]
 ret=dict(phase=0,returns=[dict(value=path([0]*4,[0]*4),gradient=[path([2,3,4,0],[5,6,7,0]),path([8,9,10,0],[11,12,13,0]),path([14,15,16,0],[17,18,19,0])])]);inp=dict(phase=0,inputs=[dict(value=path([0]*4,[1,2,3,0])),{},{}]);J=reconstruct([ret],[inp],[0,0,0,1],1,1,-1)
 assert np.array_equal(J[0][0],[[1,0,8,14],[5,3,11,17],[0,1,0,0],[0,0,0,1]])
 assert J[1][0][1,1]==3 and J[2][0][2,1]==3

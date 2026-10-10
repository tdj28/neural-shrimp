import numpy as np
from neural_shrimp.clark_rnn import inputs,iterate
from neural_shrimp.clark_rnn_boundaries import jets,augmented,qualify_boundary

def test_jets_independent_centered_map_jacobian_control():
    j,x,_=inputs(3,28173);theta=x[0];g=1.8;d=.7;y=np.r_[theta,g,d];end,u,v,_,_=jets(theta,j,g,d,3);eps=2e-5
    def first(y):return jets(y[:3],j,y[3],y[4],3)[1]
    fd=np.stack([(first(y+eps*e)-first(y-eps*e))/(2*eps) for e in np.eye(5)],axis=2)
    np.testing.assert_allclose(v,fd,atol=4e-8,rtol=4e-8)
    np.testing.assert_allclose(v,v.transpose(0,2,1),atol=1e-14)
    np.testing.assert_allclose(end,iterate(theta,j,g,d,3)[0],atol=0)
    np.testing.assert_allclose(u[:,:3],iterate(theta,j,g,d,3)[1],atol=1e-14)

def test_augmented_derivative_both_signs():
    j,x,_=inputs(2,28173);z=np.r_[x[0],.75,[.6,.8]];eps=1e-5
    for sign in [1,-1]:
        h,dh,_=augmented(z,j,1.5,3,sign)
        fd=np.column_stack([(augmented(z+eps*e,j,1.5,3,sign)[0]-augmented(z-eps*e,j,1.5,3,sign)[0])/(2*eps) for e in np.eye(len(z))])
        np.testing.assert_allclose(dh,fd,atol=1e-8,rtol=2e-8)

def test_known_scalar_flip_total_derivative_and_degenerate_fold():
    # At the origin F=(1-d)x+d*g*(-1)*tanh(x), g=2, d=2/3: mu=-1.
    a,r=qualify_boundary(np.array([0.,2/3,1.]),np.array([[-1.]]),2.,1,-1)
    assert r['status']=='numerical_boundary_candidate'
    assert abs(r['total_flip_alpha_delta']+3)<1e-12
    # delta=0 is an identity-map degeneracy, not a nondegenerate fold.
    a,r=qualify_boundary(np.array([0.,0.,1.]),np.array([[-1.]]),2.,1,1)
    assert r['status']=='unqualified'


def test_full_family_augmented_derivative_and_null():
    from neural_shrimp.clark_rnn_boundaries import family_augmented,oriented_null
    j,x,_=inputs(2,28173);z=np.r_[x[0],1.5,.75,[.6,.8]];eps=1e-5
    for sign in [1,-1]:
        h,b,_=family_augmented(z,j,3,sign)
        fd=np.column_stack([(family_augmented(z+eps*e,j,3,sign)[0]-family_augmented(z-eps*e,j,3,sign)[0])/(2*eps) for e in np.eye(len(z))])
        np.testing.assert_allclose(b,fd,atol=1e-8,rtol=2e-8)
        t,w=oriented_null(b);assert np.max(np.abs(b@t))<1e-14
        t2,_=oriented_null(b,-t);assert t2@t<-.999999


def test_fold_family_tangent_can_turn_in_parameter_without_loss_of_rank():
    from neural_shrimp.clark_rnn_boundaries import oriented_null
    # Synthetic parabola h(y0,y1)=y0²+y1: tangent at tip points alongy0.
    b=np.array([[0.,1.]]);t,w=oriented_null(b)
    assert abs(t[0])==1 and t[1]==0 and w['s'][0]==1

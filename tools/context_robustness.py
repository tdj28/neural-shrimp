"""Fixed contextual nearest-template observer under an incorrect initial state."""
import numpy as np

LABELS=np.array([-1,-1,1,1])


def success_counts(templates,actual_means,sigma,noise):
    templates=np.asarray(templates,dtype=float)
    actual=np.asarray(actual_means,dtype=float)
    noise=np.asarray(noise,dtype=float)
    if templates.ndim!=2 or templates.shape[0]!=4 or actual.ndim!=2 or len(actual)%4:
        raise ValueError('Four templates and complete four-history groups required')
    if actual.shape[1]!=templates.shape[1] or noise.ndim!=2 or noise.shape[1]!=templates.shape[1]:
        raise ValueError('Observation dimensions must match')
    if sigma<=0 or not np.isfinite(sigma) or not all(np.isfinite(x).all() for x in [templates,actual,noise]):
        raise ValueError('Finite inputs and positive sigma required')
    # Common ||noise||² and noise·actual terms cancel across candidate templates.
    offset=templates-templates[0]
    base=np.sum((actual[:,None,:]-templates[None,:,:])**2,axis=-1)
    projected=-2*sigma*(noise@offset.T)
    choice=np.argmin(base[None,:,:]+projected[:,None,:],axis=-1)
    truth=np.tile(LABELS,len(actual)//4)
    return np.sum(LABELS[choice]==truth[None,:],axis=1).astype(np.uint8)


def failed_contexts(a,b,gate):
    if a.shape!=b.shape or a.ndim!=6:
        raise ValueError('Expected epsilon, jitter, context, history, time, neuron arrays')
    errors=np.max(np.abs(np.tanh(a)-np.tanh(b)),axis=(0,1,3,4,5))
    return np.flatnonzero(errors>gate).tolist(),float(np.max(errors))


def score_counts(counts):
    # [method, condition(clean/epsilon1/epsilon2), context, noise replicate]
    denominator=np.array([4.,8.,8.])[None,:,None,None]
    fractions=counts/denominator
    per_context=fractions.mean(axis=-1)
    accuracy=per_context.mean(axis=-1)
    mcse=np.sqrt(fractions.var(axis=-1,ddof=1).sum(axis=-1)/(counts.shape[-2]**2*counts.shape[-1]))
    return per_context,accuracy,mcse

"""Exact-real Qwen primitives and forward directional derivatives in Arb balls.

Weights/scalar inputs are exact dyadics. Derived temperatures, rotary factors
and prefix activations are real expressions enclosed by Arb, never float caches.
A full interval tangent box is an operator-action enclosure, not a dense
Jacobian enclosure and not by itself a certificate. No target run on import.
"""
from dataclasses import dataclass
import math
import numpy as np
from flint import arb, arb_mat, ctx, fmpq
from ball_linear import affine


def exact(value):
    if isinstance(value, arb):
        if not value.is_finite():
            raise ValueError("finite ball required")
        return value
    if isinstance(value, (int, np.integer)):
        return arb(int(value))
    if isinstance(value, (float, np.floating)):
        if isinstance(value, np.floating) and value.dtype.itemsize>8:
            raise TypeError("wider than binary64 input would be rounded")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('finite dyadic constant required')
        n, d = value.as_integer_ratio()
        return arb(fmpq(n, d))
    raise TypeError('use int, binary float or explicitly enclosed arb')


@dataclass(frozen=True)
class Dual:
    value: arb
    tangent: arb

    @staticmethod
    def make(value, tangent=0):
        return Dual(exact(value), exact(tangent))

    def __add__(self, other):
        other = dual(other)
        return Dual(self.value + other.value, self.tangent + other.tangent)
    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.value, -self.tangent)

    def __sub__(self, other):
        return self + -dual(other)

    def __rsub__(self, other):
        return dual(other) + -self

    def __mul__(self, other):
        other = dual(other)
        return Dual(self.value*other.value, self.tangent*other.value + self.value*other.tangent)
    __rmul__ = __mul__

    def reciprocal(self):
        if self.value.contains(0):
            raise ValueError('denominator enclosure contains zero')
        r = 1/self.value
        return Dual(r, -self.tangent*r*r)

    def __truediv__(self, other):
        return self*dual(other).reciprocal()

    def exp(self):
        v = self.value.exp()
        return Dual(v, v*self.tangent)

    def rsqrt(self):
        if not self.value > 0:
            raise ValueError('positive RMS denominator not established')
        r = self.value.rsqrt()
        return Dual(r, -self.tangent*r*r*r/2)


def dual(value):
    return value if isinstance(value, Dual) else Dual.make(value)


def softmax(values):
    """Rigorous positive softmax with an arbitrary exact common shift.

    The midpoint maximum is only a conditioning choice; its derivative is not
    part of the softmax. Jacobian action uses p_i(dz_i-sum p_j dz_j).
    """
    if not values:
        raise ValueError('empty softmax')
    if any(not q.value.is_finite() or not q.tangent.is_finite() for q in values):
        raise ValueError("nonfinite softmax input")
    shift = max((q.value.mid() for q in values), key=float)
    exps = [(q.value-shift).exp() for q in values]
    denominator = sum(exps, arb(0))
    if not denominator > 0:
        raise ValueError('positive softmax denominator not established')
    p = [v/denominator for v in exps]
    average = sum((w*q.tangent for w,q in zip(p,values)), arb(0))
    return [Dual(w, w*(q.tangent-average)) for w,q in zip(p,values)]


def rmsnorm(values, weights, epsilon):
    variance = sum((q*q for q in values), dual(0))/len(values)
    inverse = (variance+exact(epsilon)).rsqrt()
    return [q*inverse*exact(w) for q,w in zip(values,weights)]


def linear_sequence(weight, sequence, bias=None, **options):
    """Pack position primal/tangent pairs as matrix columns, sharing weights."""
    if not sequence or not sequence[0]:
        raise ValueError('nonempty sequence required')
    n=len(sequence[0]); count=len(sequence)
    if any(len(q)!=n for q in sequence):
        raise ValueError('ragged sequence')
    x=arb_mat(n, 2*count)
    for j,state in enumerate(sequence):
        for i,q in enumerate(state):
            x[i,2*j]=q.value; x[i,2*j+1]=q.tangent
    y=affine(weight,x,bias,bias_columns=tuple(range(0,2*count,2)),**options)
    return [[Dual(y[i,2*j],y[i,2*j+1]) for i in range(y.nrows())] for j in range(count)]


class BallQwen:
    """One-token, fixed-position expected-embedding feedback in exact reals.

    The fixed prefix is causal and steered after the designated decoder block
    at every position. Its entire cache is recomputed in ball arithmetic.
    ctx.prec and every input array/parameter must remain unchanged for the
    lifetime of a cached prefix. The driver owns read-only arrays and does not
    mutate them; attribute identity guards catch simple rebinding, not arbitrary
    hostile in-place mutation through an external alias.
    """
    def __init__(self, config, weights, prefix_ids, steering_vector, *, steering_layer=12,
                 alpha=.071674, log10_temperature=-.115, linear_options=None, progress=None):
        self.config=dict(config); self.weights=weights
        self.d=int(config['hidden_size']); self.layers=int(config['num_hidden_layers'])
        self.heads=int(config['num_attention_heads']); self.kv_heads=int(config['num_key_value_heads'])
        self.vocab=int(config['vocab_size']); self.hdim=self.d//self.heads
        self.groups=self.heads//self.kv_heads; self.steering_layer=steering_layer
        if self.d%self.heads or self.heads%self.kv_heads or self.hdim%2:
            raise ValueError('unsupported attention dimensions')
        if (config.get('use_sliding_window',False) or not config.get('tie_word_embeddings',False)
            or config.get('hidden_act')!='silu' or config.get('attention_dropout',0.)!=0.
            or config.get('rope_scaling') is not None or config.get('partial_rotary_factor',1.)!=1.):
            raise ValueError('unsupported architecture')
        if not 0<=steering_layer<self.layers or ctx.prec<64:
            raise ValueError('valid steering layer and at least64 bit arithmetic required')
        if any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) for i in prefix_ids):
            raise TypeError("prefix token IDs must be exact integers")
        self.prefix_ids=tuple(int(i) for i in prefix_ids)
        if not self.prefix_ids or any(i<0 or i>=self.vocab for i in self.prefix_ids):
            raise ValueError('invalid prefix')
        self.vector=np.asarray(steering_vector,dtype=np.float64)
        if self.vector.shape!=(self.d,) or not np.isfinite(self.vector).all():
            raise ValueError('invalid steering vector')
        self.alpha=exact(alpha); self.log10_temperature=exact(log10_temperature)
        self.temperature=(arb(10).log()*self.log10_temperature).exp()
        self.epsilon=exact(float(config['rms_norm_eps']))
        base=exact(float(config['rope_theta']))
        if not base>0 or not self.epsilon>0:
            raise ValueError('positive normalization and rotary constants required')
        self.frequencies=[(base.log()*arb(fmpq(-2*j,self.hdim))).exp() for j in range(self.hdim//2)]
        self.scale=arb(self.hdim).rsqrt()
        self.linear_options=linear_options or {}
        self.precision=ctx.prec
        shapes={'model.embed_tokens.weight':(self.vocab,self.d), 'model.norm.weight':(self.d,)}
        intermediate=int(config['intermediate_size'])
        for index in range(self.layers):
            stem=f'model.layers.{index}.'
            for norm in ('input_layernorm','post_attention_layernorm'):
                shapes[stem+norm+'.weight']=(self.d,)
            for projection,rows in [('q',self.d),('k',self.kv_heads*self.hdim),('v',self.kv_heads*self.hdim)]:
                shapes[stem+f'self_attn.{projection}_proj.weight']=(rows,self.d)
                shapes[stem+f'self_attn.{projection}_proj.bias']=(rows,)
            shapes[stem+'self_attn.o_proj.weight']=(self.d,self.d)
            for projection in ('gate','up'):
                shapes[stem+f'mlp.{projection}_proj.weight']=(intermediate,self.d)
            shapes[stem+'mlp.down_proj.weight']=(self.d,intermediate)
        if set(weights)!=set(shapes):
            raise ValueError('tensor inventory mismatch')
        for name,shape in shapes.items():
            w=weights[name]
            if not isinstance(w,np.ndarray) or w.shape!=shape or w.dtype.kind!='f' or w.dtype.itemsize not in (2,4,8):
                raise ValueError('invalid exact-binary tensor: '+name)
            # This reads the existing array; no widened full-model copy.
            if not np.isfinite(w).all():
                raise ValueError('nonfinite weights: '+name)
        self.progress=progress or (lambda *args: None)
        self.cache_signature=(id(self.weights),id(self.vector),id(self.alpha),id(self.temperature))
        self.cache=None
        self.embedding=weights['model.embed_tokens.weight']
        if self.embedding.shape!=(self.vocab,self.d):
            raise ValueError('embedding shape mismatch')

    def linear(self, name, sequence, bias=None):
        return linear_sequence(self.weights[name],sequence,
                               self.weights[bias] if bias else None,**self.linear_options)

    def rotary(self, heads, positions):
        half=self.hdim//2
        result=[]
        for position,token in zip(positions,heads):
            phases=[position*f for f in self.frequencies]
            cs=[p.cos() for p in phases]; sn=[p.sin() for p in phases]
            result.append([[h[j]*cs[j]-h[j+half]*sn[j] for j in range(half)]+
                           [h[j+half]*cs[j]+h[j]*sn[j] for j in range(half)] for h in token])
        return result

    def block(self, hidden, index, positions, prior=None):
        name=f'model.layers.{index}.'
        normalized=[rmsnorm(h,self.weights[name+'input_layernorm.weight'],self.epsilon) for h in hidden]
        projections=[]
        for p in ('q','k','v'):
            stem=name+f'self_attn.{p}_proj.'
            q=self.linear(stem+'weight',normalized,stem+'bias')
            projections.append([[v[i:i+self.hdim] for i in range(0,len(v),self.hdim)] for v in q])
        query,key,value=projections
        query=self.rotary(query,positions); key=self.rotary(key,positions)
        current=(key,value)
        if prior is not None:
            if len(hidden)!=1:
                raise ValueError('one token required with cache')
            key=prior[0]+key; value=prior[1]+value
        mixed=[]
        for token,heads in enumerate(query):
            allowed=len(key) if prior is not None else token+1
            row=[]
            for head,q in enumerate(heads):
                kv=head//self.groups
                scores=[sum((a*b for a,b in zip(q,key[j][kv])),dual(0))*self.scale for j in range(allowed)]
                probabilities=softmax(scores)
                row.extend([sum((probabilities[j]*value[j][kv][k] for j in range(allowed)),dual(0)) for k in range(self.hdim)])
            mixed.append(row)
        attention=self.linear(name+'self_attn.o_proj.weight',mixed)
        hidden=[[a+b for a,b in zip(h,v)] for h,v in zip(hidden,attention)]
        normalized=[rmsnorm(h,self.weights[name+'post_attention_layernorm.weight'],self.epsilon) for h in hidden]
        gate=self.linear(name+'mlp.gate_proj.weight',normalized)
        up=self.linear(name+'mlp.up_proj.weight',normalized)
        activation=[[g/(1+(-g).exp())*v for g,v in zip(gs,vs)] for gs,vs in zip(gate,up)]
        down=self.linear(name+'mlp.down_proj.weight',activation)
        hidden=[[a+b for a,b in zip(h,v)] for h,v in zip(hidden,down)]
        if index==self.steering_layer:
            hidden=[[q+self.alpha*exact(v) for q,v in zip(h,self.vector)] for h in hidden]
        return hidden,current

    def prefix_cache(self):
        if ctx.prec!=self.precision:
            raise ValueError('recreate map after changing precision')
        if self.cache_signature!=(id(self.weights),id(self.vector),id(self.alpha),id(self.temperature)):
            raise ValueError('recreate map after changing cache dependencies')
        if self.cache is None:
            hidden=[[Dual.make(v) for v in self.embedding[i]] for i in self.prefix_ids]
            result=[]
            for i in range(self.layers):
                self.progress("prefix_start",i,hidden)
                hidden,kv=self.block(hidden,i,list(range(len(hidden))))
                self.progress("prefix_end",i,hidden)
                result.append(kv)
            self.cache=result
        return self.cache

    def forward(self, state, tangent=None):
        if len(state)!=self.d or tangent is not None and len(tangent)!=self.d:
            raise ValueError('full-dimensional state and tangent required')
        tangent=[0]*self.d if tangent is None else tangent
        hidden=[[Dual.make(x,v) for x,v in zip(state,tangent)]]
        for i,cache in enumerate(self.prefix_cache()):
            self.progress("step_start",i,hidden)
            hidden,_=self.block(hidden,i,[len(self.prefix_ids)],cache)
            self.progress("step_end",i,hidden)
        normalized=[rmsnorm(hidden[0],self.weights['model.norm.weight'],self.epsilon)]
        logits=self.linear('model.embed_tokens.weight',normalized)[0]
        probabilities=softmax([q/self.temperature for q in logits])
        output=linear_sequence(self.embedding.T,[probabilities],**self.linear_options)[0]
        if any(not q.value.is_finite() or not q.tangent.is_finite() for q in output):
            raise ValueError('nonfinite ball output')
        return [q.value for q in output],[q.tangent for q in output]

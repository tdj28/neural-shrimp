"""All-token rigorous expected-embedding head with bounded vocabulary batches.

This new evaluator leaves the existing BallQwen, Jet, cache and refinement
sources unchanged. Only the short forward orchestration is repeated; decoder,
attention, rotary, normalization and prefix methods remain inherited.

For stored exact binary embeddings E, hidden Jets (h,U) and positive constant
T, compute p=softmax(E h/T), F=E.T p, and
D=E.T diag(p)(E U/T)-F [p.T(E U/T)]. All signed sums use Arb. Every vocabulary
row participates; no sparse token selection or floating error bound is used.
A V-vector of primal logits/probabilities is permitted, never a V-by-K tangent
array. Extra head storage is O(V+dK+bK+bd) for token chunk b, including
bounded weight-validation/conversion and native matrix temporaries. Existing transformer-body Jet storage is unchanged.

Weights/parameters must remain immutable throughout a call and cached-prefix
lifetime. Receiver guards detect attribute rebinding, not hostile mutation
through an external writable alias. Drivers must own their readonly arrays.
No target loading or calls occur on import. This is an arithmetic kernel,
not a complete Jacobian, domain admission, or orbit certificate.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
from flint import arb, arb_mat

from ball_jet import BallQwenJet, Jet, finite_jet
from ball_linear import affine
from ball_qwen import exact


def _chunk_size(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError('positive integer token_chunk required')
    return int(value)


def streamed_head(embedding, hidden, temperature, *, token_chunk=256,
                  linear_options=None, progress=None, diagnostics=None):
    """Return (full primal vector, full d-by-K signed derivative enclosure).

    ``hidden`` is a nonempty sequence of same-K Jets. Primal and derivative
    radii are retained, including interval directions. ``temperature`` is a
    finite positive scalar enclosure constant with respect to all directions.
    Optional progress receives (pass_name, row_start, row_stop); it must not
    mutate inputs. An optional empty dict receives the all-vocabulary strict
    argmax-gap diagnostic after successful evaluation; it is never an orbit
    gate. Weight conversion is delegated to the reviewed affine
    kernel with bounded chunks, including the final transpose product.
    """
    chunk = _chunk_size(token_chunk)
    if diagnostics is not None and (type(diagnostics) is not dict or diagnostics):
        raise ValueError('diagnostics must be an empty caller-owned dict')
    if (not isinstance(embedding, np.ndarray) or embedding.ndim != 2
            or embedding.dtype.kind != 'f' or embedding.dtype.itemsize not in (2,4,8)
            or min(embedding.shape) < 1):
        raise ValueError('nonempty binary16/32/64 embedding matrix required')
    vocab, dimension = embedding.shape
    for first in range(0, vocab, chunk):
        if not np.isfinite(embedding[first:first+chunk]).all():
            raise ValueError('embedding constants must be finite')
    if not isinstance(hidden, (list, tuple)) or len(hidden) != dimension:
        raise ValueError('one full hidden Jet per embedding coordinate required')
    if any(not isinstance(q, Jet) for q in hidden):
        raise TypeError('hidden coordinates must be Jets')
    count = hidden[0].dimension
    if any(q.dimension != count or not finite_jet(q) for q in hidden):
        raise ValueError('finite hidden Jets with equal direction counts required')
    temperature = exact(temperature)
    if not temperature > 0:
        raise ValueError('positive constant temperature enclosure required')
    options = {} if linear_options is None else dict(linear_options)
    if 'bias_columns' in options:
        raise ValueError('readout has no bias; bias-column override is unsupported')
    emit = (lambda *args: None) if progress is None else progress
    values = [q.value for q in hidden]
    directions = arb_mat(dimension, count, [v for q in hidden for v in q.tangent])

    # Pass 1 retains V scalar logits, independent of direction count.
    logits = []
    for first in range(0, vocab, chunk):
        last = min(first+chunk, vocab)
        logits.extend(v/temperature for v in affine(embedding[first:last], values, **options))
        emit('head_logits', first, last)
    if len(logits) != vocab or any(not v.is_finite() for v in logits):
        raise ValueError('all finite vocabulary logits required')
    # Midpoints are exact dyadics. The selected shift is a constant conditioning
    # choice over the entire input box; it has no directional derivative.
    label_diagnostic = argmax_diagnostic(logits) if diagnostics is not None else None
    shift = max(v.mid() for v in logits)
    exps = [(v-shift).exp() for v in logits]
    denominator = sum(exps, arb(0))
    if not denominator.is_finite() or not denominator > 0:
        raise ValueError('positive softmax denominator not established')
    probabilities = [v/denominator for v in exps]
    if any(not p.is_finite() for p in probabilities):
        raise ValueError('all finite vocabulary probabilities required')
    del logits, exps

    # Pass 2 shares the exact transpose weight product over a primal column
    # and the K signed weighted-logit columns, but only for this token chunk.
    accumulated = arb_mat(dimension, count+1)
    mean = [arb(0) for _ in range(count)]
    for first in range(0, vocab, chunk):
        last = min(first+chunk, vocab)
        dz = affine(embedding[first:last], directions, **options)
        weighted = arb_mat(last-first, count+1)
        for local, p in enumerate(probabilities[first:last]):
            weighted[local,0] = p
            for k in range(count):
                value = p*(dz[local,k]/temperature)
                weighted[local,k+1] = value
                mean[k] += value
        accumulated += affine(embedding[first:last].T, weighted, **options)
        emit('head_derivatives', first, last)
    primal = [accumulated[i,0] for i in range(dimension)]
    derivative = arb_mat(dimension, count)
    for i in range(dimension):
        for k in range(count):
            derivative[i,k] = accumulated[i,k+1]-primal[i]*mean[k]
    if (any(not v.is_finite() for v in primal)
            or any(not derivative[i,k].is_finite() for i in range(dimension) for k in range(count))):
        raise ValueError('nonfinite streamed head result')
    if diagnostics is not None:
        diagnostics.update(label_diagnostic)
    return primal, derivative


def _packed(value):
    return dict(mid=[int(v) for v in value.mid().man_exp()],
                rad=[int(v) for v in value.rad().man_exp()])


def argmax_diagnostic(logits):
    """Strict all-competitor gap on the supplied finite logit enclosures.

    The candidate is a midpoint proposal. Only a strictly positive enclosed
    margin establishes its label for every logit vector in this box. This
    local diagnostic does not establish an orbit, model/domain provenance,
    or the label of a separately chosen phase box.
    """
    if not isinstance(logits, (list,tuple)) or not logits or any(not isinstance(v,arb) or not v.is_finite() for v in logits):
        raise ValueError('all finite vocabulary logit balls required')
    candidate = max(range(len(logits)), key=lambda i: logits[i].mid())
    lower = logits[candidate].lower()
    competitors = [v.upper() for i,v in enumerate(logits) if i != candidate]
    upper = max(competitors) if competitors else None
    margin = lower-upper if upper is not None else None
    return dict(candidate_index=candidate, candidate_selection='maximum exact midpoint',
        vocabulary_count=len(logits), competitor_count=len(competitors),
        candidate_lower=_packed(lower), largest_competitor_upper=_packed(upper) if upper is not None else None,
        strict_margin=_packed(margin) if margin is not None else None,
        strict_margin_established=bool(margin is not None and margin > 0),
        scope='Supplied full-vocabulary logit enclosures only; no orbit or separate phase-domain claim.',
        certificate_passed=False)


def _direction_rows(directions, dimension, count):
    """Readmit the unchanged BallQwenJet public direction contract."""
    if directions is None:
        return [(arb(0),)*count for _ in range(dimension)]
    if isinstance(directions, arb_mat):
        if (directions.nrows(), directions.ncols()) != (dimension,count):
            raise ValueError('Arb directions must have shape(d,K)')
        return [tuple(exact(directions[i,k]) for k in range(count)) for i in range(dimension)]
    if isinstance(directions, np.ndarray):
        if (directions.shape != (count,dimension) or directions.dtype.kind != 'f'
                or directions.dtype.itemsize not in (2,4,8)):
            raise ValueError('NumPy directions must be binary16/32/64 of shape(K,d)')
        if not np.isfinite(directions).all():
            raise ValueError('directions must be finite')
        return [tuple(exact(directions[k,i]) for k in range(count)) for i in range(dimension)]
    raise TypeError('directions must be an Arb matrix or a floating NumPy array')


class StreamedHeadQwenJet(BallQwenJet):
    """Inherited transformer with an independently reviewable streamed head.

    This short orchestration mirrors BallQwen.forward through final RMSNorm,
    then calls streamed_head. Existing cache precision/identity guards apply.
    Token/head progress uses a separate callback from transformer progress.
    """
    def __init__(self, *args, token_chunk=256, head_progress=None, **kwargs):
        self.token_chunk = _chunk_size(token_chunk)
        self.head_progress = head_progress
        super().__init__(*args, **kwargs)
        self._stream_embedding = self.embedding
        self.last_head_diagnostics = None

    def forward(self, state, directions=None):
        self.last_head_diagnostics = None
        if len(state) != self.d:
            raise ValueError('full-dimensional state required')
        if (self.embedding is not self._stream_embedding
                or self.weights['model.embed_tokens.weight'] is not self._stream_embedding):
            raise ValueError('recreate map after changing embedding identity')
        rows = _direction_rows(directions, self.d, self.direction_count)
        hidden = [[self.scalar(x,v) for x,v in zip(state,rows)]]
        for i,cache in enumerate(self.prefix_cache()):
            self.progress('step_start', i, hidden)
            hidden,_ = self.block(hidden, i, [len(self.prefix_ids)], cache)
            self.progress('step_end', i, hidden)
        normalized = self.normalize(hidden[0], self.weights['model.norm.weight'], self.epsilon)
        diagnostic = {}
        result = streamed_head(self.embedding, normalized, self.temperature,
            token_chunk=self.token_chunk, linear_options=self.linear_options, progress=self.head_progress,
            diagnostics=diagnostic)
        self.last_head_diagnostics = diagnostic
        return result

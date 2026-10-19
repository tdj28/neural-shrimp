"""Bind one whole-tube interval-direction call to actual full-map coverage.

Prepared receiver helper only. The caller must separately authenticate source,
assets, backend, original point evidence and prefix; this module loads no model.
"""
from flint import arb_mat,ctx
from tube_operator import admit,build,evaluate,need

DIMENSION,LAYERS,VOCABULARY,CHUNK=896,24,151936,1024


def evaluate_receiver(ball,centres,radius,save):
    """One reviewed StreamedHeadQwenJet call, with every live stage retained.

    The target driver must create the authenticated reviewed evaluator, install
    its original fixed prefix and supply an exclusive durable save callback.
    This structural helper does not certify an arbitrary callback implementation.
    """
    need(callable(save),'durable saving callback required')
    need((ball.d,ball.layers,ball.vocab,ball.direction_count,ball.token_chunk,ball.precision)
         ==(DIMENSION,LAYERS,VOCABULARY,1,CHUNK,2048),'complete2048-bit full-map allocation required')
    need((ctx.prec,ctx.threads)==(2048,1),'2048-bit one-thread context required')
    state,directions=build(centres,radius,DIMENSION)
    expected=admit(centres,radius,state,directions,DIMENSION)
    starts=[];ends=[];heads=[];actual=None
    expected_heads=[(kind,first,min(first+CHUNK,VOCABULARY))
                    for kind in ('head_logits','head_derivatives')
                    for first in range(0,VOCABULARY,CHUNK)]
    flags=dict(trajectory_enclosure=False,certificate_passed=False,
               conditional_operator_bound_only=True)
    def progress(kind,index,hidden):
        nonlocal actual
        need(type(index)is int,'exact decoder layer index required')
        if kind=='step_start':
            need(index==len(starts) and len(ends)==len(starts) and index<LAYERS,
                 'ordered complete decoder starts required')
            if index==0:
                need(actual is None and len(hidden)==1 and len(hidden[0])==DIMENSION,
                     'one complete original live receiver required')
                live=hidden[0]
                need(all(len(q.tangent)==1 for q in live),'one interval direction per live coordinate required')
                V=arb_mat(DIMENSION,1,[q.tangent[0]for q in live])
                actual=admit(centres,radius,[q.value for q in live],V,DIMENSION)
                need(actual==expected,'actual first decoder receiver differs from original tube or direction cube')
                save('actual-receiver.json',dict(actual,precision=2048,**flags))
            starts.append(index)
        elif kind=='step_end':
            need(index==len(ends) and len(starts)==len(ends)+1,
                 'ordered complete decoder ends required')
            ends.append(index)
        else:
            raise ValueError('unexpected decoder stage')
    def head_progress(kind,first,last):
        row=(kind,first,last)
        need(type(first)is int and type(last)is int and ends==list(range(LAYERS))
             and len(heads)<len(expected_heads) and row==expected_heads[len(heads)],
             'ordered full-vocabulary head coverage required')
        heads.append(row)
    old=(ball.progress,ball.head_progress)
    ball.progress,ball.head_progress=progress,head_progress
    try:
        result=evaluate(centres,radius,ball.forward,dimension=DIMENSION,
                        primal=state,directions=directions,save=save)
        need(actual is not None and starts==ends==list(range(LAYERS)),
             'missing original receiver or decoder coverage')
        need(heads==expected_heads,'both full-vocabulary head passes required')
        save('coverage.json',dict(started_layers=starts,ended_layers=ends,
             head_passes=heads,actual_receiver_equals_original=True,
             all_direction_cube_retained=True,**flags))
        return result
    finally:
        ball.progress,ball.head_progress=old

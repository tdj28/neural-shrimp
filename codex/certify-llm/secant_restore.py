"""Restore only the post-proposal numerical factors from authenticated records.

The caller authenticates the complete parent source/receipt/history first.
Exact dyadic factors are arbitrary inverse proposals. This reader neither
replays historical dense arithmetic nor asserts that an inverse is correct.
"""
from __future__ import annotations

from fractions import Fraction
from flint import arb
import center_resume as saved
import inverse_secant_fast as kernel

CAPS=dict(max_history=4,signal_ratio=128,update_norm_limit=1024,step_ratio_limit=4)
REJECTIONS={'history_cap_restart','zero_center_step','zero_secant','noise_dominated_secant',
    'nonpositive_denominator','nonfinite_update','update_norm_cap','update_arithmetic_failure'}
FALLBACKS={'step_cap_baseline_fallback','nonfinite_step_baseline_fallback'}


def exact_record(record):
    value=saved.decoded_vector([record],1,exact=True)[0]
    if saved.base.packed(value)!=record:raise ValueError('canonical exact dyadic required')
    return value


def exact_vector(records,size):
    if not isinstance(records,list) or len(records)!=size:raise ValueError('complete factor shape required')
    return [exact_record(v) for v in records]


def point_fraction(record):
    value=exact_record(record)
    return saved.dyadic(saved.base.packed(value)['mid'])


def restore_factors(decisions,proposal,*,period,dimension,caps=CAPS):
    """Retain every clear and the parent's final proposal clear, without replay.

    Only the original four-factor protocol is admitted. An update at a full
    history discards its pair after clearing; there is no second admission.
    All supplied decisions are consumed once in their externally bound order.
    """
    if (type(period)is not int or period<2 or type(dimension)is not int or dimension<1
            or type(caps)is not dict or caps!=CAPS or any(type(v)is not int for v in caps.values())):
        raise ValueError('exact four-factor dimensions and caps required')
    if not isinstance(decisions,list):raise ValueError('complete ordered decisions required')
    size=period*dimension;history=[];audit=[]
    for index,record in enumerate(decisions):
        if (type(record)is not dict or record.get('kind')!='secant'
                or record.get('certificate_passed')is not False
                or type(record.get('accepted'))is not bool
                or type(record.get('prior_history'))is not int or record['prior_history']!=len(history)
                or type(record.get('active_history'))is not int
                or type(record.get('max_history'))is not int or record['max_history']!=4):
            raise ValueError('complete chronological secant decision required')
        for key in ('signal_ratio','update_norm_limit'):
            if point_fraction(record[key])!=caps[key]:raise ValueError('different secant admission cap')
        prior=len(history);reason=record.get('reason')
        if record['accepted']:
            if reason!='accepted_midpoint_rank_one' or prior>=caps['max_history']:
                raise ValueError('accepted update cannot exceed its history cap')
            u=exact_vector(record['u_midpoints'],size);w=exact_vector(record['w_midpoints'],size)
            for exact,enclosures in ((u,record['u_enclosures']),(w,record['w_enclosures'])):
                balls=saved.decoded_vector(enclosures,size)
                for value,raw,ball in zip(exact,enclosures,balls):
                    # Compare the ORIGINAL saved midpoint, not a widened decoder
                    # midpoint that could hide a corrupted factor.
                    if saved.dyadic(raw['mid'])!=point_fraction(saved.base.packed(value)):
                        raise ValueError('factor is not its original enclosure midpoint')
                    if not ball.contains(value):raise ValueError('factor omitted by original arithmetic witness')
            bound=max(abs(point_fraction(v)) for v in record['u_midpoints'])*sum(
                (abs(point_fraction(v)) for v in record['w_midpoints']),Fraction(0))
            recorded_bound=point_fraction(record['update_norm_upper'])
            if not bound<=recorded_bound<=caps['update_norm_limit']:
                raise ValueError('retained factor exceeds the recorded update cap')
            signal=point_fraction(record['signal']);noise=point_fraction(record['noise'])
            if noise<0 or not signal>caps['signal_ratio']*noise:
                raise ValueError('accepted signal is not separated from noise')
            denominator=record['denominator']
            saved.decoded_vector([denominator],1)
            if saved.dyadic(denominator['mid'])-saved.dyadic(denominator['rad'])<=0:
                raise ValueError('accepted denominator not strictly positive')
            history.append((u,w))
        else:
            if reason not in REJECTIONS or (prior==4)!=(reason=='history_cap_restart'):
                raise ValueError('unknown or inconsistent clear/restart reason')
            history=[]
        if record['active_history']!=len(history):raise ValueError('post-update history count differs')
        audit.append(dict(decision_index=index,accepted=record['accepted'],reason=reason,
                          prior_history=prior,active_history=len(history)))
    if (type(proposal)is not dict or proposal.get('kind')!='proposal'
            or proposal.get('certificate_passed')is not False
            or type(proposal.get('prior_history'))is not int or proposal['prior_history']!=len(history)
            or type(proposal.get('active_history'))is not int
            or point_fraction(proposal['step_ratio_limit'])!=caps['step_ratio_limit']):
        raise ValueError('bound final proposal and exact step cap required')
    reason=proposal.get('reason')
    if reason in ('accelerated','step_cap_baseline_fallback'):
        norms={}
        for name in ('accelerated_correction','baseline_correction'):
            records=proposal[name];saved.decoded_vector(records,size)
            intervals=[(saved.dyadic(v['mid']),saved.dyadic(v['rad'])) for v in records]
            norms[name]=(max(abs(m)+r for m,r in intervals),
                         max(max(Fraction(0),abs(m)-r) for m,r in intervals))
        upper=point_fraction(proposal['accelerated_step_upper'])
        lower=point_fraction(proposal['baseline_step_lower'])
        if upper<norms['accelerated_correction'][0] or not 0<=lower<=norms['baseline_correction'][1]:
            raise ValueError('recorded step norms do not enclose original correction norms')
        within_cap=upper<=caps['step_ratio_limit']*lower
        if within_cap!=(reason=='accelerated'):
            raise ValueError('retained proposal decision disagrees with exact step cap')
    if reason=='accelerated':
        if not history or proposal['chosen_correction']!=proposal['accelerated_correction']:
            raise ValueError('accelerated proposal requires active factors and chosen correction')
    elif reason=='baseline_no_history':
        if history or proposal['chosen_correction']!=proposal['baseline_correction']:
            raise ValueError('baseline-no-history proposal requires empty factors')
    elif reason in FALLBACKS:
        if not history or proposal['chosen_correction']!=proposal['baseline_correction']:
            raise ValueError('fallback must clear actual active history and use baseline')
        history=[]
    else:raise ValueError('unknown final proposal reason')
    if proposal['active_history']!=len(history):raise ValueError('post-proposal history count differs')
    c=exact_vector(proposal['centers'],size)
    correction=saved.decoded_vector(proposal['chosen_correction'],size)
    candidate=exact_vector(proposal['next_centers'],size)
    if any((v+d).mid()!=n for v,d,n in zip(c,correction,candidate)):
        raise ValueError('candidate does not match saved exact-midpoint construction')
    return history,dict(schema='codex-secants-restored-proposals-v1',certificate_passed=False,
        decisions=audit,proposal_reason=reason,active_history_after_proposal=len(history),
        historical_dense_arithmetic_replayed=False,external_authentication_performed=False,
        statement='Numerical dyadic factors restored after all recorded clears; not a certified inverse or orbit.')


def install_factors(derivatives,inverse,history,*,caps=CAPS):
    """Construct an independent mutable proposal kernel from admitted factors."""
    from flint import arb_mat
    if type(caps)is not dict or caps!=CAPS or any(type(v)is not int for v in caps.values()):
        raise ValueError('exact fixed continuation caps required')
    result=kernel.InverseSecant([arb_mat(v) for v in derivatives],arb_mat(inverse),**caps)
    if not isinstance(history,list) or len(history)>caps['max_history']:
        raise ValueError('bounded admitted history required')
    copied=[]
    for pair in history:
        if not isinstance(pair,(tuple,list)) or len(pair)!=2:raise ValueError('two exact factor vectors required')
        u,w=pair
        if len(u)!=result.size or len(w)!=result.size:raise ValueError('complete factor dimensions required')
        copied.append((exact_vector([saved.base.packed(v) for v in u],result.size),
                       exact_vector([saved.base.packed(v) for v in w],result.size)))
    result.history=copied
    return result

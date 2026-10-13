"""Equivalent inverse-secant proposals with one shared scalar dot product.

The original inverse_secant.py remains immutable for completed-run ancestry.
The sole arithmetic change below hoists the identical _dot(w,ys) calculation
out of its coordinate loop; all inputs and the arithmetic context are unchanged.

Bounded inverse-secant midpoint proposals for cyclic shooting equations.

No model calls or evidence authentication occur here. Caller supplies complete
centres and residual enclosures from the same map. Active numerical history is
at most m rank-one pairs on p*n vectors; no (p*n)-square matrix is formed.
Audit decisions retain all supplied data/witnesses and grow with the number of
calls; a future driver must persist them. All outputs are proposals, never
inverse, inclusion, or orbit certificates. A/B must remain immutable.
"""
from __future__ import annotations
import json
from flint import arb, arb_mat
from proof_gates import ball_record
from shooting_correction import _matrix


def _point(value):
    if not isinstance(value,arb) or not value.is_finite() or not value.is_exact():
        raise ValueError('finite exact dyadic point required')
    return value


def _power(value, exponent):
    m,e=value.man_exp()
    return arb((int(m),int(e)+exponent))


def _difference(a,b):
    am,ae=a.man_exp();bm,be=b.man_exp();e=min(int(ae),int(be))
    result=arb(((int(am)<<(int(ae)-e))-(int(bm)<<(int(be)-e)),e))
    return _point(result)


def _upper(values):return max(v.abs_upper() for v in values)
def _lower(values):return max(v.abs_lower() for v in values)
def _dot(a,b):return sum((x*y for x,y in zip(a,b)),arb(0))
def _pack(values):return [ball_record(v) for v in values]


class InverseSecant:
    def __init__(self, derivatives, inverse_cycle, *, max_history=4,
                 signal_ratio=128, update_norm_limit=1024, step_ratio_limit=4):
        if not isinstance(derivatives,(list,tuple)) or len(derivatives)<2:
            raise ValueError('at least two phase derivative proposals required')
        if not isinstance(derivatives[0],arb_mat) or derivatives[0].nrows()<1:
            raise ValueError('nonempty square phase matrices required')
        n=derivatives[0].nrows();self.p=len(derivatives);self.n=n;self.size=n*self.p
        self.A=tuple(_matrix(a,n) for a in derivatives);self.B=_matrix(inverse_cycle,n)
        if type(max_history) is not int or not 1<=max_history<=8:
            raise ValueError('fixed history cap from one through eight required')
        self.max_history=max_history
        self.signal_ratio=self._positive(signal_ratio)
        self.update_norm_limit=self._positive(update_norm_limit)
        self.step_ratio_limit=self._positive(step_ratio_limit)
        if not self.signal_ratio>1 or not self.step_ratio_limit>=1:
            raise ValueError('signal ratio must exceed one and step ratio at least one')
        self.history=[];self.decisions=[]

    @staticmethod
    def _positive(value):
        if type(value) is int:value=arb(value)
        _point(value)
        if not value>0:raise ValueError('positive exact cap required')
        return value

    def _rows(self,rows,*,points=False):
        if not isinstance(rows,(list,tuple)) or len(rows)!=self.p:
            raise ValueError('complete phase tuple required')
        result=[]
        for row in rows:
            if not isinstance(row,(list,tuple)) or len(row)!=self.n:
                raise ValueError('all coordinates required for every phase')
            for value in row:
                if not isinstance(value,arb) or not value.is_finite():raise ValueError('finite residual balls required')
                if points:_point(value)
                result.append(value)
        return result

    def baseline(self,vector):
        """R_0 v from the same cyclic recurrence whose correction is -R_0 v."""
        if len(vector)!=self.size or any(not isinstance(v,arb) or not v.is_finite() for v in vector):
            raise ValueError('complete finite flattened vector required')
        h=[arb_mat(self.n,1,vector[i*self.n:(i+1)*self.n]) for i in range(self.p)]
        s=arb_mat(self.n,1)
        for a,v in zip(self.A,h):s=a*s+v
        delta=self.B*s;result=[]
        for a,v in zip(self.A,h):
            result.extend(-delta[j,0] for j in range(self.n));delta=a*delta+v
        if any(not v.is_finite() for v in result):raise ValueError('nonfinite implicit inverse action')
        return result

    def apply(self,vector):
        result=self.baseline(vector)
        for u,w in self.history:
            coefficient=_dot(w,vector)
            result=[a+b*coefficient for a,b in zip(result,u)]
        if any(not v.is_finite() for v in result):raise ValueError('nonfinite low-rank inverse action')
        return result

    def update(self,old_centers,old_residuals,new_centers,new_residuals,*,evidence):
        """Admit one exact-centre secant or reset to deterministic baseline.

        Residual midpoints guide proposals only. The full difference balls and
        arithmetic enclosures are retained; signal-to-radius admission uses
        their actual outward radii. Collinear secants are algebraically valid
        for rank-one inverse Broyden (no least-squares division is used).
        """
        c0=self._rows(old_centers,points=True);c1=self._rows(new_centers,points=True)
        h0=self._rows(old_residuals);h1=self._rows(new_residuals)
        if type(evidence) is not dict or not evidence:raise ValueError('explicit caller evidence labels required')
        evidence=json.loads(json.dumps(evidence,allow_nan=False))
        record=dict(kind='secant',evidence=evidence,certificate_passed=False,
            old_centers=_pack(c0),new_centers=_pack(c1),old_residuals=_pack(h0),new_residuals=_pack(h1),
            prior_history=len(self.history),signal_ratio=ball_record(self.signal_ratio),
            update_norm_limit=ball_record(self.update_norm_limit),max_history=self.max_history)
        def reject(reason):
            self.history.clear();record.update(accepted=False,reason=reason,active_history=0)
            self.decisions.append(record);return record
        if len(self.history)>=self.max_history:return reject('history_cap_restart')
        s=[_difference(a,b) for a,b in zip(c1,c0)]
        if all(v==0 for v in s):
            record['center_difference']=_pack(s);return reject('zero_center_step')
        yball=[a-b for a,b in zip(h1,h0)];y=[v.mid() for v in yball]
        signal=_upper(y);noise=max(v.rad() for v in yball)
        record.update(center_difference=_pack(s),residual_difference=_pack(yball),signal=ball_record(signal),noise=ball_record(noise))
        if not signal>0:return reject('zero_secant')
        if not signal>self.signal_ratio*noise:return reject('noise_dominated_secant')
        mantissa,exponent=signal.man_exp();shift=-(abs(int(mantissa)).bit_length()-1+int(exponent))
        ys=[_power(v,shift) for v in y];ss=[_power(v,shift) for v in s]
        denominator=_dot(ys,ys);record.update(scale_power=shift,denominator=ball_record(denominator))
        if not denominator>0:return reject('nonpositive_denominator')
        try:
            ry=self.apply(ys);u_ball=[a-b for a,b in zip(ss,ry)]
            w_ball=[v/denominator for v in ys]
            if any(not v.is_finite() for v in u_ball+w_ball):return reject('nonfinite_update')
            u=[v.mid() for v in u_ball];w=[v.mid() for v in w_ball]
            bound=_upper(u)*sum((v.abs_upper() for v in w),arb(0))
            record.update(u_enclosures=_pack(u_ball),w_enclosures=_pack(w_ball),
                u_midpoints=_pack(u),w_midpoints=_pack(w),update_norm_upper=ball_record(bound.abs_upper()))
            if not bound<=self.update_norm_limit:return reject('update_norm_cap')
            coefficient=_dot(w,ys)
            residual=[a+b*coefficient-c for a,b,c in zip(ry,u,ss)]
            record['scaled_secant_defect']=_pack(residual)
        except (ArithmeticError,ValueError,OverflowError) as error:
            record['arithmetic_error']=repr(error);return reject('update_arithmetic_failure')
        self.history.append((u,w));record.update(accepted=True,reason='accepted_midpoint_rank_one',active_history=len(self.history))
        self.decisions.append(record);return record

    def propose(self,centers,residuals,*,evidence):
        c=self._rows(centers,points=True);h=self._rows(residuals)
        if type(evidence) is not dict or not evidence:raise ValueError('explicit caller evidence labels required')
        evidence=json.loads(json.dumps(evidence,allow_nan=False))
        baseline=[-v for v in self.baseline(h)]
        record=dict(kind='proposal',evidence=evidence,certificate_passed=False,centers=_pack(c),residuals=_pack(h),
            baseline_correction=_pack(baseline),step_ratio_limit=ball_record(self.step_ratio_limit),prior_history=len(self.history))
        reason='baseline_no_history';chosen=baseline
        if self.history:
            try:
                candidate=[-v for v in self.apply(h)]
                record['accelerated_correction']=_pack(candidate)
                upper,reference=_upper(candidate),_lower(baseline)
                record.update(accelerated_step_upper=ball_record(upper),baseline_step_lower=ball_record(reference))
                if upper<=self.step_ratio_limit*reference:
                    chosen=candidate;reason='accelerated'
                else:self.history.clear();reason='step_cap_baseline_fallback'
            except (ArithmeticError,ValueError,OverflowError) as error:
                self.history.clear();reason='nonfinite_step_baseline_fallback';record['arithmetic_error']=repr(error)
        points=[(x+d).mid() for x,d in zip(c,chosen)]
        if any(not v.is_finite() or not v.is_exact() for v in points):raise ValueError('nonfinite midpoint proposal')
        record.update(reason=reason,chosen_correction=_pack(chosen),next_centers=_pack(points),active_history=len(self.history))
        self.decisions.append(record)
        return dict(centers=[points[i*self.n:(i+1)*self.n] for i in range(self.p)],record=record)

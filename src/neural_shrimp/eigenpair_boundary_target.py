"""Exploratory event flags applied only after durable accepted native proofs."""
import math
from .rigorous_eigenpair_locus import require


class BoundaryTargetReached(RuntimeError):
    def __init__(self,flags):self.flags=flags;super().__init__(', '.join(flags))


def event_flags(diagnostics):
    spectrum=diagnostics.get('spectrum',{});fold=diagnostics.get('fold',{})
    if 'unresolved' in spectrum or 'unresolved' in fold:return ['saved_diagnostics_unresolved']
    values=spectrum.get('eigenvalues',[]);index=spectrum.get('critical_nearest_index')
    if len(values)!=13 or type(index)is not int or not 0<=index<13:return ['saved_spectral_shape_unresolved']
    values=[complex(*v) for v in values]
    if not all(math.isfinite(v.real) and math.isfinite(v.imag) for v in values):return ['saved_spectral_finiteness_unresolved']
    residuals=spectrum.get('relative_residuals',[])
    if len(residuals)!=13 or not all(math.isfinite(v) and 0<=v<1e-8 for v in residuals) or abs(values[index]-1)>=1e-6 or not 0<=spectrum.get('mode_residual',math.inf)<1e-8:return ['saved_mode_residual_unresolved']
    other=[v for i,v in enumerate(values)if i!=index];flags=[]
    if min(abs(v-1)for v in other)<=.03:flags.append('nontrivial_plus_one_gap_at_most_0.03')
    if max(abs(v)for v in other)>=.97:flags.append('nontrivial_unit_circle_margin_at_most_0.03')
    c=fold.get('second_derivative_contraction')
    if type(c) not in (int,float) or not math.isfinite(c):flags.append('saved_fold_coefficient_unresolved')
    elif c>=-1:flags.append('fold_coefficient_proxy_reaches_minus_one_or_above')
    return flags

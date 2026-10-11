"""One fresh chart with outward-rounded proposed radii and the last proved parent."""
from fractions import Fraction
from .rigorous_eigenpair_locus import require
from .rigorous_eigenpair_path import quadratic_tube
from .eigenpair_wider import wider_inputs


def outward_dyadic(value,bits=128):
    v=Fraction(value);require(v>0 and type(bits)is int and 16<=bits<=192,'positive fixed-precision rounding')
    exponent=v.numerator.bit_length()-v.denominator.bit_length()
    if v<Fraction(2)**exponent:exponent-=1
    step=Fraction(2)**(exponent-bits+1);ratio=v/step
    return ((ratio.numerator+ratio.denominator-1)//ratio.denominator)*step


def repair_inputs(parent,proposal):
    initial,kw=wider_inputs(parent);scale=Fraction(2)**parent['settings']['radius_exponent'];old=[scale*Fraction(v) for v in parent['weights']]
    requested=list(map(Fraction,proposal['exact_radii']))
    require(list(map(Fraction,proposal['old_exact_radii']))==old and len(requested)==len(old),'proposal exact old radii')
    require(all(v>=old[i]>0 for i,v in enumerate(requested)),'proposal cannot shrink anchor allowance')
    new=[outward_dyadic(v) for v in requested];maximum=max(new);exponent=-100
    while Fraction(2)**exponent<maximum:exponent+=1
    require(-100<=exponent<=-1,'bounded outward radii')
    kw.update(radius_exponent=exponent,weights=[str(v/Fraction(2)**exponent) for v in new])
    audit={'rounding_significand_bits':128,'requested_exact_radii':[str(v) for v in requested],'rounded_exact_radii':[str(v) for v in new],'maximum_rounding_relative_increase':str(max(new[i]/requested[i] for i in range(len(new)))),'parent_right':parent['parameter_endpoints'][1],'parameter':kw['parameter'],'halfwidth_exponent':kw['halfwidth_exponent'],'radius_exponent':exponent,'weights':kw['weights'],'scope':'Proposal only. Every rounded radius is at least its requested and inherited radius; full whole-domain proof and anchor required.'}
    return initial,kw,audit


def repair(model,parent,proposal,*,tick=lambda:None,observe=lambda k,v:None):
    initial,kw,audit=repair_inputs(parent,proposal);observe('repair_proposal',audit)
    chart,_=quadratic_tube(model,initial,**kw,tick=tick,observe=observe)
    return chart

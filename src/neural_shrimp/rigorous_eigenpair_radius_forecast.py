"""Proposal-only fixed recent radius increments; all fresh proof gates unchanged."""
from fractions import Fraction
import numpy as np
from flint import ctx
from .rigorous_eigenpair_locus import require
from .rigorous_eigenpair_march import compact,next_inputs,midpoint_spectrum
from .rigorous_eigenpair_path import quadratic_tube
from .eigenpair_radius_diagnostic import diagnose
from .eigenpair_inflation_repair import outward_dyadic
from .eigenpair_gauge_diagnostics import saved_predictor_diagnostics
from .eigenpair_boundary_diagnostics import fold_indicator


class DiagnosticObserverFailure(RuntimeError):
    """Durability/resource exceptions must not be treated as numeric abstention."""


def radius_settings(proposal,old_weights,old_exponent):
    old=[Fraction(v)*Fraction(2)**old_exponent for v in old_weights]
    require(list(map(Fraction,proposal['old_exact_radii']))==old,'exact current radius identity')
    requested=list(map(Fraction,proposal['exact_radii']))
    require(len(requested)==len(old) and all(v>=old[i]>0 for i,v in enumerate(requested)),'nonshrinking proposal')
    new=[outward_dyadic(v) for v in requested];exponent=-100
    while Fraction(2)**exponent<max(new):exponent+=1
    require(exponent<=-1,'bounded proposed radius')
    return {'radius_exponent':exponent,'weights':[str(v/Fraction(2)**exponent) for v in new],'exact_radii':[str(v) for v in new]}


def march(model,parent,fixed_increment,*,accepted_cap=4,tick=lambda:None,observe=lambda k,v:None,tube=quadratic_tube,proposer=diagnose):
    require(type(accepted_cap)is int and 1<=accepted_cap<=4,'fixed accepted ceiling')
    previous=compact(parent);increment=list(map(Fraction,fixed_increment));require(len(increment)==len(previous['weights']) and all(v>=0 for v in increment),'fixed nonnegative native increment');attempts=[];accepted=[];calls=0;repairs=0
    def diagnostic_observe(kind,value):
        try:observe(kind,value)
        except BaseException as exc:raise DiagnosticObserverFailure(type(exc).__name__+': '+str(exc)) from exc
    with ctx.workprec(previous['settings']['precision']):
        for index in range(accepted_cap):
            initial,kwargs=next_inputs(previous)
            old=[Fraction(v)*Fraction(2)**kwargs['radius_exponent'] for v in kwargs['weights']]
            requested=[old[i]+2*increment[i] for i in range(len(old))]
            require(all(old[i]<=v<=old[i]*Fraction(17,16) for i,v in enumerate(requested)),'bounded forecast: at most1/16 relative increase per chart')
            updated=radius_settings({'old_exact_radii':list(map(str,old)),'exact_radii':list(map(str,requested))},kwargs['weights'],kwargs['radius_exponent'])
            observe('initial_radius_forecast',{'accepted_index':index,'fixed_increment':list(map(str,increment)),'inherited_exact_radii':list(map(str,old)),'settings':updated,'scope':'Fixed twice-recent increment, never feedback from forecast output. Proposal only; fresh whole tube and full anchor decide.'})
            kwargs.update(radius_exponent=updated['radius_exponent'],weights=updated['weights'])
            for attempt in range(2):
                call=calls;calls+=1;require(calls<=2*accepted_cap,'fixed eight-call ceiling')
                observe('march_call_start',{'call':call,'accepted_index':index,'attempt':attempt,'initial':initial,'parameter':kwargs['parameter'],'parent_right':previous['parameter_endpoints'][1],'radius_exponent':kwargs['radius_exponent'],'weights':kwargs['weights']})
                chart,_=tube(model,initial,**kwargs,tick=tick,observe=observe)
                row={'call':call,'index':index,'attempt':attempt,'gates':chart['gates'],'passed':chart['passed'],'parameter_endpoints':chart['parameter_endpoints']};attempts.append(row)
                diagnostics={}
                for name,fn in [('spectrum',lambda:midpoint_spectrum(chart,diagnostic_observe)),('fold',lambda:fold_indicator(chart,diagnostic_observe)),('predictor',lambda:saved_predictor_diagnostics(chart))]:
                    try:diagnostics[name]=fn()
                    except (ValueError,np.linalg.LinAlgError,OverflowError,ZeroDivisionError) as exc:diagnostics[name]={'unresolved':str(exc),'scope':'No certified diagnostic conclusion.'}
                observe('march_diagnostics',{'call':call,'diagnostics':diagnostics})
                if chart['passed']:
                    saved=compact(chart);observe('march_accepted',{'call':call,'index':index,'chart':saved,'diagnostics':diagnostics})
                    previous=saved;accepted.append(row);break
                onlyroot=chart['gates']=={'uniform_normalized_eigenpair_locus':False,'primitive_period':True,'anchor_identified':True}
                if not onlyroot or attempt==1:
                    return {'completed':False,'accepted':accepted,'attempts':attempts,'calls':calls,'repairs':repairs,'last_chart':previous,'stop':'unresolved chart; failed tail never advances'}
                proposal=proposer(chart,observe);updated=radius_settings(proposal,kwargs['weights'],kwargs['radius_exponent'])
                observe('march_radius_rounding',{'call':call,'settings':updated,'scope':'Outward nonshrinking radii only; fresh entire tube must qualify.'})
                kwargs.update(radius_exponent=updated['radius_exponent'],weights=updated['weights']);repairs+=1
                del chart
    return {'completed':True,'accepted':accepted,'attempts':attempts,'calls':calls,'repairs':repairs,'last_chart':previous,'stop':'fixed four-chart cap'}

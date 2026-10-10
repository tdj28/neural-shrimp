"""Four-chart continuation with at most one nonshrinking repair per chart."""
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


def radius_settings(proposal,old_weights,old_exponent):
    old=[Fraction(v)*Fraction(2)**old_exponent for v in old_weights]
    require(list(map(Fraction,proposal['old_exact_radii']))==old,'exact current radius identity')
    requested=list(map(Fraction,proposal['exact_radii']))
    require(len(requested)==len(old) and all(v>=old[i]>0 for i,v in enumerate(requested)),'nonshrinking proposal')
    new=[outward_dyadic(v) for v in requested];exponent=-100
    while Fraction(2)**exponent<max(new):exponent+=1
    require(exponent<=-1,'bounded proposed radius')
    return {'radius_exponent':exponent,'weights':[str(v/Fraction(2)**exponent) for v in new],'exact_radii':[str(v) for v in new]}


def march(model,parent,*,accepted_cap=4,tick=lambda:None,observe=lambda k,v:None,tube=quadratic_tube,proposer=diagnose):
    require(type(accepted_cap)is int and 1<=accepted_cap<=4,'fixed accepted ceiling')
    previous=compact(parent);attempts=[];accepted=[];calls=0;repairs=0
    with ctx.workprec(previous['settings']['precision']):
        for index in range(accepted_cap):
            initial,kwargs=next_inputs(previous)
            for attempt in range(2):
                call=calls;calls+=1;require(calls<=2*accepted_cap,'fixed eight-call ceiling')
                observe('march_call_start',{'call':call,'accepted_index':index,'attempt':attempt,'initial':initial,'parameter':kwargs['parameter'],'parent_right':previous['parameter_endpoints'][1],'radius_exponent':kwargs['radius_exponent'],'weights':kwargs['weights']})
                chart,_=tube(model,initial,**kwargs,tick=tick,observe=observe)
                row={'call':call,'index':index,'attempt':attempt,'gates':chart['gates'],'passed':chart['passed'],'parameter_endpoints':chart['parameter_endpoints']};attempts.append(row)
                diagnostics={}
                for name,fn in [('spectrum',lambda:midpoint_spectrum(chart,observe)),('fold',lambda:fold_indicator(chart,observe)),('predictor',lambda:saved_predictor_diagnostics(chart))]:
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

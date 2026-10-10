"""Stream a fixed finite sequence of fully witnessed arclength charts.

No chart retry, parameter search, gauge change or stability claim. A chart
must be durably emitted before its endpoint can anchor the next one.
"""
from fractions import Fraction
from flint import ctx
from .rigorous_orbits import exact
from .rigorous_connection import require,restore_ball
from .rigorous_connection_frames import restore_matrix,fraction_mid
from .rigorous_arclength_connection import arclength_tube

COMPACT=('passed','gates','statement','settings','normal','origin','center',
         'parameter_center','exact_parameter_endpoints','predictor_slope',
         'predictor_curvature','exact_weights','endpoint_root_enclosures')


def compact_chart(chart):return {key:chart[key] for key in COMPACT}


def next_arguments(previous):
    """Exact inherited gauge and shared endpoint; no native phase changes."""
    settings=previous['settings'];delta=Fraction(2)**settings['halfwidth_exponent']
    center=Fraction(previous['parameter_center']);shared=center+delta
    require(previous['exact_parameter_endpoints']==[str(center-delta),str(shared)],'predecessor exact endpoints')
    with ctx.workprec(settings['precision']):
        flat=[restore_ball(v) for v in previous['center']];s=restore_matrix(previous['predictor_slope']);t=restore_matrix(previous['predictor_curvature'])
        d=exact(str(2*delta))
        initial=[str(fraction_mid(flat[i]+s[i,0]*d+t[i,0]*d*d/2)) for i in range(len(flat))]
    return dict(initial=initial,normal=previous['normal'],origin=previous['origin'],parameter=str(center+2*delta),
                weights=previous['exact_weights'],period=previous['statement']['period'],
                anchor={'state_box':previous['endpoint_root_enclosures'][1],'exact_parameter':str(shared)})


def stream_charts(model,jet_model,previous,*,count=16,synthetic_profile=False,tick=lambda:None,observe=lambda k,r:None,emit_chart=lambda i,r:None):
    require(type(count) is int and 1<=count<=16,'fixed chart cap')
    require(previous['passed'] or synthetic_profile,'proved predecessor required')
    require(all(previous['gates'].values()) or synthetic_profile,'all predecessor gates required')
    settings=previous['settings'];summaries=[];start=previous['exact_parameter_endpoints'][1]
    for index in range(count):
        tick();arguments=next_arguments(previous)
        result=arclength_tube(model,jet_model,**arguments,**settings,tick=tick,observe=observe)
        # emit_chart must return only after fsync+rename+payload identity. If it
        # raises, this result cannot become a predecessor; partial raw remains.
        saved=emit_chart(index,result)
        summary={'index':index,'passed':result['passed'],'gates':result['gates'],'artifact':saved,
                 'parameter_center':result['parameter_center'],'exact_parameter_endpoints':result['exact_parameter_endpoints'],
                 'contraction':result['contraction'],'coordinate_inclusions':sum(result['coordinate_inclusions']),
                 'anchor_inclusions':sum(result['anchor_check'].get('coordinate_inclusions',[])),
                 'anchor_parameter_inside':result['anchor_check'].get('parameter_inside',False),
                 'proper_divisors_excluded':[d['divisor'] for d in result['divisors'] if d['excluded']],
                 'hidden_rate_endpoint_enclosures':[e[-1] for e in result['endpoint_root_enclosures']]}
        summaries.append(summary);observe('chart_committed',summary)
        if not result['passed'] and not synthetic_profile:break
        previous=compact_chart(result)
        del result
    connected=[v for v in summaries if v['passed']]
    return {'schema':'neural-shrimp-streamed-arclength-v1','requested_charts':count,'completed_charts':len(summaries),
            'passed':len(summaries)==count and all(v['passed'] for v in summaries),'qualified_charts':len(connected),
            'charts':summaries,'new_exact_gauge_endpoints':[start,connected[-1]['exact_parameter_endpoints'][1]] if connected else None,
            'scope':'same native primitive branch through exact shared-gauge containment; no attraction or fold identity'}

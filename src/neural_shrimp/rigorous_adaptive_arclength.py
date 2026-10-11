"""Continue from a proved endpoint with explicitly separate new error-box data.

The predecessor is never relabeled with the proposed radii. Every new chart
reacquires its entire changed domain through the unchanged chart kernel.
"""
from .rigorous_connection import require
from .rigorous_orbits import exact
from .rigorous_arclength_connection import arclength_tube
from .rigorous_arclength_batch import compact_chart,next_arguments


def stream_radii(model,jet_model,previous,*,new_weights,new_radius_exponent,count=16,
                 synthetic_profile=False,tick=lambda:None,observe=lambda k,r:None,emit_chart=lambda i,r:None):
    require(type(count) is int and 1<=count<=16,'fixed chart cap')
    require(type(new_radius_exponent) is int and -100<=new_radius_exponent<=-1,'new radius exponent')
    require(previous['passed'] and all(previous['gates'].values()) or synthetic_profile,'proved predecessor required')
    require(len(new_weights)==len(previous['center']),'new weight dimension')
    values=[exact(v) for v in new_weights]
    require(all(v.is_exact() and v.is_finite() and 0<v<=1 for v in values),'exact positive normalized new weights')
    settings=dict(previous['settings']);settings['radius_exponent']=new_radius_exponent
    summaries=[];start=previous['exact_parameter_endpoints'][1]
    for index in range(count):
        tick();arguments=next_arguments(previous);arguments['weights']=new_weights
        result=arclength_tube(model,jet_model,**arguments,**settings,tick=tick,observe=observe)
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
        previous=compact_chart(result);del result
    qualified=[v for v in summaries if v['passed']]
    return {'schema':'neural-shrimp-adapted-radii-stream-v1','requested_charts':count,'completed_charts':len(summaries),
        'passed':len(summaries)==count and all(v['passed'] for v in summaries),'qualified_charts':len(qualified),'charts':summaries,
        'new_exact_gauge_endpoints':[start,qualified[-1]['exact_parameter_endpoints'][1]] if qualified else None,
        'scope':'new whole-domain enclosures with separate proposed radii; same primitive branch, no attraction or fold identity'}

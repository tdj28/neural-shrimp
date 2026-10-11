"""Two unchanged Krawczyk chart proofs starting at an inherited exact gauge."""
from fractions import Fraction
from flint import ctx
from .rigorous_orbits import exact
from .rigorous_connection import require,restore_ball
from .rigorous_connection_frames import fraction_mid,restore_matrix
from .rigorous_arclength_connection import arclength_tube


def two_shifted_charts(model,jet_model,inputs,*,synthetic_profile=False,tick=lambda:None,observe=lambda k,r:None,**settings):
    shared=Fraction(inputs['anchor']['exact_parameter']);delta=Fraction(2)**settings['halfwidth_exponent']
    require(exact(str(shared)).is_exact(),'exact inherited start gauge')
    first=arclength_tube(model,jet_model,inputs['initial'],normal=inputs['normal'],origin=inputs['origin'],parameter=str(shared+delta),
        weights=inputs['weights'],period=inputs['period'],anchor=inputs['anchor'],tick=tick,observe=observe,**settings)
    observe('chart_completed',{'chart':0,'result':first});charts=[first]
    if first['passed'] or synthetic_profile:
        with ctx.workprec(settings.get('precision',256)):
            flat=[restore_ball(v) for v in first['center']];slope=restore_matrix(first['predictor_slope']);curv=restore_matrix(first['predictor_curvature']);d=exact(str(2*delta))
            proposed=[str(fraction_mid(flat[i]+slope[i,0]*d+curv[i,0]*d*d/2)) for i in range(len(flat))]
            anchor={'state_box':first['endpoint_root_enclosures'][1],'exact_parameter':str(shared+2*delta)}
            second=arclength_tube(model,jet_model,proposed,normal=inputs['normal'],origin=inputs['origin'],parameter=str(shared+3*delta),
                weights=inputs['weights'],period=inputs['period'],anchor=anchor,tick=tick,observe=observe,**settings)
        observe('chart_completed',{'chart':1,'result':second});charts.append(second)
    passed=len(charts)==2 and all(c['passed'] for c in charts)
    return {'schema':'neural-shrimp-two-shifted-arclength-tubes-v1','passed':passed,'charts':charts,
        'connected_parameter_endpoints':[str(shared),str(shared+4*delta)] if passed else None,
        'scope':'same native primitive branch through exact inherited gauge and full endpoint joins; no attraction or distant fold identity'}

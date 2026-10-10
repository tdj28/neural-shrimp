"""One fresh proof from a proved partial prefix; failed chart supplies radii only."""
from fractions import Fraction
from .rigorous_eigenpair_locus import require
from .rigorous_eigenpair_march import next_inputs
from .rigorous_eigenpair_inflation_march import radius_settings
from .rigorous_eigenpair_path import quadratic_tube


def repair_inputs(parent,failed,proposal):
    initial,kw=next_inputs(parent)
    require(failed['passed'] is False and failed['gates']=={'uniform_normalized_eigenpair_locus':False,'primitive_period':True,'anchor_identified':True},'failed root only')
    for key in ('period','dimension','sigma','shooting_dimension','normal','origin'):
        require(failed[key]==parent[key],'same native failed domain '+key)
    for key in ('precision','halfwidth_exponent','newton_steps'):
        require(failed['settings'][key]==parent['settings'][key],'same settings '+key)
    center=Fraction(kw['parameter']);delta=Fraction(2)**kw['halfwidth_exponent']
    require(failed['parameter_endpoints']==[str(center-delta),str(center+delta)],'same target interval')
    previous=[Fraction(v)*Fraction(2)**parent['settings']['radius_exponent'] for v in parent['weights']]
    old=[Fraction(v)*Fraction(2)**failed['settings']['radius_exponent'] for v in failed['weights']]
    require(len(old)==len(previous)==parent['shooting_dimension'] and all(v>=previous[i]>0 for i,v in enumerate(old)),'failed radii preserve parent allowance')
    require(len(failed['coordinate_radii'])==len(old),'saved radii dimension')
    for i,v in enumerate(failed['coordinate_radii']):
        require(v['rad'][0]==0 and Fraction(v['mid'][0])*Fraction(2)**v['mid'][1]==old[i],'exact saved failed radius')
    updated=radius_settings(proposal,failed['weights'],failed['settings']['radius_exponent'])
    kw.update(radius_exponent=updated['radius_exponent'],weights=updated['weights'])
    audit=dict(updated,parent_right=parent['parameter_endpoints'][1],parameter=kw['parameter'],parameter_endpoints=failed['parameter_endpoints'],scope='Same interval, inherited nonshrinking radius floor; only the proved endpoint is the anchor. Every whole-domain gate must be freshly checked.')
    return initial,kw,audit


def repair(model,parent,failed,proposal,*,tick=lambda:None,observe=lambda k,v:None,tube=quadratic_tube):
    initial,kw,audit=repair_inputs(parent,failed,proposal);observe('repair_proposal',audit)
    chart,_=tube(model,initial,**kw,tick=tick,observe=observe)
    return chart

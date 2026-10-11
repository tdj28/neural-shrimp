"""Preparatory chart35 changed-gauge ancestry; no launch or proof acquisition.

The existing independent rotated-root certificate identifies the same known
chart34 endpoint through its own fresh residual localization and full705
projected uniqueness membership. No scalar equality across gauges is assumed.
"""
from fractions import Fraction as F
from pathlib import Path
import subprocess,hashlib
import bind_native705_cubic_panel as P
G,M,O,need,read=P.G,P.M,P.O,P.need,P.read


def checked_artifact(ir,manifest,name):
    need(name in manifest and O.entry(ir/name)==manifest[name],'literal rotated-join artifact '+name)
    return read(ir/name)


def rotated_join(index,row,ir,proof,ib,inputs,predecessor,tick):
    need(row['ordinal']==35 and row['degree']==3,'fixed first changed-gauge cubic root')
    previous=M.select(index,34)
    need(predecessor['index']==previous and predecessor['complete_complement_endpoints']==previous['parameter_endpoints'],'complete preceding chart34 stable cover prerequisite')
    prior,oldrun,oldir,oldrt,olddomain,oldinputs,oldproof,oldib=M.metadata(index,34,tick)
    need(prior==previous and inputs['map_inputs']==oldinputs['map_inputs'],'same physical native map across changed gauge')
    spec=ib['card']['old_independent'];need({k:spec[k]for k in('root','source','outer','worker')}=={k:previous['independent'][k]for k in('root','source','outer','worker')},'literal preceding independent cubic prefix')
    om=read(oldir/'manifest.json')['files'];oldend=checked_artifact(oldir,om,'connected-endpoint.json')
    need(O.entry(oldir/'connected-endpoint.json')['sha256']==spec['endpoint']and oldend==dict(chart=oldproof['chart'],endpoint=oldproof['endpoints'][1]),'exact independently proved full705 prior endpoint')
    im=read(ir/'manifest.json')['files'];fresh=checked_artifact(ir,im,'fresh-endpoint-localization.json');oldloc=checked_artifact(ir,im,'old-endpoint-localization.json');membership=checked_artifact(ir,im,'new-root/projected-membership.json');runtime=checked_artifact(ir,im,'runtime.json')
    need(fresh['passed']and fresh['old_parameter']==previous['parameter_endpoints'][1]and fresh['old_domain']==previous['producer']['domain']['identity'],'same independently localized chart34 endpoint and original fullE')
    need(len(fresh['point'])==len(fresh['known'])==len(oldloc['known'])==len(oldend['endpoint'])==705,'all coordinates retained through both localizations')
    for v in fresh['point']:G.U.M.point(v)
    for vals in(fresh['known'],oldloc['known'],oldend['endpoint']):
        for v in vals:G.U.M.interval(v)
    for name in('tools/reviews/review_rotated705.py','tools/reviews/review_rotated705_math.py'):
        pin=runtime['sources'][name];need(O.entry(ir/'source'/name)==pin and hashlib.sha256(subprocess.check_output(['git','show',row['independent']['source']+':'+name],cwd=P.S.ROOT)).hexdigest()==pin['sha256'],'exact independently reviewed endpoint and projected-membership source')
    need(membership==proof['membership']and membership['passed']and membership['coordinates']==705 and membership['parameter_inside']and membership['known_parameter']is None,'literal independently recomputed changed-gauge uniqueness membership')
    need(len(membership['coordinate_inclusions'])==705 and all(v is True for v in membership['coordinate_inclusions']),'strict full705 projected membership')
    lo,hi=G.PREVIOUS.bounds(membership['projection']);left,right=map(F,row['parameter_endpoints'])
    need(left<=lo<=hi<=right,'entire previous stable endpoint projection inside current complete domain')
    need(proof['chart']['normal']!=oldproof['chart']['normal']or proof['chart']['origin']!=oldproof['chart']['origin'],'literal changed gauge; no scalar identity shortcut')
    tick()
    return dict(previous=predecessor,old_endpoint=om['connected-endpoint.json'],old_independent=spec,fresh_localization=im['fresh-endpoint-localization.json'],old_localization=im['old-endpoint-localization.json'],projected_membership=im['new-root/projected-membership.json'],projection=[str(lo),str(hi)],known_parameter=None,scope='Same independently known full705 chart34 endpoint, narrowed by the authenticated independent residual/localization proof, identified inside chart35 by full705 projected uniqueness. Scalar gauges differ; no interpolation or same-scalar assumption.')

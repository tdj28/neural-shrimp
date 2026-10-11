"""Existing cubic root/inverse families with literal independently proved joins."""
from fractions import Fraction as F
from pathlib import Path
from flint import arb,arb_mat
import bind_native705_cubic_complement as G
P,U,D,O,B,M,need,read=G.P,G.U,G.D,G.O,G.B,G.M,G.need,G.read
S=G.S
from neural_shrimp.cubic_complement_geometry import residual_derivative
exact_inverse=G.exact_inverse


def metadata(card,predecessor,tick):
    index=read(S.ROOT/M.CARD);ordinal=card['ordinal'];need(ordinal in (33,34),'first reviewed multi-chart family33–34')
    need(card['root_index']==M.select(index,ordinal),'literal independently indexed cubic root')
    row,run,ir,rt,domain,inp,proof,ib=M.metadata(index,ordinal,tick)
    need(row['degree']==3 and all(v is True for v in proof['gates'].values()),'independently complete cubic root')
    previous=M.select(index,ordinal-1);need(predecessor['index']==previous and predecessor['complete_complement_endpoints']==previous['parameter_endpoints'],'entire preceding stable root cover')
    need(row['independent']['root']==previous['independent']['root']and row['independent']['source']==previous['independent']['source'],'same independently ordered three-chart chain')
    oldpath=ir/previous['independent']['proof']['path'];need(O.entry(oldpath)==previous['independent']['proof']['identity'],'exact previous full705 independent root')
    old=read(oldpath);M.proof_gates(old,previous);chart=proof['chart'];oldchart=old['chart'];endpoint=read(ir/f"connected-endpoint-{ordinal-32:03d}.json")
    need(endpoint==dict(chart=oldchart,endpoint=old['endpoints'][1]),'literal actual full705 prior endpoint used by original verifier')
    need(chart['normal']==oldchart['normal']and chart['origin']==oldchart['origin']and proof['anchor']['known_parameter']==oldchart['parameter_endpoints'][1],'exact gauge and proved prior parameter before scalar membership')
    lo,hi=G.PREVIOUS.bounds(proof['anchor']['projection']);left,right=map(F,row['parameter_endpoints'])
    need(left<=lo<=hi<=right and proof['anchor']['passed']and len(proof['anchor']['coordinate_inclusions'])==705 and all(proof['anchor']['coordinate_inclusions']),'entire all705 joined root projection inside complete new domain')
    need(card['parameter_endpoints']==row['parameter_endpoints'],'new wholechart domain fixed')
    ip=ir/Path(row['independent']['proof']['path']).parent
    events=sorted((run/'events').glob('*-cubic_predictor_return.json.gz'));need(len(events)==1 and str(events[0].relative_to(run))==card['predictor']['path']and O.entry(events[0])==card['predictor']['identity'],'unique source-authenticated finite C0/predictor')
    manifest=read(ir/'manifest.json')['files']
    for name,pin in card['coefficients'].items():need(manifest[str(Path(row['independent']['proof']['path']).parent/name)]==pin and O.entry(ip/name)==pin,'literal independent coefficient family')
    expected={f'{label}_coefficient_{k}.json.gz'for label,count in [('defect',4),('forcing',6),('residual',5)]for k in range(count)}|{'inverse_linear_proposal.json.gz'}
    need(set(card['coefficients'])==expected,'complete independent D0..3/f0..5/R0..4 and C1')
    return dict(index=row,run=run,ir=ir,coefficient_dir=ip,runtime=rt,domain=domain,inputs=inp,proof=proof,independent_binding=ib,predictor_path=events[0],prior=predecessor,anchor_projection=[str(lo),str(hi)])


def bind(w,card,predecessor):
    data=metadata(card,predecessor,w.tick);row=data['index'];domain=data['domain'];proof=data['proof'];inp=data['inputs'];ip=data['coefficient_dir'];proposal=read(data['predictor_path']);cfg=row['producer']['settings']
    for k in ('center','slope','curvature','cubic'):need(domain[k]==proposal[k],'same emitted predictor and acquired fullE '+k)
    c,s,a,b=[[U.R.exact(U.M.point(v))for v in domain[k]]for k in('center','slope','curvature','cubic')]
    normal=[U.R.exact(F(x))for x in proof['chart']['normal']];origin=[U.R.exact(F(x))for x in proof['chart']['origin']]
    radii=[U.R.exact(U.M.point(v))for v in proof['root']['radii']];weights=[v/arb((1,-10))for v in radii]
    need(all(len(v)==705 for v in(c,s,a,b,normal,origin,weights))and all(v.is_exact()and 0<v<=1 for v in weights),'entire original cubic family and weights')
    original=arb_mat([[arb(0,r)]for r in radii]);image=arb_mat([[U.R.original_ball(v)]for v in proof['root']['image_correction']]);P.exact_coverage(domain['error'],[original[i,0]for i in range(705)],'literal fullE contains inherited error')
    C0=exact_inverse(proposal['preconditioner_exact']);C1=P.small_matrix(ip/'inverse_linear_proposal.json.gz',705)
    need(all(v.is_exact()for rr in C1.tolist()for v in rr),'literal exact independent C1')
    defect=[P.small_matrix(ip/f'defect_coefficient_{k}.json.gz',705)for k in range(4)];forcing=[P.small_matrix(ip/f'forcing_coefficient_{k}.json.gz',1)for k in range(6)];res=[P.small_matrix(ip/f'residual_coefficient_{k}.json.gz',1)for k in range(5)]
    E,metrics=M.OLD.fixed_frames(inp['map_inputs'],read(S.ROOT/M.CARD));model=B.TwoRateCuspMap(inp['map_inputs'])
    payload=dict(index=row,prior_card=data['independent_binding']['card'],predictor=card['predictor'],coefficient_artifacts=card['coefficients'],prior_stable_authority=data['prior'],prior_projection=data['anchor_projection'],parameter_endpoints=card['parameter_endpoints'],original_parameter_endpoints=row['parameter_endpoints'],scope='Same independently proved full705 cubic root, source-authenticated affine inverse/defect and quartic residual. No new root or fixed-C replacement; original fullE domains retained.')
    w.save('input-binding.json',payload)
    return dict(model=model,c=c,s=s,a=a,b=b,normal=normal,origin=origin,sc=U.R.exact(F(cfg['parameter'])),halfwidth=F(2)**cfg['halfwidth_exponent'],weights=weights,original=original,image=image,C=[C0,C1],ds=defect,ff=forcing,rp=residual_derivative(res),E=E,metrics=metrics,domain=domain)

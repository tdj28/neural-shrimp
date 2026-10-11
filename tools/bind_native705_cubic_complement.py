"""Source-bound cubic known-root family and its original polynomial inverse."""
from fractions import Fraction as F
from pathlib import Path
from flint import arb,arb_mat
import probe_native705_second_quarter as S
import check_native705_chart31_boundary_hypotheses as PREVIOUS
from neural_shrimp.cubic_complement_geometry import residual_derivative
P,U,D,O,B,need,read=S.P,S.U,S.D,S.O,S.B,S.need,S.read
M=P.M


def exact_inverse(rows,n=705):
    need(type(rows)is list and len(rows)==n and all(type(r)is list and len(r)==n for r in rows),'complete finite C0 proposal')
    need(all(type(v)is list and len(v)==2 and all(type(k)is int for k in v)for r in rows for v in r),'literal dyadic proposal')
    C=arb_mat([[arb(tuple(v))for v in row]for row in rows]);need(all(v.is_finite()and v.is_exact()for row in C.tolist()for v in row),'exact finite original C0');return C


def metadata(card,tick):
    index=read(S.ROOT/M.CARD);need(card['root_index']==M.select(index,card['ordinal'])and card['ordinal']==32,'literal first cubic32 index')
    row,run,ir,rt,domain,inp,proof,ib=M.metadata(index,card['ordinal'],tick)
    need(row['degree']==3 and all(v is True for v in proof['gates'].values()),'independently complete cubic root')
    ip=ir/Path(row['independent']['proof']['path']).parent
    events=sorted((run/'events').glob('*-cubic_predictor_return.json.gz'));need(len(events)==1 and str(events[0].relative_to(run))==card['predictor']['path']and O.entry(events[0])==card['predictor']['identity'],'unique source-authenticated finite C0/predictor')
    manifest=read(ir/'manifest.json')['files']
    for name,pin in card['coefficients'].items():need(manifest[str(Path(row['independent']['proof']['path']).parent/name)]==pin and O.entry(ip/name)==pin,'literal independent coefficient family')
    expected={f'{label}_coefficient_{k}.json.gz'for label,count in [('defect',4),('forcing',6),('residual',5)]for k in range(count)}|{'inverse_linear_proposal.json.gz'}
    need(set(card['coefficients'])==expected,'complete independent D0..3/f0..5/R0..4 and C1')
    prior=PREVIOUS.check();need(prior==read(S.ROOT/'docs/reviews/native705-chart31-boundary-hypotheses.json')and prior['passed']and prior['chart']==31,'complete independently stable prior chart')
    need(ib['card']['anchor']==dict(PREVIOUS.SPECS['root'],endpoint='41a904368aeb9458297bf3a436474698d0d95beced993a01102961be13fda078'),'same exact previous root endpoint authority')
    anchor=proof['anchor'];lo,hi=PREVIOUS.bounds(anchor['projection']);left,right=map(F,card['parameter_endpoints']);need(left<=lo<=hi<=right and anchor['passed']and all(anchor['coordinate_inclusions']),'entire prior stable endpoint projection in proposed cubic portion')
    return dict(index=row,run=run,ir=ir,coefficient_dir=ip,runtime=rt,domain=domain,inputs=inp,proof=proof,independent_binding=ib,predictor_path=events[0],prior=prior,anchor_projection=[str(lo),str(hi)])


def bind(w,card):
    data=metadata(card,w.tick);row=data['index'];domain=data['domain'];proof=data['proof'];inp=data['inputs'];ip=data['coefficient_dir'];proposal=read(data['predictor_path']);cfg=row['producer']['settings']
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

"""Bind eight independently connected quadratic/cubic705 roots and Taylor data."""
import gc
from fractions import Fraction as F
from pathlib import Path
from flint import arb,arb_mat
import bind_native705_complement_march as OLD
H=OLD.H;B,T,L,D,U,O,need,read=OLD.B,OLD.T,OLD.L,OLD.D,OLD.U,OLD.O,OLD.need,OLD.read
CARD='docs/plans/native705-later-complement-inputs.json'

def select(card,ordinal):
    need(type(ordinal)is int and 31<=ordinal<=38,'fixed later eight charts')
    need([r['ordinal']for r in card['charts']]==list(range(31,39)),'complete fixed later-root index')
    row=card['charts'][ordinal-31];cfg=row['producer']['settings']
    need(row['degree']==(2 if ordinal==31 else 3)and cfg['period']==18 and cfg['precision']==256 and cfg['radius_exponent']==-10 and cfg['halfwidth_exponent']==(-5 if ordinal<=34 else -6),'literal original later-root family')
    h=F(2)**cfg['halfwidth_exponent'];s=F(cfg['parameter'])
    need(row['parameter_endpoints']==[str(s-h),str(s+h)]and row['initial_depth']==cfg['halfwidth_exponent']+7,'exact original domain and measured initial width')
    return row

def proof_gates(proof,row):
    root=proof['root'];chart=proof['chart'];cfg=row['producer']['settings']
    need(proof['passed']and root['passed']and root['dimension']==705 and len(root['coordinate_inclusions'])==705 and all(v is True for v in root['coordinate_inclusions']),'all705 independently strict root inclusions')
    need(chart['shooting_dimension']==705 and chart['parameter_center']==cfg['parameter']and chart['parameter_endpoints']==row['parameter_endpoints']and chart['settings']=={k:cfg[k]for k in('halfwidth_exponent','radius_exponent')},'same independently proved full parameter domain')
    a=proof.get('anchor',proof.get('membership'));need(a['passed']and a['parameter_inside']and a['coordinates']==705,'full independently proved prior-root membership')
    if row['ordinal']==31:need(set(a)=={'coordinates','parameter_inside','passed','projection'},'literal older independently checked all-coordinate summary')
    else:need(len(a['coordinate_inclusions'])==705 and all(v is True for v in a['coordinate_inclusions']),'all later full-coordinate membership bits')
    div=proof.get('primitive',proof.get('proper_divisors'))
    if isinstance(div,list):div={str(v['divisor']):v['coordinates']for v in div}
    need(set(div)=={'1','2','3','6','9'}and all(v and all(type(i)is int and 0<=i<13 for i in v)for v in div.values()),'all proper18 divisors independently excluded')
    frame=proof.get('frame_nonzero',proof.get('frame_nonzero_coordinates',proof.get('independent_frame_coordinates')))
    need(frame and all(type(i)is int and 0<=i<13 for i in frame),'independent realified frame')
    if 'gates'in proof:need(all(v is True for v in proof['gates'].values()),'all actual independent curve gates')
    return chart

def declared_producer(row,card):
    p=row['producer'];ordinal=row['ordinal'];key='new_producer'if ordinal==35 else'producer';ref=card[key]
    need(ref['root']==p['root']and ref['outer']==p['outer']and ref['source']==p['source'],'literal independently authenticated producer parent')
    if ordinal==31:need(ref['worker']==p['worker'],'literal single producer')
    elif ordinal<=34:need(ref['workers'][ordinal-32]==p['worker'],'literal selected cubic-prefix worker')
    elif ordinal==35:need(ref['worker']==p['worker'],'literal rotated worker')
    else:
        need(card['selected_attempts']==[1,2,4],'literal accepted maintained prefix with failure retained')
        need(ref['workers'][[1,2,4][ordinal-36]-1]==dict(attempt=[1,2,4][ordinal-36],worker=p['worker']),'literal selected maintained worker')

def metadata(card,ordinal,tick):
    row=select(card,ordinal);p=row['producer'];i=row['independent'];run=Path(p['run']);ir=Path(i['root'])/'run'
    O.verify(Path(p['root']),p['outer']);O.verify(run,p['worker']);rt=D.B.archived(run,p['source'],'runtime.json','source');tick()
    O.verify(Path(i['root']),i['outer']);O.verify(ir,i['worker']);it=D.B.archived(ir,i['source'],'runtime.json','source');io=read(ir/'outcome.json')
    need(io['passed']and io['connected']and io['native_calls']==0,'independently connected saved native705 proof')
    need(O.entry(ir/'input-binding.json')==i['input_binding'],'literal independent source/domain card');ib=read(ir/'input-binding.json');need(ib['card']==i['card'],'entire literal independent card');declared_producer(row,ib['card'])
    need(rt['settings']==p['settings']and rt['threads']==dict.fromkeys(O.THREADS,'1'),'literal original source settings')
    for name in B.NATIVE+T.PATH_FILES:
        need(O.entry(H.ROOT/name)==rt['sources'][name]==O.entry(run/'source'/name),'same executed native/path source '+name)
    need(all(rt['sources'][n]['sha256']==v['sha256']for n,v in ib['source_native'].items()),'independent actual native source equality')
    need(O.entry(run/'certificate.json.gz')==p['certificate']and O.entry(run/'inputs.json')==p['inputs'],'original complete evidence identities')
    dom=run/p['domain']['path'];need(O.entry(dom)==p['domain']['identity'],'literal independently authenticated domain event');domain=read(dom);inp=read(run/'inputs.json')
    need(inp['settings']==p['settings']and domain['parameter']==p['settings']['parameter']and all(domain[k]==p['settings'][k]for k in('halfwidth_exponent','radius_exponent')),'actual original model/domain settings')
    path=ir/i['proof']['path'];need(O.entry(path)==i['proof']['identity'],'selected independent root artifact');proof=read(path);proof_gates(proof,row)
    for spec in i['coefficients']:need(O.entry(ir/spec['path'])==spec['identity'],'independently reconstructed Taylor column')
    need(len(i['coefficients'])==row['degree']+2,'complete residual Taylor order')
    return row,run,ir,rt,domain,inp,proof,ib

def coefficients(ir,specs,n=705):
    cols=[]
    for spec in specs:
        raw=read(ir/spec['path']);need(raw['encoding']=='mid-rad-dyadic-pairs'and len(raw['rows'])==n and all(len(v)==1 for v in raw['rows']),'entire independent residual column')
        col=[]
        for item in raw['rows']:
            m,r=item[0];need(all(type(v)is int for v in m+r)and len(m)==len(r)==2 and r[0]>=0,'literal finite dyadic ball encoding')
            z=arb(arb(tuple(m)),arb(tuple(r)));need(z.is_finite(),'finite independently derived coefficient');col.append(z)
        cols.append(col)
    return arb_mat([[col[k]for col in cols]for k in range(n)])

def known_chart(card,ordinal,tick):
    row,run,ir,rt,domain,inp,proof,ib=metadata(card,ordinal,tick);cfg=row['producer']['settings'];chart=proof['chart']
    vals={k:[U.R.exact(U.M.point(v))for v in domain[k]]for k in('center','slope','curvature')}
    vals['cubic']=[U.R.exact(U.M.point(v))for v in domain['cubic']]if row['degree']==3 else[arb(0)]*705
    radius=arb((1,-10));radii=[U.R.exact(U.M.point(v))for v in proof['root']['radii']];weights=[v/radius for v in radii]
    need(all(len(v)==705 for v in vals.values())and len(weights)==705 and all(v.is_exact()and 0<v<=1 for v in weights),'full exact predictor and inherited independent correction radii')
    # The original independent verifier reconstructed these coefficients directly
    # from this authenticated native domain, without parsing producer certificates.
    for old,r in zip(domain['error'],radii,strict=True):
        lo,hi=U.M.interval(old);q=U.M.point(D.ball_json(r));need(lo<=-q and q<=hi,'original derivative correction domain contains known root radii')
    normal=[U.R.exact(F(v))for v in chart['normal']];origin=[U.R.exact(F(v))for v in chart['origin']];need(len(normal)==len(origin)==705 and any(normal),'same independently authenticated affine gauge')
    model_inputs=inp['map_inputs'];need(model_inputs['network']['d']==2 and model_inputs['network']['width']==3 and model_inputs['network']['bias']is True and 1<=len(model_inputs['x'])<=24,'original native13 profile-size model')
    E,metrics=OLD.fixed_frames(model_inputs,card);cc=coefficients(ir,row['independent']['coefficients']);tick()
    args=dict(**vals,weights=weights,original_halfwidth=arb((1,cfg['halfwidth_exponent'])),radius=radius,center_coefficients=cc,whole_coefficients=cc)
    binding=dict(index=row,original_run=str(run),independent_input_card=ib['card'],same_root=True,coefficients='Independent480-bit native residual Taylor columns, restored outward at384. Same indexed source/domain/root; producer derived certificate unparsed.',scope='Original independently proved primitive18 root and prior full705 membership in its literal gauge. No new root or cross-gauge interpolation.')
    del proof,domain;gc.collect();return B.TwoRateCuspMap(model_inputs),args,normal,origin,U.R.exact(F(cfg['parameter'])),E,metrics,binding

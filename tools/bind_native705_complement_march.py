"""Literal binding of24 existing independently connected quadratic705 root charts."""
import gc,hashlib,json
from pathlib import Path
from fractions import Fraction as F
from flint import arb
import probe_native705_complement_halves as H
W=H.W;B,T,L,D,U,O,need,read=H.B,H.T,H.L,H.D,H.U,H.O,H.need,H.read
CARD='docs/plans/native705-complement-march-inputs.json'

def select(card,ordinal):
    need(type(ordinal)is int and 7<=ordinal<=30,'fixed remaining24 only')
    need([r['ordinal']for r in card['charts']]==list(range(7,31)),'entire fixed24 root index')
    row=card['charts'][ordinal-7];need(row['unit']==(ordinal-7)%8+1,'fixed batch-local native unit')
    need(row['settings']['halfwidth_exponent']==-7 and row['settings']['radius_exponent']==-10 and row['settings']['period']==18 and row['settings']['precision']==256 and row['settings']['newton_steps']==2,'exact original quadratic settings')
    return row

def fixed_frames(inp,card):
    frozen=card['frozen_basis'];fr=Path(frozen['root']);O.verify(fr,frozen['outer']);O.verify(fr/'run',frozen['worker']);D.B.archived(fr/'run',frozen['source'],'runtime.json','source')
    fo=read(fr/'run/outcome.json');fl=read(fr/'launcher-result.json');fb=read(fr/'run/bindings.json');prop=read(fr/'run/events/001-proposal.json')
    need(not fo['passed']and fo['status']=='completed_sufficient_bound_unresolved'and fo['native_calls']==0 and fl['exit_code']==2 and not fl['owned_live_pids']and prop['strategy']=='periodic_lyapunov_balanced_orthogonal','literal finite E/B proposal provenance')
    original=card['physical_origin'];rr=Path(original['root']);need(fb['card']['producer']==original,'same physical source of proposals');O.verify(rr,original['outer']);O.verify(rr/'run',original['worker']);D.B.archived(rr/'run',original['source'],'runtime.json','source')
    need(inp==read(rr/'run/inputs.json')['map_inputs'],'same literal native objective and both rate variables')
    E=[U.R.matrix(v)for v in prop['phase_complements']];metrics=[D.exact_float_matrix(D.np.asarray(v))for v in prop['upper_cholesky_factors']]
    need(len(E)==len(metrics)==18 and all(v.is_exact()and v.is_finite()for m in E+metrics for row in m.tolist()for v in row),'all18 exact finite proposals')
    return E,metrics

def known_chart(card,ordinal,tick):
    row=select(card,ordinal);run=Path(row['producer_root']);pr=run.parents[1];unit=row['unit'];O.verify(pr,row['producer_outer']);O.verify(run,row['producer_manifest']['sha256']);rt=D.B.archived(run,row['producer_source'],'runtime.json','source')
    out=read(run/'outcome.json');po=read(pr/'outcome.json');need(po['commit']==row['producer_source']and po['passed']and po['connected_count']==8 and not po['owned_live_pids'],'complete original rotated batch')
    need(len(po['workers'])==8 and [v['unit']for v in po['workers']]==list(range(1,9))and all(v['exit_code']==0 and not v['owned_live_pids']for v in po['workers']),'eight original owned roots')
    need(out['passed']and out['unit']==unit and out['ticks']==253 and out['strict_inclusions']==705 and out['events']['direction_return']==144 and out['events']['pal_path_phase_return']==108,'complete original fresh quadratic acquisitions')
    independent=row['independent'];ir=Path(independent['root']);O.verify(ir,independent['outer']['sha256']);O.verify(ir/'run',independent['worker']['sha256']);D.B.archived(ir/'run',independent['source'],'runtime.json','source')
    need(O.entry(ir/'run/input-binding.json')==independent['input_binding'],'literal independently reviewed input binding')
    io=read(ir/'run/outcome.json');il=read(ir/'launcher-result.json');ib=read(ir/'run/input-binding.json');ic=ib['card']
    need(io['passed']and io['connected']and io['connected_count']==8 and io['native_calls']==0 and il['exit_code']==0 and not il['owned_live_pids'],'complete independently connected native batch')
    need(ic['root']==str(pr)and ic['source']==row['producer_source']and ic['outer']==row['producer_outer']and ic['qualified_prefix']==ic['expected_workers']==8 and len(ic['charts'])==len(ib['charts'])==8,'same exact independent producer scope')
    chosen=ib['charts'][unit-1];need(chosen==dict(root=str(run),manifest=row['producer_manifest'],certificate=row['certificate'])and O.entry(run/'certificate.json.gz')==row['certificate'],'same actual independent full certificate')
    need(ic['charts'][unit-1]==dict(path=f'chart{unit:03d}/run',unit=unit,worker=row['producer_manifest']['sha256']),'same selected original native unit')
    up=ir/f'run/chart-{unit}/uniform.json';jp=ir/f'run/chart-{unit}/join.json';need(O.entry(up)==independent['uniform']and O.entry(jp)==independent['join'],'exact selected independent uniform root and join')
    uniform=read(up);join=read(jp);need(uniform['inclusions']==705 and all(v for k,v in uniform['gates'].items()if k!='anchor_identified')and join['passed']and join['coordinates']==705,'full independent root and prior705 membership')
    # Rotated first chart has a full projected interval; never invent an exact scalar.
    need(row['same_gauge']==(unit!=1)and io['proofs'][unit-1]['same_gauge']==row['same_gauge'],'actual preserved gauge relation')
    chart=read(run/'certificate.json.gz');ci=read(run/'chart-inputs.json');need(all(chart[k]==ci[k]for k in['normal','origin','weights']),'full actual gauge and weights')
    need(chart['schema']=='neural-shrimp-native-unit-pair-pal-v1'and(chart['dimension'],chart['period'],chart['shooting_dimension'])==(13,18,705),'full native quadratic705 family')
    need(chart['settings']=={k:row['settings'][k]for k in['precision','newton_steps','halfwidth_exponent','radius_exponent']}and chart['parameter_center']==row['settings']['parameter']and chart['parameter_endpoints']==row['parameter_endpoints'],'exact original per-chart parameter domain')
    need(io['proofs'][unit-1]['parameter_endpoints']==chart['parameter_endpoints']and io['proofs'][unit-1]['unit']==unit,'same independently proved interval')
    need(join.get('parameter')==chart['parameter_endpoints'][0]if unit>1 else join.get('parameter_inside')is True,'actual exact-gauge or projected-gauge join')
    need(all(chart['gates'][k]for k in ['uniform_full_native_unit_pair_locus','primitive_period','realified_frame_independent','whole_angle_positive']),'all original root/primitive/frame/angle gates')
    need({k:hashlib.sha256(json.dumps(chart[k],sort_keys=True,separators=(',',':')).encode()).hexdigest()for k in ['normal','origin','weights']}==row['geometry_hashes'],'exact frozen affine geometry')
    c,s,a=[[U.R.exact(U.M.point(v))for v in chart[k]]for k in['center','slope','curvature']];w=[U.R.exact(F(v))for v in chart['weights']]
    need(len(c)==len(s)==len(a)==len(w)==705 and all(0<v<=1 for v in w),'all705 exact coefficients and positive weights')
    need([U.M.point(v)for v in chart['coordinate_radii']]==[F(1,1024)*F(v)for v in chart['weights']],'literal original correction radii')
    inp=read(run/'inputs.json')['map_inputs'];need(inp['network']['d']==2 and inp['network']['width']==3 and inp['network']['bias']is True and 1<=len(inp['x'])<=24,'same profile-sized native13 model')
    for name in B.NATIVE+T.PATH_FILES:need(O.entry(H.ROOT/name)==rt['sources'][name]==O.entry(run/'source'/name),'unchanged native/path source '+name)
    E,metrics=fixed_frames(inp,card);normal=[U.R.exact(F(v))for v in chart['normal']];origin=[U.R.exact(F(v))for v in chart['origin']]
    need(len(normal)==len(origin)==705 and any(normal),'exact full original affine gauge')
    args=dict(center=c,slope=s,curvature=a,weights=w,original_halfwidth=arb((1,-7)),radius=arb((1,-10)),center_coefficients=U.R.matrix(chart['quadratic_center_coefficients']),whole_coefficients=U.R.matrix(chart['quadratic_whole_coefficients']))
    binding=dict(card=card,ordinal=ordinal,index=row,map_inputs=inp,independent_input_card=ic,original_run=str(run),independent_uniform=O.entry(up),independent_join=O.entry(jp),same_root=True,scope='Same literal independently proved primitive18 root in its original affine gauge. Restriction/root correction inherited; no new root or scalar interpolation across gauges.')
    del chart;gc.collect();tick();return B.TwoRateCuspMap(inp),args,normal,origin,U.R.exact(F(row['settings']['parameter'])),E,metrics,binding

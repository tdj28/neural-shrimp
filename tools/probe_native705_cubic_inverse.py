#!/usr/bin/env python3
"""Single source-bound cubic705 pilot at fixed h=1/8, rho=2^-8."""
import profile_native705_cubic_inverse as P
import probe_native705_compact_inverse as N
from neural_shrimp.rigorous_cubic_correlated_pal import cubic_correlated_tube
from neural_shrimp.rigorous_orbits import vector_json
C=N.C;ROOT=C.ROOT;BASE=P.BASE;TARGET=BASE/'attempt001-target/run'
SETTINGS=dict(C.SETTINGS,halfwidth_exponent=-3,radius_exponent=-8,compact_evidence=True,predictor_degree=3,residual_degree=4)
SOURCES=list(dict.fromkeys(N.SOURCES+P.SOURCES+['tools/probe_native705_cubic_inverse.py','tools/launch_native705_cubic_inverse.py','tests/test_native705_cubic_inverse_driver.py','docs/plans/native705-cubic-target.md','docs/reviews/native705-compact-target.json']))
COMPACT_PRIOR=N.BASE/'attempt001-target'
CPINS=dict(outer='76c5decd85de28edfdaedcb35ff969a25067e9dc9ced63ec13b8c6ec753bab85',worker='1b45c3ac083ff675687db899c4442ce046b8c396dd3aa2cc17a786dfe141f6bd',source='917641f164f23d9be855b865c58564babd67b2ca')
original_parent=C.bind_parent
CUBIC=None

def admit_profile(out):
 C.require(out['passed']is True and out['target_acquired']is False and out['ticks']==629,'complete new cubic synthetic629 profile')
 e=out['events'];expected={'direction_return':180,'pal_path_phase_return':108,'pal_chart_return':1,'gradient_path_input':144,'gradient_path_return':144,'gradient_path_assembly_return':4,'cubic_correlated_chart_return':2,'cubic_correlated_domain':2,'cubic_path_input':144,'cubic_path_return':144,'cubic_path_assembly_return':4,'profile_gradient_input':24,'profile_gradient_return':24,'profile_quartic_input':24,'profile_quartic_return':24}
 C.require(all(e.get(k)==v for k,v in expected.items()),'complete actual cubic/quartic profile native acquisitions')
 C.require(0<out['elapsed_seconds']<240 and 0<out['peak_rss_kib']<6*1024**2,'full profile actual limits')
 forecast=P.projection(out['chart_times'],out['gradient_times'],out['path_times'],out['peak_rss_kib'],out['stored_bytes_before_final'])
 C.require(forecast==out['projection']and forecast['resource_admitted'],'same-source measured cubic forecast');return forecast

def bind_profile(args,identity):
 binding=N.bind_profile(args,identity)
 C.inventory(COMPACT_PRIOR,CPINS['outer']);C.inventory(COMPACT_PRIOR/'run',CPINS['worker'])
 old=C.read(COMPACT_PRIOR/'run/outcome.json');rt=C.read(COMPACT_PRIOR/'run/runtime.json')
 C.require(rt['commit']==CPINS['source']and rt['settings']==N.SETTINGS and old['passed']and old['connected']and old['ticks']==148 and old['strict_inclusions']==old['anchor_inclusions']==705,'complete unchanged native compact measured predecessor')
 binding['retained_compact_proof']=dict(root=str(COMPACT_PRIOR),**CPINS)
 return binding

def cubic_proposal(C0,coeff,tick=lambda:None):
 C.require(C0.nrows()==C0.ncols()==len(coeff)and all(len(row)==4 for row in coeff),'full original cubic residual proposal')
 residual=C.arb_mat([[C.decode(row[3])]for row in coeff]);tick();v=-C0*residual
 return [v[i,0].mid()for i in range(len(coeff))]

def bind_parent(tick,save,identity):
 global CUBIC
 inp,p,anchor=original_parent(tick,save,identity)
 pre=C.arb_mat([[C.decode(v)for v in row]for row in p['preconditioner']]);CUBIC=cubic_proposal(pre,p['quadratic_center_coefficients'],tick)
 save('cubic-proposal.json',{'cubic':vector_json(CUBIC),'quadratic_center_residual_cubic':vector_json([C.decode(row[3])for row in p['quadratic_center_coefficients']]),'scope':'Finite midpoint proposal only. All new cubic-path coefficients and whole quartic remainder are freshly reacquired; no cancellation is presumed.'})
 return inp,p,anchor

def curve(model,center,slope,curvature,C0,**kw):
 C.require(CUBIC is not None and len(CUBIC)==len(center),'prepared source-bound finite cubic proposal')
 return cubic_correlated_tube(model,center,slope,curvature,CUBIC,C0,**kw)

def configure():
 global CUBIC
 CUBIC=None;C.P=P;C.BASE=BASE;C.TARGET=TARGET;C.SETTINGS=SETTINGS;C.SOURCES=SOURCES;C.COMPACT=True;C.TICK_CAP=149;C.FINAL_EVENT='cubic_correlated_chart_return';C.PATH_EVENT='cubic_path_return';C.PATH_ASSEMBLY_EVENT='cubic_path_assembly_return';C.bind_profile=bind_profile;C.admit_profile=admit_profile;C.bind_parent=bind_parent;C.correlated_tube=curve

def main(args):configure();return C.main(args)
parser=C.parser
if __name__=='__main__':raise SystemExit(main(parser().parse_args()))

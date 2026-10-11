#!/usr/bin/env python3
"""Fixed four double-width705 proofs, then separate exact-gauge native joins."""
from fractions import Fraction
from flint import arb
import probe_native_unit_pair_pal_batch as B
from neural_shrimp.rigorous_unit_pair_pal import anchor_test
ROOT=B.ROOT;C=B.C;P=B.P;BASE=P.BASE/'batch002';entry=B.entry;read=B.read;inventory=B.inventory
PARENT=P.BASE/'batch001';PIN='2c116e332c8c8e85edda2d0c14329f9ee97ec0f8c1ab006f78cda3e99deb8cff'
DIAG=P.BASE/'batch001-diagnosis'
DPINS={'source': '78d246c2d1076db3b70b24ff73e4efc99f82338f', 'outer': 'd7a7b08969618b48897a56e0405dbfeea714196c650785e663451eb905a3c6af', 'worker': '08af12caf0a460a6b0652cd51452d266622c9db70777e3f97a04f439189cd435'}
AUDIT=B.Path('/data2/neural-shrimp-data/audits/native705-batch-independent/attempt001')
APINS={'source':'0cbabdf36b18121f86e36944fad10509bf562417','outer':'6382f1d06e34ced6cd252d7b6b5dc1d986017ef4c844d02b0a932569538adaea','worker':'d2bc72f5782ffbae819689926124656d460479dad70d2fc692824b0fed605815'}
OLD_BIND=B.bind_parent
SOURCES=list(dict.fromkeys(B.SOURCES+['tools/probe_native_unit_pair_pal_wide.py','tools/launch_native_unit_pair_pal_wide.py','tests/test_native_unit_pair_pal_wide.py','docs/plans/native-unit-pair-pal-wide.md']))

def settings(unit):
 B.require(type(unit)is int and 1<=unit<=4,'fixed four wider proposals')
 return dict(period=18,precision=256,parameter=str(Fraction(255+128*unit,4096)),halfwidth_exponent=-6,radius_exponent=-10,newton_steps=2)

def bind_parent():
 model,first,prior=OLD_BIND();del first
 inventory(PARENT,PIN);out=read(PARENT/'outcome.json');B.require(out['passed']and out['connected_count']==4 and out['connected_right']=='319/4096'and out['commit']=='cb83dcc7677e5e67fabe7e2f86fd8e7255395265'and not out['owned_live_pids'],'fixed complete prior4chain')
 B.require(read(PARENT/'parent-binding.json')==prior,'same independent initial705 root and join')
 peaks=[];times=[];sizes=[]
 for unit,(owner,join)in enumerate(zip(out['workers'],out['joins'],strict=True),1):
  B.require(owner['unit']==join['unit']==unit and owner['exit_code']==0 and not owner['owned_live_pids']and owner['stop_reason']is None and join['passed']and join['coordinates']==705,'all prior owned workers and native joins')
  p=PARENT/f'chart{unit:03d}';inventory(p/'run',join['worker_manifest']['sha256']);C.archive_bound(p,out['commit']);run=read(p/'run/outcome.json');rt=read(p/'run/runtime.json')
  expected=dict(period=18,precision=256,parameter=str(Fraction(31+64*unit,4096)),halfwidth_exponent=-7,radius_exponent=-10,newton_steps=2)
  B.require(rt['settings']==expected and run['passed']and run['ticks']==253 and run['strict_inclusions']==705 and run['status']=='uniform705_unjoined','full original source-indexed native root')
  B.require(read(p/'run/inputs.json')['map_inputs']==model,'same actual model throughout chain');j=read(PARENT/f'join-{unit:03d}.json');B.require(j['passed']and j['coordinate_inclusions']==[True]*705,'retained full-coordinate original membership')
  peaks.append(run['peak_rss_kib']);times.append(run['elapsed_seconds']);sizes.append(run['stored_bytes_before_final'])
 B.require(len(times)==4,'exact full prior four-chart inventory')
 inventory(DIAG,DPINS['outer']);inventory(DIAG/'run',DPINS['worker']);C.archive_bound(DIAG,DPINS['source']);dl=read(DIAG/'launcher-result.json');dd=read(DIAG/'run/outcome.json')
 B.require(dl['exit_code']==0 and not dl['owned_live_pids']and dd['passed']and dd['native_calls']==0 and dd['geometry_points']==53 and dd['resolvent_available'],'fixed saved conditioning and geometry proposal')
 forecast={'seconds':2*max(times)+20,'rss_kib':2*max(peaks)+128*1024,'bytes':2*max(sizes)+8*1024**2}
 B.require(forecast['seconds']<240 and forecast['rss_kib']<6*1024**2 and forecast['bytes']<B.CAP,'measured same253call resource admission')
 chart=read(PARENT/'chart004/run/certificate.json.gz');B.require(B.uniform_admission(chart,B.decode(chart['whole_box'][-1])>0)and chart['parameter_endpoints']==['255/4096','319/4096'],'last complete native parent')
 inventory(AUDIT,APINS['outer']);inventory(AUDIT/'run',APINS['worker']);C.archive_bound(AUDIT,APINS['source']);audit=read(AUDIT/'run/outcome.json');al=read(AUDIT/'launcher-result.json');ab=read(AUDIT/'run/input-binding.json');ae=read(AUDIT/'run/connected-endpoint.json')
 B.require(al['exit_code']==0 and not al['owned_live_pids']and audit['passed']and audit['connected']and audit['connected_count']==4 and audit['connected_parameter_endpoints']==['-1/4096','319/4096'],'independent full4 connected705 audit')
 B.require(ab['batch']==str(PARENT)and ab['batch_manifest']==entry(PARENT/'manifest.json')and ab['producer_source']==out['commit'],'exact audited batch identity')
 B.require(ae['chart']['normal']==chart['normal']and ae['chart']['origin']==chart['origin']and ae['chart']['parameter_endpoints']==chart['parameter_endpoints']and len(ae['endpoint'])==705,'same exact audited endpoint gauge')
 chart['endpoint_root_boxes'][1]=ae['endpoint']
 return model,chart,{'independent_batch':dict(root=str(AUDIT),**APINS),'initial_independent':prior,'prior_batch':{'root':str(PARENT),'outer':PIN,'source':out['commit']},'diagnosis':dict(root=str(DIAG),**DPINS),'forecast':forecast,'scope':'Exact original gauge; numerical width/radius adaptation only. Full new root proofs and ordered joins decide accepted prefix.'}

def seed(parent,unit):
 cfg=settings(unit);delta=B.exact(cfg['parameter'])-B.exact(parent['parameter_center'])
 initial=[B.exact_mid(B.decode(c)+B.decode(s)*delta+B.decode(k)*delta*delta/2)for c,s,k in zip(parent['center'],parent['slope'],parent['curvature'],strict=True)]
 weights=[str(Fraction(3,4)*Fraction(w))for w in parent['weights']]
 B.require(len(initial)==len(weights)==705 and all(0<Fraction(w)<=1 for w in weights),'fixed positive full weights')
 return initial,list(parent['normal']),list(parent['origin']),weights

def ordered_join(previous,current):
 B.require(previous['normal']==current['normal']and previous['origin']==current['origin'],'identical authenticated gauge')
 B.require(previous['period']==current['period']==18 and previous['dimension']==current['dimension']==13,'same full native family')
 s=previous['parameter_endpoints'][1];B.require(Fraction(s)==Fraction(current['parameter_endpoints'][0]),'exact adjoining closed boundary')
 B.require(current['settings']=={'precision':256,'newton_steps':2,'halfwidth_exponent':-6,'radius_exponent':-10},'exact wider-domain halfwidth')
 a={'box':[B.decode(v)for v in previous['endpoint_root_boxes'][1]],'dimension':13,'period':18,'exact_parameter':s}
 return anchor_test(a,[B.decode(v)for v in current['center']],[B.decode(v)for v in current['slope']],[B.decode(v)for v in current['curvature']],[B.exact(v)for v in current['normal']],[B.exact(v)for v in current['origin']],B.exact(current['parameter_center']),arb((1,-6)),[B.decode(v)for v in current['coordinate_radii']],13,18)

# Binding-only adapter: keep the original acquisition/resource/finalization loop.
B.BASE=BASE;B.SOURCES=SOURCES;B.settings=settings;B.seed=seed;B.bind_parent=bind_parent
require=B.require;uniform_admission=B.uniform_admission;decode=B.decode
if __name__=='__main__':raise SystemExit(B.main(B.parser().parse_args()))

"""One fresh same-gauge descending native470 chart from an audited endpoint."""
import gzip,hashlib,json,subprocess,time
from fractions import Fraction as F
from pathlib import Path
from flint import arb,ctx
import probe_native470_first_flip as C
from neural_shrimp.rigorous_orbits import exact,ball_json
CARD='docs/plans/native470-advancing-inputs.json';OUT=C.OUT.parents[2]/'signed-flip-advancing001/attempt001/run'
SETTINGS=dict(C.SETTINGS,halfwidth_exponent=-5)
need,read,entry=C.need,C.read,C.entry

def check_geometry(chart):
 need(chart['passed']and chart['shooting_dimension']==470 and chart['period']==18 and chart['sigma']==-1,'full old signed470 chart')
 need(chart['normal']==['0']*468+['1','0']and F(chart['parameter_center'])==0 and chart['parameter_endpoints']==['-1/256','1/256']and chart['settings']['halfwidth_exponent']==-8,'literal old native hidden gauge')
 return '-9/256','-1/256'

def retained(w,card):
 scout=dict(card['authority'],status='reviewed');import scout_native470_flip as S
 old=S.SETTINGS
 try:S.SETTINGS=scout['settings'];S.binding(w,scout,metadata_only=True)
 finally:S.SETTINGS=old
 proof=scout['proof'];ind=scout['independent'];p=Path(proof['root']);a=Path(ind['root']);need(entry(a/'run/independent512.json')==ind['proof'],'literal independently recomputed full470 endpoint')
 oldo=proof['outcome'];total=sum(v['bytes']for v in read(p/'manifest.json')['files'].values());floor=dict(seconds=time.monotonic()-w.start+2*oldo['seconds']+15,rss_kib=2*oldo['peak_rss_kib']+256*1024,raw_bytes=2*total+16*1024**2);need(floor['seconds']<240 and floor['rss_kib']<4*1024**2 and floor['raw_bytes']<768*1024**2,'actual prior complete native cost floor');w.save('actual-cost-floor.json',floor)
 for n in C.L.K.NATIVE:need(entry(C.ROOT/n)==read(p/'run/runtime.json')['sources'][n],'same actual native equations as independently known endpoint')
 w.advance=(p,a,proof,ind);w.save('continuation-binding.json',dict(authority=scout,scope='Exact full independently recomputed left root; literal unchanged gauge allows exact previous scalar only. Fresh full470 root/primitive/anchor gates remain mandatory.'))

def setup(w,known,inp):
 p,a,proof,ind=w.advance;oldinp=read(p/'run/target-inputs.json');need(oldinp['map_inputs']==inp,'identical full target map')
 with gzip.open(p/'run/target-certificate.json.gz','rb')as f:b=f.read(256*1024**2+1)
 need(len(b)<=256*1024**2,'bounded original certificate');chart=json.loads(b);parameter,oldparameter=check_geometry(chart);audit=read(a/'run/independent512.json');need(audit['passed']and all(audit['gates'].values())and audit['inclusions']==470 and audit['anchor_coordinates']==470 and len(audit['recomputed_endpoints'])==2,'complete independent uniform root and original junction membership')
 need(C.WEIGHTS==chart['weights']and chart['normal']==oldinp['normal']and chart['origin']==oldinp['origin'],'literal old gauge/weights');normal=chart['normal'];origin=chart['origin']
 with ctx.workprec(512):box=[C.original_ball(v)for v in audit['recomputed_endpoints'][0]]
 need(len(box)==470,'all470 independently enclosed endpoint coordinates')
 with ctx.workprec(256):
  d=exact(parameter);initial=[C.exact_mid(C.original_ball(c)+C.original_ball(s)*d+C.original_ball(k)*(d*d)/2)for c,s,k in zip(chart['center'],chart['slope'],chart['curvature'],strict=True)]
 anchor=dict(box=box,exact_parameter=oldparameter,period=18,sigma=-1,dimension=13);known=dict(box=audit['recomputed_endpoints'][0],source=ind,dimension=13,period=18,sigma=-1,shooting_dimension=470,exact_parameter=oldparameter)
 w.save('initial-proposal.json',dict(initial=initial,parameter=parameter,old_parameter=oldparameter,normal=normal,origin=origin,halfwidth='1/32',known=known,scope='Old quadratic extrapolation is a finite proposal only; two fresh Newton proposals and full new-domain acquisitions follow.'))
 return known,initial,normal,origin,parameter,anchor

def install():
 C.CARD=CARD;C.OUT=OUT;C.SETTINGS=SETTINGS;C.WEIGHTS=read(C.ROOT/CARD)['settings']['weights'];C.EXTRA_BINDING=retained;C.TARGET_SETUP=setup
 C.FILES=sorted(set(C.FILES+[CARD,'docs/plans/native470-advancing.md','docs/reviews/native470-independent001.json','docs/reviews/native470-weighted-flip002.json','tools/probe_native470_advancing.py','tools/launch_native470_advancing.py','tests/test_native470_advancing.py','tools/scout_native470_flip.py','tests/test_native470_flip_scout.py','docs/plans/native470-flip-scout.md','docs/plans/native470-flip-scout-inputs.json','docs/reviews/native470-first-diagnosis001.json','src/neural_shrimp/unit_pair_scout.py','src/neural_shrimp/signed_flip_scout.py']))
if __name__=='__main__':install();raise SystemExit(C.main())

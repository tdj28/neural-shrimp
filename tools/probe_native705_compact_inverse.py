#!/usr/bin/env python3
"""One fresh doubled parameter/eightfold-error compact705 proof attempt."""
import profile_native705_compact_inverse as P
import probe_native705_correlated_inverse as C
ROOT=C.ROOT
BASE=P.BASE;TARGET=BASE/'attempt001-target/run'
SETTINGS=dict(C.SETTINGS,halfwidth_exponent=-4,radius_exponent=-7,compact_evidence=True)
MARGIN=C.W.BASE.parent/'correlated001/attempt001-margins'
MPIN='c9fa346257944a1c1218805abcd4592869731c1dbe3764c3b5ff2ebd7e54759b'
MWORKER='c8099cb03ff80264a273c5f235a8697b567796643795e7101e2ae7aaabc04c44'
PRIOR=C.W.BASE.parent/'correlated001/attempt001-target'
PPIN='28393e38f8d3fadfa1355f759439d574639cb4255ca20e1d6971085caeb5971c'
PWORKER='da963cfe0011d6d5325a13ad055333511d47b9f9083f2909a02dd552089afbdb'
SOURCES=list(dict.fromkeys(C.SOURCES+P.SOURCES+['tools/probe_native705_compact_inverse.py','tools/launch_native705_compact_inverse.py','tests/test_native705_compact_inverse_driver.py','docs/plans/native705-compact-target.md','docs/reviews/native705-correlated-margins.json','docs/reviews/native705-correlated-target.json']))
original_profile=C.bind_profile

def actual_forecast(measured,prior):
 C.require(prior['passed']and prior['connected']and prior['ticks']==148 and prior['strict_inclusions']==prior['anchor_inclusions']==705,'complete previous actual native proof')
 seconds=max(measured['one_fresh_chart_seconds'],1.25*prior['seconds']+10)
 rss=max(measured['rss_kib'],int(1.25*prior['peak_rss_kib'])+128*1024)
 storage=max(measured['compressed_storage_bytes'],int(1.25*prior['stored_bytes_before_final'])+8*1024**2)
 C.require(seconds<220 and rss<6*1024**2 and storage<1024**3,'fresh measured and prior actual native cost admission')
 return dict(seconds=seconds,rss_kib=rss,raw_bytes=storage,scope='Conservative operational forecast only; changed domains require complete fresh native acquisitions.')

def bind_profile(args,identity):
 binding=original_profile(args,identity)
 for root,outer,worker in [(MARGIN,MPIN,MWORKER),(PRIOR,PPIN,PWORKER)]:C.inventory(root,outer);C.inventory(root/'run',worker)
 md=C.read(MARGIN/'run/outcome.json');old=C.read(PRIOR/'run/outcome.json')
 C.require(md['passed']and md['native_calls']==0 and md['dimension']==705,'completed saved-only width diagnosis')
 C.require(C.read(PRIOR/'run/runtime.json')['commit']=='99b6f71e61b85cc8e201fa0b0f198a15ebc3a5c2','exact first native source')
 C.require(C.read(MARGIN/'run/runtime.json')['commit']=='b3b456a4d22ef2e80a7d5f32dc04d73398ae14a5','exact saved diagnosis source')
 binding.update(prior_native=dict(root=str(PRIOR),outer=PPIN,worker=PWORKER),saved_diagnosis=dict(root=str(MARGIN),outer=MPIN,worker=MWORKER),actual_forecast=actual_forecast(binding['forecast'],old))
 return binding

def configure():
 C.P=P;C.BASE=BASE;C.TARGET=TARGET;C.SETTINGS=SETTINGS;C.SOURCES=SOURCES;C.COMPACT=True;C.bind_profile=bind_profile

def main(args):configure();return C.main(args)
parser=C.parser
if __name__=='__main__':raise SystemExit(main(parser().parse_args()))

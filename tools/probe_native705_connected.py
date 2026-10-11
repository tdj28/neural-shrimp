#!/usr/bin/env python3
"""Source-reviewed331-call kernel with maintained-gauge continuation cards."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from fractions import Fraction as F
import probe_native705_adaptive as W
import localize_cubic705_endpoint_point as E
import refine_native705_rechart_endpoint as REF
from neural_shrimp.rigorous_cubic_correlated_pal import cubic_anchor_test
from neural_shrimp.rigorous_orbits import exact
from flint import arb
ROOT=W.ROOT;BASE=W.P.BASE/'maintained002';CAP=W.CAP;EXPECTED=W.EXPECTED
read=W.read;entry=W.entry;inventory=W.inventory;need=W.require;decode=W.decode
SOURCES=list(dict.fromkeys(W.SOURCES+E.SOURCES+REF.SOURCES+['src/neural_shrimp/cubic_continuation_policy.py','tools/probe_native705_maintained.py','tools/launch_native705_maintained.py','tests/test_cubic_continuation_policy.py','tests/test_native705_maintained.py','docs/plans/native705-maintained-continuation.md','docs/plans/native705-maintained-policy.json','docs/reviews/native705-cubic-endpoint-point001.json','tools/probe_native705_connected.py','tools/launch_native705_connected.py','tools/probe_native705_endpoint_fallback.py','tests/test_native705_connected.py','docs/plans/native705-connected-policy.json','docs/plans/native705-connected-continuation.md','docs/reviews/native705-rechart-refinement002.json']))

def settings(exponent,parameter):
 need(type(exponent)is int and -9<=exponent<=-3,'fixed bounded width range');q=F(parameter);need(q.denominator&(q.denominator-1)==0,'exact dyadic gauge parameter')
 return dict(period=18,precision=256,parameter=str(q),halfwidth_exponent=exponent,radius_exponent=-10,newton_steps=2,predictor_degree=3,residual_degree=4)

def validate_card(card,source,attempt):
 need(card['source']==source and card['attempt']==attempt and type(attempt)is int and 1<=attempt<=24,'literal bounded indexed source')
 need(card['kernel']=='fresh-cubic705-maintained-v1'and card['settings']==settings(card['settings']['halfwidth_exponent'],card['settings']['parameter']),'full native331 unchanged kernel/settings')
 need(card['mode']in ['same_gauge','rotated_gauge']and all(len(card[k])==705 for k in ['normal','origin','weights','initial'])and len(card['anchor']['box'])==705,'entire native continuation geometry')
 need(all(0<F(v)<=1 for v in card['weights'])and all(W.exact_proposal(v)for k in ['normal','origin','initial']for v in card[k]),'positive weights and finite exact proposals')
 anchor=card['anchor']
 if card['mode']=='same_gauge':need(anchor['normal']==card['normal']and anchor['origin']==card['origin']and 'exact_parameter'in anchor,'authenticated literal same gauge')
 else:need('exact_parameter'not in anchor,'rotated gauge must project entire known box')
 need(card['policy']==entry(BASE/'policy.json'),'same sealed policy');return card

def join(anchor,chart):
 a={'box':[decode(v)for v in anchor['box']],'dimension':13,'period':18}
 if 'exact_parameter'in anchor:
  need(anchor['normal']==chart['normal']and anchor['origin']==chart['origin'],'literal known-root gauge matches actual tube');a['exact_parameter']=anchor['exact_parameter']
 return cubic_anchor_test(a,*[[decode(v)for v in chart[k]]for k in ['center','slope','curvature','cubic']],*[[exact(v)for v in chart[k]]for k in ['normal','origin']],exact(chart['parameter_center']),arb((1,chart['settings']['halfwidth_exponent'])),[arb((1,-10))*exact(v)for v in chart['weights']],13,18)

def main(args):
 # Binding-only adapter: the already reviewed scientific331-call worker is
 # unchanged; only literal policy/path/card validator are selected here.
 W.BASE=BASE;W.SOURCES=SOURCES;W.validate_card=validate_card
 return W.main(args)
if __name__=='__main__':raise SystemExit(main(W.parser().parse_args()))

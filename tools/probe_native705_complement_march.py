#!/usr/bin/env python3
"""Bounded exact restriction adapter around the reviewed fresh-local-J worker."""
import argparse,gc,json,subprocess
from fractions import Fraction as F
from pathlib import Path
import bind_native705_complement_march as M
from neural_shrimp.complement_partition import cell
H=M.H;ROOT=H.ROOT;need=H.need;read=H.read;D=H.D;U=H.U
BASE=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/complement/march001')
CARD=M.CARD
FILES=sorted(set(H.FILES+[CARD,'docs/plans/native705-complement-march.md','src/neural_shrimp/complement_partition.py','tests/test_complement_partition.py','tools/bind_native705_complement_march.py','tools/probe_native705_complement_march.py','tools/launch_native705_complement_march.py','tests/test_native705_complement_march.py']))


def assignment(card,job,attempt,source):
    need(type(attempt)is int and 1<=attempt<=48 and job['attempt']==attempt and job['source_commit']==source,'exclusive fixed48 same-source assignment')
    row=M.select(card,job['ordinal']);c=cell(row['settings']['parameter'],'1/128',job['cell']['depth'],job['cell']['index'])
    need(job['cell']==c and job['root']==str(BASE/f'chart{attempt:03d}/run')and job['input_index']==H.O.entry(ROOT/CARD),'literal canonical cell/root/source index')
    need(job['bounds']==dict(worker_seconds=400,worker_rss_kib=4*1024**2,worker_raw_bytes=384*1024**2,native_calls=360),'fixed worker cap')
    return row,c


def bind(card,job,attempt,source,tick):
    row,c=assignment(card,job,attempt,source);need(read(BASE/f'chart{attempt:03d}/run/profile.json')['passed'],'complete per-worker and48-unit measured admission');model,a,normal,origin,oldsc,E,metrics,parent=M.known_chart(card,job['ordinal'],tick)
    restricted=H.restrict_quadratic(a['center'],a['slope'],a['curvature'],a['weights'],a['original_halfwidth'],a['radius'],U.R.exact(F(c['shift'])),U.R.exact(F(c['halfwidth'])),a['center_coefficients'],a['whole_coefficients'])
    need(all(v>0 for v in restricted['fresh_domain'][-3:])and abs(restricted['fresh_domain'][-1]).upper()<=H.arb('1/4'),'positive physical rates/angle and exact terminal series domain')
    rp=restricted.pop('residual_derivative');sc=U.R.exact(F(c['center']));need(sc-oldsc==U.R.exact(F(c['shift'])),'literal original affine parameter translation')
    binding=dict(assignment=job,assignment_artifact=H.O.entry(BASE/f'chart{attempt:03d}/attempt-card.json'),cell=c,parameter_endpoints=c['endpoints'],original=parent,scope='Exact subset of one independently connected full705 root. Original Taylor coefficients and all root corrections inherited, fresh local J/C with zero Newton, all705 tangent and11 complementary modes; no new root or interpolation across gauges.')
    del a;gc.collect();tick();return model,restricted,rp,normal,origin,sc,E,metrics,binding


def main(attempt):
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();card=read(ROOT/CARD);job=read(BASE/f'chart{attempt:03d}/attempt-card.json');_,c=assignment(card,job,attempt,source)
    original={k:getattr(H,k)for k in ['BASE','CARD','FILES','geometry','bind','preserved_profile','save_bound']}
    def geometry(unit):need(unit==attempt,'same assigned cell');return c['center'],c['endpoints']
    def preprofile(actual):need(actual==card,'same literal fixed24 source index');assignment(actual,job,attempt,source)
    def save(out,name,value,used,closing=False):
        if name=='profile.json':
            total_time=12*value['forecast_seconds']+600;total_raw=48*value['forecast_bytes']+64*1024**2
            value=dict(value,passed=bool(value['passed']and total_time<8400 and total_raw<16*1024**3),complete48_forecast_seconds=total_time,complete48_forecast_bytes=total_raw)
        if name=='runtime.json':value=dict(value,assignment=job)
        if name=='outcome.json':value=dict(value,assignment=job,profile_native_calls=min(value['native_calls'],180),target_native_calls=max(0,value['native_calls']-180))
        original['save_bound'](out,name,value,used,closing)
    try:
        H.BASE=BASE;H.CARD=CARD;H.FILES=FILES;H.geometry=geometry;H.preserved_profile=preprofile;H.save_bound=save
        H.bind=lambda actual,unit,tick:bind(actual,job,unit,source,tick)
        return H.main(attempt)
    finally:
        for k,v in original.items():setattr(H,k,v)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,required=True);a=p.parse_args();raise SystemExit(main(a.attempt))

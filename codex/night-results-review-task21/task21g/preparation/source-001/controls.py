"""One manufactured-only control invocation; no target or producer imports."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import time
import traceback


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--core',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(25,25))
    nice=os.getpriority(os.PRIO_PROCESS,0)
    if nice>10:
        raise ValueError('cannot restore exact nice10')
    if nice<10:
        os.nice(10-nice)
    out=Path(args.out);out.mkdir(exist_ok=False)
    begin=time.perf_counter();rows=[]
    try:
        spec=importlib.util.spec_from_file_location('task21g_own',args.core)
        c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);np=c.np
        def require(v):
            if not v: raise AssertionError('manufactured expectation failed')
        def check(name,fn):
            t=time.perf_counter()
            try:
                fn();rows.append({'name':name,'passed':True,'seconds':time.perf_counter()-t})
            except BaseException:
                rows.append({'name':name,'passed':False,'traceback':traceback.format_exc(),'seconds':time.perf_counter()-t})
        def single():
            r=c.prominence_profile([0,1,2],[0,1,0])
            require(len(r['initial']['interior'])==1 and r['initial']['interior'][0]['kind']=='maximum')
            require(len(r['events'])==1 and r['events'][0]['edge_height']==1 and r['events'][0]['remaining_count']==0)
            a=c.prominence_profile([0,1,2],[0,-1,0]);require(a['initial']['interior'][0]['kind']=='minimum')
        check('single extremum and reflected orientation preserve count',single)
        def plateau():
            r=c.extrema([0,1,1,0]);require(r['interior']==[{'left':1,'right':2,'value':1.,'index':1,'kind':'maximum','left_excursion':1.,'right_excursion':1.}])
            require(not c.extrema([0,1,1])['interior'] and not c.extrema([1,1,0])['interior'])
            require(not c.extrema([1,1,1])['interior'])
        check('exact plateau rule and no endpoint turning points',plateau)
        def multi():
            r=c.prominence_profile([0,1,2,3,4],[0,2,0,3,1])
            require(len(r['initial']['interior'])==3 and [e['remaining_count'] for e in r['events']]==[1,0])
            require(r['events'][0]['edge']==[0,1] and r['endpoint_only']==[0,4])
            require(all(a['effective_threshold']<=b['effective_threshold'] for a,b in zip(r['events'],r['events'][1:])))
        check('complete multiextremum hierarchy, left tie and protected endpoints',multi)
        def tiny():
            require(len(c.extrema([0,1e-200,0])['interior'])==1)
            r=c.prominence_profile([0,1,2],[0,2,0],(0,4))
            require(r['events'][0]['range_normalized_threshold']==.5)
        check('no slope-product underflow and raw-pair normalization',tiny)
        def gaps():
            x=np.r_[np.linspace(0,.1,20),np.linspace(.9,1,20)]
            r=c.partition(x,x*x,np.arange(40),16,'width')
            require(len(r['runs'])==2 and len(r['empty_bins'])==12)
            require(sum(row['count'] for row in r['rows'])==40)
        check('width gaps remain separate occupied runs with all pairs retained',gaps)
        def ties():
            x=np.repeat([0.,.5,1.],20);r=c.partition(x,np.arange(60,dtype=float),np.arange(60),16,'count')
            used=[row for row in r['rows'] if row['count']]
            require(len(used)==3 and all(row['count']==20 for row in used))
            require([row['bin'] for row in used]==[2,7,13])
            require(all(row['min_x']==row['max_x'] for row in used))
        check('equal-count midrank bins keep exact ties together',ties)
        def parity():
            ti=np.arange(80);x=np.repeat([0.,1.],40);y=(ti%2).astype(float)
            r=c.partition(x,y,ti,16,'width')
            used=[row for row in r['rows'] if row['count']]
            require(len(used)==2 and all(row['odd_minus_even_median_y']==1 for row in used))
            require(all(row['parity'][0]['x']['quantiles'][3]==row['parity'][1]['x']['quantiles'][3] for row in used))
        check('parity strand diagnostic retains both x and y conditional locations',parity)
        def undefined():
            r=c.analyze_sequence([1.,1.,1.]);require(all(p['status']=='undefined_x_range' for p in r['partitions']))
            require(r['literal_producer_count']['count']==0)
            r=c.analyze_sequence([0.,1.,float('nan'),2.,3.])
            require(r['finite_pairs']==2 and r['nonfinite_pair_indices'].tolist()==[1,2])
            require(r['literal_producer_count']['status']=='undefined_nonfinite_or_empty_input')
        check('constant and nonfinite sequence dispositions without finite fills',undefined)
        def literal():
            x=np.repeat(np.linspace(0,1,40),6);y=-(x-.5)**2
            r=c.literal_producer_count(x,y)
            require(r['count']==1 and all(row['kept'] for row in r['rows']))
            r=c.literal_producer_count(x,x);require(r['count']==0)
        check('literal producer counter expressed independently on manufactured curves',literal)
        t=time.perf_counter()
        result=c.analyze_sequence(np.sin(np.arange(1000)*.13)+.01*np.cos(np.arange(1000)*.31))
        require(result['pairs']==999 and len(result['partitions'])==10)
        require(all(sum(r['count'] for r in p['rows'])==999 for p in result['partitions']))
        encoded=json.dumps(c.plain(result),allow_nan=False,separators=(',',':'))
        profile={'manufactured_sequence_states':1000,'partitions':10,'serialized_bytes':len(encoded.encode()),
                 'seconds_through_serialization':time.perf_counter()-t,'peak_rss_kib_observed':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        passed=all(row['passed'] for row in rows)
        report={'core_sha256':hashlib.sha256(Path(args.core).read_bytes()).hexdigest(),'manufactured_only':True,
                'target_or_producer_imported':False,'controls':rows,'all_passed':passed,'profile':profile,
                'seconds_before_report_write':time.perf_counter()-begin,
                'scope':'CPU1/nice10/1GiB AS/25s CPU, external30s deadline; peak observed before report write.'}
        (out/'report.json').write_text(json.dumps(c.plain(report),indent=2,allow_nan=False)+'\n')
        print(json.dumps({'all_passed':passed,'controls':len(rows),'profile':profile}));return 0 if passed else 1
    except BaseException:
        (out/'failure.json').write_text(json.dumps({'controls':rows,'traceback':traceback.format_exc()},indent=2)+'\n')
        raise


if __name__=='__main__':
    raise SystemExit(main())

"""Manufactured-only controls. Prepared; execute only after coordinator admission."""
import argparse
import importlib.util
import json
import resource
import tempfile
import time
from pathlib import Path

import numpy as np


def load_core():
    spec=importlib.util.spec_from_file_location('own_pm_core',Path(__file__).with_name('core.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def row(i,L,batch,ans='1',gold='1',text=None,kind='pm'):
    return {'index':i,'layout':L,'batch':batch,'row':i%L,'prompt_len':30,
        'answer':ans,'gold':gold,'correct':ans is not None and ans==gold,
        'text':text if text is not None else ('The answer is '+str(ans) if kind=='pm' else '\\boxed{'+str(ans)+'}'),
        'top1':[10,11,12],'switch_step':3,'forced_switch':False}


def load_fixture(c,tmp,name,items,kind='pm',layouts=(8,16)):
    p=tmp/(name+'.jsonl');p.write_text(''.join(json.dumps(r)+'\n' for r in items))
    cfg={'layouts':list(layouts),'out':name,'mode':'soft','arm':'soft','top_n':15,
         'soft_steps':64,'temperature':.6,'model':'manufactured','task':'gsm8k'}
    rs,lay,extra,diag=c.load_rows(p,cfg,kind)
    return {'rows':rs,'layouts':lay,'extra':extra,'cfg':cfg,'diagnostics':diag}


def main(out):
    out.mkdir(exist_ok=False)
    c=load_core();start=time.perf_counter();checks=[]
    def check(name,condition):
        checks.append({'name':name,'passed':bool(condition)})
    check('PM canonical integer and last-cue behavior',c.answer('12 then answer is 2.0 and 7','pm')=='2' and c.answer('x -1,234.','pm')=='-1234')
    check('PM2 nested boxed content and last box',c.answer('\\boxed{8} then \\boxed{\\text{USD} 2.0}','pm2')=='2')
    check('byte marker view stays separate',c.answer('\\boxed{Ġ12}','pm2')=='Ġ12' and c.answer(c.byte_marker_text('\\boxed{Ġ12}'),'pm2')=='12')
    check('missing extraction',c.answer('no numeric answer','pm') is None and c.answer('no numeric answer','pm2') is None)
    check('exact paired binomial',c.exact_mcnemar(0,0)==1 and c.exact_mcnemar(3,0)==.25 and c.exact_mcnemar(1,1)==1)
    check('Wilson empty explicit',c.wilson(0,0)==[None,None])
    with tempfile.TemporaryDirectory(dir=out) as directory:
        tmp=Path(directory)
        items=[row(0,8,0,None,text='none'),row(0,16,0,None,text='none'),row(1,8,0,'2'),row(1,16,0,'1'),row(2,8,1)]
        f=load_fixture(c,tmp,'pairing',items)
        check('missing answers agree and unpaired retained',not f['rows'][0]['flip'] and f['rows'][1]['flip'] and f['diagnostics']['unpaired']==[{'index':2,'present_layouts':[8]}])
        f=load_fixture(c,tmp,'duplicate',items+[row(1,16,0,'2')])
        check('conflicting duplicate retained with literal last write',not f['rows'][1]['flip'] and len(f['diagnostics']['duplicates_last_write_retained'])==1 and not f['diagnostics']['duplicates_last_write_retained'][0]['equal_saved_fields']['answer'])
        a={2:{'cluster':9,'flip':True,'gold':'1'},1:{'cluster':3,'flip':False,'gold':'1'}}
        check('sorted versus encounter cluster ordering',list(c.group_counts(a,'flip'))==[3,9] and list(c.group_counts(a,'flip','encounter'))==[9,3])
        b={1:{'cluster':3,'flip':True,'gold':'2'},3:{'cluster':4,'flip':False,'gold':'1'}}
        value,diag=c.pairwise(a,b,'flip',np.random.default_rng(c.SEED))
        check('paired intersection and contrary labels retained',value['problems']==1 and value['difference']==-1 and diag['left_only']==[2] and diag['right_only']==[3] and diag['gold_mismatch_ids']==[1])
        f=load_fixture(c,tmp,'label',[{**row(0,8,0),'correct':False},row(0,16,0)])
        check('wrong stored correctness is exposed, not overwritten',not f['rows'][0]['correct_a'] and not f['diagnostics']['record_checks'][0]['stored_label_consistent'])
        huge='x'*600
        f=load_fixture(c,tmp,'suffix',[row(0,1,0,'3',text=huge,kind='pm2'),row(0,8,0,'3',text=huge,kind='pm2')],kind='pm2',layouts=(1,8))
        check('suffix incompleteness stays explicit',all(r['may_be_truncated_suffix'] and not r['suffix_extraction_matches'] for r in f['diagnostics']['record_checks']))
        # A complete manufactured PM/PM2 workload, retaining all reports through serialization.
        pm={};names=['top15-s64','replay','greedy','top1-s64','full-s64','top15-s16','full-s16']
        for k,name in enumerate(names):
            rows=[row(i,L,i//16,'2' if L==8 and i%(k+2)==0 else '1') for L in [8,16] for i in range(256)]
            pm[name]=load_fixture(c,tmp,name,rows)
        pm2={}
        for cell,n in [('A-GSM8K',256),('A-MATH',64),('B-GSM8K',256),('B-MATH',64),('C-GSM8K',256),('D-GSM8K',256)]:
            pair={};layouts=(1,8) if n==64 else (8,16)
            for k,mode in enumerate(['soft','plain']):
                rows=[row(i,L,i if n==64 else i//16,'2' if L==layouts[0] and i%(k+3)==0 else '1',kind='pm2') for L in layouts for i in range(n)]
                pair[mode]=load_fixture(c,tmp,cell+mode,rows,'pm2',layouts)
            pm2[cell]={'role':'replication' if cell.startswith('D') else 'primary',**pair}
        p1,d1=c.analyze_pm(pm);p2,d2=c.analyze_pm2(pm2,[('A-GSM8K','D-GSM8K')])
        result={'controls':checks,'PM':p1,'PM2':p2,'PM_pairing':d1,'PM2_pairing':d2,
            'row_diagnostics':{'PM':{n:x['diagnostics'] for n,x in pm.items()},
                               'PM2':{n:{m:x[m]['diagnostics'] for m in ['soft','plain']} for n,x in pm2.items()}},
            'scope':'Manufactured data only, complete source schedule. No original inputs or producer code.'}
        (out/'manufactured.json').write_text(json.dumps(result,allow_nan=False,separators=(',',':'))+'\n')
    elapsed=time.perf_counter()-start
    result={'groups':len(checks),'passed':sum(r['passed'] for r in checks),'checks':checks,
            'seconds_through_manufactured_serialization':elapsed,'rss_KiB_observed_before_status':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'serialized_bytes':(out/'manufactured.json').stat().st_size,'computational_completed':True,
            'scientific_claims_accepted':False,'resource_profile_only':True}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    if not all(x['passed'] for x in checks):raise SystemExit(1)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();main(Path(args.out))

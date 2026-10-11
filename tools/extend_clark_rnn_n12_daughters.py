#!/usr/bin/env python3
"""Fixed native continuation from both exact saved symmetry-broken daughters."""
import argparse,hashlib,importlib.util,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('daughters',ROOT/'tools/explore_clark_rnn_n12_daughters.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
B,S,R,M=D.B,D.S,D.R,D.M
need=D.need
PIN='dffc426ffc015f5bacd75ddfeaa5c4234442ca58a9810798d6faad465066ef3f'
EXECUTION='db96b28b5eb110c98444048bb14ec3c492775a57'
START=.835078108268263
AMPLITUDES=[START+.1*k for k in range(1,33)]
EXTRA=D.EXTRA+['tools/extend_clark_rnn_n12_daughters.py','tools/launch_clark_rnn_n12_daughter_extension.py','tests/test_clark_rnn_n12_daughter_extension.py','docs/plans/clark-rnn-n12-daughter-extension.md']

def crossing(previous,current):
    a,b=previous['maximum_native_radius'],current['maximum_native_radius']
    if not((a<1 and b>=1)or(a>1 and b<=1)):return None
    direction='loss'if a<1 else'gain'
    modes=[v['leading_real_mode']for point in [previous,current]for v in point['spectra']]
    valid=all(v is not None for v in modes)and len({v['sign']for v in modes})==1
    sign=modes[0]['sign']if valid else None
    return dict(direction=direction,kind='actual_native6_minus_bracket'if sign==-1 else'actual_native6_plus_bracket'if sign==1 else'other_native6_stability_crossing',previous=previous,current=current)

def verify_input():
    parent=R.RAW/'n12-p6-daughters-002';need(S.sha(parent/'manifest.json')==PIN,'exact daughter002 outer')
    inv=json.loads((parent/'manifest.json').read_text())['files'];need(set(inv)=={str(p.relative_to(parent))for p in parent.rglob('*')if p.is_file()and p!=parent/'manifest.json'},'complete daughter inventory')
    for name,item in inv.items():
        p=parent/name;need(not Path(name).is_absolute()and '..'not in Path(name).parts and not p.is_symlink()and p.stat().st_size==item['bytes']and S.sha(p)==item['sha256']and not p.stat().st_mode&0o222,'immutable daughter hash')
    launch=json.loads((parent/'launch-end.json').read_text());runtime=json.loads((parent/'run/runtime-start.json').read_text())
    need(launch['source_commit']==EXECUTION and launch['exit_code']==0 and launch['owned_group_empty'],'exact completed parent execution')
    for name,value in runtime['source']['sha256'].items():
        need(hashlib.sha256(subprocess.check_output(['git','show',EXECUTION+':'+name],cwd=ROOT)).hexdigest()==value,'historical parent source')
    result=json.loads((parent/'run/result.json').read_text());need(result['status']=='bounded_daughter_unit_completed'and len(result['branches'])==2,'completed fixed daughter unit')
    j,originals,binding=B.verify_input(R.RAW/'n12-p6-signed-critical-001');seeds=[]
    for tag,(om,oz,oc),branch in zip(B.TAGS,originals,result['branches']):
        need(branch['source_tag']==tag and branch['status']=='fixed_amplitudes_completed'and len(branch['points'])==25 and len(branch['associations'])==24,'fixed complete branch')
        need(all(v['passed']for v in branch['points'])and branch['birth']['passed']and branch['original_association']['passed'],'parent qualification and birth')
        need(all(a['forward']['passed']and a['reverse_native_passed']and all(v['passed']for v in a['reverse'])for a in branch['associations']),'complete parent joins')
        last=branch['points'][-1];need(last['amplitude']==START and last['gain']==om['gain'],'fixed physical endpoint')
        matches=[p for p in(parent/'run').glob('*daughter_point.json')if json.loads(p.read_text())['tag']==last['tag']];need(len(matches)==1,'unique saved endpoint')
        p=matches[0];meta=json.loads(p.read_text());need(all(meta[k]==v for k,v in last.items()),'exact endpoint result binding');a=p.with_name(meta['array_file']);need(S.sha(a)==meta['array_sha256'],'endpoint array pin')
        with np.load(a)as f:z=f['physical'].copy();cycles=[f['cycle_x'].copy(),f['cycle_y'].copy()];q=f['projection'].copy()
        need(np.array_equal(q,oz[13:])and z[-1]==last['delta']and np.array_equal(cycles[0][0],z[:12])and np.array_equal(cycles[1][0],z[12:24]),'exact original projection and native endpoint')
        meta['projection']=q;meta['source_tag']=tag;seeds.append((meta,z,cycles))
    return j,seeds,dict(parent_manifest=PIN,original_critical_manifest=B.PIN,matrix_input_sha256=binding['matrix_input_sha256'],fixed_start_amplitude=START,amplitudes=AMPLITUDES,targets=B.TAGS)

class Work(D.Work):
    def branch(self,meta,parent,original_cycles,j,tag):
        g=meta['gain'];q=meta['projection'];result={'source_tag':tag,'gain':g,'points':[],'associations':[],'status':'incomplete'}
        try:
            z,h,b,a=self.correct_pair(parent,parent,j,g,START,q,tag+'-extension-anchor');row,cycles=self.qualify(z,h,j,g,START,q,tag+'-extension-anchor')
            joins=[B.native_join(old,new,parent[-1],z[-1])for old,new in zip(original_cycles,cycles)];result['saved_endpoint_joins']=joins;need(all(v['passed']for v in joins),'exact saved native daughter endpoints')
            need(row['native_stability']==meta['native_stability']and abs(row['maximum_native_radius']-meta['maximum_native_radius'])<1e-8,'saved stability reproduction')
            result['points'].append(row)
            for i,amp in enumerate(AMPLITUDES,1):
                oldamp=result['points'][-1]['amplitude'];guess=self.predict(z,b,amp-oldamp,tag+f'-extension-step{i}')
                nz,nh,nb,na=self.correct_pair(guess,z,j,g,amp,q,tag+f'-extension-point{i}');new,cy=self.qualify(nz,nh,j,g,amp,q,tag+f'-extension-point{i}')
                forward=D.paired_join(cycles,cy);result['pending']={'point':new,'forward':forward};S.save(self.out/(tag+'-progress.json'),result);need(forward['passed'],'untransformed paired forward phases')
                rg=self.predict(nz,nb,oldamp-amp,tag+f'-extension-reverse{i}');rz,rh,rb,ra=self.correct_pair(rg,nz,j,g,oldamp,q,tag+f'-extension-reverse{i}')
                reverse_native=[self.native(rz[k:k+12],j,g,float(rz[-1]),tag+f'-extension-reverse{i}-{k}')for k in [0,12]]
                reverse=[B.native_join(cycles[k],reverse_native[k]['cycle'],z[-1],rz[-1])for k in range(2)]
                assoc={'index':i,'forward':forward,'reverse':reverse,'reverse_native_passed':all(v['passed']for v in reverse_native)}
                self.emit('daughter_association',dict(assoc,previous=z,current=nz,reverse_state=rz,previous_cycle_x=cycles[0],previous_cycle_y=cycles[1],current_cycle_x=cy[0],current_cycle_y=cy[1]))
                need(assoc['reverse_native_passed']and all(v['passed']for v in reverse),'paired reverse full native join')
                previous=result['points'][-1];result['points'].append(new);result['associations'].append(assoc);result.pop('pending');z,b,cycles=nz,nb,cy
                boundary=crossing(previous,new)
                if boundary is not None:result['candidate_boundary']=boundary;result['status']='stopped_at_native_stability_crossing';break
                S.save(self.out/(tag+'-progress.json'),result)
            else:result['status']='fixed_extension_completed'
        except(np.linalg.LinAlgError,ValueError,FloatingPointError)as e:result.update(status='unresolved_retained',error=type(e).__name__+': '+str(e))
        S.save(self.out/(tag+'-progress.json'),result);return result

def main(out):
    identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p)for p in EXTRA});need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in R.M.THREADS),'CPU1 nice10')
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT,S.RAW_CAP=out,256*1024**2;w,booked=Work(out),[0];signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True))
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists()else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite raw cap')
        booked[0]=new
    S.reserve=reserve;S.save(out/'runtime-start.json',{'source':identity,'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ[k]for k in R.M.THREADS},'started_unix':time.time(),'python':sys.version,'numpy':np.__version__});result={'status':'incomplete','branches':[]}
    try:
        need(int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024>=2*1024**3,'2GiB shared headroom')
        j=np.eye(12)*.2+np.ones((12,12))*.025;x=np.arange(1,13)/20;q=np.eye(12)[0];times={k:[]for k in ['native','pair','eigen','solve']};sizes={k:[]for k in times}
        def measure(kind,f):
            before,t=booked[0],time.monotonic();v=f();times[kind].append(time.monotonic()-t);sizes[kind].append(booked[0]-before);return v
        for k in range(10):
            native=measure('native',lambda:w.it(x,j,2.,.8,'synthetic_profile'));pair=measure('pair',lambda:w.pair(np.r_[x,-x,.8],j,2.,.02,q,'synthetic_profile'))
            def eigen():
                e,v=np.linalg.eig(native[1]);w.emit('profile_eigen',{'matrix':native[1],'eigenvalues':e,'eigenvectors':v})
            def solve():
                matrix,rhs=pair[1],-pair[0];v=np.linalg.solve(matrix,rhs);w.emit('profile_solve',{'matrix':matrix,'rhs':rhs,'solution':v})
            measure('eigen',eigen);measure('solve',solve)
        t,b=({k:max(v)for k,v in a.items()}for a in[times,sizes]);seconds=2*(1280*(t['pair']+t['solve'])+512*(t['native']+4*t['eigen']))+20
        raw=1280*(b['pair']+b['solve']+4096)+512*(2*b['native']+4*b['eigen']+4096)+16*1024**2;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+256*1024**2
        gate=all(np.isfinite(v)and v>0 for a in times.values()for v in a)and seconds<=180 and raw<=256*1024**2 and peak<=768*1024**2;S.save(out/'profile.json',{'times':times,'bytes':sizes,'projected_seconds':seconds,'projected_raw_bytes':raw,'projected_peak_bytes':peak,'gate':bool(gate),'target_unopened':True});need(gate,'measured paired resource admission')
        j,seeds,binding=verify_input();S.save(out/'input-binding.json',binding)
        for meta,z,cycle in seeds:
            row=w.branch(meta,z,cycle,j,meta['source_tag']);result['branches'].append(row);S.save(out/'progress.json',result)
        w.check();result['status']='bounded_daughter_unit_completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=w.native_calls,half_calls=w.augmented_calls,total_calls=w.calls,scope='Numerical Gperiod2 daughters on full native primitive6 orbits only. Native stability crossings are proposals, not qualified boundaries or native12/24 joins. No interval, generalizedflip, shrimp or utility claim.');S.save(out/'result.json',result,final=True);S.manifest(out)
    return 0 if result['status']=='bounded_daughter_unit_completed'else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);sys.exit(main(p.parse_args().out))

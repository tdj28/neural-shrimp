#!/usr/bin/env python3
"""Fixed native daughter-fold and signed-half flip curves at identical gains."""
import argparse,hashlib,importlib.util,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('daughter_fold',ROOT/'tools/qualify_clark_rnn_n12_daughter_fold.py');F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
D,B,S,R,M=F.D,F.B,F.S,F.R,F.M
need=F.need
PIN='be4b98a64abe2a908d4e773112a82d53d90204ae4ee116d1e538278e5dd1e1af'
EXECUTION='69f320c46749cc421ddb3517e2aeed9c592b0388'
STEPS=48
END_GAIN=4.08
EXTRA=F.EXTRA+['tools/continue_clark_rnn_n12_daughter_fold.py','tools/launch_clark_rnn_n12_fold_continuation.py','tests/test_clark_rnn_n12_fold_continuation.py','docs/plans/clark-rnn-n12-fold-continuation.md']

def guard(z,anchor):
    return bool(np.isfinite(z).all()and .6<z[12]<1 and abs(z[12]-anchor[12])<=.02 and max(abs(z[:12]-anchor[:12]))<=.1*max(1.,max(abs(anchor[:12])))and z[13:]@anchor[13:]>0 and np.linalg.norm(z[13:]-anchor[13:])<.5)

def comparison(fold,flip,projection):
    x,y=fold[:12],fold[12:24];scale=max(1.,max(abs(fold[:24])),max(abs(flip[:12])))
    return {'signed_amplitude':float(projection@(x-y)/2),'partner_distance':float(max(abs(x-y))/scale),
            'midpoint_to_signed_root':float(max(abs((x+y)/2-flip[:12]))/scale),
            'x_to_signed_root':float(max(abs(x-flip[:12]))/scale),'y_to_signed_root':float(max(abs(y-flip[:12]))/scale),
            'fold_minus_signed_leak':float(fold[-1]-flip[12]),'scope':'same-gain raw native phase distances; no phase/sign minimization or connection assertion'}

def verify_input():
    j,old,binding=F.verify_input();parent=R.RAW/'n12-p6-daughter-fold-001'
    need(S.sha(parent/'manifest.json')==PIN,'exact native fold inventory')
    inv=json.loads((parent/'manifest.json').read_text())['files']
    need(set(inv)=={str(p.relative_to(parent))for p in parent.rglob('*')if p.is_file()and p!=parent/'manifest.json'},'complete native fold inventory')
    for name,item in inv.items():
        p=parent/name;need(not Path(name).is_absolute()and '..'not in Path(name).parts and not p.is_symlink()and p.stat().st_size==item['bytes']and S.sha(p)==item['sha256']and not p.stat().st_mode&0o222,'immutable native fold bytes')
    launch=json.loads((parent/'launch-end.json').read_text());runtime=json.loads((parent/'run/runtime-start.json').read_text());need(launch['source_commit']==EXECUTION and launch['exit_code']==0 and launch['owned_group_empty'],'exact completed native fold')
    for name,value in runtime['source']['sha256'].items():need(hashlib.sha256(subprocess.check_output(['git','show',EXECUTION+':'+name],cwd=ROOT)).hexdigest()==value,'historical fold source')
    result=json.loads((parent/'run/result.json').read_text());need(result['status']=='bounded_daughter_unit_completed'and len(result['branches'])==1,'single exact fold')
    row=result['branches'][0];need(row['status']=='numerically_qualified_native6_daughter_fold'and all(c['passed']for c in row['coefficients'])and all(a['forward']['passed']and all(v['passed']for v in a['reverse'])for a in row['bracket_joins']),'qualified native fold with original bracket joins')
    def record(kind,tag=None):
        paths=[p for p in(parent/'run').glob('*'+kind+'.json')if tag is None or json.loads(p.read_text())['tag']==tag];need(len(paths)==1,'unique saved '+kind)
        md=json.loads(paths[0].read_text());arr=paths[0].with_name(md['array_file']);need(S.sha(arr)==md['array_sha256'],'saved '+kind+' array')
        with np.load(arr)as a:return md,{k:a[k].copy()for k in a.files}
    md,arrays=record('native_daughter_fold_result');need(all(md[k]==v for k,v in row.items()),'exact result binding')
    _,critical=record('native_fold_criticality',B.TAGS[1]+'-x');z=critical['physical'];need(np.array_equal(z[:13],np.r_[arrays['physical'][:12],arrays['physical'][-1]]),'same fold augmented state')
    jj,seeds,bb=B.verify_input(R.RAW/'n12-p6-signed-critical-001');need(np.array_equal(j,jj),'same exact map for fold and signed flip');signed=seeds[1]
    need(row['gain']==signed[0]['gain']and row['source_tag']==signed[0]['tag'],'same original gain and native daughter source')
    projection=old[0][0]['endpoints'][0]['projection'];need(np.array_equal(projection,signed[1][13:]),'literal original daughter projection')
    return j,z,[arrays['cycle_x'],arrays['cycle_y']],signed,projection,dict(parent_manifest=PIN,execution=EXECUTION,prior=binding,signed_parent=B.PIN,matrix_input_sha256=bb['matrix_input_sha256'],retained_failed_original_global_join=True)

class Work(F.Work):
    def charge(self,units):
        self.check();need(self.augmented_calls+units<=2560,'2560 total derivative half-units');self.augmented_calls+=units;self.calls+=units
    def aug(self,z,j,g,tag):
        self.charge(2);self.native_aug_calls+=1;t=time.monotonic();h,b,a=M.augmented(z,j,g,6,1);self.emit('native_fold_augmented',dict(a,h=h,dh=b,gain=g,tag=tag));return h,b,a,time.monotonic()-t
    def half(self,z,j,g,sign,tag):
        need(sign==-1,'fixed signed flip only');self.charge(1);t=time.monotonic();h,b,a=B.C.critical_fixed_gain(z,j,g,-1,3);self.emit('signed_half',dict(a,h=h,dh=b,sign=-1,tag=tag));return h,b,a,time.monotonic()-t
    def third(self,x,q,j,g,d,tag):
        need(self.third_calls<60,'60 signed third returns');self.charge(1);self.third_calls+=1;t=time.monotonic();c,stages=B.C.third_directional(x,q,j,g,d,3);self.emit('signed_third',dict(tag=tag,theta=x,direction=q,gain=g,delta=d,third=-c,stages=stages));return -c,time.monotonic()-t
    def predict_gain(self,z,b,a,dg,tag):return B.Work.predict(self,z,b,a,dg,tag)
    def correct(self,guess,anchor,j,g,tag,fold):
        z=guess.copy()
        for k in range(12):
            self.emit('critical_attempt',dict(tag=tag,iteration=k,gain=g,fold=fold,physical=z,anchor=anchor));need(guard(z,anchor),'native state/leak/continued eigenvector guard')
            h,b,a,_=(self.aug if fold else lambda z,j,g,tag:self.half(z,j,g,-1,tag))(z,j,g,tag)
            need(np.isfinite(h).all()and np.isfinite(b).all(),'finite critical equations')
            if max(abs(h))<1e-13:return z,h,b,a
            need(k<11,'fixed critical iteration cap');direction=np.linalg.solve(b,-h);self.emit('critical_solve',dict(tag=tag,iteration=k,matrix=b,rhs=-h,direction=direction));z+=direction
        raise AssertionError('unreachable')
    def fold_point(self,z,h,a,j,g,projection,tag):
        # F6 derivative acquisition contains the literal F3 image at cycle[3].
        x,d,q=z[:12],float(z[12]),z[13:];y=-a['cycle'][3];physical=np.r_[x,y,d];amp=float(projection@(x-y)/2)
        ph,pb,pa,_=self.pair(physical,j,g,amp,projection,tag+'-pair');row,cycles=D.Work.qualify(self,physical,ph,j,g,amp,projection,tag)
        critical=[]
        for partner in ['x','y']:
            for suffix in ['', '-adjacent']:
                values=self.contexts[tag+'-'+partner+suffix]['eigenvalues'];idx=int(np.argmin(abs(values-1)));other=np.delete(values,idx)
                item=dict(partner=partner,phase=suffix,critical_error=float(abs(values[idx]-1)),other_plus_distance=float(min(abs(other-1))),complement_radius=float(max(abs(other))))
                critical.append(item);need(item['critical_error']<1e-7 and item['other_plus_distance']>1e-4 and item['complement_radius']<1-1e-5,'four native simple+1/stable-complement contexts')
        qy=pa['first_x'][:,:12]@q;need(np.isfinite(qy).all()and np.linalg.norm(qy)>1e-8,'literal nonzero partner mode');qy/=np.linalg.norm(qy);zy=np.r_[y,d,qy]
        hy,by,ay,_=self.aug(zy,j,g,tag+'-partner');need(max(abs(hy))<1e-11,'literal partner augmented root')
        coefficients=[]
        for partner,zz,aa in [('x',z,a),('y',zy,ay)]:
            report,w,bqq=F.fold_coefficients(aa,zz[13:]);self.emit('native_fold_criticality',dict(report,tag=tag+'-'+partner,physical=zz,left=w,bqq=bqq,parameter_derivative=aa['first'][:,13]));need(report['passed'],'both native generic fold coefficients');coefficients.append(dict(partner=partner,**report))
        row=dict(row,critical_spectra=critical,coefficients=coefficients,augmented_residual=float(max(abs(h))))
        self.emit('fold_curve_point',dict(row,physical=z,paired_physical=physical,partner_mode=qy,cycle_x=cycles[0],cycle_y=cycles[1],projection=projection))
        return row,physical,cycles
    def signed_point(self,z,h,b,a,j,g,tag):return B.Work.qualify(self,z,h,b,a,j,g,tag)
    def reverse_fold(self,z,a,j,g,tag):
        x,y=z[:12],-a['cycle'][3];native=[self.native(v,j,g,float(z[12]),tag+'-'+str(k))for k,v in enumerate([x,y])]
        return native

def run_curves(w,j,z,oldcycles,signed,projection,result):
    g0=signed[0]['gain'];gains=np.linspace(g0,END_GAIN,STEPS+1);need(g0>END_GAIN,'fixed decreasing gains');fz,fh,fb,fa=w.correct(z,z,j,g0,'fold00',True)
    fr,fp,fc=w.fold_point(fz,fh,fa,j,g0,projection,'fold00');sz,sh,sb,sa=w.correct(signed[1],signed[1],j,g0,'signed00',False);sr,sn=w.signed_point(sz,sh,sb,sa,j,g0,'signed00')
    joins=[B.native_join(oldcycles[k],fc[k],z[12],fz[12])for k in range(2)]+[B.native_join(signed[2],sn['cycle'],signed[1][12],sz[12])]
    need(all(v['passed']for v in joins)and np.linalg.norm(fz[13:]-z[13:])<1e-7 and np.linalg.norm(sz[13:]-signed[1][13:])<1e-7,'exact two saved critical seeds/modes')
    result.update(gains=gains.tolist(),start_joins=joins,points=[dict(index=0,gain=g0,fold=fr,signed=sr,comparison=comparison(fp,sz,projection))],associations=[]);S.save(w.out/'progress.json',result)
    for i,g in enumerate(gains[1:],1):
        oldg=gains[i-1];nfz,nfh,nfb,nfa=w.correct(w.predict_gain(fz,fb,fa,g-oldg,f'fold{i}-predict'),fz,j,g,f'fold{i:02d}',True)
        nfr,nfp,nfc=w.fold_point(nfz,nfh,nfa,j,g,projection,f'fold{i:02d}')
        nsz,nsh,nsb,nsa=w.correct(w.predict_gain(sz,sb,sa,g-oldg,f'signed{i}-predict'),sz,j,g,f'signed{i:02d}',False);nsr,nsn=w.signed_point(nsz,nsh,nsb,nsa,j,g,f'signed{i:02d}')
        forward_fold=D.paired_join(fc,nfc);forward_signed=M.phase_match(sn['cycle'],nsn['cycle']);point=dict(index=i,gain=float(g),fold=nfr,signed=nsr,comparison=comparison(nfp,nsz,projection));result['pending']=dict(point=point,forward_fold=forward_fold,forward_signed=forward_signed);S.save(w.out/'progress.json',result)
        need(forward_fold['passed']and forward_signed['passed'],'all unchanged forward native phases')
        rfz,rfh,rfb,rfa=w.correct(w.predict_gain(nfz,nfb,nfa,oldg-g,f'fold{i}-reverse-predict'),nfz,j,oldg,f'fold{i}-reverse',True);rfn=w.reverse_fold(rfz,rfa,j,oldg,f'fold{i}-reverse')
        rsz,rsh,rsb,rsa=w.correct(w.predict_gain(nsz,nsb,nsa,oldg-g,f'signed{i}-reverse-predict'),nsz,j,oldg,f'signed{i}-reverse',False);rsn=w.native(rsz[:12],j,oldg,float(rsz[12]),f'signed{i}-reverse-native')
        reverse=[B.native_join(fc[k],rfn[k]['cycle'],fz[12],rfz[12])for k in range(2)]+[B.native_join(sn['cycle'],rsn['cycle'],sz[12],rsz[12])];qe=[float(np.linalg.norm(rfz[13:]-fz[13:])),float(np.linalg.norm(rsz[13:]-sz[13:]))]
        assoc=dict(index=i,forward_fold=forward_fold,forward_signed=forward_signed,reverse=reverse,reverse_q_errors=qe,all_reverse_native_passed=all(v['passed']for v in rfn+[rsn]))
        w.emit('joint_curve_association',dict(assoc,previous_fold=fz,current_fold=nfz,reverse_fold=rfz,previous_signed=sz,current_signed=nsz,reverse_signed=rsz,previous_cycle_x=fc[0],previous_cycle_y=fc[1],previous_signed_cycle=sn['cycle'],current_cycle_x=nfc[0],current_cycle_y=nfc[1],current_signed_cycle=nsn['cycle'],reverse_cycle_x=rfn[0]['cycle'],reverse_cycle_y=rfn[1]['cycle'],reverse_signed_cycle=rsn['cycle']))
        need(assoc['all_reverse_native_passed']and all(v['passed']for v in reverse)and max(qe)<=1e-7,'three full native cycle/leak and both mode reverse joins')
        result['points'].append(point);result['associations'].append(assoc);result.pop('pending');fz,fb,fa,fc=nfz,nfb,nfa,nfc;sz,sb,sa,sn=nsz,nsb,nsa,nsn;S.save(w.out/'progress.json',result)
    need(len(result['points'])==49 and len(result['associations'])==48,'fixed complete joint unit')

def profile(w,booked,out):
    j=np.eye(12)*.2+np.ones((12,12))*.025;x=np.arange(1,13)/20;q=np.eye(12)[0];z=np.r_[x,.8,q];times={k:[]for k in ['native','derivative','eigen','solve']};sizes={k:[]for k in times}
    def measure(kind,f,charge=1):
        before,t=booked[0],time.monotonic();v=f();times[kind].append((time.monotonic()-t)/charge);sizes[kind].append((booked[0]-before)/charge);return v
    for k in range(10):
        native=measure('native',lambda:w.it(x,j,2.,.8,'synthetic_profile'))
        def eigen():
            e,v=np.linalg.eig(native[1]);w.emit('profile_eigen',dict(matrix=native[1],eigenvalues=e,eigenvectors=v))
        measure('eigen',eigen)
        for call,charge in [(lambda:w.pair(np.r_[x,-x,.8],j,2.,.02,q,'synthetic_profile'),2),(lambda:w.aug(z,j,2.,'synthetic_profile'),2),(lambda:w.half(z,j,2.,-1,'synthetic_profile'),1)]:
            ret=measure('derivative',call,charge)
            def solve():
                b,rhs=ret[1],-ret[0];v=np.linalg.solve(b,rhs);w.emit('profile_solve',dict(matrix=b,rhs=rhs,solution=v))
            measure('solve',solve)
        measure('derivative',lambda:w.third(x,q,j,2.,.8,'synthetic_profile'))
    t,b=({k:max(v)for k,v in a.items()}for a in[times,sizes]);seconds=2*(2560*(t['derivative']+3*t['solve']+4*t['eigen'])+512*(t['native']+4*t['eigen']))+20
    raw=2560*(b['derivative']+3*b['solve']+4*b['eigen']+4096)+512*(2*b['native']+4*b['eigen']+4096)+16*1024**2;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+256*1024**2
    gate=all(np.isfinite(v)and v>0 for a in times.values()for v in a)and seconds<=180 and raw<=256*1024**2 and peak<=768*1024**2
    S.save(out/'profile.json',dict(times=times,bytes=sizes,projected_seconds=seconds,projected_raw_bytes=raw,projected_peak_bytes=peak,gate=bool(gate),target_unopened=True,units='derivative times/bytes per charged half-return; profile calls consume total quotas'));need(gate,'measured joint resource admission')

def main(out):
    identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p)for p in EXTRA});need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in R.M.THREADS),'CPU1 nice10')
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT,S.RAW_CAP=out,256*1024**2;w,booked=Work(out),[0];signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True))
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists()else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite raw cap')
        booked[0]=new
    S.reserve=reserve;S.save(out/'runtime-start.json',dict(source=identity,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),threads={k:os.environ[k]for k in R.M.THREADS},started_unix=time.time(),python=sys.version,numpy=np.__version__));result={'status':'incomplete','points':[],'associations':[]}
    try:
        need(int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024>=2*1024**3,'2GiB current shared headroom');profile(w,booked,out)
        j,z,cycles,signed,projection,binding=verify_input();S.save(out/'input-binding.json',dict(binding,steps=STEPS,end_gain=END_GAIN));run_curves(w,j,z,cycles,signed,projection,result);w.check();result['status']='completed_fixed_joint_native_curves'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=w.native_calls,derivative_half_units=w.augmented_calls,native_augmented_calls=w.native_aug_calls,third_calls=w.third_calls,total_calls=w.calls,scope='Numerical full-native6 daughter-fold and same-gain signed-half flip curves. All native phases and reverse joins retained. Distances toward organizer are diagnostics, not endpoint identity, interval connection or full shrimp proof. Original failed global signed-branch join remains failed.');S.save(out/'result.json',result,final=True);S.manifest(out)
    return 0 if result['status']=='completed_fixed_joint_native_curves'else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);sys.exit(main(p.parse_args().out))

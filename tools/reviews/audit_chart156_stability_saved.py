"""Independent saved reductions for all seven new chart156 attraction slices."""
from audit_refined_tangent_saved import *
import audit_refined_tangent_saved as R
import gc,re
RAW=BASE/'s2-period18-chart-stability-2026-10-10/attempt-001-target'
OUTER='e826d904baa76a9c915091d0e31dce08e118974166816af4f05e22f501d2ca24'
RUN='b073c1181e7603ec26549fbc4a147c4f091740aa972989ec38c1725dc65cbd5e'
EXEC='c891f4d6'
REVIEW=Path('/data2/neural-shrimp-data/audits/s2-period18-refined-tangent/attempt-002')
REVIEW_PIN='bbf6ba536e862b2d3a1b8394473535234800f299d2bd11dbe410f1860e143ec5'
AUDIT_OUT=None

def coverage(rows):
    pairs=sorted((Fraction(a),Fraction(b)) for a,b in rows)
    need(len(pairs)==8 and pairs[0][0]==Fraction(11,32) and pairs[-1][1]==Fraction(45,128),'exact chart endpoints')
    need(all(b-a==Fraction(1,1024) for a,b in pairs) and all(pairs[i][1]==pairs[i+1][0] for i in range(7)),'no gap/overlap')
    return ['11/32','45/128']

def grouped(root,index):
    records={};paths=sorted(root.glob(f'slice-{index:02d}-stage-*.json.gz'))
    need(len(paths)==161,'all161 slice stages')
    for offset,p in enumerate(paths):
        m=re.fullmatch(r'slice-\d+-stage-(\d+)-(.+)\.json\.gz',p.name);need(m is not None and int(m[1])==161*index+offset,'global stage order')
        records.setdefault(m[2],[]).append(read(p))
    return records

def reduce_slice(inp,report,records,ends):
    chart=inp['chart'];base=report['base'];saved=report['tangent_refinement'];p=18;n=13
    need(report['passed'] and report['status']=='complete_refined_slice_bounds' and base['status']=='complete_slice_bounds' and saved['passed'],'complete slice prerequisites')
    def one(name):
        need(len(records[name])==1,'unique '+name);return records[name][0]
    start=one('base_whole_slice_start');whole=records['base_whole_slice_intersection'][-1];center=records['base_center_slice_intersection'][-1];physical=one('base_physical_inputs')
    need(len(records['base_whole_slice_intersection'])==len(records['base_center_slice_intersection'])==3 and whole['iteration']==center['iteration']==2,'fixed six intersections')
    need(report['exact_endpoints']==base['exact_endpoints']==physical['exact_endpoints']==start['exact_endpoints']==ends,'same exact slice')
    need(start['normal']==chart['normal'] and start['origin']==chart['origin'] and start['preconditioner']==chart['preconditioner'] and start['inherited_image']==chart['image_correction'],'same native root/gauge')
    need(whole['retained_error']==base['whole_localization']['final_error'] and center['retained_error']==base['center_localization']['final_error'],'known-root localization identities')
    center_parameter=str((Fraction(ends[0])+Fraction(ends[1]))/2);need(base['center_localization']['exact_endpoints']==[center_parameter]*2 and physical['center_parameter']==center_parameter,'same exact center')
    pre=mat(start['preconditioner']);weights=[exact(v) for v in chart['exact_weights']];old=base['uniform_tangent'];tv=tangent(pre,mat(whole['jacobian']),weights,old);velocity=mat(old['derivative_box'])
    need(clean(physical['actual_uniform_tangent'])==[enc(velocity[i,0]) for i in range(235)],'actual old true derivative')
    need(clean(physical['actual_whole_box'])==[enc(ball(v)) for v in base['whole_localization']['final_state_box']] and clean(physical['actual_center_box'])==[enc(ball(v)) for v in base['center_localization']['final_state_box']],'actual localized boxes')
    rhs=arb_mat([[int(i==234)] for i in range(235)]);u0,q0=implicit(pre,mat(center['jacobian']),rhs,weights,saved['center_derivative'])
    phases=records['base_phase_derivatives'];need([v['phase'] for v in phases]==list(range(18)),'native first phases')
    net=inp['model_inputs']['network'];hidden=(net['d']+int(net['bias']))*net['width'];need(hidden==9 and hidden+net['width']+int(net['bias'])==13,'hidden prefix')
    f,details=forcing([(mat(v['whole_J_mixed_diagnostic']),mat(v['whole_total_Jprime'])) for v in phases],velocity,ball(physical['actual_whole_box'][-1]),hidden)
    need(serial(f)==clean(saved['forcing']),'same independently derived curvature forcing')
    for j,(hh,bp,ff) in enumerate(details):
        row=saved['native_forcing_rows'][j];need(row['phase']==j and serial(hh)==clean(row['hidden_hessian_rows']) and serial(bp)==clean(row['parameter_column_derivative']) and serial(ff)==clean(row['forcing']),'both rate cross terms')
    u2,q2=implicit(pre,mat(whole['jacobian']),-f,weights,saved['second_derivative']);delta=exact(str((Fraction(ends[1])-Fraction(ends[0]))/2));path=u0+u2*arb(0,delta)
    u1=arb_mat([[velocity[i,0].intersection(path[i,0])] for i in range(235)]);need(serial(path)==clean(saved['centered_derivative_enclosure']) and serial(u1)==clean(saved['refined_derivative_box']),'centered derivative bound')
    need(clean(one('tangent_refinement'))==clean(saved),'retained refinement identity')
    actual=one('refined_inputs');decoded=mat(saved['refined_derivative_box'])
    need(actual['exact_endpoints']==ends and actual['halfwidth']=='1/2048' and clean(actual['actual_whole_box'])==[enc(ball(v)) for v in physical['actual_whole_box']] and clean(actual['actual_uniform_tangent'])==[enc(decoded[i,0]) for i in range(235)],'actual refined derivative domain')
    frames=one('base_frames');need(frames['cyclic_last_index']==0 and [v['phase'] for v in frames['witnesses']]==list(range(18)),'same native frame order')
    bases=[mat(v['B']).transpose() for v in frames['witnesses']];fresh=records['refined_phase'];need([v['phase'] for v in fresh]==list(range(18)),'complete new mixed phases')
    rows=[]
    for j,v in enumerate(fresh):
        need(v['center_J']==phases[j]['center_J'],'unchanged native center Jacobian');rows.append((mat(v['center_J']),mat(v['whole_total_Jprime'])))
    final=norm_values(report['norm_cocycle'],records['refined_norm_phase'],rows,bases,delta,ends)
    return {'passed':True,'exact_endpoints':ends,'first_norm_passed':base['norm_cocycle']['passed'],'first_tangent':tv,'center_neumann':q0,'second_neumann':q2,'norm':final}

def norm_values(saved,phase_records,rows,bases,delta,ends):
    need(saved['passed'] and saved['phases']==18,'complete new case');need(ends==[str(Fraction(v)) for v in ends] and ball(saved['halfwidth'])==delta,'fixed final norm domain')
    frames=[b.transpose() for b in bases];inverses=[b.inv() for b in frames];n=13
    for b,inv in zip(frames,inverses):need(inf(eye(n)-mid(inv)*b)<1,'fixed frame invertibility')
    cs=[eye(n),arb_mat(n,n),arb_mat(n,n)];eps=arb(0);gamma=arb(1)
    need([v['phase'] for v in phase_records]==list(range(18)),'all phase proposals')
    for j,(j0,jp) in enumerate(rows):
        a0=mid(j0);d=jp*delta;a1=mid(d);error=j0-a0+(d-a1)*arb(0,1);left,right=frames[(j+1)%18],inverses[j]
        a=[left*x*right for x in [a0,a1,arb_mat(n,n)]];ea=frob(left*error*right);record=phase_records[j];g=ball(record['operator_bound']['gamma'])
        need(g.is_exact() and g>=0,'fixed operator proposal');pd(eye(n)*(g*g)-a[0].transpose()*a[0],record['operator_bound']['positive'])
        cs,eps,gamma=advance(a,ea,g,cs,eps,gamma)
    product_identity=([serial(v) for v in cs]==clean(saved['coefficients']) and enc(eps)==clean(saved['remainder_norm_upper']))
    m=saved['metric'];p=mat(m['P']);need(all(v.is_exact() for row in p.tolist() for v in row) and all(p[i,j]==p[j,i] for i in range(n) for j in range(n)),'exact symmetric fixed P')
    pc=pd(p,m['P_positive']);point=mid(cs[0]);dc=pd(p-point.transpose()*p*point,m['point_D_positive']);need(inf(p)*inf(p.inv())<2**40 and inf(p-point.transpose()*p*point-eye(n))<arb('1/100'),'metric proposal guard')
    ds=[arb_mat(n,n) for _ in range(5)];ds[0]=p+arb_mat(n,n)
    for i in range(3):
        for j in range(3):ds[i+j]-=cs[i].transpose()*p*cs[j]
    norm=gamma+frob(cs[1])+frob(cs[2]);penalty=(inf(p)*(2*norm*eps+eps*eps)).upper();complete=polynomial(ds)-eye(n)*penalty
    decrease_identity=(enc(penalty)==clean(m['remainder_penalty']) and serial(complete)==clean(m['complete_D']))
    return {'passed':True,'public_decode_note':'Fresh Jprime consumed live by producer; reconstructed radii may safely widen. Byte identities are diagnostic only; all independently recomputed strict inequalities remain gates.','product_byte_identity':product_identity,'decrease_byte_identity':decrease_identity,'original_remainder':clean(saved['remainder_norm_upper']),'original_penalty':clean(m['remainder_penalty']),'exact_endpoints':ends,'phase_operator_gates':18,'remainder_norm_upper':enc(eps),'remainder_penalty':enc(penalty),'P':pc,'point_D':dc,'complete_D':pd(complete,m['complete_D_positive'])}

def authenticated_reused_inputs(inventories):
    prior,_,_=authenticate(*R.PINS[2]);inventories.append(prior)
    return read(R.PINS[2][0]/'inputs.json.gz')

def audit():
    commit=subprocess.check_output(['git','rev-parse',EXEC],cwd=ROOT,text=True).strip();inv=[inventory(RAW,OUTER),inventory(REVIEW,REVIEW_PIN)]
    a,rt,out=authenticate(RAW/'run',RUN,commit);inv.append(a);launch=read(RAW/'launcher-result.json')
    need(launch['exit_code']==0 and launch['source_commit']==commit and not launch['owned_live_pids'] and 0<launch['elapsed_seconds']<337,'producer bounded launch')
    need(out['passed'] and out['completed_slices']==7 and out['stages']==1127 and out['complete_coverage']==['11/32','45/128'],'complete producer seven slices')
    need(out['counts']=={'map_attempted':1134,'map_returned':1134,'unit_attempted':2268,'unit_returned':2268,'mixed_attempted':252,'mixed_returned':252},'all original acquisition counts')
    root=RAW/'run';inp=read(root/'inputs.json.gz');expected=[[str(Fraction(k,1024)),str(Fraction(k+1,1024))] for k in range(358,351,-1)];need(inp['slices']==expected,'fixed seven slices')
    chart=inp['chart'];need(chart['passed'] and all(chart['gates'].values()) and chart['statement']['period']==18 and chart['statement']['phase_dimension']==13 and chart['exact_parameter_endpoints']==['11/32','45/128'],'inherited native primitive root chart')
    reused=read(REVIEW/'run/result.json');need(reused['passed'] and reused['case']['exact_endpoints']==['359/1024','45/128'],'completed independent reused eighth')
    oldinp=authenticated_reused_inputs(inv);need(inp['chart']==oldinp['chart'] and inp['model_inputs']==oldinp['model_inputs'],'same independently reviewed reused chart/model')
    need(inp['reused']['manifest']==R.PINS[0][1] and inp['reused']['source_commit']==R.PINS[0][2] and inp['reused']['exact_endpoints']==reused['case']['exact_endpoints'],'exact reused source/result binding')
    results=[]
    for index,ends in enumerate(expected):
        report=read(root/f'slice-{index:02d}-result.json.gz');records=grouped(root,index);result=reduce_slice(inp,report,records,ends);put(AUDIT_OUT/f'case-{index:02d}.json',result);results.append(result);del report,records;gc.collect()
    domain=coverage([v['exact_endpoints'] for v in results]+[reused['case']['exact_endpoints']])
    return {'passed':True,'inventories':inv,'cases':results,'reused_independent_audit':REVIEW_PIN,'covered_exact_chart':domain,'scope':'Independent saved first/second true derivative and complete norm inequalities on seven slices plus independently reviewed eighth. Inherited authenticated smooth primitive-root/localization and analytic derivative enclosures, shared Arb. No new acquisition or whole-branch/neutral-boundary/geometry/utility claim.'}
def main(out):
    global AUDIT_OUT
    AUDIT_OUT=out
    need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean committed review')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();status={'passed':False};code=1
    put(out/'runtime-start.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'checker_sha256':sha(Path(__file__).read_bytes()),'helper_sha256':sha(Path(A.__file__).read_bytes()),'tangent_helper_sha256':sha(Path(R.__file__).read_bytes()),'pid':os.getpid(),'ppid':os.getppid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'internal_seconds':60,'raw_cap_bytes':8*1024**2,'rss_cap_kib':2*1024**2,'post_outcome':True})
    def stop(*_):raise TimeoutError('fixed saved-audit deadline')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,60)
    try:
        need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
        with ctx.workprec(256):status=audit()
        code=0
    except BaseException as e:status.update(error={'type':type(e).__name__,'message':str(e)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);status.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code)
        if status['peak_rss_kib']>=2*1024**2:status.update(passed=False,exit_code=1,error={'type':'MemoryError','message':'audit RSS exceeded'});code=1
        put(out/'result.json',status)
        files={str(p.relative_to(out)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in out.rglob('*') if p.is_file()}
        need(sum(v['bytes'] for v in files.values())<8*1024**2,'raw audit cap');put(out/'manifest.json',{'files':files})
    print(json.dumps(status));return code
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);raise SystemExit(main(p.parse_args().out))

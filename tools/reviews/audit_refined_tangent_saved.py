"""Independent post-outcome true-tangent and norm proof reductions; saved only."""
from audit_norm_cocycle_saved import *
import audit_norm_cocycle_saved as A
RAW=BASE/'s2-period18-refined-tangent-cocycle-2026-10-10/attempt-001-target'
OUTER='57d7602e1eca501e0226b7dbd39956b45a24bf1a10cb73a65f354f6e0de9985a'
PINS=[(RAW/'run','37a1b50124970d35ea11cf6c6815da80a194f6a81d6abf2484ca87fabb88c42a','a42bc33fba39fbf3b1c99440e13c81a804ec46fd'),
(BASE/'s2-period18-tangent-variation-2026-10-10/attempt-001/run','00740edbfdbf33ccc47af79ecc3751d239176c9160e66b5ad888b4a4491a2e77','db95fc152ce4253c598719b05a85480aa2bdd28a'),
(BASE/'s2-period18-doubled-stability-slice-2026-10-10/attempt-001-target/run','ac7c9e89a602e4e0f8d120ece947b2827a95fc854a358a0c3046771d10b3bf77','b837c59318aeebd3de17673fa74ca5c1f2c2d4e3')]
TANGENT_OUTER='2e14ae28f6d139164b4f0f9747cb319e7bb83742305e99653857f536851f942e'

def implicit(pre,jac,rhs,weights,saved):
    n=jac.nrows();need(pre.nrows()==n==len(weights) and all(w.is_exact() and w>0 for w in weights),'implicit dimensions/weights')
    d=eye(n)-pre*jac;q=max((sum((abs(d[i,j]).upper()*weights[j] for j in range(n)),arb(0))/weights[i]).upper() for i in range(n))
    need(q<1 and enc(q)==clean(saved['weighted_neumann_bound']),'strict saved implicit Neumann bound')
    v=pre*rhs;e=d*v;radius=(max((abs(e[i,0]).upper()/weights[i]).upper() for i in range(n))/(1-q)).upper()
    y=arb_mat([[arb(v[i,0],weights[i]*radius)] for i in range(n)])
    need(serial(v)==clean(saved['preconditioned_rhs']) and serial(y)==clean(saved['initial_box']) and enc(radius)==clean(saved['weighted_tail_radius']),'initial implicit enclosure')
    need(len(saved['refinements'])==3,'fixed three refinements')
    for row in saved['refinements']:
        image=v+d*y;need(serial(image)==clean(row['image']),'implicit image')
        y=arb_mat([[y[i,0].intersection(image[i,0])] for i in range(n)])
        need(serial(y)==clean(row['retained']),'implicit retained box')
    need(serial(y)==clean(saved['box']),'final implicit box');return y,enc(q)

def forcing(phases,velocity,rate,hidden):
    n=phases[0][0].nrows();size=len(phases)*n+1;need(rate>0 and velocity.nrows()==size,'actual positive rate/velocity')
    result=arb_mat(size,1);details=[]
    for j,(jac,prime) in enumerate(phases):
        v=arb_mat([[velocity[j*n+i,0]] for i in range(n)])
        hess=arb_mat([[((int(i==k)-jac[i,k])/rate) if i<hidden else arb(0) for k in range(n)] for i in range(n)])
        bp=-hess*v;f=prime*v+bp*velocity[size-1,0]
        for i in range(n):result[j*n+i,0]=f[i,0]
        details.append((hess,bp,f))
    return result,details

def norm_case(root,rows,bases,delta):
    saved=read(root/'result.json.gz');need(saved['passed'] and saved['phases']==18,'complete new case');need(saved['exact_endpoints']==['359/1024','45/128'] and ball(saved['halfwidth'])==delta,'fixed final norm domain')
    frames=[b.transpose() for b in bases];inverses=[b.inv() for b in frames];n=13
    for b,inv in zip(frames,inverses):need(inf(eye(n)-mid(inv)*b)<1,'fixed frame invertibility')
    cs=[eye(n),arb_mat(n,n),arb_mat(n,n)];eps=arb(0);gamma=arb(1)
    phase_records=[read(p) for p in sorted(root.glob('stage-*-norm_phase.json.gz'))]
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
    return {'passed':True,'public_decode_note':'Fresh Jprime consumed live by producer; reconstructed radii may safely widen. Byte identities are diagnostic only; all independently recomputed strict inequalities remain gates.','product_byte_identity':product_identity,'decrease_byte_identity':decrease_identity,'original_remainder':clean(saved['remainder_norm_upper']),'original_penalty':clean(m['remainder_penalty']),'exact_endpoints':saved['exact_endpoints'],'phase_operator_gates':18,'remainder_norm_upper':enc(eps),'remainder_penalty':enc(penalty),'P':pc,'point_D':dc,'complete_D':pd(complete,m['complete_D_positive'])}

def audit():
    inv=[inventory(RAW,OUTER),inventory(PINS[1][0].parent,TANGENT_OUTER)];rt=[];out=[]
    for p,pin,commit in PINS:
        a,b,c=authenticate(p,pin,commit);inv.append(a);rt.append(b);out.append(c)
    root,tan,old=[p[0] for p in PINS];launch=read(RAW/'launcher-result.json')
    need(launch['source_commit']==PINS[0][2] and launch['exit_code']==0 and not launch['owned_live_pids'] and 0<launch['elapsed_seconds']<47,'target bounded launcher')
    need(out[0]['passed'] and out[0]['stages']==75 and out[0]['mixed_counts']=={'attempted':18,'returned':18,'cap':18} and rt[0]['model_C1_calls']==0,'complete fixed mixed acquisition')
    need(out[1]['conditional_bound_passed'] and rt[1]['model_calls']==rt[1]['derivative_calls']==0,'saved tangent premise')
    need(out[2]['result_status']=='complete_slice_bounds' and out[2]['stages']==85 and out[2]['completed_intersections']==6 and out[2]['completed_physical_phases']==18,'complete original doubled acquisition')
    inp=read(old/'inputs.json.gz');base=read(old/'slice-result.json.gz');chart=inp['chart'];saved=read(tan/'tangent-result.json.gz');new=read(root/'inputs.json.gz')
    need(chart['passed'] and all(chart['gates'].values()) and chart['statement']['period']==18 and chart['statement']['phase_dimension']==13,'inherited normalized primitive branch')
    start=read(old/'stage-000-whole_slice_start.json.gz');whole=read(old/'stage-003-whole_slice_intersection.json.gz');center=read(old/'stage-008-center_slice_intersection.json.gz');physical=read(old/'stage-009-physical_inputs.json.gz')
    ends=['359/1024','45/128'];need(inp['exact_endpoints']==base['exact_endpoints']==physical['exact_endpoints']==start['exact_endpoints']==new['exact_endpoints']==ends,'same exact doubled domain')
    need(start['normal']==chart['normal'] and start['origin']==chart['origin'] and start['preconditioner']==chart['preconditioner'] and start['inherited_image']==chart['image_correction'],'same inherited root/gauge')
    need(whole['iteration']==center['iteration']==2 and whole['retained_error']==base['whole_localization']['final_error'] and center['retained_error']==base['center_localization']['final_error'],'third localization identity')
    center_parameter=str((Fraction(ends[0])+Fraction(ends[1]))/2)
    need(base['center_localization']['exact_endpoints']==[center_parameter]*2 and physical['center_parameter']==center_parameter,'same gauge center')
    pre=mat(start['preconditioner']);weights=[exact(v) for v in chart['exact_weights']];old_tangent=base['uniform_tangent'];first=tangent(pre,mat(whole['jacobian']),weights,old_tangent);velocity=mat(old_tangent['derivative_box'])
    need(clean(physical['actual_uniform_tangent'])==[enc(velocity[i,0]) for i in range(235)],'actual old true tangent')
    need(clean(physical['actual_whole_box'])==[enc(ball(v)) for v in base['whole_localization']['final_state_box']],'actual whole root domain')
    need(clean(physical['actual_center_box'])==[enc(ball(v)) for v in base['center_localization']['final_state_box']],'actual center root domain')
    rhs=arb_mat([[int(i==234)] for i in range(235)]);u0,q0=implicit(pre,mat(center['jacobian']),rhs,weights,saved['center_derivative'])
    phases=[read(old/f'stage-{10+j:03d}-phase_derivatives.json.gz') for j in range(18)];need([v['phase'] for v in phases]==list(range(18)),'native old phase order')
    net=inp['model_inputs']['network'];hidden=(net['d']+int(net['bias']))*net['width'];need(hidden==9 and hidden+net['width']+int(net['bias'])==13,'native packing')
    f,details=forcing([(mat(v['whole_J_mixed_diagnostic']),mat(v['whole_total_Jprime'])) for v in phases],velocity,ball(physical['actual_whole_box'][-1]),hidden)
    need(serial(f)==clean(saved['forcing']),'independent full U-doubleprime forcing')
    for j,(hh,bp,ff) in enumerate(details):
        row=saved['native_forcing_rows'][j];need(row['phase']==j and serial(hh)==clean(row['hidden_hessian_rows']) and serial(bp)==clean(row['parameter_column_derivative']) and serial(ff)==clean(row['forcing']),'both hidden-rate cross terms')
    u2,q2=implicit(pre,mat(whole['jacobian']),-f,weights,saved['second_derivative']);path=u0+u2*arb(0,exact('1/2048'));u1=arb_mat([[velocity[i,0].intersection(path[i,0])] for i in range(235)])
    need(serial(path)==clean(saved['centered_derivative_enclosure']) and serial(u1)==clean(saved['refined_derivative_box']),'centered true tangent intersection')
    need(new['model_inputs']==inp['model_inputs'] and new['whole_box']==physical['actual_whole_box'] and new['bindings']=={'original_manifest':PINS[2][1],'tangent_manifest':PINS[1][1]},'same model/domain provenance')
    # Producer explicitly decodes and re-encodes saved tangent before use.
    refined_decoded=mat(saved['refined_derivative_box']);need(clean(new['whole_tangent'])==[enc(refined_decoded[i,0]) for i in range(235)],'actual refined tangent input')
    actual=read(root/'stage-000-actual_inputs.json.gz');need(new['halfwidth']==actual['halfwidth']=='1/2048' and actual['exact_endpoints']==ends,'fixed actual derivative parameter domain');need(clean(actual['actual_whole_box'])==[enc(ball(v)) for v in new['whole_box']] and clean(actual['actual_uniform_tangent'])==[enc(ball(v)) for v in new['whole_tangent']],'new actual decoded derivative domain')
    frames=read(old/'stage-029-frames.json.gz');need(frames['cyclic_last_index']==0 and [v['phase'] for v in frames['witnesses']]==list(range(18)),'same native cyclic frames')
    bases=[mat(v['B']).transpose() for v in frames['witnesses']];need([serial(b) for b in bases]==clean(new['bases']),'unchanged frame proposals')
    rows=[]
    for j in range(18):
        v=read(root/f'stage-{j+1:03d}-refined_phase.json.gz');need(v['phase']==j and v['center_J']==phases[j]['center_J']==new['center_J'][j],'same native center Jacobian')
        rows.append((mat(v['center_J']),mat(v['whole_total_Jprime'])))
    case=norm_case(root,rows,bases,exact('1/2048'))
    return {'passed':True,'inventories':inv,'first_tangent':first,'center_neumann':q0,'second_neumann':q2,'tangent_coordinates':235,'case':case,'scope':'Independent saved U-doubleprime/centered Uprime and full norm metric arithmetic. Inherits authenticated same-gauge smooth primitive-root/localization and analytic mixed derivative enclosures; shared Arb, no new derivative or decomposition. Only this fixed doubled slice, not chart/whole-branch/boundary attraction.'}
def main(out):
    need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean committed review')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();status={'passed':False};code=1
    put(out/'runtime-start.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'checker_sha256':sha(Path(__file__).read_bytes()),'helper_sha256':sha(Path(A.__file__).read_bytes()),'pid':os.getpid(),'ppid':os.getppid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'internal_seconds':60,'raw_cap_bytes':8*1024**2,'rss_cap_kib':2*1024**2,'post_outcome':True})
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

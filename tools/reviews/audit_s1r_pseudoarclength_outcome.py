"""Prospective saved-only pseudo-arclength audit; no numerical target execution."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREADS:os.environ[key]='1'
import importlib.util,json,math,re,subprocess,time,signal,argparse,platform
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('saved_math',Path(__file__).with_name('pseudoarclength_saved_math.py'));M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
G=M.G
require,array,exact,scalar,compare,close_array=M.require,M.array,M.exact,M.scalar,M.compare,M.close_array
norm,dot=M.norm,M.dot
CAPS=dict(analytic=256,map=256,solve=64,svd=102,eig=34)
RAW=Path('/data2/neural-shrimp-data/explore/s1r-pseudoarclength/attempt-001');LAUNCH=RAW.with_name('attempt-001-launcher')
PLAN='docs/plans/s1r-pseudoarclength-inputs.json'

class Replay:
    """Validate causal event arithmetic and witnessed-state advancement separately."""
    def __init__(self,initial,hidden=64):
        self.initial=initial;self.hidden=hidden;self.n=len(initial['pair'][0]);self.output=initial['output'];self.ops={};self.pending=None;self.assembly_starts={}
        self.attempted=dict.fromkeys(CAPS,0);self.returned=self.attempted.copy();self.maps={};self.assemblies={};self.svds={};self.spectra={};self.tangents={};self.states={};self.transport={}
        self.correctors={};self.qualifiers={};self.nodes={};self.matches=[];self.common={};self.edges={};self.witnessed=None;self.accepted=[];self.next_s=None;self.failures=0
        self.anchor=np.r_[initial['pair'][initial['carried']],initial['h']];self.linear={};self.trials={};self.current_edge=None;self.event_count=0;self.need_halving=False;self.terminal=False;self.blocking=None;self.failed_gates=[]
    def operation(self,kind,p):
        ident=p['id'];require(type(ident)is int and ident>=0,'integer operation ID')
        if kind=='operation_start':
            require(self.pending is None and ident==len(self.ops) and p['kind'] in CAPS,'serial single operation')
            k=p['kind'];self.attempted[k]+=1;require(self.attempted[k]<=CAPS[k] and p['attempted']==self.attempted,'attempted operation cap/mirror')
            self.ops[ident]=dict(kind=k,inputs=p['inputs']);self.pending=ident
        else:
            require(self.pending==ident and p['kind']==self.ops[ident]['kind'],'matching operation completion');op=self.ops[ident];self.pending=None
            if kind=='operation_exception':op['exception']=p['error'];return
            require(kind=='operation_return','recognized operation event');k=op['kind'];self.returned[k]+=1;require(p['returned']==self.returned,'returned counters');op['value']=p['value'];v=op['value'];i=op['inputs']
            if k=='analytic':
                array(i['theta'],(self.n,));array(v['gradient'],(self.n,));array(v['hessian'],(self.n,self.n));require(type(v['loss'])in(float,int) and math.isfinite(v['loss']),'finite numeric analytic loss')
            elif k=='map':
                require(i['analytic_id']<ident,'prior analytic operation');a=self.ops[i['analytic_id']];require(a['kind']=='analytic' and 'value'in a and exact(a['inputs']['theta'],i['theta']),'exact cached derivative input')
                rates=np.r_[np.full(self.hidden,10.**float(i['h'])),np.full(self.n-self.hidden,10.**self.output)];require(exact(i['rates'],rates),'actual binary64 rate map')
                phase=dict(theta=i['theta'],image=v['image'],jacobian=v['J'],parameter=v['c'],**a['value']);G.phase_check(i['theta'],rates,a['value'],phase,self.hidden);self.maps[ident]=dict(**phase,h=i['h'],id=ident)
            elif k=='svd':
                saved=dict(matrix=i['matrix'],U=v['U'],singular_values=v['s'],Vh=v['Vh']);self.svds[ident]=saved
                # Residual checks are validated at the corresponding durable gate.
            elif k=='eig':
                M.complex_array(v['values'],(self.n,));M.complex_array(v['left'],(self.n,self.n));M.complex_array(v['right'],(self.n,self.n))
            elif k=='solve':
                c=self.correctors[i['corrector']];iteration=i['iteration'];require(type(iteration)is int and 0<=iteration<8,'fixed Newton iteration cap');require(iteration==len(c['solves']),'sequential Newton iterations')
                require(exact(i['z'],c['current']),'Newton current state');a=self.assemblies[f"corrector{i['corrector']}:iteration{iteration}"]
                gv,j=M.bordered(i['z'],a['R'],a['A'],a['b'],c['plane_origin'],c['plane_tangent'],c['offset']);close_array(i['G'],gv,'bordered residual');close_array(i['J'],j,'bordered derivative');array(v['step'],(self.n+1,));c['solves'].append(ident)
    def gate(self,passed,label,stop=False):
        if not passed:self.failed_gates.append(dict(label=label,event=self.event_count,stop=stop));self.blocking=(label,stop)
    def consume(self,kind,p):
        self.event_count+=1
        if self.blocking is not None:
            require(kind in('corrector_end','qualifier_end','common_end','edge_end')and not p.get('accepted',p.get('converged',p.get('passed',False))),'failed numerical gate only unwinds')
        if kind.startswith('operation_'):return self.operation(kind,p)
        if kind=='cost_gate':
            expected=2*(1.1480005020275712*max(256/12,102/24)+38.103352084057406*(16/473))+5
            compare(p,dict(projected_seconds=expected,limit_seconds=60.,passed=expected<=60.,scope='historical heuristic only'))
        elif kind=='cache_hit':
            op=self.ops[p['id']];require(op['kind']==p['kind'] and 'value'in op and exact(op['inputs']['theta'],p['theta']),'prior exact cache hit')
            if p['kind']=='map':require(np.float64(op['inputs']['h']).tobytes()==np.float64(p['h']).tobytes(),'exact phase-cache h')
        elif kind=='assembly_start':
            require(p['id']==len(self.assembly_starts)and p['id']<256,'assembly serial cap');require(p['purpose']not in self.assembly_starts,'unique assembly start');self.assembly_starts[p['purpose']]=p
        elif kind=='assembly':
            start=self.assembly_starts[p['purpose']];G.exact_tree({k:p[k]for k in start},start,'assembly immutable start');
            first,second=self.maps[p['first']],self.maps[p['second']];require(np.float64(first['h']).tobytes()==np.float64(p['h']).tobytes()==np.float64(second['h']).tobytes(),'same actual hidden rate')
            a=M.assembly(p['theta'],first,second);require(exact(p['map1'],first['image'])and exact(p['map2'],second['image']),'saved actual native maps');compare(p,{k:a[k] for k in ('R','M','A','b')},'native assembly');require(p['purpose']not in self.assemblies,'unique assembly purpose');self.assemblies[p['purpose']]=dict(**p,J0=a['J0'],c0=a['partial_h_F'])
        elif kind=='svd_checks':
            op=self.ops[p['id']];require(op['kind']=='svd' and op['inputs']['purpose']==p['purpose'],'SVD purpose');s=self.svds[p['id']];d=M.svd_residuals(s['matrix'],s)
            V=s['Vh'][:len(s['singular_values'])].T;mat=s['matrix'];U=s['U'];sv=s['singular_values'];sc=max(1.,norm(mat))
            compare(p,dict(right=norm(mat@V-U*sv)/sc,left=norm(mat.T@U-V*sv)/sc,reconstruction=d['reconstruction'],orthogonality=d['orthogonality'],rank=d['rank']),'SVD diagnostics');s['checked']=True;s['passed']=max(p[k]for k in('right','left','reconstruction','orthogonality'))<=1e-10;self.gate(s['passed'],'SVD residual',True)
        elif kind=='tangent':
            sid=p['svd_id'];s=self.svds[sid];require(s.get('checked') and self.ops[sid]['inputs']['purpose']==p['purpose'],'checked tangent decomposition')
            node,phase=p['purpose'].split(':tangent');a=self.assemblies[node+':'+phase];close_array(s['matrix'],np.column_stack((a['A'],a['b'])),'actual native augmented tangent matrix');d=M.tangent_check(s['matrix'],s,p['reference']);compare(p,dict(tangent=d['tangent'],cosine=d['secant_cosine'],residual=d['raw_residual']),'oriented tangent');self.tangents[p['purpose']]=p;self.gate(p['residual']<=1e-10,'null residual',True)
        elif kind=='spectrum_checks':
            eid=p['eig_id'];op=self.ops[eid];require(op['kind']=='eig' and op['inputs']['purpose']==p['purpose'],'spectrum purpose');ns,cs=self.svds[p['norm_svd_id']],self.svds[p['condition_svd_id']];require(ns.get('checked') and cs.get('checked'),'checked norm/condition SVD')
            node,phase=p['purpose'].split(':phase');close_array(op['inputs']['matrix'],self.assemblies[node+':'+phase]['M'],'actual native spectral matrix');d=M.spectrum(op['inputs']['matrix'],op['value'],ns,cs);compare(p,dict(matrix_norm=d['norm'],condition=d['condition_m_minus_i'],right_residual=d['right_residual'],left_residual=d['left_residual'],radius=d['radius'],stability=d['stability']),'native spectrum');self.spectra[p['purpose']]=dict(saved=p,values=op['value']['values'],diagnostic=d);self.gate(max(float(np.max(p['right_residual'])),float(np.max(p['left_residual'])))<=1e-10,'eigen residual',True)
        elif kind=='state_checks':
            pair=array(p['pair'],(2,self.n));aa=[self.assemblies[p['purpose']+':'+str(i)] for i in (0,1)]
            for i,a in enumerate(aa):require(exact(a['theta'],pair[i]) and np.float64(a['h']).tobytes()==np.float64(p['h']).tobytes(),'state native inputs')
            images=np.stack([self.maps[a['first']]['image'] for a in aa]);second=np.stack([self.maps[a['second']]['image'] for a in aa]);d=M.state_checks(pair,images,second);compare(p,d,'physical pair gates');self.states[p['purpose']]=dict(**p);self.gate(p['passed'],'physical state')
        elif kind=='qualifier_start':
            ident=p['id'];require(ident==f'node-{len(self.qualifiers):04d}' and len(self.qualifiers)<17,'qualifier serial cap');require(type(p['carried_phase'])is int and p['carried_phase']in(0,1) and p['accepted']is False,'initial qualifier state');array(p['pair'],(2,self.n));self.qualifiers[ident]=dict(p)
            if len(self.qualifiers)==1:require(p['purpose']=='anchor' and exact(p['pair'],self.initial['pair']) and p['h']==self.initial['h'] and p['carried_phase']==self.initial['carried'],'sole saved anchor')
            else:
                c=self.correctors[max(self.correctors)];require(c.get('end',{}).get('converged') and c['purpose']==p['purpose'] and p['carried_phase']==0 and exact(p['pair'][0],c['end']['state'][:-1]) and p['h']==c['end']['state'][-1],'qualifier from exact converged leg');require(any(exact(v['theta'],p['pair'][0]) and np.float64(v['h']).tobytes()==np.float64(p['h']).tobytes() and exact(v['image'],p['pair'][1])for v in self.maps.values()),'candidate second phase is actual map');q=self.qualifiers[ident];q['corrector_id']=max(self.correctors)
        elif kind=='phase_transport':
            ident=p['node'];q=self.qualifiers[ident];carried=q['carried_phase'];a=self.assemblies[ident+':'+str(carried)];b=self.assemblies[ident+':'+str(1-carried)];t=self.tangents[ident+':tangent'+str(carried)];u=self.tangents[ident+':tangent'+str(1-carried)]
            expected_reference=self.initial['secant'] if q['purpose']=='anchor' else self.correctors[max(self.correctors)]['orientation_reference'];close_array(t['reference'],expected_reference,'tangent carried orientation')
            raw=np.r_[a['J0']@t['tangent'][:-1]+a['c0']*t['tangent'][-1],t['tangent'][-1]];close_array(u['reference'],raw,'other native transported orientation');d=M.transport(t['tangent'],u['tangent'],a['J0'],a['c0'],np.column_stack((b['A'],b['b'])))
            compare(p,dict(carried_phase=carried,primary=t['tangent'],other=u['tangent'],propagated=raw,residual=d['residual'],cosine=d['cosine']),'transport');self.transport[ident]=p;self.gate(p['residual']<=1e-10 and abs(1-p['cosine'])<=1e-7,'native tangent transport',True)
        elif kind=='qualifier_end':
            q=self.qualifiers[p['id']];keys=[k for k in q if k not in('accepted','corrector_id')];G.exact_tree({k:p[k]for k in keys},{k:q[k]for k in keys},'qualifier immutable fields')
            require('end'not in q,'one qualifier end');q['end']=p
            if p['accepted']:
                ident=p['id'];require(ident in self.transport and self.states[ident]['passed'],'complete qualified state');require(self.transport[ident]['residual']<=1e-10 and abs(1-self.transport[ident]['cosine'])<=1e-7,'accepted transport gates');
                for native in(0,1):
                    t=self.tangents[ident+':tangent'+str(native)];sp=self.spectra[ident+':phase'+str(native)]['saved'];require(t['residual']<=1e-10 and self.svds[t['svd_id']]['passed'] and all(self.svds[sp[k]]['passed']for k in('norm_svd_id','condition_svd_id')) and max(float(np.max(sp['right_residual'])),float(np.max(sp['left_residual'])))<=1e-10,'all accepted decomposition/eigen gates');
                compare(p['state_checks'],{k:self.states[ident][k]for k in('closure','transitions','separation','passed')},'node state gate')
                for i in(0,1):compare(p['spectra'][i],{k:v for k,v in self.spectra[ident+':phase'+str(i)]['saved'].items()if k!='purpose'},'node spectrum')
                rows=p['spectrum_assignment_rows'];cols=p['spectrum_assignment_cols'];require(np.asarray(rows).dtype.kind in('i','u')and list(rows)==list(range(self.n)),'ordered complete assignment rows');assignment=cols;error=M.spectral_assignment(self.spectra[ident+':phase0']['values'],self.spectra[ident+':phase1']['values'],assignment);scalar(p['spectrum_error'],error,'spectrum assignment error')
                close_array(p['tangent'],self.transport[ident]['primary'],'node tangent');require(exact(p['z'],np.r_[p['pair'][p['carried_phase']],p['h']]),'native node coordinates');self.nodes[ident]=p
        elif kind=='initial_step':
            anchor=self.nodes[p['anchor']];require(self.witnessed is None and anchor['purpose']=='anchor','one initial step');scalar(p['saved_h_component'],self.initial['saved_h_component']);require(abs(anchor['tangent'][-1]-p['saved_h_component'])<=2e-11 and abs(p['saved_h_component'])>=1e-4,'initial tangent binding');s=min(.005,.000125/abs(p['saved_h_component']));scalar(p['s'],s,'initial fixed arc');self.witnessed=anchor['id'];self.next_s=s
        elif kind=='corrector_start':
            ident=p['id'];require(ident==len(self.correctors) and ident<16 and self.current_edge is not None,'corrector serial cap');array(p['state'],(self.n+1,));require(exact(p['state'],p['predictor']) and p['converged']is False,'initial corrector state');scalar(p['ell'],norm(p['predictor']-p['seed']),'predictor length');require(p['ell']>0,'positive predictor')
            e=self.edges[self.current_edge];parent=self.nodes[e['parent']];leg=len(e['correctors']);require(leg<4 and p['purpose']==('direct','half','half_endpoint','reverse')[leg],'fixed four legs');z,t=parent['z'],parent['tangent'];require(exact(p['plane_origin'],z) and exact(p['plane_tangent'],t),'fixed parent plane')
            if leg<2:seed=z;offset=e['s']/(2 if leg==1 else 1);pred=z+offset*t;orientation=t
            else:
                source=self.nodes[e['qualified'][1 if leg==2 else 0]];seed=source['z'];offset=e['s']if leg==2 else 0.;pred,_=M.predictor(seed,source['tangent'],z,t,offset);orientation=source['tangent']
            require(exact(p['seed'],seed)and exact(p['orientation_reference'],orientation),'leg seed/carried tangent');close_array(p['predictor'],pred,'plane-intersection predictor');scalar(p['offset'],offset,'parent-plane offset');self.correctors[ident]=dict(**p,current=p['state'].copy(),solves=[]);e['correctors'].append(ident)
        elif kind=='linear_checks':
            op=self.ops[p['solve_id']];i,v=op['inputs'],op['value'];require(op['kind']=='solve' and i['corrector']==p['corrector'] and i['iteration']==p['iteration'],'solve gate association');d=M.linear_check(i['G'],i['J'],v['step'],i['z']);relative=float(np.max(np.abs(v['step'][:-1])))/max(1.,float(np.max(np.abs(i['z'][:-1]))))
            compare(p,dict(error=d['error'],forcing=d['true_relative_forcing'],nearzero=d['true_relative_forcing']is None,backward=d['normwise_backward_error_frobenius'],passed=d['passed'],slope=d['slope'],theta_relative_correction=relative,h_correction=float(abs(v['step'][-1]))),'true bordered solve gate');self.linear[(p['corrector'],p['iteration'])]=dict(**p,data=d,op=op);self.gate(p['passed'],'bordered solve')
        elif kind=='armijo_trial':
            key=(p['corrector'],p['iteration']);lin=self.linear[key];require(lin['passed'] and lin['slope']<0,'qualified descent before trial');op=lin['op'];i,v=op['inputs'],op['value'];halves=self.trials.setdefault(key,[]);require(p['halving']==len(halves) and len(halves)<13 and not any(x['accepted']for x in halves),'first accepted fixed halving');state=i['z']+2.**-p['halving']*v['step'];require(exact(p['state'],state),'actual trial input');scalar(p['alpha'],2.**-p['halving']);local=abs(state[-1]-self.anchor[-1])<=.001 and norm(state[:-1]-self.anchor[:-1])<=.02
            if 'G'in p:
                require(local,'local trial');a=self.assemblies[f"corrector{p['corrector']}:trial{p['iteration']}:{p['halving']}"];gv,_=M.bordered(state,a['R'],a['A'],a['b'],self.correctors[p['corrector']]['plane_origin'],self.correctors[p['corrector']]['plane_tangent'],self.correctors[p['corrector']]['offset']);close_array(p['G'],gv,'trial residual');d=M.trial_decision(i['G'],i['J'],v['step'],gv,p['halving']);compare(p,dict(merit=d['trial_merit'],old_merit=d['merit'],slope=d['slope'],accepted=d['accepted']),'Armijo actual slope')
            else:require(p['accepted']is False and ('error'in p),'retained rejected invalid trial')
            halves.append(p)
            if p['accepted']:self.correctors[p['corrector']]['current']=state
        elif kind=='corrector_end':
            c=self.correctors[p['id']];require('end'not in c,'one corrector end');require(exact(p['state'],c['current']),'retained last accepted Newton state');c['end']=p
            if p['converged']:
                lin=self.linear[(p['id'],len(c['solves'])-1)];op=lin['op'];i=op['inputs'];require(exact(p['state'],i['z']) and not any(v['accepted']for v in self.trials.get((p['id'],len(c['solves'])-1),[])),'converged evaluated state without final micro-step');require(lin['passed'] and lin['theta_relative_correction']<=1e-10 and lin['h_correction']<=1e-10 and abs(i['G'][-1])<=1e-10 and M.normalized(self.assemblies[f"corrector{p['id']}:iteration{len(c['solves'])-1}"]['map2'],i['z'][:-1])<=1,'corrector convergence gates');require(M.locality(p['state'],c['predictor'],c['seed'],self.anchor)['passed'],'converged predictor locality');require(p['iterations']==len(c['solves']),'corrector iteration mirror')
        elif kind=='pair_match':
            parent,child=self.nodes[p['parent']],self.nodes[p['child']];d=M.phase_match(parent['pair'],child['pair'],p['replay']);compare(p,d,'phase matching');require(not d['passed'] or(parent['carried_phase']^d['phase'])==child['carried_phase'],'native phase carry');self.matches.append(p);self.gate(p['passed'],'phase match',p['cost_gap']<.5*p['separation'])
            e=self.edges[self.current_edge]
            if p['passed'] and not p['replay'] and child['id']not in e['qualified']:e['qualified'].append(child['id'])
        elif kind=='common_start':
            ident=p['id'];require(ident==len(self.common) and ident<8,'common-control cap');ref,cand=self.nodes[p['reference']],self.nodes[p['candidate']];scalar(p['canonical_h'],ref['h']);scalar(p['candidate_original_h'],cand['h']);scalar(p['h_difference'],abs(ref['h']-cand['h']));self.common[ident]=dict(p)
        elif kind=='common_end':
            start=self.common[p['id']];require('end'not in start,'one common end');G.exact_tree({k:p[k]for k in start if k!='passed'},{k:v for k,v in start.items()if k!='passed'},'common immutable endpoints');start['end']=p
            if p['passed']:
                require(start['h_difference']<=1e-10,'preliminary hidden agreement');state=self.states['common'+str(p['id'])];ref,cand=self.nodes[p['reference']],self.nodes[p['candidate']];require(exact(state['pair'],cand['pair']) and np.float64(state['h']).tobytes()==np.float64(ref['h']).tobytes() and state['passed'],'byte-identical common control actual reevaluation');scalar(p['canonical_h'],ref['h']);compare(p['state_checks'],{k:state[k]for k in('closure','transitions','separation','passed')});compare(p['match'],M.phase_match(ref['pair'],cand['pair'],True));require(p['match']['passed'],'common replay match')
        elif kind=='edge_start':
            ident=p['id'];require(not self.terminal and not self.need_halving and len(self.accepted)<2,'no edge after terminal/without required half');require(ident==len(self.edges) and ident<4 and self.current_edge is None and p['accepted']is False and p['nodes']==[],'edge serial cap');require(p['parent']==self.witnessed,'rollback to last witnessed');scalar(p['s'],self.next_s,'fixed/halved arc');self.edges[ident]=dict(**p,correctors=[],qualified=[]);self.current_edge=ident
        elif kind=='edge_end':
            e=self.edges[p['id']];require(self.current_edge==p['id']and'end'not in e,'one current edge end');e['end']=p;self.current_edge=None
            if self.blocking is not None and not self.blocking[1]:self.blocking=None
            # Every claimed node must be a qualified result of one of this edge's legs.
            candidates=[q['end']['id']for q in self.qualifiers.values() if 'end'in q and q['end']['accepted'] and q.get('corrector_id')in e['correctors']]
            require(p['nodes']==candidates and len(p['nodes'])<=4,'edge qualified nodes')
            if p['accepted']:
                require(len(e['correctors'])==len(p['nodes'])==4 and p['direct']==p['nodes'][0] and len(self.accepted)<2,'four-leg witnessed acceptance');direct,half,end,reverse=(self.nodes[k]for k in p['nodes']);parent=self.nodes[e['parent']]
                for a,b,replay in((parent,direct,False),(parent,half,False),(half,end,False),(direct,end,True),(parent,reverse,True)):
                    require(any(x['parent']==a['id']and x['child']==b['id']and x['replay']==replay and x['passed']for x in self.matches),'all edge phase links')
                for a,b in((direct,end),(parent,reverse)):require(any(x.get('end',{}).get('passed')and x['reference']==a['id']and x['candidate']==b['id']for x in self.common.values()),'both common-control witnesses')
                self.accepted.append(p['id']);self.witnessed=p['direct'];self.failures=0;self.terminal=len(self.accepted)==2
            else:self.failures+=1;require(self.failures<=2,'at most two failed attempts per edge');self.need_halving=self.failures==1;self.terminal=self.failures==2
        elif kind=='halve':
            require(self.current_edge is None and self.need_halving and not self.terminal and self.failures==1 and p['parent']==self.witnessed,'single retry from witnessed parent');self.need_halving=False;scalar(p['new_s'],self.next_s/2,'fixed halving');self.next_s/=2
        elif kind in('bound_inputs','terminal'):pass
        else:raise ValueError('unknown event '+kind)
    def summary(self):
        qualified=list(self.nodes);witnessed=[self.initial.get('anchor_id','node-0000')]+[self.edges[i]['end']['direct']for i in self.accepted]
        return dict(events=self.event_count,failed_gates=self.failed_gates,attempted=self.attempted,returned=self.returned,pending_operation=self.pending,accepted_edges=self.accepted,last_witnessed=self.witnessed,qualified_nodes=qualified,qualified_unwitnessed_nodes=[i for i in qualified if i not in witnessed],correctors=len(self.correctors),qualifiers=len(self.qualifiers),common_checks=len(self.common),attempted_edges=len(self.edges))

PRODUCER_FILES=('src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','src/neural_shrimp/period2_branch.py','src/neural_shrimp/period2_branch_io.py','src/neural_shrimp/s1r_gap_diagnostic.py','src/neural_shrimp/s1r_pseudoarclength.py','tools/audit_s1r_branch_outcome.py','tools/explore_s1r_pseudoarclength.py','tools/launch_s1r_pseudoarclength.py',PLAN,'docs/plans/s1r-pseudoarclength-next.md','docs/plans/s1r-pseudoarclength-implementation.md')
AUDIT_FILES=('tools/reviews/audit_s1r_pseudoarclength_outcome.py','tools/reviews/pseudoarclength_saved_math.py','tools/reviews/audit_s1r_lower_gap_outcome.py','tools/reviews/launch_s1r_pseudoarclength_audit.py','tests/reviews/test_s1r_pseudoarclength_audit.py','docs/reviews/s1r-pseudoarclength-audit-card.md','docs/reviews/s1r-pseudoarclength-producer-source.json','tests/reviews/fixtures/pseudoarclength-synthetic-two-edges.json.gz')
FROZEN='docs/reviews/s1r-pseudoarclength-producer-source.json'

def source_check(sources,execution):
    freeze=G.read(ROOT/FROZEN);require(set(sources)==set(freeze['files']),'exact full producer source inventory')
    for name,identity in sources.items():
        require(identity==freeze['files'][name],'frozen reviewed source '+name)
        for commit in(execution,freeze['commit']):
            data=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT);require(identity==dict(bytes=len(data),sha256=G.digest(data)),'execution/source unchanged bytes '+name)

def original_inputs(plan):
    old=plan['historical_collection'];folder=Path(old['collection_root']);files=G.inventory(folder,old['collection_manifest']['sha256']);require(files['manifest.json']==old['collection_manifest'],'historical manifest identity')
    for name,pin in old['collection_metadata'].items():require(files[name]==pin,'historical metadata')
    runtime=G.read(folder/'runtime-start.json');require(runtime['source_commit']==old['producer_execution_commit'],'historical execution')
    for name,pin in runtime['sources'].items():
        for commit in(old['producer_execution_commit'],old['producer_reviewed_source']):
            data=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT);require(pin==dict(bytes=len(data),sha256=G.digest(data)),'historical source bytes')
    values={};encoded={}
    for name,pin in plan['selected_records'].items():
        require(files[name]==pin['json'] and files[pin['arrays_file']]==pin['arrays'],'fixed selected record');record,payload=G.decode_record(folder,name);values[name]=payload;encoded[name]=record['payload']
    binding=values['00000-bound_inputs.json'];pre=values['04820-node.json'];last=values['04918-node.json'];edge=values['05002-edge.json'];failure=values['05267-edge.json']
    require(binding['model']==dict(activation='tanh',bias=False,d=8,in_scale=.5,loss_scale=1.,out_scale=.125,ridge=.001,width=8),'exact train model');array(binding['x'],(72,8));array(binding['y'],(72,))
    for spec in plan['native_input_slots']:
        raw=encoded[spec['record']];require(all(raw[k]==spec[k]for k in('pair','params','rates')),'fixed native array descriptors')
    require(pre['accepted']is True and last['accepted']is True and edge['status']=='accepted' and edge['parent']==pre['id'] and edge['direct']==last['id'] and failure['parent']==last['id'],'saved qualified lineage')
    # Historical HiddenBridge aligned the parent pair before matching: match.phase
    # is the child's native carried bit; XOR gives native-pair exchange only.
    pp,carried=edge['parent_phase'],edge['parent_match']['phase'];require(type(pp)is int and type(carried)is int and pp in(0,1)and carried in(0,1)and failure['parent_phase']==carried,'historical native carry')
    require(last['params'][0]==.992 and last['params'][1]==.6271428571428572,'fixed anchor rates');require(exact(last['rates'],np.array([10.**float(x)for x in last['params']])),'historical actual rates')
    secant=np.r_[last['pair'][carried]-pre['pair'][pp],last['params'][0]-pre['params'][0]]
    history={str(folder):dict(manifest=files['manifest.json'],files=len(files),bytes=sum(x['bytes']for x in files.values()))}
    for d in plan['gap_diagnostic']['raw'].values():
        fs=G.inventory(Path(d['root']),d['manifest']['sha256']);require(fs['manifest.json']==d['manifest'],'gap manifest identity');history[d['root']]=dict(manifest=d['manifest'],files=len(fs),bytes=sum(x['bytes']for x in fs.values()))
    a=plan['gap_audit'];fs=G.inventory(Path(a['root']),a['manifest_sha256']);history[a['root']]=dict(manifest=fs['manifest.json'],files=len(fs),bytes=sum(x['bytes']for x in fs.values()))
    for item in plan['initial_raw_tangent_summaries'].values():
        path=Path(plan['gap_diagnostic']['raw']['attempt-001']['root'])/item['source_file'];require(G.entry(path)==item['source_identity'],'historical tangent summary identity');raw=G.read(path)['payload'];expected=item['summary'];require(raw['rank']==expected['rank']and {k:raw['null_tangent'][k]for k in expected['null_tangent']}==expected['null_tangent'],'historical tangent summary mirror')
    saved=plan['initial_raw_tangent_summaries']['native_phase_'+str(carried)]['summary']['null_tangent']['h_component']
    initial=dict(pair=last['pair'],h=float(last['params'][0]),output=float(last['params'][1]),carried=carried,secant=secant,saved_h_component=saved)
    bound=dict(model=binding['model'],identity=binding['identity'],x=binding['x'],y=binding['y'],pair=last['pair'],params=last['params'],rates=last['rates'],carried_phase=carried,secant=secant,saved_h_component=saved)
    return initial,bound,history

def records(folder,files,result,initial,bound,hidden=64):
    journal=[]if'journal.jsonl'not in files else[json.loads(x)for x in(folder/'journal.jsonl').read_text().splitlines()];require(len(journal)<=10000,'journal record ceiling')
    replay=Replay(initial,hidden);used=set();errors=[];last_arithmetic_error=None
    payloads=[]
    for index,row in enumerate(journal):
        kind=row['kind'];name=f'{index:05d}-{kind}.json';require(re.fullmatch('[A-Za-z0-9_]+',kind)and row==dict(sequence=index,kind=kind,file=name),'causal journal names');record,p=G.decode_record(folder,name);require(record['kind']==kind and record['sequence']==index,'record metadata mirror');used.add(name)
        if'arrays_file'in record:used.add(record['arrays_file'])
        payloads.append((kind,p))
    # The target stream must bind historical inputs and pass its predeclared
    # cost screen before any numerical/scientific event. A pre-engine write
    # failure may retain no header or only the bound-input header, never science.
    if payloads:
        require(payloads[0][0]=='bound_inputs','mandatory first bound-input record')
        require(sum(k=='bound_inputs'for k,_ in payloads)==1,'single bound-input header')
    if len(payloads)>1:
        require(payloads[1][0]=='cost_gate'and sum(k=='cost_gate'for k,_ in payloads)==1,'single pre-science cost gate')
    if result['status']in('two_edges_complete','unresolved_gap'):
        require(len(payloads)>2,'complete target requires both headers and science')
    # Numerical invalid returns can be durably retained at a failed terminal
    # operation. They never qualify any dependent state. No earlier corruption
    # may be excused by a later failure.
    complete=result['status']in('two_edges_complete','unresolved_gap')
    for index,(kind,p)in enumerate(payloads):
        if kind=='bound_inputs':require(index==0,'bound input first');G.exact_tree(p,bound,'historical input binding')
        try:replay.consume(kind,p)
        except (ValueError,ArithmeticError,KeyError)as error:
            suffix=payloads[index+1:]
            allowed_terminal=(not complete and kind=='operation_return' and all(k in('corrector_end','qualifier_end','common_end','edge_end')and not v.get('accepted',v.get('converged',v.get('passed',False)))for k,v in suffix))
            if not allowed_terminal:raise
            last_arithmetic_error=dict(index=index,kind=kind,error=type(error).__name__+': '+str(error));errors.append(last_arithmetic_error);break
    extras=set(files)-used-{'runtime-start.json','input-plan.json','journal.jsonl','checkpoint.json','result.json','manifest.json'}
    require(all(re.fullmatch(rf'{len(journal):05d}-[A-Za-z0-9_]+\.(json|npz)(\.tmp)?',s)or s=='checkpoint.json.tmp'for s in extras),'only next interrupted record extras')
    if complete:require(not extras and replay.pending is None and not errors,'complete durable event inventory')
    claimed_a=result['operation_attempted'];claimed_r=result['operation_returned']
    if claimed_a==claimed_r==result['stage_counts']=={}:
        require(not complete and not replay.ops and not replay.qualifiers,'engine not initialized');return dict(**replay.summary(),arithmetic_complete=False,uninitialized_engine=True,interrupted_record_files=sorted(extras))
    da={k:claimed_a.get(k,0)-replay.attempted[k]for k in CAPS};dr={k:claimed_r.get(k,0)-replay.returned[k]for k in CAPS}
    require(set(claimed_a)==set(claimed_r)==set(CAPS)and all(type(claimed_a[k])is int and type(claimed_r[k])is int and 0<=claimed_r[k]<=claimed_a[k]<=CAPS[k]for k in CAPS),'claimed operation counts/caps')
    require(all(v==0 for v in da.values())or(not complete and replay.pending is None and sum(da.values())==1 and all(v in(0,1)for v in da.values())),'single unjournaled attempted operation boundary')
    pending_kind=None if replay.pending is None else replay.ops[replay.pending]['kind'];require(all(v==0 for v in dr.values())or(not complete and pending_kind is not None and dr=={k:int(k==pending_kind)for k in CAPS}),'single returned-before-emit boundary')
    expected_stages=dict(correctors=len(replay.correctors),qualifiers=len(replay.qualifiers),common=len(replay.common),edges=len(replay.edges),assemblies=len(replay.assembly_starts));sc=result['stage_counts'];require(set(sc)==set(expected_stages),'stage count keys');diff={k:sc[k]-expected_stages[k]for k in sc};require(all(v==0 for v in diff.values())or(not complete and sum(diff.values())==1 and all(v in(0,1)for v in diff.values())),'stage increment-before-start boundary')
    # edge_end is durable before in-memory advancement; a terminal storage error
    # may retain exactly one witnessed edge ahead of result, never an unwitnessed node.
    accepted=replay.accepted;witnessed=replay.witnessed;lag=False
    if result['accepted_edges']!=accepted or result['last_witnessed']!=witnessed:
        require(not complete and bool(journal)and journal[-1]['kind']=='edge_end'and bool(accepted)and replay.edges[accepted[-1]]['end']['accepted'],'terminal edge advancement boundary');previous=accepted[:-1];parent=replay.edges[accepted[-1]]['parent'];require(result['accepted_edges']==previous and result['last_witnessed']==parent,'exact prior witnessed result');lag=True
    if result['status']=='two_edges_complete':require(len(accepted)==2 and replay.current_edge is None,'two witnessed edges')
    if result['status']=='unresolved_gap':require(replay.failures==2 and len(accepted)<2 and replay.current_edge is None,'second failed edge terminates')
    checkpoint_lag=False
    if'checkpoint.json'in files:
        cp=G.read(folder/'checkpoint.json');last=None if not journal else journal[-1]
        if cp!=dict(status=result['status'],latest=last):
            require(not complete and bool(journal)and cp==dict(status=result['status'],latest=journal[-2]if len(journal)>1 else None),'one terminal checkpoint lag');checkpoint_lag=True
    return dict(**replay.summary(),claimed_attempted=claimed_a,claimed_returned=claimed_r,unvalidated_terminal_return=errors,interrupted_record_files=sorted(extras),terminal_edge_advancement_lag=lag,terminal_checkpoint_lag=checkpoint_lag,shared_primitive='Saved analytic gradients/Hessians reused; no independent derivative regeneration.',arithmetic_complete=not errors and not extras and replay.pending is None and all(v==0 for v in da.values()) and all(v==0 for v in dr.values()) and all(v==0 for v in diff.values()) and not lag and not checkpoint_lag)

def producer_provenance(pins,emit):
    require(pins['source_review_commit']=='9df5e138fbba1b283c18a8d47f49ebe9a69b15ae','fixed producer source review');lf=G.inventory(LAUNCH,pins['launcher_manifest_sha256']);ls=G.read(LAUNCH/'runtime-start.json');lr=G.read(LAUNCH/'receipt.json');require(set(lf)=={'launcher-source.py','runtime-start.json','stdout.log','receipt.json','manifest.json'},'exact launcher inventory')
    for v in(ls,lr):require(v['source_commit']==pins['execution_commit']and v['review_commit']==pins['source_review_commit'],'execution/review mirrors')
    source_check(ls['sources'],pins['execution_commit']);require(lf['launcher-source.py']==ls['sources']['tools/launch_s1r_pseudoarclength.py'],'archived launcher bytes');require(ls['out']==str(RAW)and ls['launcher']==str(LAUNCH),'fixed producer paths')
    require(ls['command']==['timeout','--signal=TERM','--kill-after=5s','75s','nice','-n','10','/data2/neural-shrimp/.venv/bin/python','tools/explore_s1r_pseudoarclength.py','--out',str(RAW),'--review-commit',pins['source_review_commit']],'fixed producer argv')
    require(ls['nice']>=10 and all(ls['thread_env'][k]=='1'for k in THREADS)and ls['gpu_used']is False,'resource ownership settings')
    for k,v in dict(external_seconds=75,kill_grace_seconds=5,supervisor_seconds=80.,rss_cap_bytes=2*1024**3,store_bytes=127*1024**2,launcher_bytes=1024**2,log_bytes=128*1024).items():require(ls[k]==v,'configured cap '+k)
    require(ls['group_pid']==lr['group_pid']and(ls['group_pid']is None or type(ls['group_pid'])is int and ls['group_pid']>0),'owned group mirror');require(not lr['owned_live_pids_remaining'],'owned producer exit')
    require(type(lr['elapsed_seconds'])in(float,int)and math.isfinite(lr['elapsed_seconds'])and 0<=lr['elapsed_seconds']<=81.1,'actual launcher duration');require(type(lr['max_group_rss_bytes'])is int and lr['max_group_rss_bytes']>=0,'actual monitored RSS')
    rf=G.directory(RAW)if RAW.exists()else{};require(lr['raw_inventory']==rf and lr['raw_file_count']==len(rf),'complete raw launcher mirror');require(sum(v['bytes']for v in rf.values())==lr['raw_bytes']<=127*1024**2 and sum(v['bytes']for v in lf.values())<=1024**2 and lf['stdout.log']['bytes']<=128*1024 and lr['within_store_cap']is True,'actual byte caps')
    pin=pins['raw_manifest_sha256']
    if pin is None:require('manifest.json'not in rf,'null pin only absent manifest')
    else:require(rf['manifest.json']['sha256']==pin,'external raw manifest pin')
    if lr['raw_manifest_verified']is not True:
        require(lr['successful']is False,'incomplete cannot succeed')
        if pin is not None:
            try:valid=G.read(RAW/'manifest.json')['files']=={k:v for k,v in rf.items()if k!='manifest.json'}
            except(ValueError,KeyError,TypeError):valid=False
            require(not valid,'reported manifest failure is real')
        emit('provenance.json',dict(pins=pins,raw_inventory=rf,launcher_inventory=lf,launcher=lr));return rf,None,dict(verdict='incomplete_inventory_only',arithmetic_qualified=False)
    require(G.inventory(RAW,pin)==rf,'full scientific inventory');rt=G.read(RAW/'runtime-start.json');result=G.read(RAW/'result.json');source_check(rt['sources'],pins['execution_commit']);require(rt['sources']==ls['sources'],'worker/launcher source identity')
    for k in('schema','source_commit','review_commit','sources','limits','tolerances','fold_verified','lower_association'):require(result[k]==rt[k],'runtime/result '+k)
    require(rt['source_commit']==pins['execution_commit']and rt['review_commit']==pins['source_review_commit']and rt['schema']=='s1r-pseudoarclength-v1','execution source/schema');require(rt['status']=='running'and rt['accepted_edges']==[]and rt['last_witnessed']is None,'initial runtime state')
    require(rt['numpy']=='2.5.3'and rt['scipy']=='1.18.1'and rt['nice']>=10 and all(rt['thread_env'][k]=='1'for k in THREADS)and rt['gpu']is False and rt['paid']is False,'worker libraries/resources')
    freeze=G.read(ROOT/FROZEN);require(rt['limits']==freeze['limits']and rt['tolerances']==freeze['tolerances'],'frozen numerical settings');require(result['fold_verified']is False and result['lower_association']is False,'no fold/ancestry claim');require(result['status']in('two_edges_complete','unresolved_gap','stopped','failed_or_interrupted'),'known producer status')
    normal=result['status']!='failed_or_interrupted';require(lr['scientific_complete']is normal,'normal completion mirror');duration=result['elapsed_seconds'];require(type(duration)in(float,int)and math.isfinite(duration)and 0<=duration<=lr['elapsed_seconds'],'worker duration')
    times=[datetime.fromisoformat(t)for t in(ls['started_utc'],rt['started_utc'],result['finished_utc'],lr['finished_utc'])];require(all(t.tzinfo is not None for t in times)and times==sorted(times),'ordered execution timestamps');require(type(rt['pid'])is int and rt['pid']>0 and rt['pid']!=ls['group_pid'],'worker PID')
    seen=lr['observed_owned_pids'];require(seen==sorted(set(seen))and all(type(i)is int and i>0 for i in seen),'owned PID set')
    if normal:require(lr['successful']is True and lr['process_exit']==0 and lr['failure']is None and not lr['log_truncated']and rt['pid']in seen and lr['max_group_rss_bytes']<=2*1024**3,'bounded completed producer execution')
    else:require(lr['successful']is False,'failure not success')
    emit('provenance.json',dict(pins=pins,raw_inventory=rf,launcher_inventory=lf,launcher=lr,producer_runtime=rt,worker_seconds=duration,finalization_seconds_beyond_cooperative_cap=max(0.,duration-60.)))
    return rf,result,None

def audit(pins,emit):
    files,result,early=producer_provenance(pins,emit)
    if early is not None:return early
    freeze=G.read(ROOT/FROZEN);plan=json.loads(subprocess.check_output(['git','show',freeze['commit']+':'+PLAN],cwd=ROOT))
    if'input-plan.json'not in files:
        require(result['status']=='failed_or_interrupted'and not any(re.fullmatch(r'\d{5}-.*',k)for k in files),'no scientific events without input binding');return dict(verdict='incomplete_provenance_only',arithmetic_qualified=False,producer_status=result['status'])
    require(G.read(RAW/'input-plan.json')==plan,'frozen design/input plan');initial,bound,history=original_inputs(plan);emit('historical-input-inventory.json',history)
    checks=records(RAW,files,result,initial,bound);emit('arithmetic-audit.json',checks)
    return dict(verdict='saved_outcome_audit_pass'if checks['arithmetic_complete']else'incomplete_terminal_return_preserved',producer_status=result['status'],arithmetic_qualified=checks['arithmetic_complete'],checks=checks,scope='Saved arithmetic/provenance only; shared derivative primitive and source-bound full n-mode eigensystem. No independent model, derivative, solve, SVD or eig. Accepted local witnessed edges are not a fold, lower-family connection, boundary, training or inference utility claim. Combined chart/predictor/globalization change is not causally separated.')

canonical,digest,entry,read=G.canonical,G.digest,G.entry,G.read


def serializable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serializable(v) for v in value]
    return value

def run(out,pins_path,pins_sha,source):
    require(os.getpriority(os.PRIO_PROCESS,0)>=10 and not out.exists(),'nice10 exclusive output')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==source,'exact audit source')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean audit checkout')
    sources={}
    for name in AUDIT_FILES:
        b=(ROOT/name).read_bytes();require(b==subprocess.check_output(['git','show',source+':'+name],cwd=ROOT),'committed audit file');sources[name]=dict(bytes=len(b),sha256=digest(b))
    require(entry(pins_path)['sha256']==pins_sha,'external metadata hash');pins=read(pins_path)
    require(set(pins)=={'execution_commit','source_review_commit','raw_manifest_sha256','launcher_manifest_sha256'},'complete pin schema')
    for key,value in pins.items():
        if key=='raw_manifest_sha256' and value is None:continue
        require(isinstance(value,str) and re.fullmatch('[0-9a-f]{'+('40' if key.endswith('commit') else '64')+'}',value),'full metadata pin')
    for key in ('execution_commit','source_review_commit'):subprocess.run(['git','cat-file','-e',pins[key]+'^{commit}'],cwd=ROOT,check=True)
    out.mkdir(parents=True,exist_ok=False)
    def emit(name,value,final=False):
        require(Path(name).name==name,'safe output name');data=canonical(serializable(value));limit=(15 if final else 14)*1024**2
        require(sum(p.stat().st_size for p in out.iterdir())+len(data)<=limit,'audit storage limit')
        with (out/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        fd=os.open(out,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    started=time.monotonic();record=dict(status='running',audit_source_commit=source,audit_source_inventory=sources,pins=pins,pins_file=entry(pins_path),pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),thread_env={k:os.environ[k] for k in THREADS},numpy=np.__version__,python=platform.python_version(),started_utc=datetime.now(timezone.utc).isoformat(),gpu=False,paid=False,new_model_evaluations=0,new_derivatives=0,new_svds=0,new_eigensystems=0,new_solves=0)
    emit('runtime-start.json',record)
    def stop(*_):raise TimeoutError('30-second saved pseudoarclength audit cap')
    old_alarm=signal.signal(signal.SIGALRM,stop);old_term=signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,30.)
    try:record.update(audit(pins,emit),status='finished')
    except BaseException as error:record.update(status='failed',verdict='audit_failed',error=type(error).__name__+': '+str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_alarm);signal.signal(signal.SIGTERM,old_term)
        record.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        emit('result.json',record,True);emit('manifest.json',dict(files={p.name:entry(p) for p in sorted(out.iterdir())}),True)
        for p in out.iterdir():p.chmod(0o444)
        out.chmod(0o555)
    print(json.dumps({k:record.get(k) for k in ('status','verdict','error','elapsed_seconds','producer_status','arithmetic_qualified')},sort_keys=True))
    return 0 if record['status']=='finished' else 2

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--pins',type=Path,required=True);parser.add_argument('--pins-sha256',required=True);parser.add_argument('--source-commit',required=True);args=parser.parse_args();raise SystemExit(run(args.out,args.pins,args.pins_sha256,args.source_commit))

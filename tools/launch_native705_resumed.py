#!/usr/bin/env python3
"""Fixed24-step resume of maintained cubic705 continuation from actual endpoint."""
import probe_native705_resumed as W
import bind_native705_resumed as B
import launch_native705_connected as L
ROOT=W.ROOT;POLICY=ROOT/'docs/plans/native705-resumed-policy.json'
LIMITS=dict(L.LIMITS,initial_halfwidth=-3)

def policy(path):
 W.need(path.resolve()==POLICY,'literal resumed policy');p=W.read(path)
 W.need(p['schema']=='native705-resumed-v1'and p['status']=='measured-reviewed'and p['limits']==LIMITS and p['initial_parent']==B.PARENT and p['initial_state']==list(B.STATE),'fixed bounded reviewed resumed packet')
 W.need(L.hashlib.sha256(L.subprocess.check_output(['git','show','HEAD:'+str(path.resolve().relative_to(ROOT))],cwd=ROOT)).hexdigest()==W.entry(path)['sha256'],'same exact committed resume policy');return p

def initial_parent(identity,check):
 return B.bind(identity,check,lambda name,value:L.save(W.BASE,name,value))

def main(args):
 W.install();L.W=W;L.BASE=W.BASE;L.LIMITS=LIMITS;L.INITIAL_STATE=B.STATE;L.policy=policy;L.initial_parent=initial_parent
 L.WORKER_PATH=ROOT/'tools/probe_native705_resumed.py';L.FALLBACK_PATH=ROOT/'tools/probe_native705_resumed_fallback.py'
 return L.main(args)
if __name__=='__main__':raise SystemExit(main(L.parser().parse_args()))

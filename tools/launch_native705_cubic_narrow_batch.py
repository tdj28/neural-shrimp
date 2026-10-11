#!/usr/bin/env python3
"""Single explicit four-chart h1/32 retry from unchanged independent known root."""
import probe_native705_cubic_narrow_batch as N
import launch_native705_cubic_batch as L
from pathlib import Path
PARENT=N.W.P.BASE/'batch001';PIN='a3675ba78c61b9c5b24b500ca049aa4bb70b3ba813db5e44e5729377c2356b49'
DIAG=N.W.P.BASE/'attempt001-diagnosis';DPIN='ba6fc179a49417e9bfcd01fe7bea4b95eacb627ca4c2864a070dc03c1a301738'
_original=L.committed_plan

def plan(path):
 p=_original(path);L.need(p['batch']==1 and p['retained_failure']=={'root':str(PARENT),'outer':PIN}and p['diagnosis']=={'root':str(DIAG),'outer':DPIN},'fixed same-root smaller-domain amendment')
 N.W.inventory(PARENT,PIN);old=N.W.read(PARENT/'outcome.json');L.need(not old['passed']and old['connected_count']==0 and not old['owned_live_pids']and old['source']=='0f5955721f5219c9ca9b2f672f490675c61f9c2f','complete original failed new-center batch')
 N.W.inventory(DIAG,DPIN);d=N.W.read(DIAG/'run/outcome.json');L.need(d['passed']and d['native_calls']==0 and len(d['charts'])==4,'complete saved four-chart diagnosis')
 return p

def main(args):
 N.configure();L.BASE=N.BASE;L.committed_plan=plan;return L.main(args)
if __name__=='__main__':raise SystemExit(main(L.parser().parse_args()))

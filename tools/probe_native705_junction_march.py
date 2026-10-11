#!/usr/bin/env python3
"""Same331-operation worker with a fixed completed-packet endpoint authority."""
import probe_native705_connected as C
ROOT=C.ROOT;BASE=C.W.P.BASE/'maintained004'
import probe_native705_resumed as PREV
import probe_native705_junction_lift as LIFT
SOURCES=list(dict.fromkeys(PREV.SOURCES+LIFT.FILES+['tools/bind_native705_junction_march.py','tools/probe_native705_junction_march.py','tools/launch_native705_junction_march.py','tools/probe_native705_junction_fallback.py','tools/native705_completion_event.py','tests/test_native705_junction_march.py','docs/plans/native705-junction-policy.json','docs/plans/native705-junction-march.md','docs/reviews/native705-maintained003.json','docs/reviews/native705-junction-lift003.json']))
for _name in ['W','REF','need','read','entry','inventory','CAP','EXPECTED','settings','decode','join','validate_card']:
 globals()[_name]=getattr(C,_name)
def install():C.BASE=BASE;C.SOURCES=SOURCES
def main(args):install();return C.main(args)
if __name__=='__main__':raise SystemExit(main(W.parser().parse_args()))

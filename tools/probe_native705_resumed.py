#!/usr/bin/env python3
"""Same331-operation worker with a fixed completed-packet endpoint authority."""
import probe_native705_connected as C
ROOT=C.ROOT;BASE=C.W.P.BASE/'maintained003'
SOURCES=list(dict.fromkeys(C.SOURCES+['tools/bind_native705_resumed.py','tools/probe_native705_resumed.py','tools/launch_native705_resumed.py','tools/probe_native705_resumed_fallback.py','tests/test_native705_resumed.py','docs/plans/native705-resumed-policy.json','docs/plans/native705-resumed-continuation.md','docs/reviews/native705-maintained002.json']))
for _name in ['W','REF','need','read','entry','inventory','CAP','EXPECTED','settings','decode','join','validate_card']:
 globals()[_name]=getattr(C,_name)
def install():C.BASE=BASE;C.SOURCES=SOURCES
def main(args):install();return C.main(args)
if __name__=='__main__':raise SystemExit(main(W.parser().parse_args()))

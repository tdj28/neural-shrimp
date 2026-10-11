#!/usr/bin/env python3
"""One fixed smaller-domain new-center cubic batch; all failed wider evidence kept."""
import probe_native705_cubic_batch as W
BASE=W.P.BASE/'narrow001'
SOURCES=list(dict.fromkeys(W.SOURCES+['tools/probe_native705_cubic_narrow_batch.py','tools/launch_native705_cubic_narrow_batch.py','tests/test_native705_cubic_narrow_batch.py','docs/plans/native705-cubic-narrow-batch.md','docs/reviews/native705-cubic-batch001-failure.json','docs/reviews/native705-cubic-batch-diagnosis.json']))

def configure():
 W.BASE=BASE;W.HALFWIDTH=-5;W.WORKER='tools/probe_native705_cubic_narrow_batch.py';W.SOURCES=SOURCES

def main(args):configure();return W.main(args)
if __name__=='__main__':configure();raise SystemExit(main(W.parser().parse_args()))

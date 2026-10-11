#!/usr/bin/env python3
"""Unchanged endpoint refinement bound to the resumed packet's exact policy."""
import probe_native705_junction_march as W
import probe_native705_endpoint_fallback as F

def main(args):
 W.install();F.W=W;F.POLICY_PATH=W.ROOT/'docs/plans/native705-junction-policy.json'
 return F.main(args)
if __name__=='__main__':raise SystemExit(main(F.parser().parse_args()))

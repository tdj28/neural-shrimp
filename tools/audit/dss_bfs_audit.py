#!/usr/bin/env python3
"""Usage: python tools/audit/dss_bfs_audit.py PROGRESS.jsonl

Reconstruct the old breadth-first continuation (every converged cell seeded its neighbours) from a dss progress.jsonl
and flag primitive cells whose parent had slipped onto a divisor-period orbit (possible wrong orbit)."""
import json, sys
import numpy as np
path = sys.argv[1]
recs = [json.loads(l) for l in open(path) if l.startswith('{"i"')]
first_seen = {}
status = {}
order = {}
for k, r in enumerate(recs):
    c = (r["i"], r["j"])
    if c in order:
        continue                                    # old code: one attempt per cell
    order[c] = k
    conv = r["converged"]
    split = r.get("split", 0.0)
    status[c] = "x" if not conv else ("o" if split > 1e-7 else "d")
parent = {}
for c, k in order.items():
    cands = [(order[n], n) for n in ((c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1))
             if n in order and order[n] < k and status[n] in ("o", "d")]
    parent[c] = min(cands)[1] if cands else None
# taint: a primitive cell is suspect if any ancestor in its parent chain was collapsed
def tainted(c):
    p = parent[c]
    while p is not None:
        if status[p] == "d":
            return True
        p = parent[p]
    return False
o = [c for c in order if status[c] == "o"]
sus = [c for c in o if tainted(c)]
same = {(r["i"], r["j"]): r.get("same_tokens") for r in recs if "same_tokens" in r}
print(f"{path.split('/')[-2]}: cells {len(order)}, primitive {len(o)}, collapsed {sum(s=='d' for s in status.values())}, "
      f"failed {sum(s=='x' for s in status.values())}, primitive with a collapsed ancestor {len(sus)}"
      + (f" (same tokens {sum(1 for c in sus if same.get(c))})" if sus else ""))
if sus:
    print("   suspect cells:", sorted(sus)[:20], "..." if len(sus) > 20 else "")

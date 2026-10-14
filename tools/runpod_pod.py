"""Task-owned RunPod pod lifecycle for approved runs (GPU playbook sections 3-6).

Subcommands:
  inventory                      read-only list of every pod on the account
  create --gpu-type ID ...       create ONE on-demand pod and record its ID in the
                                 local ledger before doing anything else
  status --pod ID                runtime state and the public SSH endpoint
  terminate --pod ID             terminate a pod, but only one this tool created
                                 (listed in the ledger); then verify it is gone by
                                 direct lookup and full inventory

The API key is read from the gitignored .env inside this process and never
printed or passed on a command line. The ledger lives outside Git at
DATA/runpod/pods.jsonl (one JSON object per event).
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
LEDGER = DATA / "runpod" / "pods.jsonl"
API = "https://api.runpod.io/graphql"
# Every pod of this project is named with this prefix; other pods on a shared
# account are never listed in detail, inspected or touched.
POD_PREFIX = "neural-shrimp-"


def api_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("RUNPOD_API_KEY="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    raise SystemExit("RUNPOD_API_KEY missing in .env")


def graphql(query, variables=None):
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    request = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {api_key()}",
        "User-Agent": "neural-shrimp-pod/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise SystemExit(f"RunPod HTTP error {error.code}")
    if data.get("errors"):
        raise SystemExit("RunPod API errors: " + "; ".join(e.get("message", "?") for e in data["errors"]))
    return data["data"]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record(event):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as handle:
        handle.write(json.dumps({"time": now(), **event}) + "\n")


def owned_pods():
    if not LEDGER.exists():
        return set()
    created = {json.loads(line)["pod_id"] for line in LEDGER.read_text().splitlines()
               if json.loads(line).get("event") == "created"}
    return created


def inventory():
    pods = graphql("query { myself { clientBalance currentSpendPerHr pods { id name desiredStatus costPerHr "
                   "gpuCount machine { gpuDisplayName } } } }")["myself"]
    return pods


def cmd_inventory(args):
    me = inventory()
    owned = owned_pods()
    ours = [p for p in me["pods"] if (p.get("name") or "").startswith(POD_PREFIX)]
    others = len(me["pods"]) - len(ours)
    print(f"balance ${me['clientBalance']:.2f}; account spend ${me['currentSpendPerHr']:.2f}/h; "
          f"{others} pod(s) of other projects (not listed)")
    for p in ours:
        tag = "in ledger" if p["id"] in owned else "NOT in ledger"
        print(f"  {p['id']} {p['name']} {p['desiredStatus']} ${p['costPerHr']}/h [{tag}]")


def cmd_create(args):
    if not args.name.startswith(POD_PREFIX):
        raise SystemExit(f"pod names must start with {POD_PREFIX!r}")
    pubkey = Path(args.pubkey_file).read_text().strip()
    if not pubkey.startswith("ssh-"):
        raise SystemExit("public key file does not look like an SSH public key")
    before = {p["id"] for p in inventory()["pods"]}
    record({"event": "inventory_before_create", "pod_ids": sorted(before)})
    mutation = """mutation($input: PodFindAndDeployOnDemandInput) {
      podFindAndDeployOnDemand(input: $input) { id name imageName costPerHr desiredStatus machineId } }"""
    variables = {"input": {
        "cloudType": args.cloud, "gpuCount": int(getattr(args, "gpu_count", 1)), "volumeInGb": 0,
        "containerDiskInGb": args.disk_gb,
        "minVcpuCount": args.min_vcpu, "minMemoryInGb": args.min_memory_gb, "gpuTypeId": args.gpu_type,
        "name": args.name, "imageName": args.image, "dockerArgs": "", "ports": "22/tcp",
        "volumeMountPath": "/workspace", "env": [{"key": "PUBLIC_KEY", "value": pubkey}]}}
    pod = graphql(mutation, variables)["podFindAndDeployOnDemand"]
    if not pod or not pod.get("id"):
        raise SystemExit("no pod was created (capacity unavailable?)")
    record({"event": "created", "pod_id": pod["id"], "gpu_type": args.gpu_type, "cloud": args.cloud,
            "gpu_count": int(getattr(args, "gpu_count", 1)),
            "cost_per_hr": pod.get("costPerHr"), "image": args.image, "name": args.name,
            "commit": args.commit, "approval": args.approval})
    print(json.dumps({"pod_id": pod["id"], "cost_per_hr": pod.get("costPerHr")}))


def pod_status(pod_id):
    query = """query($input: PodFilter) { pod(input: $input) { id name desiredStatus costPerHr
      machine { gpuDisplayName } runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }"""
    return graphql(query, {"input": {"podId": pod_id}})["pod"]


def cmd_status(args):
    pod = pod_status(args.pod)
    if not pod:
        print(json.dumps({"pod_id": args.pod, "found": False}))
        return
    ssh = None
    for port in ((pod.get("runtime") or {}).get("ports") or []):
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            ssh = {"ip": port["ip"], "port": port["publicPort"]}
    print(json.dumps({"pod_id": pod["id"], "status": pod["desiredStatus"], "cost_per_hr": pod["costPerHr"],
                      "gpu": (pod.get("machine") or {}).get("gpuDisplayName"),
                      "uptime": (pod.get("runtime") or {}).get("uptimeInSeconds"), "ssh": ssh}))


def cmd_terminate(args):
    if args.pod not in owned_pods():
        raise SystemExit(f"refusing: {args.pod} was not created by this tool (not in {LEDGER})")
    current = pod_status(args.pod)
    if current and not (current.get("name") or "").startswith(POD_PREFIX):
        raise SystemExit(f"refusing: {args.pod} is not named with {POD_PREFIX!r}")
    graphql("""mutation($input: PodTerminateInput!) { podTerminate(input: $input) }""",
            {"input": {"podId": args.pod}})
    record({"event": "terminate_requested", "pod_id": args.pod})
    for _ in range(12):
        time.sleep(5)
        direct = pod_status(args.pod)
        listed = args.pod in {p["id"] for p in inventory()["pods"]}
        if not direct and not listed:
            record({"event": "terminated_verified", "pod_id": args.pod})
            print(json.dumps({"pod_id": args.pod, "terminated": True, "direct_lookup": None, "in_inventory": False}))
            return
    record({"event": "terminate_unverified", "pod_id": args.pod})
    raise SystemExit(f"termination of {args.pod} NOT verified; check the RunPod console now")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    create = sub.add_parser("create")
    create.add_argument("--gpu-type", required=True)
    create.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    create.add_argument("--image", required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--pubkey-file", required=True)
    create.add_argument("--disk-gb", type=int, default=40)
    create.add_argument("--min-vcpu", type=int, default=8)
    create.add_argument("--min-memory-gb", type=int, default=32)
    create.add_argument("--commit", required=True)
    create.add_argument("--approval", required=True)
    status = sub.add_parser("status")
    status.add_argument("--pod", required=True)
    terminate = sub.add_parser("terminate")
    terminate.add_argument("--pod", required=True)
    args = parser.parse_args()
    {"inventory": cmd_inventory, "create": cmd_create, "status": cmd_status, "terminate": cmd_terminate}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

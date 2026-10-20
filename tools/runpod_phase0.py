"""Orchestrate RunPod Phase 0 end to end (approved: at most [amount redacted]; see docs/consults/approvals.md).

create pod (named neural-shrimp-...) -> wait for SSH -> upload `git archive` of
a pushed commit and the sm_89 reference -> run tools/runpod_phase0_runner.sh
under a wall-time limit -> retrieve and hash-verify the results -> terminate the
pod and verify it is gone. Termination runs in a `finally` block, so any
failure still tears the pod down. Every step is appended to the pod ledger
(DATA/runpod/pods.jsonl); no credential is sent to the pod.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import runpod_pod as rp  # noqa: E402

KEY = Path.home() / ".ssh" / "neural_shrimp_runpod_ed25519"
UPLOAD_PATHS = ("src", "tools", "tests", "native", "experiments/trainability", "pyproject.toml", "uv.lock")


def ssh_base(ip, port):
    known = rp.DATA / "runpod" / "known_hosts"
    return ["ssh", "-i", str(KEY), "-p", str(port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known}", f"root@{ip}"]


def wait_for_ssh(pod_id, deadline):
    while time.time() < deadline:
        status = rp.pod_status(pod_id)
        ssh = None
        for port in ((status or {}).get("runtime") or {}).get("ports") or []:
            if port.get("privatePort") == 22 and port.get("isIpPublic"):
                ssh = (port["ip"], port["publicPort"])
        if ssh:
            probe = subprocess.run(ssh_base(*ssh) + ["echo ready"], capture_output=True, text=True, timeout=60)
            if probe.returncode == 0 and "ready" in probe.stdout:
                return ssh
        time.sleep(15)
    raise TimeoutError("SSH endpoint not ready before the deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-type", required=True, nargs="+",
                        help="ordered fallbacks, CLOUD:TYPE (e.g. COMMUNITY:'NVIDIA H100 PCIe'); "
                             "only creation is retried, never anything after a pod exists")
    parser.add_argument("--max-rate", type=float, default=3.5, help="refuse pods listed above this $/h")
    parser.add_argument("--image", default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    parser.add_argument("--name", default="neural-shrimp-phase0")
    parser.add_argument("--commit", required=True, help="pushed commit to upload")
    parser.add_argument("--reference", default=str(rp.DATA / "runpod/phase0/reference-sm89"))
    parser.add_argument("--out", default=str(rp.DATA / "runpod/phase0"))
    parser.add_argument("--ready-minutes", type=float, default=15)
    parser.add_argument("--run-minutes", type=float, default=60)
    parser.add_argument("--approval", default="docs/consults/approvals.md#2026-10-07-runpod-phase-0-approved")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", args.commit], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    pushed = subprocess.run(["git", "branch", "-r", "--contains", head], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    if not pushed:
        raise SystemExit(f"commit {head} is not on any remote branch; push it first")
    pod_id, chosen = None, None
    for entry in args.gpu_type:
        cloud, _, gpu = entry.partition(":")
        create = argparse.Namespace(gpu_type=gpu, cloud=cloud, image=args.image, name=args.name,
                                    pubkey_file=str(KEY) + ".pub", disk_gb=40, min_vcpu=4, min_memory_gb=16,
                                    commit=head, approval=args.approval)
        try:
            rp.cmd_create(create)
        except SystemExit as error:
            if "instances available" in str(error):
                rp.record({"event": "no_capacity", "gpu_type": gpu, "cloud": cloud})
                print(f"no capacity: {cloud} {gpu}", flush=True)
                continue
            raise
        last = [json.loads(l) for l in rp.LEDGER.read_text().splitlines()][-1]
        pod_id, chosen = last["pod_id"], entry
        break
    if pod_id is None:
        raise SystemExit("no capacity for any listed GPU type; no pod was created")
    started = time.time()
    result = {"pod_id": pod_id, "gpu_type": chosen, "commit": head, "started": started}
    rate = [json.loads(l) for l in rp.LEDGER.read_text().splitlines()][-1].get("cost_per_hr") or 0
    if rate > args.max_rate:
        rp.cmd_terminate(argparse.Namespace(pod=pod_id))
        raise SystemExit(f"pod rate ${rate}/h above --max-rate; terminated")
    try:
        ip, port = wait_for_ssh(pod_id, started + 60 * args.ready_minutes)
        rp.record({"event": "ssh_ready", "pod_id": pod_id, "seconds_after_create": round(time.time() - started)})
        base = ssh_base(ip, port)
        # Only what the runner needs (about 5 MB); the full tree carries ~1.8 GB of
        # committed evidence that would take ~15 minutes on a 2 MB/s uplink.
        archive = subprocess.run(["git", "archive", "--format=tar", head, *UPLOAD_PATHS], cwd=ROOT,
                                 capture_output=True, check=True).stdout
        subprocess.run(base + ["mkdir -p /root/neural-shrimp && tar -x -C /root/neural-shrimp"], input=archive,
                       check=True, timeout=600)
        reference = subprocess.run(["tar", "-c", "-C", str(Path(args.reference).parent), Path(args.reference).name],
                                   capture_output=True, check=True).stdout
        subprocess.run(base + ["tar -x -C /root"], input=reference, check=True, timeout=600)
        rp.record({"event": "uploaded", "pod_id": pod_id, "archive_bytes": len(archive), "reference_bytes": len(reference)})
        run = subprocess.run(base + [f"REF=/root/{Path(args.reference).name} bash /root/neural-shrimp/tools/runpod_phase0_runner.sh"],
                             capture_output=True, text=True, timeout=60 * args.run_minutes)
        (out / "runner-stdout.txt").write_text(run.stdout + "\n--- stderr ---\n" + run.stderr)
        remote_hash = next((line.split()[0] for line in run.stdout.splitlines() if line.endswith("phase0-results.tgz")), None)
        subprocess.run(["scp", "-i", str(KEY), "-P", str(port), "-o", "BatchMode=yes",
                        "-o", f"UserKnownHostsFile={rp.DATA / 'runpod' / 'known_hosts'}",
                        f"root@{ip}:/root/phase0-results.tgz", str(out / "phase0-results.tgz")], check=True, timeout=600)
        local_hash = hashlib.sha256((out / "phase0-results.tgz").read_bytes()).hexdigest()
        result.update(runner_returncode=run.returncode, remote_sha256=remote_hash, local_sha256=local_hash,
                      hash_verified=remote_hash == local_hash)
        rp.record({"event": "retrieved", "pod_id": pod_id, "hash_verified": remote_hash == local_hash,
                   "sha256": local_hash})
    finally:
        rp.cmd_terminate(argparse.Namespace(pod=pod_id))
        result["wall_seconds"] = round(time.time() - started)
        (out / "phase0-result.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result))


if __name__ == "__main__":
    main()

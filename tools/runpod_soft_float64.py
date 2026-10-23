"""Float64 soft-token continuation on one RunPod pod, end to end.

create a pod (named neural-shrimp-..., ID recorded in the ledger) -> wait for SSH -> upload `git archive` of a
pushed commit (only the soft-loop tools and the runner) and the start data (concept vectors, saved branches)
-> run tools/runpod_soft_runner.sh JOBSET under a wall-time limit, streaming its output to a local log ->
retrieve and hash-verify results.tgz and unpack it under DATA/explore -> terminate the pod and verify it is
gone. Termination runs in a `finally` block, so any failure still tears the pod down. The run refuses to start
when the pod's hourly rate times the time limits exceeds --max-cost. No credential is sent to the pod; the
model is downloaded there from the public Hugging Face hub at the pinned revision.
"""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import runpod_pod as rp  # noqa: E402
from runpod_phase0 import KEY, ssh_base, wait_for_ssh  # noqa: E402

UPLOAD_PATHS = ("tools/huginn_chaos_map.py", "tools/runpod_huginn_chaos_runner.sh", "docs/huginn-chaos-map-plan.json", "tools/explore_llm_soft_loop.py", "tools/explore_llm_attractors.py", "tools/explore_llm_phase_map.py",
                "tools/bifurcation_curves.py", "tools/explore_latent_loop.py", "tools/runpod_soft_runner.sh",
                "tools/explore_llm_msm.py", "tools/llm_feigenbaum.py", "tools/doubly_superstable.py", "tools/pi_f64_twins.py", "tools/pi_harness.py", "tools/pi_matrix_run.sh", "tools/pi_matrix_analyze.py", "tools/pi_reason.py", "tools/pi_reason_f64.py", "tools/pi_generate_diag.py", "docs/pi/llama-diag-indices.json", "tools/pi_matrix2_analyze.py", "tools/damping_locate.py", "tools/moe_routing.py", "tools/olmoe_float64.py", "docs/pi-gsm8k-test.json", "tools/behavior_zones.py", "docs/bz-pairs.json", "docs/bz-tasks.json", "tools/behavior_prompts.py", "docs/bp-tasks.json", "tools/two_block_map.py", "tools/steer_gain.py", "tools/repro_noise.py", "tools/repro_task.py", "tools/crossover.py", "tools/turning_scan.py", "tools/soft_dialogue.py", "docs/dlg", "tools/reservoir_capacity.py", "docs/tp", "tools/period_colors.py", "docs/xo-cells.json", "docs/xo7-cells.json", "docs/xo-cells-gratitude.json", "docs/xo-cells-random1.json", "docs/repro7-cells.json", "docs/repro-l8-cells.json", "docs/repro70-cells.json", "docs/bpc-tasks.json", "docs/bpcs-tasks.json", "docs/repro-task-problems.json", "tools/return_map.py", "tools/flip_normal_form.py", "tools/hostility_score.py",
                "tools/huginn_relax_eval.py", "tools/huginn_edge.py", "tools/p1_committor.py", "tools/p1_detection.py", "tools/p1_select_dose.py", "tools/cusp_check.py", "tools/llm_sae_readout.py", "tools/cusp_unfolding.py", "tools/pilot_stage2.py",
                "tools/steering_controls.py", "docs/steering-controls-selection.json", "tools/qwen35_float64.py",
                "tools/selfref_features.py", "tools/hidden_chaos.py", "tools/gain_split.py", "tools/start_state.py", "docs/h2-horizon-cells.json",
                "docs/steering-controls-selection-70b.json", "docs/prefix-scale-prefixes.json", "docs/plateau-cells-z70.json", "docs/plateau-cells-z128.json", "docs/plateau-cells-l31-70b.json", "docs/q72fix-cells.json",
                "tools/sheets_map.py", "tools/sheets_transect.py", "tools/sheets_edge.py", "tools/p5_temperature.py", "tools/p5_dss_temperature.py", "tools/long_transients.py", "tools/cert_remote.py", "tools/cert_capacity.py", "tools/shrimp_centre.py", "tools/norm_bounds.py", "tools/follow_attractor.py", "tools/leg_centerline.py", "tools/nilpotent_track.py", "tools/monodromy_svd.py",
                "docs/nb-cells.json", "docs/nb-cells-cross-a.json", "docs/nb-cells-cross-b.json", "tools/bound_hierarchy.py", "docs/bh/bh-cells.json", "tools/attractor_portrait.py", "tools/unstable_manifold.py", "tools/lyap_spectrum.py", "tools/basin_slice.py", "tools/attractor_sweep.py",
                "docs/lt-cells.json",
                "tools/prefix_transition.py", "tools/soft_thinking_twins.py", "docs/st72-cells-p3.json",
                "tools/lyapunov_spectrum.py", "docs/tc-spectrum-cells.json", "docs/ky-cells-philosophy.json", "docs/ky-cells-hostility.json", "tools/cusp_locate.py", "tools/locate_curve_candidates.py", "tools/repro_reset.py", "docs/rr-problems.json", "tools/window_census.py", "tools/window_closure.py", "docs/oc", "docs/cl", "docs/wp-line-cells.json", "docs/fl15-line-cells.json", "docs/fl7-line-cells.json",
                "tools/symbolic_dynamics.py", "tools/sd_verify_words.py", "tools/kite_basin_check.py",
                "tools/hook_edge_track.py", "tools/hook_saddle_check.py", "tools/real_inference_twins.py", "docs/ri/ri1-cells.json",
                "tools/hook_tip_strip.py", "tools/hook_tip_check.py", "tools/hook_tip_fold_points.py",
                "tools/baby_shrimp_orbits.py", "tools/baby_shrimp_geometry.py", "docs/baby-shrimp-zoom/geometry-plan.json", "docs/baby-shrimp-zoom/geometry-amended-plan.json", "tools/baby_shrimp_crossing.py", "docs/baby-shrimp-zoom/crossing-plan.json", "tools/baby_shrimp_transect.py", "tools/baby_shrimp_trajectories.py", "docs/baby-shrimp-zoom/jones-transect-plan.json",
                "docs/baby-shrimp-zoom/crossing-curve-plan.json", "docs/baby-shrimp-zoom/crossing-receipt.json",
                "docs/baby-shrimp-zoom/junction-scan-plan.json",
                "docs/baby-shrimp-zoom/grandchild-scan-plan.json",
                "tools/baby_shrimp_saddle_memory.py", "tools/baby_shrimp_memory.py",
                "tools/baby_shrimp_memory_scan.py", "docs/baby-shrimp-zoom/memory-junction-scan-plan.json",
                "docs/baby-shrimp-zoom/memory-local-extension-receipt.json",
                "tools/trace_saddle_continuation.py", "tools/trace_pair_sensitivity.py",
                "docs/baby-shrimp-zoom/saddle-memory-plan.json",
                "pyproject.toml")
DATA_PATHS = ("explore/llm-phase-map/vectors-l12", "explore/llm-soft-loop/branch-p8-a0.0717-up2",
              "explore/llm-soft-loop/branch-p8-a0.0717-down", "explore/llm-soft-loop/branch-p2-t-0.115-alpha-up")


GATED_JOBSETS = ("repro70", "reprollama", "llama8b", "llamamap", "llamactrl", "selfref", "bigsweep", "bigctrl", "bigmap", "gainsplit", "bighidden", "scaleset", "startstate", "upperset", "damping")                     # job sets that download gated weights with the human's HF token


def hf_token():
    """The Hugging Face token from the gitignored .env (never printed, logged or recorded)."""
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    raise SystemExit("HF_TOKEN missing in .env (needed for gated weights)")


def archives(commit, extra=(), huginn_only=False):
    selected = ("tools/huginn_chaos_map.py", "tools/runpod_huginn_chaos_runner.sh", "tools/runpod_soft_runner.sh",
                "tools/explore_latent_loop.py", "docs/huginn-chaos-map-plan.json", "docs/huginn-chaos-zoom-plan.json",
                "docs/huginn-chaos-tight-zoom-plan.json") if huginn_only else UPLOAD_PATHS
    code = subprocess.run(["git", "archive", "--format=tar", commit, *selected], cwd=ROOT, capture_output=True,
                          check=True).stdout
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for rel in (*( () if huginn_only else DATA_PATHS), *extra):
            tar.add(rp.DATA / rel, arcname=rel)
    return code, buf.getvalue()


def own_creation(name, since):
    """The newest 'created' record of the pod named `name` written after `since` (epoch seconds); refuses if absent."""
    from datetime import datetime
    mine = []
    for line in rp.LEDGER.read_text().splitlines():
        e = json.loads(line)
        if e.get("event") == "created" and e.get("name") == name and \
                datetime.fromisoformat(e["time"]).timestamp() >= since - 1:
            mine.append(e)
    if not mine:
        raise SystemExit(f"a pod named {name} was requested but no creation record of it was found; check the inventory")
    return mine[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu-type", nargs="+", default=["COMMUNITY:NVIDIA H100 NVL", "COMMUNITY:NVIDIA H100 PCIe",
                                                          "SECURE:NVIDIA H100 NVL", "COMMUNITY:NVIDIA A100 80GB PCIe"],
                        help="ordered fallbacks, CLOUD:TYPE; only creation is retried")
    parser.add_argument("--max-rate", type=float, required=True, help="refuse pods listed above this hourly rate")
    parser.add_argument("--max-cost", type=float, required=True,
                        help="refuse to run when rate x (ready + run + 10 minutes) exceeds this")
    parser.add_argument("--image", default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    parser.add_argument("--name", default="neural-shrimp-soft64")
    parser.add_argument("--commit", required=True, help="pushed commit to upload")
    parser.add_argument("--jobset", default="skeleton", choices=("smoke", "tbmulti", "rc", "hedge", "dialogue", "repro70", "reprollama", "tp", "xo", "reprotask", "repro", "mech", "dss", "spectrum", "flipnf", "bz", "bp", "tb", "retmap", "skeleton", "maps", "p4a", "highT", "scanband", "p9s", "shrimpcurves", "kitecurves", "p4curves", "scale15", "p6z3a", "p6z3curves", "p1ladder", "cascade", "edges", "superstable", "p1control", "scan32", "p1detect", "huginn", "hverify", "hnormal", "hnormal2", "p1commit", "scan64", "scale7", "huginnA", "hcurves", "hverify7", "p1x", "cusps", "scalex", "huginnS", "hcascade", "cubic", "pairmap", "llama8b", "unfold", "pilot1", "pilot2", "llamamap", "llamactrl", "anymap", "qwen35moe", "selfref", "morph", "morphfill", "hidden", "horizon", "bigsweep", "bigctrl", "bigmap", "gainsplit", "bighidden", "scaleset", "prefixscale", "startstate", "upperset", "prefix72", "prefixcoarse", "q72fix", "prefix72base", "window", "softthink", "softthink72", "tangentcheck", "codim2", "hiddencells", "foldlaw15", "foldlaw7", "fm5", "fm5c", "fm5d", "rr", "shrimp", "pif64", "damping", "moe", "pm", "pm2", "pm2f64", "pm2diag", "preason", "pm2f64x", "sheets", "transects", "p5temp", "p5dss", "ltrans", "certcal", "shrimpc", "fm5t", "nbounds", "fa", "nt", "fm5x", "fmg", "rk", "certpack", "fm5l", "bh", "eslseq", "toolseq"))
    parser.add_argument("--jobset-args", default="", help="arguments passed to the runner after the job set name")
    parser.add_argument("--ready-minutes", type=float, default=15)
    parser.add_argument("--disk-gb", type=int, default=60, help="container disk (large checkpoints need more)")
    parser.add_argument("--gpu-count", type=int, default=1, help="GPUs per pod (large models spread over them)")
    parser.add_argument("--min-memory-gb", type=int, default=32, help="minimum host RAM of the pod")
    parser.add_argument("--min-vcpu", type=int, default=8, help="minimum vCPUs of the pod (consumer hosts offer fewer)")
    parser.add_argument("--run-minutes", type=float, required=True)
    parser.add_argument("--out", default=str(rp.DATA / "runpod" / "soft64"))
    parser.add_argument("--approval", required=True)
    parser.add_argument("--dry-run", action="store_true", help="build the uploads and report their sizes only")
    parser.add_argument("--upload-timeout", type=float, default=600,
                        help="seconds allowed for uploading the start data (large warm-start states need more)")
    parser.add_argument("--extra-data", nargs="*", default=[],
                        help="further DATA-relative paths to upload (e.g. branches that curves start from)")
    args = parser.parse_args()
    head = subprocess.run(["git", "rev-parse", args.commit], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    pushed = subprocess.run(["git", "branch", "-r", "--contains", head], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    if not pushed:
        raise SystemExit(f"commit {head} is not on any remote branch; push it first")
    huginn_only = args.jobset == "toolseq" and " huginn_chaos_map.py " in args.jobset_args
    code, data = archives(head, args.extra_data, huginn_only=huginn_only)
    # the 10 minutes cover the default upload allowance and the retrieval; a longer upload allowance is added
    worst = args.max_rate * (args.ready_minutes + args.run_minutes + 10
                             + max(0.0, args.upload_timeout - 600) / 60) / 60
    plan = {"commit": head, "jobset": args.jobset, "code_bytes": len(code), "data_bytes": len(data),
            "code_sha256": hashlib.sha256(code).hexdigest(), "data_sha256": hashlib.sha256(data).hexdigest(),
            "extra_data": list(args.extra_data), "jobset_args": args.jobset_args,
            "worst_case_cost_at_max_rate": round(worst, 2), "max_cost": args.max_cost}
    print(json.dumps(plan), flush=True)
    if worst > args.max_cost:
        raise SystemExit("time limits times the maximum rate exceed --max-cost; shorten the run")
    if args.dry_run:
        return
    # one folder per run: launches in the same second used to share it (and overwrite each other's receipts), so the
    # folder is created atomically and a clash takes the next free suffix
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out, k = Path(args.out) / stamp, 1
    while True:
        try:
            out.mkdir()
            break
        except FileExistsError:
            out, k = Path(args.out) / f"{stamp}-{k}", k + 1
    pod_id, chosen = None, None
    for entry in args.gpu_type:
        cloud, _, gpu = entry.partition(":")
        create = argparse.Namespace(gpu_type=gpu, cloud=cloud, image=args.image, name=args.name,
                                    gpu_count=args.gpu_count,
                                    pubkey_file=str(KEY) + ".pub", disk_gb=args.disk_gb, min_vcpu=args.min_vcpu,
                                    min_memory_gb=args.min_memory_gb,
                                    commit=head, approval=args.approval)
        attempt = time.time()
        try:
            rp.cmd_create(create)
        except SystemExit as error:
            if "instances available" in str(error) or "no pod was created" in str(error):
                rp.record({"event": "no_capacity", "gpu_type": gpu, "cloud": cloud})
                print(f"no capacity: {cloud} {gpu}", flush=True)
                continue
            raise
        # this run's own creation record: the ledger is shared by parallel launches, so its last line can belong to
        # another run (whose pod would then be adopted here, and its own left running)
        last = own_creation(args.name, attempt)
        pod_id, chosen = last["pod_id"], entry
        break
    if pod_id is None:
        raise SystemExit("no capacity for any listed GPU type; no pod was created")
    started = time.time()
    rate = last.get("cost_per_hr")
    if rate is None:
        raise SystemExit(f"no hourly rate in the creation record of pod {pod_id}")
    result = {"pod_id": pod_id, "gpu_type": chosen, "rate": rate, "commit": head, "jobset": args.jobset,
              "started": started, "out": str(out), "upload": {k: plan[k] for k in ("code_bytes", "data_bytes",
              "code_sha256", "data_sha256", "extra_data", "jobset_args")}}
    try:
        if rate > args.max_rate:
            raise SystemExit(f"pod rate {rate}/h above --max-rate")
        ip, port = wait_for_ssh(pod_id, started + 60 * args.ready_minutes)
        rp.record({"event": "ssh_ready", "pod_id": pod_id, "seconds_after_create": round(time.time() - started)})
        base = ssh_base(ip, port)
        subprocess.run(base + ["mkdir -p /root/neural-shrimp && tar -x -C /root/neural-shrimp"], input=code,
                       check=True, timeout=600)
        subprocess.run(base + ["mkdir -p /root/neural-shrimp-data && tar -x -C /root/neural-shrimp-data"], input=data,
                       check=True, timeout=args.upload_timeout)
        rp.record({"event": "uploaded", "pod_id": pod_id, "archive_bytes": len(code), "data_bytes": len(data)})
        with open(out / "runner-stdout.txt", "w") as log:              # streamed: readable while the run goes
            remote = f"bash /root/neural-shrimp/tools/runpod_soft_runner.sh {args.jobset} {args.jobset_args}".strip()
            token = hf_token() if args.jobset in GATED_JOBSETS else None
            if token:            # gated downloads: the token travels on stdin, never on a command line or in a log
                remote = "IFS= read -r HF_TOKEN; export HF_TOKEN; " + remote
            proc = subprocess.Popen(base + [remote], stdin=subprocess.PIPE if token else subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
            if token:
                proc.stdin.write((token + "\n").encode())
                proc.stdin.close()
            try:
                proc.wait(timeout=60 * args.run_minutes)
            except subprocess.TimeoutExpired:
                proc.kill()
                rp.record({"event": "run_timeout", "pod_id": pod_id, "minutes": args.run_minutes})
        text = (out / "runner-stdout.txt").read_text()
        remote_hash = next((line.split()[0] for line in text.splitlines() if line.endswith("/root/results.tgz")), None)
        if remote_hash is None:                                   # timed out or failed: pack what exists
            pack = subprocess.run(base + ["tar czf /root/results.tgz -C /root/neural-shrimp-data "
                                          "explore/llm-soft-loop/pod64 && sha256sum /root/results.tgz"],
                                  capture_output=True, text=True, timeout=600)
            remote_hash = (pack.stdout.split() or [None])[0]
        subprocess.run(["scp", "-i", str(KEY), "-P", str(port), "-o", "BatchMode=yes",
                        "-o", f"UserKnownHostsFile={rp.DATA / 'runpod' / 'known_hosts'}",
                        f"root@{ip}:/root/results.tgz", str(out / "results.tgz")], check=True, timeout=900)
        local_hash = hashlib.sha256((out / "results.tgz").read_bytes()).hexdigest()
        result.update(runner_returncode=proc.returncode, remote_sha256=remote_hash, local_sha256=local_hash,
                      hash_verified=remote_hash == local_hash)
        rp.record({"event": "retrieved", "pod_id": pod_id, "hash_verified": remote_hash == local_hash,
                   "sha256": local_hash})
        if remote_hash == local_hash:
            with tarfile.open(out / "results.tgz") as tar:
                # raw outputs are never overwritten: if any retrieved file already exists locally (the shared pod
                # logs excepted), the whole archive goes to the run's own folder instead, and the clash is reported
                clash = [m.name for m in tar.getmembers() if m.isfile() and (rp.DATA / m.name).exists()
                         and not m.name.startswith("explore/llm-soft-loop/pod64/logs/")]
                if clash:
                    result["extract_clash"] = clash[:50]
                    tar.extractall(out / "extracted", filter="data")
                else:
                    tar.extractall(rp.DATA, filter="data")
    finally:
        rp.cmd_terminate(argparse.Namespace(pod=pod_id))
        result["wall_seconds"] = round(time.time() - started)
        (out / "result.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

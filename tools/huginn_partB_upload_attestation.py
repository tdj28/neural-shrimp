"""Rebuild the Part B upload archives (code: git archive of the uploaded commit with the orchestrator's UPLOAD_PATHS
at launch; data: tar of its DATA_PATHS plus --extra-data, built exactly as tools/runpod_soft_float64.archives) and
compare their sizes with the sizes recorded at launch."""
import ast, hashlib, io, json, subprocess, sys, tarfile
from pathlib import Path

ROOT = Path("/data2/neural-shrimp")
DATA = Path("/data2/neural-shrimp-data")
RUNS = [  # run, orchestrator revision at launch, uploaded commit, recorded code_bytes, data_bytes, extra data
    ("TB1 partB-0 (pod gscnb3fjkt93ep)", "82a4136d", "9805ed32055e7ef790df0cc67316357c1067aef0", 460800, 6103040,
     ["explore/gsm8k"]),
    ("TB2 partB-100 (pod ps5pnfy9xnitr1)", "82a4136d", "9805ed32055e7ef790df0cc67316357c1067aef0", 460800, 6103040,
     ["explore/gsm8k"]),
    ("TB3 partB-96 (pod 2la4td179fx2kf)", "62e0a33e", "705e8dc9dd8e6ced6bc4a1a8b01b8d6ddf3da34e", 471040, 6103040,
     ["explore/gsm8k"]),
]


def constants(rev):
    src = subprocess.run(["git", "show", f"{rev}:tools/runpod_soft_float64.py"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and \
                node.targets[0].id in ("UPLOAD_PATHS", "DATA_PATHS"):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def sha(b):
    return hashlib.sha256(b).hexdigest()


report = []
for name, rev, commit, code_bytes, data_bytes, extra in RUNS:
    c = constants(rev)
    code = subprocess.run(["git", "archive", "--format=tar", commit, *c["UPLOAD_PATHS"]], cwd=ROOT,
                          capture_output=True, check=True).stdout
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for rel in (*c["DATA_PATHS"], *extra):
            tar.add(DATA / rel, arcname=rel)
    data = buf.getvalue()
    members = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.startswith("explore/gsm8k/"):
                members[m.name] = sha(tar.extractfile(m).read())
    code_files = {}
    with tarfile.open(fileobj=io.BytesIO(code)) as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name in ("tools/huginn_relax_eval.py", "tools/runpod_soft_runner.sh"):
                code_files[m.name] = sha(tar.extractfile(m).read())
    report.append({"run": name, "orchestrator_revision_at_launch": rev, "uploaded_commit": commit,
                   "code_archive": {"rebuilt_bytes": len(code), "recorded_bytes": code_bytes,
                                    "bytes_match": len(code) == code_bytes, "sha256": sha(code),
                                    "members_sha256": code_files},
                   "data_archive": {"rebuilt_bytes": len(data), "recorded_bytes": data_bytes,
                                    "bytes_match": len(data) == data_bytes, "sha256": sha(data),
                                    "paths": [*c["DATA_PATHS"], *extra], "gsm8k_members_sha256": members}})
print(json.dumps(report, indent=1))

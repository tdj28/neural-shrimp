"""Single all-synthetic complete-shape cost check, not target data."""
import importlib.util
import json
import resource
import time
from pathlib import Path

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("synthetic_fixtures", here / "test_reduce.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rows, progress, problems = module.fixture()
start = time.perf_counter()
report = module.r.calculate(rows, progress, problems, draws=10000)
serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
metadata = {"scope": "synthetic complete832+54+208 records; eight10000-draw endpoints; no target file/model/producer access",
            "elapsed_through_serialization_seconds": time.perf_counter() - start,
            "observed_rss_through_serialization_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "serialized_science_report_bytes": len(serialized.encode()), "coverage": report["coverage"]}
print(json.dumps(metadata, indent=2))

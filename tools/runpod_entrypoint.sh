#!/usr/bin/env bash
set -euo pipefail
set +x
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# No API keys, cloud calls, installation, or moving branch checkout.
# All identity, approval, quotation and hash arguments must be supplied explicitly.
exec python3 -B "${SCRIPT_DIR}/prepare_runpod.py" run --bundle-root "${SCRIPT_DIR}/.." "$@"

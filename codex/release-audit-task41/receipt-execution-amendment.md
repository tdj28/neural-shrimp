# Receipt-only launcher amendment, 20 October 2026

The completed README scan and eight selected tests are unchanged.

The first invocation of `newcomer_receipt.py` used an enclosing 1 GiB
address-space limit, 60 CPU seconds, 90 wall seconds, one numerical thread
and nice 10. The host invocation returned exit 2 immediately with a Go
runtime allocation failure in the launcher. The terminal tool retains
that failure; no `newcomer-evidence.json` was created. The error output is
not a model or scientific-test result.

For a distinct receipt-only invocation, increase the enclosing address-space
limit to 4 GiB, retaining the other limits and the identical checkpointed
source `bbf3fd5062ef0accafff30a510f2e98f45ad6508`.
The successful saved-data audit already used this address-space envelope
with the required host `uv` launcher. The receipt checker only reads completed
outputs, reads test logs, validates the Git tag and hashes downloaded files;
it does not call the model or repeat the scan or tests.

Separately, the first sandboxed `uv` preparation call earlier in Task 41
encountered the host's Snap/AppArmor launch restriction. Subsequent required
`uv` calls used authorized host execution. These host-wrapper limitations
are disclosed and are not classified as failures of the project's README.

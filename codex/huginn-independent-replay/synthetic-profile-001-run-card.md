# Task 8 synthetic full-size cost profile 001

Approved by Root on13 October2026 after complete source/plan/test read. The
statistical reducer was independently compared to the frozen9805ed32 contract;
45 combined pure controls pass0.72s. This profile reads no target evidence and
runs no producer code, tokenizer, model or network operation.

- Plan: synthetic-profile-001-plan.json, SHAa7835860f336319ee8437c26bf6a0a7b1a9c57b2e1071a39fd57ff3dc8ab3e49.
- Reducer: statistics.py, SHA0a76f6ad2949d2234aa560ec4e9267f4a3fe429332b1c33c1b85d3551498581c.
- Driver: profile.py, SHA551e80dd18eb57f9fbb5b5b0b7602e958676b94dcedbee53f347ab55f6006b59.
- Output: /data2/neural-shrimp-data/codex/huginn-independent-replay/synthetic-profile-001, created exclusively.
- Scope: deterministic invented A64x4 with2000 bootstrap draws and B200x4 with10000 draws; both complete outputs retained. Their numerical values are synthetic, never scientific evidence.
- CPU: one worker plus one supervisor reservation, nice10, OMP/OpenBLAS/MKL/NumExpr threads1. No parallel children. Task11 has confirmed it will wait for closure; Task1's seven-slot reservation is unchanged.
- Hard worker AS4GiB; external wall watchdog120s with TERM followed by at most5s cleanup grace. Driver also checks elapsed time and memory floor every1s via SIGALRM.
- Start only with MemAvailable>=20GiB; runtime floor16GiB. Capture fresh CPU/RAM/disk and read-only GPU inventory before dispatch; GPU stays untouched.
- Privately checkpoint source/plan/card/log first. Verify exact Git blobs and remote ancestry, pass exact sourcecommit/planSHA environment, and retain original worker PID/birth identity before source execution. Source bytes checked before and after.
- Preserve any refusal, exception, timeout and partial outputs. No automatic retry. Verify original worker absence after the reaped wrapper. Retain stdout/stderr, tool exit status, elapsed/maxRSS and sampled headroom; no future target replay follows automatically.

The external dispatch uses the frozen plan/output above with
`timeout --signal=TERM --kill-after=5s 120s`, `nice -n10`, hard
`prlimit --as=4294967296:4294967296 --cpu=120`, and the project Python. A small
stdlib bootstrap records its own PID/start/nice identity, then runs the exact
hash-checked profile path in the same process. No target computation occurs in
that bootstrap. The owning tool waits/reaps the wrapper and checks the recorded
worker identity after completion. Actual input admission/replay is a later
reviewed phase requiring immutable retrieval and termination receipts.

# Task 18 run-002: narrow operational recovery review

Disposition: accept the concrete recovery source and outer dispatch, subject to the ordinary reviewed-plan admission, private checkpoint, fresh preflight, and subsequent actual startup/closure checks. This is not an execution or numerical-result acceptance. No target computation, process action, pod access, or Git mutation was performed in this review.

The original run-001 receipt SHA-256 `353d1ad1c3dbd5acb805a60c2b8972d24ba175de512aa7511100bfb2ecadab0c` and saved closure report `c62624fcaa02df4a72b4717c8703d7a92d229cd0f405a242535c21893f1fa9aa` were read. They retain the startup BrokenPipeError, worker termination with exit -15, no numerical result, and absence of both original recorded births/groups/sessions. The outer tool's exit 0 was dispatch completion, not experiment success. These records must remain unchanged.

All six files in `/tmp/task18-recovery-002/release-manifest.json` matched their declared byte counts and hashes. Direct source comparison established that run.py and launch.py are identical to the captured run-001 sources after replacing only the literal string `run-001` with `run-002`. Analysis, original ManualQwen, and ownership-helper bytes are unchanged. The new prepared plan differs from the admitted old plan only in its admission field and the worker hash induced by that output-path literal. Scientific parameters, tolerances, seeds, operation counts, and resource caps are unchanged.

Reviewed hashes:

- run.py `80bf95a99738cc05e4d3480994051fc684666230a4e06aebca5719af1b1d23c0`
- launch.py `980275e4d82728f49b004e79ecaaa42e57e283a348cc0fb4862213650d2cda4e`
- prepared plan `cc5aa9e3938f34f68aaaec66372603877febddc7f0995092637b5039dc8e7c69`
- prepared card `24ccb53baa31ceb1d09d4d5aab465ee6fd9795b1a60122a2229299f1b599943f`
- run-002-dispatch.sh `5f126aa40d247aa85fd546cc00f53a6cf048808315bd3e1ee300309ce99cf710`
- metadata-only preflight `7fc17e94d0cd1405bc7c0b6a222849f48b44742b38d71ad20d714bc59ee84ac3`

The outer script creates a fresh supervision directory, copies itself and its source/plan arguments, then redirects stdout and stderr to persistent files before `exec setsid --wait nice ... launch.py`. Those descriptors belong to regular retained files rather than the tool's output pipe. Therefore the startup and final launcher prints no longer depend on the tool pipe's lifetime. When setsid must fork because it is a process-group leader, `--wait` keeps the caller alive to wait for the launched session and return its exit status; when it can directly execute, the executing process remains the tool child. The launcher independently retains its worker Popen handle, birth-bound group/session identity, deadline, and bounded cleanup. None of that lifecycle source changed.

The new preflight binds dispatch bytes to the exact Git source, verifies the pinned original closure, refuses an existing run-002 or supervision root, and requires clean source/private ancestry and fresh capacity. The card's placeholder for the separately pinned preflight should be resolved in the final admission record, without changing the accepted scientific design. Final closure should retain the actual waiter identity if a fork occurred, the tool session's eventual exit status, both persistent launcher logs, and the actual launcher/worker closure. The small outer logs live outside the worker attempt inventory and therefore need their separate supervision inventory. No further source defect was found in the requested recovery scope.

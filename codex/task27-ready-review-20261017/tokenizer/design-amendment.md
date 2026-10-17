# Tokenizer import recovery profile-002 — prepared, 17 October 2026

This amendment preserves profile-001 and its failed attempt. It changes dependency admission and the bounded resource ceiling, not the tokenizer path, decoder algorithm, answer extraction, conditions, row selection, controls or scientific comparisons. No package import, profile, control, target decode, model or GPU operation has been run during this preparation; Python syntax was parsed only.

## Why the original profile stopped

The original run under source9775a9b6638ee255ad7723649846e6c3849c095a stopped before controls or decoding: AutoTokenizer imports PreTrainedConfig, GenerationConfig and logits_process, which imports the installed Torch. `libtorch_cpu.so` failed to map under the worker's1GiB address-space limit. The full failure SHAabf01061 and receipt68da91e3 are pinned in import-supplement.json. Inference observed exit1, reaping/group absence and external1.260671972s. Original files are unchanged. This is an import/resource failure, not evidence about the roundtrip or decoder differences.

Pinned Transformers5.19 import_utils.py194–202 discovers installed Torch>=2.5 and does not honor USE_TORCH. Its TRANSFORMERS_DISABLE_TORCH_CHECK flag controls a tensor-assertion helper only. Thus the original design's assertion that all backends were disabled was false for this package; after increasing AS, its blanket no-Torch-module assertion would still refuse. The original document remains intact with this explicit correction. The static diagnosis note's provisional8GiB proposal is superseded by Root's explicit12GiB recovery ceiling.

## Exact source delta

- `LIMITS.address_space_bytes`:1GiB →12GiB; fixed output parent run-001 →run-002. Fresh profile is run-002/tokenizer-profile, with profile-002-card.json. Future actual is run-002/tokenizer with prepared run-002-card.json; it remains unadmitted pending profile measurement.
- Replace the ineffective USE_TORCH=0 environment entry with CUDA_VISIBLE_DEVICES=''. Retain local-files-only, trust_remote_code=False, offline/telemetry and single-thread settings. Other historical framework environment entries remain but are not claimed to disable the installed Torch route.
- Replace the prohibition on an imported Torch module with a passive initialization-state record. Before load, after load/before bulk decode, and after bulk decode, record whether Torch is imported, its module/version, and `torch.cuda.is_initialized()` if imported. This function reads the existing flag; no availability/device-count query is made. Any observed initialized state or nonempty CUDA visibility refuses. The loader's finite roundtrip is inside the before/after-load interval.
- Add these observations and the limited runtime-attestation scope to the report. All existing roundtrip/decoder/row/count/comparison semantics and completion criteria stay unchanged. Allowing Torch import does not permit calling model/weight loaders, generation, tensor arithmetic, CUDA init/allocation or GPU queries.

Installed Torch metadata version2.14.1 and source version2.14.1+cu130 are newly bound, along with eight small import/availability source and metadata files. Package import may map CUDA libraries even with no GPU work; no claim is made that this is a CPU-only binary distribution. Passive false CUDA initialization is a bounded runtime check, not a proof that no CUDA library was mapped. The supplement and existing inputs deliberately do not attest every native/transitive binary byte or original historical runtime.

No package downgrade/install, Hub-ID substitution, alternate tokenizer, custom legacy decoder, module monkeypatch or import-hiding trick is used. The exact local snapshot and both AutoTokenizer calls/conditional generic fallback remain as before.

## Concrete prepared cards and bounds

Both new cards remain status=prepared and refuse execution. Inference owns exact adoption/private checkpoint, supervisor and any admission. Proposed profile002: one CPU/nice10,55CPU seconds/60external wall,12GiB hard worker AS,64MiB science-output+streams,128MiB declared input ceiling; startup available RAM28GiB, runtime floor16GiB, disk free16GiB. Parent supervisor softAS1GiB/hardAS12GiB is required so the child can receive12GiB. The separately prepared supervisor is Inference's responsibility; card metadata itself does not apply kernel limits. No hard RSS limit or measured peak is claimed.

The12GiB ceiling is a bounded, unmeasured recovery allowance, not a success guarantee. Static large shared-object sizes explain why original1GiB AS was unsuitable but are not summed live memory or a measured forecast. The first failed attempt supplies only failure cost; it supplies no full decode cost. Any further failure is retained without automatic retries/escalation. The unchanged55/60s and64MiB caps retain a finite stop.

Common pins now44 files/53,926,710B; actual-only18 files/2,969,707B. Common includes the added import-source supplement, prior preparation/diagnosis and original failure/receipt. Only metadata/source files were hashed during this preparation; target records/publication leaves remain unopened. The actual mode still has a separate prepared card and cannot inherit resource approval automatically.

Profile002 runs the unchanged nine focused manufactured groups once, followed by the unchanged512 x4096-ID manufactured encode/decode/serialization exercise. None of those controls ran in001. Report/rows retain full current-runtime decoder traces, with no requirement that the legacy roundtrip fail or that later published counts agree. Outputs remain controls.json,rows.json,report.json,manifest.json; failure.json on failure. Computational completion remains distinct from scientific acceptance. Actual mode does not repeat controls and remains separately reviewable after measured profile002.

## Ownership and immutable scope

The old source/card/design/release and failure stay at their original paths. NEW preparation is only `/tmp/task27-tokenizer-recovery-prep`. `run.diff` and both card diffs identify every change. controls.py and pm_core.py are byte-identical to the original accepted preparation. No tests or model/runtime calls were executed here. Root must review exact new bytes before Inference makes a status-only admission/checkpoint and launches the sole bounded recovery profile.

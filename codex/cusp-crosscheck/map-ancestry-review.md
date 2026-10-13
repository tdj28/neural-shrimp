# Task14 declared-map ancestry: metadata/source check

13 October 2026. The archived curve metadata identifies the same declared
full-state soft-feedback map as the accepted independent Task7 implementation,
with return period **7** for this new family. No model or producer tool was
constructed or executed; no state/coefficient array values or Part A data were
read. This is input-recipe qualification, not a numerical endpoint result.

`hub7-foldS-plus/curve.json`, SHA256
`0bf6f562b6343e21e7fe35a55c109407d92e0a1cc02983607f8ea28ce408e321`,
points to `llm-soft-loop/pod64/hverify/branch-hub7-Tdown`. Its `branch.json` is
62,053 bytes, SHA256
`afff7e464e79a4802fca65308452e716b8a19a34f2491c0e660985fd9b83ccba`.
Those bytes equal the unique member
`explore/llm-soft-loop/pod64/hverify/branch-hub7-Tdown/branch.json` in the
already admitted V11 returned archive, SHA256
`9df8795f541bc3717efb643a4bba7514a649f203369af17407367c3a65b1255b`.
Both records declare period7; the branch's input period0 means automatic
initial detection, not the final return-map period.

The branch records model=`qwen0.5b`, layer=12, prompt=`open`, opener=`Hello!`,
window=1, vectors=`llm-phase-map/vectors-l12-p1b`, and
concept=`hostility-vs-kindness`. The source `curve` function explicitly inherits
these fields (`explore_llm_soft_loop.py:606–611`). The curve uses alpha_scale10,
so theta=(log10T,10 alpha). Physical alpha derivatives require that conversion.
The producer branch and curve declare CUDA/float64; the independent check uses
the accepted CPU/all-float64 implementation, with measured precision gates.

The pinned producer source resolves qwen0.5b to Qwen2.5-0.5B-Instruct and forms
the same system/user chat prompt with an assistant generation prefix.
`explore_llm_phase_map.py:78–97` adds the unrenormalized vector after zero-based
decoder block12. `load_vectors` at280–282 loads the selected NPZ array unchanged;
`soft_family` applies steering to both the prefix and live state and reads out
all vocabulary probabilities. These operations match the independent recipe.
The supplemental producer phase-map source SHA256 is
`28234a411b4012ee85d28216fc3cd7b72df192a32d76f3322787126bddd71d5a`,
from commit cb6a105c; the other exact source hashes are in the retained report.

The current tokenizer/config metadata match the previously pinned small asset
hashes, and the hostile NPZ matches
`d18395239dba63c7deacb8aa9431d768269617c9cc44a98f89cc1e22423b5958`.
The independent source's37 integer IDs decode, using only stored vocabulary
metadata, to the exact expected system/open + user/Hello! + assistant-prefix
text. No new BPE encoding or tokenizer construction was performed.

**Authority limit:** the producer branch metadata does not itself store resolved
checkpoint/tokenizer revisions or its actual37 IDs. The returned result archive
also does not contain the vector NPZ; the original upload metadata names its
folder. Therefore this check establishes declared recipe agreement and current
pinned-asset consistency, not a newly reconstructed proof of every historical
producer asset byte. The future independent loader must verify its actual
checkpoint/config/vector pins and use the recorded37 IDs. The previous accepted
Task7 map authority remains the basis for those constants.

Retained source-only audit:

- `/tmp/task14-map-ancestry-check.py`: `6419acca3bada0f52205c79031f4128dd19f1faebfbc7c6429c308a661392a58`.
- `/tmp/task14-map-ancestry-check.json`: `35c41bf933941cec2d7a4ac271a05c60383138460316bf9b3f3c89f7512cf82f`.

The own checker passed in0.20s at nice10, using only stdlib metadata parsing,
source-AST literals and byte hashing. No private cost fields are copied into the
report. No target/profile launch or coefficient conclusion follows from this
check.

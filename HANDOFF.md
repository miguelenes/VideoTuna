<!-- Paste into Augment Chat -->

# Handoff from Cursor
> 10 messages | ~344 tokens | Projects/VideoTuna | branch `main`
>
> Conditional Residual Handoff — transmits what the repo can't tell you (decisions, dead-ends, constraints, uncommitted diff), not the code itself.

## ⚡ Paste this first

Continue in Augment. Use the file list and next actions to resume the implementation.

```text
I'm resuming a previous Cursor session on Projects/VideoTuna. You have the repository — read it for anything not stated here. This handoff carries only what the code itself cannot tell you.

TASK
Pin SimpleTuner upstream SHA on next sync
Migrate third_party/flux/ → videotuna/vendor/simpletuner/ via submodule
Remove cogvideo_sat after SAT deprecation
First-party Flux LoRA trainer to drop the 71-file snapshot
Original task: @/home/menes/.cursor/projects/home-menes-Projects-VideoTuna/terminals/10.txt:9-239

STATE
- Branch `main` · 25 uncommitted file(s)
UNCOMMITTED (in-flight — not on HEAD, you can't see this by reading committed code)
- M README.md
- M poetry.lock
- M pyproject.toml
- M tests/conftest.py
- M tests/test_import_smoke.py
- M uv.lock
- D videotuna/models/flux/__init__.py
- D videotuna/models/flux/__main__.py
- D videotuna/models/flux/api.py
- D videotuna/models/flux/cli.py
- D videotuna/models/flux/flux_math.py
- D videotuna/models/flux/model.py
- D videotuna/models/flux/modules/autoencoder.py
- D videotuna/models/flux/modules/conditioner.py
- D videotuna/models/flux/modules/layers.py
- D videotuna/models/flux/sampling.py
- D videotuna/models/flux/util.py
- D videotuna/third_party/flux/convert_parquet_to_images.py
- M videotuna/third_party/flux/data_backend/factory.py
- D videotuna/third_party/flux/training/quantisation/peft_workarounds.py
NEXT
- Run the relevant build, lint, or test command before calling the handoff complete.
- Preserve existing user changes and avoid reverting unrelated work.
VERIFY
- No verification command was captured — run the project build/lint/test before finishing.

SYNTHESIS — before you change anything, restate in one line: (a) the task, and (b) the one constraint you must not break. Then proceed.
```

---

## 🧠 Decision log

_No explicit decisions were captured in the transcript._

## 🛑 Dead-ends — do not redo

_None captured._

## 📌 Constraints

_None explicitly stated._

## 🔀 In-flight (uncommitted) state

Branch: `main`

Uncommitted changes:
- `M README.md`
- `M poetry.lock`
- `M pyproject.toml`
- `M tests/conftest.py`
- `M tests/test_import_smoke.py`
- `M uv.lock`
- `D videotuna/models/flux/__init__.py`
- `D videotuna/models/flux/__main__.py`
- `D videotuna/models/flux/api.py`
- `D videotuna/models/flux/cli.py`
- `D videotuna/models/flux/flux_math.py`
- `D videotuna/models/flux/model.py`
- `D videotuna/models/flux/modules/autoencoder.py`
- `D videotuna/models/flux/modules/conditioner.py`
- `D videotuna/models/flux/modules/layers.py`
- `D videotuna/models/flux/sampling.py`
- `D videotuna/models/flux/util.py`
- `D videotuna/third_party/flux/convert_parquet_to_images.py`
- `M videotuna/third_party/flux/data_backend/factory.py`
- `D videotuna/third_party/flux/training/quantisation/peft_workarounds.py`
- `?? .gemini/`
- `?? .jolli/`
- `?? docs/vendor-policy.md`
- `?? tests/test_flux_training_config.py`
- `?? videotuna/third_party/flux/VENDOR.md`

```
README.md                                          |   27 +-
 poetry.lock                                        | 1070 +++-
 pyproject.toml                                     |   88 +-
 tests/conftest.py                                  |    6 +-
 tests/test_import_smoke.py                         |   37 +-
 uv.lock                                            | 5379 +++++++++++++++++++-
 videotuna/models/flux/__init__.py                  |   11 -
 videotuna/models/flux/__main__.py                  |    4 -
 videotuna/models/flux/api.py                       |  200 -
 videotuna/models/flux/cli.py                       |  272 -
 videotuna/models/flux/flux_math.py                 |   32 -
 videotuna/models/flux/model.py                     |  126 -
 videotuna/models/flux/modules/autoencoder.py       |  338 --
 videotuna/models/flux/modules/conditioner.py       |   45 -
 videotuna/models/flux/modules/layers.py            |  278 -
 videotuna/models/flux/sampling.py                  |  140 -
 videotuna/models/flux/util.py                      |  210 -
 .../third_party/flux/convert_parquet_to_images.py  |   44 -
 videotuna/third_party/flux/data_backend/factory.py |    9 +-
 .../flux/training/quantisation/peft_workarounds.py |  421 --
 20 files changed, 6310 insertions(+), 2427 deletions(-)
```

## 📁 Files in play (pointers — read the live files, this is just the index)

Modified:
_none captured_

Read / explored:
_none captured_

## ⌨️ Commands run

_none captured_

**Verify:** _none captured — run build/lint/test before finalizing._

## 🎯 Task

**Continue:** Pin SimpleTuner upstream SHA on next sync
Migrate third_party/flux/ → videotuna/vendor/simpletuner/ via submodule
Remove cogvideo_sat after SAT deprecation
First-party Flux LoRA trainer to drop the 71-file snapshot

**Original request:** @/home/menes/.cursor/projects/home-menes-Projects-VideoTuna/terminals/10.txt:9-239

## 💬 Recent exchange (tail)

**You**: Provide me with 3 comprehensive prompts, to run in plan model to setup amdu rocm support, imrpove nvidia support and use cpu. Also, be thorough on how to improve integration with the current system.

**You**: This is too slow poetry run pytest tests/test_diffusers_video_flow.py

**You**: @videotuna/third_party Is there a better way than doing this in our repo ? Provide me with a prompt to re-organize and improve the dependencies, management, etc

**You**: Consume this article https://bitmovin.com/blog/ai-video-research/ , suggest me 10 improvements you would do on this codebase based on the information.

**You**: Provide me with 3 comprehensive prompts, to run in plan mode to setup amdu rocm support, imrpove nvidia support and use cpu. Also, be thorough on how to improve integration with the current system.

**You**: Pin SimpleTuner upstream SHA on next sync Migrate third_party/flux/ → videotuna/vendor/simpletuner/ via submodule Remove cogvideo_sat after SAT deprecation First-party Flux LoRA trainer to drop the 71-file snapshot

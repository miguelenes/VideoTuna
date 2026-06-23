# PrivTune — Contributing

## Coding conventions

### Code style

- **Ruff:** Select rules `E`, `F`, `C90`, `I`; line-length = 88; target = py311; max-complexity = 19
- **isort:** First-party known = `["videotuna"]` (enforced via ruff `I` rule)
- **Indentation:** 4 spaces, LF line endings, UTF-8 (see `.editorconfig`)
- **Comments:** Never add code comments unless explicitly requested
- **Emojis:** Never add emojis to files unless explicitly requested

### Device handling

- Use `videotuna/utils/device_utils.py` (`detect_compute_backend()`, `resolve_inference_device()`) for all compute backend detection
- Never call `torch.cuda.is_available()` directly in flow code

### Settings

- All environment configuration goes through `videotuna/settings.py` (`PrivTuneSettings`, pydantic-settings)
- Environment variables use `VIDEOTUNA_*` prefix. No `PRIVTUNE_*` aliases.
- Add new settings by adding a field to `PrivTuneSettings` with `env_prefix` already configured

### CLI

- All CLI commands use **cyclopts** framework
- Follow the pattern in `videotuna/cli/inference_app.py` / `videotuna/cli/inference_options.py`
- New inference entry points: register in `pyproject.toml` `[tool.poetry.scripts]`, wire through `videotuna/cli/inference_app.py`

### Imports

- Import path: `videotuna.*` (Poetry package name is `privtune`, but import path is `videotuna`)
- First-party imports sorted via ruff `I` rule (isort-compatible)
- **Vendor boundary:** First-party code must not import `easydict` (used only by vendor configs in `videotuna/models/wan/wan/configs/`)

### Configs

- Training configs: `configs/domain/` (JSON for Flux, YAML for Wan)
- Inference presets: `configs/inference/presets/` (YAML)
- Configs use Pydantic validation schemas defined alongside the respective trainers

### Two stacks rule

- Flux T2I and Wan T2V/I2V use different training stacks ([ADR-001](docs/decisions/0001-dual-training-stacks.md))
- **Do not** refactor one phase to match the other's stack without superseding ADR-001
- **Never mix** `wanvideo.py` (Lightning/Wan 2.1) with `diffusers_video.py` (Diffusers/Wan 2.2+Flux) in the same execution path

## Branching and PR conventions

- Branch from `main`
- PRs merge into `main`
- CI runs on every PR (CPU smoke tests + lint + coverage gate)
- GPU nightly tests run on a schedule (Monday 08:00 UTC)
- Ruff auto-fixes are applied by CI on PRs from the same repository

## Testing expectations

- All changes must pass `poetry run lint && poetry run format-check && poetry run coverage-gate`
- New features should include or update tests
- Follow existing test patterns (pytest, appropriate markers)
- GPU-only logic should use `@pytest.mark.gpu`
- ROCm-specific logic should use `@pytest.mark.rocm`
- Add to `CI_SMOKE_TESTS` list in `scripts/__init__.py` if the test should run in CI
- Coverage floor: 35% on training/ + utils/ + flow/ + cli/

### Test by change area

| Change area | Test files to run |
|-------------|-------------------|
| Wan 2.2 presets / bridge | `test_wan_inference_presets.py`, `test_wan_lora_bridge.py`, `test_wan_i2v_lora_bridge.py` |
| `diffusers_video` flow | `test_diffusers_video_flow.py` |
| Device / attention | `test_device_utils.py`, `test_attention_backend.py` |
| Inference CLI / memory | `test_inference_optimization.py` |
| Flux LoRA training | `test_flux_lora_train_smoke.py`, `test_flux_lora_features.py` |
| Wan training | `test_wan_training_step.py`, `test_wan_train_smoke.py` |
| Config validation | `test_domain_finetune_configs.py` |
| Import boundaries | `test_import_smoke.py`, `test_vendor_import_boundary.py` |

## Formatting expectations

- Run `poetry run format` before committing to auto-format
- CI runs `poetry run format-check` and will fail on unformatted code
- Ruff auto-fixes (`--fix`) are applied by CI on same-repo PRs

## Review/merge hygiene

- Scoped diffs: change only what the task requires. No incidental refactoring.
- Never commit `.env`, checkpoints, `outputs/`, `results/`, `data/`, weights, or secrets
- Read `docs/vendor-policy.md` before modifying vendored code (`videotuna/models/wan/wan/`)
- For Wan LoRA bridge changes, verify remap coverage >= 90% (`test_wan_lora_bridge.py`)
- Update `docs/MODEL_VERSIONS.md` when upgrading model versions or dependencies

## Validation checklist (pre-merge)

```shell
poetry run lint
poetry run format-check
poetry run coverage-gate
```

Additional checks for specific changes:

```shell
# Wan bridge changes
poetry run test tests/test_wan_lora_bridge.py -q
poetry run test tests/test_wan_i2v_lora_bridge.py -q

# Config changes
poetry run test tests/test_domain_finetune_configs.py -q

# Flux training changes
poetry run test tests/test_flux_lora_train_smoke.py -q

# Wan training changes
poetry run test tests/test_wan_training_step.py -q
```

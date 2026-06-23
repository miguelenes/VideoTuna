# PrivTune — Agent Notes

## Rules AI agents must follow

1. **Always use `poetry run <command>`** — never invoke Python scripts directly. Exceptions: `poetry run pytest <path>` for single-file tests.
2. **Device detection:** Always use `videotuna/utils/device_utils.py`. Never write `torch.cuda.is_available()` in flow code.
3. **Settings:** All env config through `videotuna/settings.py` (`PrivTuneSettings`). Settings are `VIDEOTUNA_*` prefix, not `PRIVTUNE_*`.
4. **Never add comments** to code unless explicitly requested.
5. **Never add emojis** to files unless explicitly requested.
6. **Scoped diffs:** Change only what the task requires. No incidental refactoring.
7. **Never commit:** `.env`, checkpoints, `outputs/`, `results/`, `data/`, weights, or secrets.

## Safe vs unsafe files

### Safe to modify (normal development)

| Path | Notes |
|------|-------|
| `videotuna/training/flux_lora/` | First-party Flux trainer. Entire tree is safe. |
| `videotuna/training/wan_lora/` | Wan LoRA config only. Safe. |
| `videotuna/flow/diffusers_video.py` | Diffusers inference flow. Safe. |
| `videotuna/cli/` | CLI entry points and options. Safe. |
| `videotuna/utils/` | Most utilities are safe. See exceptions below. |
| `videotuna/data/` | Dataset code. Safe. |
| `videotuna/base/` | Base classes. Safe with caution. |
| `videotuna/settings.py` | Settings schema. Safe. |
| `configs/` | Config files. Safe. |
| `tests/` | Tests. Safe. |
| `scripts/__init__.py` | Poetry command implementations. Safe with caution. |

### Proceed with caution

| Path | Reason |
|------|--------|
| `videotuna/utils/wan_lora_bridge.py` | Critical path between training and validation. Remap coverage must stay >= 90%. Verify with `test_wan_lora_bridge.py`. |
| `videotuna/flow/wanvideo.py` | Lightning/native Wan 2.1 training loop. GPU-only. Deep-seated assumptions. |
| `videotuna/utils/callbacks.py` | `LoraModelCheckpoint`, `WanCheckpoint`, DeepSpeed checkpoint export. Affects all training output. |
| `scripts/__init__.py` | Central command dispatch. Install commands affect system packages. CI smoke test list lives here. |

### Do NOT modify without reading policy

| Path | Policy |
|------|--------|
| `videotuna/models/wan/wan/` | Vendored upstream code. Read `docs/vendor-policy.md` first. Apache 2.0 licensed. |
| `videotuna/models/wan/` | Contains VENDOR.md. Read vendor policy before any change. |

### Never modify

| Path | Reason |
|------|--------|
| `videotuna/models/wan/LICENSE` | Third-party license file |
| `.env` | Personal credentials (should not be committed) |
| `poetry.lock` | Managed by Poetry (`poetry lock` / `poetry update`) |
| `uv.lock` | Managed by uv |

## Common failure modes

### CPU-torch swap needed after `poetry install`

**Symptom:** `torch.cuda.is_available()` errors or CUDA not available warnings on a CPU-only machine.

**Fix:** Every `poetry install` on a CPU-only machine re-installs CUDA torch from the lockfile. Re-run:
```shell
poetry run install-cpu-torch
poetry run verify-cpu-torch
```

### `poetry run test <path>` runs all tests

**Symptom:** Running `poetry run test tests/test_foo.py` runs the entire suite instead of just `test_foo.py`.

**Cause:** The `test` poetry script appends args to `pytest tests`.

**Fix:** Use `poetry run pytest tests/test_foo.py -q` to run a single file.

### Wan bridge remap coverage drop

**Symptom:** `validate-domain-t2v` produces garbage or fails to load trained checkpoints.

**Cause:** Changes to `wan_lora_bridge.py` dropped remap coverage below 90%.

**Fix:** Run `poetry run test tests/test_wan_lora_bridge.py -q` after any bridge change. Coverage target: >= 90%.

### DeepSpeed install fails

**Symptom:** `poetry run install-deepspeed` fails with CUDA/nvcc errors.

**Fix:** Ensure CUDA toolkit is installed and `nvcc` is on `PATH`. On conda: `conda install -c nvidia cuda-toolkit=12.6`. On bare metal: install system CUDA toolkit.

### Flash-attn on ROCm

**Symptom:** `install-flash-attn` errors on AMD GPU.

**Fix:** ROCm does not support flash-attn in PrivTune. Set `VIDEOTUNA_ATTN_BACKEND=sdpa`. Never run `install-flash-attn` or `install-flash-attn-rocm`.

### Pre-existing test failures

Known baseline failures (not environment issues):
- `tests/datasets/test_dataset_from_csv.py` — `PosixPath` bug in `videotuna/data/datasets.py`
- `test_wan_checkpoint.py::test_wan_from_pretrained_missing_dir` — depends on diffusers/network behavior
- `poetry run lint` reports ~1000 pre-existing ruff errors (legacy code, not regressions)

## Areas requiring extra caution

### ROCm (AMD GPU)

- Use `VIDEOTUNA_ATTN_BACKEND=sdpa`. Flash-attn is not supported.
- Wan training requires CUDA. ROCm is inference + Flux training only.
- After `poetry install`, re-run `poetry run install-rocm` (same CPU-torch swap issue as CPU).

### Docker

- The Compose service is `privtune` (not `videotuna` — that's a deprecated alias).
- User mapping via `HOST_UID`/`HOST_GID`.
- Source code is bind-mounted; dependencies are built inside the container.

### Cloud provisioning (Vast.ai)

- Scripts in `cloud/vast/` are for provisioning cloud GPU instances.
- `.env.cloud.example` has cloud-specific environment template.
- Do not modify `cloud/vast/bootstrap.sh` or `provisioning.yaml` without understanding the full provisioning flow.

## How to verify changes

### Standard verification

```shell
poetry run lint
poetry run format-check
poetry run coverage-gate
```

### Testing specific changes

| Change | Verification |
|--------|-------------|
| Any Python change | `poetry run lint`, `poetry run format-check`, `poetry run coverage-gate` |
| Wan bridge change | + `poetry run test tests/test_wan_lora_bridge.py -q` |
| Diffusers flow change | + `poetry run test tests/test_diffusers_video_flow.py -q` |
| Device utils change | + `poetry run test tests/test_device_utils.py -q` |
| Attention backend change | + `poetry run test tests/test_attention_backend.py -q` |
| Inference CLI / memory | + `poetry run test tests/test_inference_optimization.py -q` |
| Flux LoRA training | + `poetry run test tests/test_flux_lora_train_smoke.py -q` |
| Wan LoRA training | + `poetry run test tests/test_wan_training_step.py -q` |
| Config changes | + `poetry run test tests/test_domain_finetune_configs.py -q` |
| Import / vendor changes | + `poetry run test tests/test_import_smoke.py -q` |
| Poetry scripts changes | + `poetry run test tests/test_poetry_scripts.py -q` |

## Preferred and avoided commands

### Preferred commands

| Scenario | Command |
|----------|---------|
| Run tests | `poetry run pytest <path> -q` (for single files) |
| Run all tests | `poetry run test -q` |
| Lint | `poetry run lint` |
| Format check | `poetry run format-check` |
| Auto-format | `poetry run format` |
| Type check | `poetry run type-check` |
| Coverage gate | `poetry run coverage-gate` |
| Coverage report (no gate) | `poetry run coverage-report` |
| DeepSpeed install | `poetry run install-deepspeed` |
| Install CPU torch | `poetry run install-cpu-torch` |
| Verify CPU torch | `poetry run verify-cpu-torch` |

### Commands to avoid

| Command | Why |
|---------|-----|
| `python videotuna/...` | Use `poetry run <script>` or ensure virtualenv is active |
| `pip install ...` | Use `poetry add ...` for permanent deps |
| `torch.cuda.is_available()` in flow code | Use `videotuna/utils/device_utils.py` instead |
| `install-flash-attn` on ROCm | ROCm does not support flash-attn; use `VIDEOTUNA_ATTN_BACKEND=sdpa` |
| Committing `.env` | Secrets must never be committed |
| Modifying `videotuna/models/wan/wan/` directly | Must read `docs/vendor-policy.md` first |

## Related documentation

| Doc | Purpose |
|-----|---------|
| `AGENTS.md` | Canonical agent instructions |
| `ARCHITECTURE.md` | System architecture |
| `CONTEXT.md` | Repo overview |
| `CONTRIBUTING.md` | Coding conventions and PR process |
| `DEVELOPMENT.md` | Setup, build, test commands |
| `docs/runbooks/domain-adult-finetune.md` | Full training runbook |
| `docs/decisions/0001-dual-training-stacks.md` | ADR: dual training stacks |
| `docs/vendor-policy.md` | Vendored upstream code policy |
| `docs/MODEL_VERSIONS.md` | Model version pins |
| `.env.example` | Environment variable reference |

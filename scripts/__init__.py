"""
Poetry commands
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime

current_time = datetime.now().strftime("%Y%m%d%H%M%S")


def _require_cuda_backend(installer_name: str) -> None:
    """Abort when the active PyTorch build is ROCm (CUDA-only installer)."""
    try:
        from videotuna.utils.device_utils import detect_compute_backend

        if detect_compute_backend() == "rocm":
            print(
                f"{installer_name} is not supported on AMD ROCm.\n"
                "Use VIDEOTUNA_ATTN_BACKEND=sdpa for attention on ROCm.\n"
                "See docs/install-rocm.md.",
                file=sys.stderr,
            )
            sys.exit(1)
    except ImportError:
        pass


def install_deepspeed():
    """
    Install DeepSpeed with CUDA 12.6 toolkit support (rebuilds against the active torch).

    When conda is unavailable, skips the CUDA toolkit step and installs via pip.
    If deepspeed>=0.19.2 is already importable, exits successfully without rebuilding.
    """
    _require_cuda_backend("install-deepspeed")
    try:
        import deepspeed
        from packaging.version import Version

        if Version(deepspeed.__version__) >= Version("0.19.2"):
            print(
                f"deepspeed {deepspeed.__version__} already installed "
                "(>= 0.19.2); skipping rebuild."
            )
            return
    except ImportError:
        pass

    if shutil.which("conda"):
        command_install_cuda_toolkit = [
            "conda",
            "install",
            "cuda-toolkit=12.6",
            "-c",
            "conda-forge",
            "-c",
            "nvidia",
            "-y",
        ] + sys.argv[1:]
        result_cuda_toolkit = subprocess.run(command_install_cuda_toolkit, check=False)
        if result_cuda_toolkit.returncode != 0:
            print(
                "conda cuda-toolkit install failed; continuing with pip-only "
                "deepspeed install.",
                file=sys.stderr,
            )
    else:
        print(
            "conda not found; skipping cuda-toolkit install. "
            "If the pip build fails, install CUDA/nvcc or use conda.",
            file=sys.stderr,
        )

    pip = [sys.executable, "-m", "pip"]
    subprocess.run([*pip, "uninstall", "deepspeed", "-y"], check=False)

    env = os.environ.copy()
    env["DS_BUILD_CPU_ADAM"] = "1"
    env["BUILD_UTILS"] = "1"
    result_deepspeed = subprocess.run(
        [*pip, "install", "deepspeed==0.19.2"],
        check=False,
        env=env,
    )
    exit(result_deepspeed.returncode)


def _python_wheel_tag() -> str:
    major, minor = sys.version_info[:2]
    return f"cp{major}{minor}"


def install_flash_attn():
    """
    Install flash-attn for PyTorch 2.6 + CUDA 12.6 (cxx11 ABI wheels).

    Tries a prebuilt wheel first (no compiler or conda required). Falls back to a
    source build only when the wheel is unavailable.
    """
    _require_cuda_backend("install-flash-attn")
    subprocess.run([sys.executable, "-m", "pip", "install", "ninja"], check=False)

    wheel_tag = _python_wheel_tag()
    flash_attn_wheel = (
        "https://github.com/Dao-AILab/flash-attention/releases/download/"
        f"v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-"
        f"{wheel_tag}-{wheel_tag}-linux_x86_64.whl"
    )
    result_wheel = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            flash_attn_wheel,
            "--no-build-isolation",
        ],
        check=False,
    )
    if result_wheel.returncode == 0:
        exit(0)

    if shutil.which("conda"):
        result_nvcc = subprocess.run(
            [
                "conda",
                "install",
                "-c",
                "nvidia",
                "cuda-nvcc=12.6",
                "-y",
            ]
            + sys.argv[1:],
            check=False,
        )
        if result_nvcc.returncode != 0:
            exit(result_nvcc.returncode)
    elif shutil.which("nvcc") is None:
        print(
            "Prebuilt flash-attn wheel install failed and nvcc was not found.\n"
            "Install the CUDA toolkit (nvcc), or use conda:\n"
            "  conda install -c nvidia cuda-nvcc=12.6",
            file=sys.stderr,
        )
        exit(result_wheel.returncode)

    result_flash = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "flash-attn==2.7.4.post1",
            "--no-build-isolation",
        ],
        check=False,
    )
    exit(result_flash.returncode)


_ROCM_TORCH_INDEX = "https://download.pytorch.org/whl/rocm6.2.4"
_CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
_CUDA_ONLY_PACKAGES = (
    "xformers",
    "bitsandbytes",
    "xfuser",
    "triton",
    "nvidia-cublas-cu12",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-cusparselt-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-nvtx-cu12",
)


def install_rocm():
    """
    Install PyTorch 2.6 + torchvision 0.21 for ROCm 6.2.4 and remove CUDA-only wheels.

    Run after: poetry install -E rocm
    """
    pip = [sys.executable, "-m", "pip"]
    for pkg in _CUDA_ONLY_PACKAGES:
        subprocess.run([*pip, "uninstall", pkg, "-y"], check=False)
    result = subprocess.run(
        [
            *pip,
            "install",
            "torch==2.6.0",
            "torchvision==0.21.0",
            "--index-url",
            _ROCM_TORCH_INDEX,
        ],
        check=False,
    )
    if result.returncode != 0:
        exit(result.returncode)
    try:
        from videotuna.utils.device_utils import describe_compute_environment

        print(describe_compute_environment())
    except ImportError:
        import torch

        print(
            f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}, "
            f"hip: {getattr(torch.version, 'hip', None)}"
        )
    exit(0)


def install_cpu_torch():
    """Install CPU-only PyTorch 2.6 wheels (no CUDA/ROCm)."""
    pip = [sys.executable, "-m", "pip"]
    for pkg in _CUDA_ONLY_PACKAGES:
        subprocess.run([*pip, "uninstall", pkg, "-y"], check=False)
    result = subprocess.run(
        [
            *pip,
            "install",
            "torch==2.6.0",
            "torchvision==0.21.0",
            "--index-url",
            _CPU_TORCH_INDEX,
        ],
        check=False,
    )
    exit(result.returncode)


def install_flash_attn_rocm():
    """
    flash-attn is not officially supported on ROCm in VideoTuna.

    Use VIDEOTUNA_ATTN_BACKEND=sdpa instead. See docs/install-rocm.md.
    """
    print(
        "flash-attn is not supported on AMD ROCm in VideoTuna.\n"
        "Use: export VIDEOTUNA_ATTN_BACKEND=sdpa\n"
        "For experimental upstream builds, see "
        "https://github.com/Dao-AILab/flash-attention",
        file=sys.stderr,
    )
    sys.exit(1)


def code_format(check=False):
    """
    Run the code formatting
    """
    commands = [["isort", "."], ["black", "."]]
    return_code = 0

    for command in commands:
        if check:
            command.append("--check")
        process = subprocess.run(command, check=False)
        if process.returncode > 0:
            return_code = process.returncode
            break

    exit(return_code)


def code_format_check():
    """
    Check the code formatting (useful with CI)
    """
    code_format(check=True)


def lint():
    """
    Run the linter
    """
    result = subprocess.run(
        ["ruff", "check", "videotuna", "tests"] + sys.argv[1:], check=False
    )
    exit(result.returncode)


def test():  # pragma: no cover
    """
    Run all unittests
    """
    os.environ["ENV"] = "test"
    result = subprocess.run(["pytest", "tests"] + sys.argv[1:], check=False)
    exit(result.returncode)


def coverage_report():
    """
    Run all unittests with coverage
    """
    os.environ["ENV"] = "test"
    result = subprocess.run(
        ["coverage", "run", "-m", "pytest", "--junitxml", "report.xml"], check=False
    )
    if result.returncode > 0:
        exit(result.returncode)
    result = subprocess.run(["coverage", "report", "-m"], check=False)
    exit(result.returncode)


def type_check():
    """
    Run the type checking
    """
    result = subprocess.run(["mypy", "videotuna", "tests"], check=False)
    exit(result.returncode)


def inference_cogvideo_i2v_diffusers():
    result = subprocess.run(
        [
            "python",
            "scripts/inference_new.py",
            "--config",
            "configs/inference/cogvideox_i2v_5b.yaml",
            "--dtype",
            "fp16",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_cogvideo_i2v_lora():
    savedir = f"results/inference/i2v/cogvideox-i2v-lora-{current_time}"

    result = subprocess.run(
        [
            "python3",
            "scripts/inference_new.py",
            "--config",
            "configs/inference/cogvideox_i2v_5b.yaml",
            "--lorackpt",
            "{YOUR_LORA_CKPT_PATH}",
            "--prompt_dir",
            "{YOUR_PROMPT_DIR}",
            "--savedir",
            savedir,
            "--denoiser_precision",
            "bf16",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_cogvideo_lora():
    savedir = f"results/t2v/{current_time}-cogvideo"
    result = subprocess.run(
        [
            "python3",
            "scripts/inference_new.py",
            "--config",
            "configs/inference/cogvideox_t2v_5b.yaml",
            "--lorackpt",
            "{YOUR_LORA_CKPT_PATH}",
            "--prompt_file",
            "inputs/t2v/prompts.txt",
            "--savedir",
            savedir,
            "--seed",
            "6666",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_cogvideo_t2v_diffusers():
    result = subprocess.run(
        [
            "python",
            "scripts/inference_new.py",
            "--config",
            "configs/inference/cogvideox_t2v_2b.yaml",
            "--dtype",
            "fp16",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_cogvideox1_5_t2v():
    result = subprocess.run(
        [
            "python",
            "scripts/inference_new.py",
            "--config",
            "configs/inference/cogvideox1.5_t2v_5b.yaml",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_cogvideox1_5_i2v():
    result = subprocess.run(
        [
            "python",
            "scripts/inference_new.py",
            "--config",
            "configs/inference/cogvideox1.5_i2v_5b.yaml",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_dc_i2v_576x1024():
    ckpt = "checkpoints/dynamicrafter/i2v_576x1024/model.ckpt"
    config = "configs/002_dynamicrafter/dc_i2v_1024.yaml"
    prompt_dir = "inputs/i2v/576x1024"
    savedir = "results/dc-i2v-576x1024"

    result = subprocess.run(
        [
            "python3",
            "scripts/inference.py",
            "--mode",
            "i2v",
            "--ckpt_path",
            ckpt,
            "--config",
            config,
            "--prompt_dir",
            prompt_dir,
            "--savedir",
            savedir,
            "--bs",
            "1",
            "--height",
            "576",
            "--width",
            "1024",
            "--fps",
            "10",
            "--seed",
            "123",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_flux_schnell():
    command_schnell = [
        "python",
        "scripts/inference_new.py",
        "--config",
        "configs/inference/flux1_schnell.yaml",
    ] + sys.argv[1:]

    result_schnell = subprocess.run(command_schnell, check=False)
    exit(result_schnell.returncode)


def inference_flux_dev():
    command_dev = [
        "python",
        "scripts/inference_new.py",
        "--config",
        "configs/inference/flux1_dev.yaml",
    ] + sys.argv[1:]

    result_dev = subprocess.run(command_dev, check=False)
    exit(result_dev.returncode)


def inference_flux_lora():
    os.environ["lora_ckpt"] = "{YOUR_LORA_CKPT_PATH}"
    result = subprocess.run(
        [
            "python",
            "scripts/inference_new.py",
            "--config",
            "configs/inference/flux1_dev.yaml",
            "--lorackpt",
            os.environ["lora_ckpt"],
            "--savedir",
            "results/t2i/flux-lora/",
            "--enable_sequential_cpu_offload",
            "--enable_vae_tiling",
            "--enable_vae_slicing",
            "--dtype",
            "fp16",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_hunyuan_t2v():
    ckpt = "checkpoints/hunyuanvideo/HunyuanVideo"
    dit_weight = (
        "checkpoints/hunyuanvideo/HunyuanVideo/"
        "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"
    )
    config = "configs/007_hunyuanvideo/hunyuanvideo_t2v.yaml"
    prompt_file = "inputs/t2v/prompts.txt"
    savedir = "results/t2v/hunyuanvideo/720P"
    result = subprocess.run(
        [
            "python3",
            "scripts/inference_new.py",
            "--ckpt_path",
            ckpt,
            "--dit_weight",
            dit_weight,
            "--config",
            config,
            "--prompt_file",
            prompt_file,
            "--savedir",
            savedir,
            "--height",
            "720",
            "--width",
            "1280",
            "--frames",
            "129",
            "--seed",
            "44",
            "--num_inference_steps",
            "50",
            "--enable_vae_tiling",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_hunyuan_t2v_diffusers():
    result = subprocess.run(
        [
            "python",
            "scripts/inference_cogvideo.py",
            "--ckpt_path",
            "checkpoints/hunyuanvideo/HunyuanVideo",
            "--config",
            "configs/007_hunyuanvideo/hunyuanvideo_t2v_diffuser.yaml",
            "--prompt_file",
            "inputs/t2v/hunyuanvideo/tyler_swift_video/labels.txt",
            "--savedir",
            f"results/t2v/hunyuanvideo-{current_time}",
            "--bs",
            "1",
            "--height",
            "256",
            "--width",
            "256",
            "--fps",
            "16",
            "--seed",
            "6666",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_mochi():
    result = subprocess.run(
        [
            "python3",
            "scripts/inference_new.py",
            "--config",
            "configs/inference/mochi_t2v.yaml",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_flux2_dev():
    result = subprocess.run(
        [
            "python",
            "scripts/inference_new.py",
            "--config",
            "configs/inference/flux_dev.yaml",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_flux2_klein_9b():
    result = subprocess.run(
        [
            "python",
            "scripts/inference_new.py",
            "--config",
            "configs/inference/flux2_klein_9b.yaml",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_wan2_2_t2v_720p():
    result = subprocess.run(
        [
            "python",
            "scripts/inference_new.py",
            "--config",
            "configs/inference/wan2_2_t2v_a14b.yaml",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_wan2_2_i2v_720p():
    result = subprocess.run(
        [
            "python",
            "scripts/inference_new.py",
            "--config",
            "configs/inference/wan2_2_i2v_a14b.yaml",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_hunyuan1_5_t2v():
    result = subprocess.run(
        [
            "python",
            "scripts/inference_new.py",
            "--config",
            "configs/inference/hunyuanvideo1.5_t2v_720p.yaml",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_hunyuan1_5_i2v():
    result = subprocess.run(
        [
            "python",
            "scripts/inference_new.py",
            "--config",
            "configs/inference/hunyuanvideo1.5_i2v_720p.yaml",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_ltx_t2v():
    result = subprocess.run(
        [
            "python",
            "scripts/inference_new.py",
            "--config",
            "configs/inference/ltx_video.yaml",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_opensora_v2():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "videotuna.models.opensora.inference_entry",
            "--config",
            "configs/003_opensora/opensorav2/inference/256px.py",
            "--save-dir",
            "results/t2v/opensora-v2-256px",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_opensora_v10_16x256x256():
    ckpt = "checkpoints/open-sora/t2v_v10/OpenSora-v1-HQ-16x256x256.pth"
    config = "configs/003_opensora/opensorav10_256x256.yaml"
    prompt_file = "inputs/t2v/prompts.txt"
    res_dir = f"results/t2v/{current_time}-opensorav10-HQ-16x256x256"
    result = subprocess.run(
        [
            "python3",
            "scripts/inference.py",
            "--seed",
            "123",
            "--mode",
            "t2v",
            "--ckpt_path",
            ckpt,
            "--config",
            config,
            "--savedir",
            res_dir,
            "--n_samples",
            "3",
            "--bs",
            "2",
            "--height",
            "256",
            "--width",
            "256",
            "--unconditional_guidance_scale",
            "7.0",
            "--ddim_steps",
            "50",
            "--ddim_eta",
            "1.0",
            "--prompt_file",
            prompt_file,
            "--fps",
            "8",
            "--frames",
            "16",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_v2v_ms():
    from .inference_v2v_ms import Settings, inference_v2v_ms

    settings = Settings(
        input_dir="inputs/v2v/001",
        output_dir=f"results/v2v/{current_time}-v2v-modelscope-001",
    )
    inference_v2v_ms(settings=settings)


def inference_vc1_i2v_320x512():
    ckpt = "checkpoints/videocrafter/i2v_v1_512/model.ckpt"
    config = "configs/000_videocrafter/vc1_i2v_512.yaml"
    prompt_dir = "inputs/i2v/576x1024"
    savedir = "results/i2v/vc1-i2v-320x512"
    result = subprocess.run(
        [
            "python3",
            "scripts/inference.py",
            "--mode",
            "i2v",
            "--ckpt_path",
            ckpt,
            "--config",
            config,
            "--prompt_dir",
            prompt_dir,
            "--savedir",
            savedir,
            "--bs",
            "1",
            "--height",
            "320",
            "--width",
            "512",
            "--fps",
            "8",
            "--seed",
            "123",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_stepvideo_t2v_544x992():
    ckpt = "checkpoints/stepvideo/stepvideo-t2v/"
    config = "configs/009_stepvideo/stepvideo_t2v.yaml"
    prompt_file = "inputs/t2v/prompts.txt"
    savedir = "results/t2v/stepvideo"
    result = subprocess.run(
        [
            "python3",
            "scripts/inference_new.py",
            "--ckpt_path",
            ckpt,
            "--config",
            config,
            "--prompt_file",
            prompt_file,
            "--savedir",
            savedir,
            "--height",
            "544",
            "--width",
            "992",
            "--frames",
            "51",
            "--seed",
            "44",
            "--num_inference_steps",
            "50",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_wanvideo_i2v_720p():
    ckpt = "checkpoints/wan/Wan2.1-I2V-14B-720P/"
    config = "configs/008_wanvideo/wan2_1_i2v_14B_720P.yaml"
    prompt_dir = "inputs/i2v/576x1024"
    savedir = "results/i2v/wanvideo/720P"
    result = subprocess.run(
        [
            "python3",
            "scripts/inference_new.py",
            "--ckpt_path",
            ckpt,
            "--config",
            config,
            "--prompt_dir",
            prompt_dir,
            "--savedir",
            savedir,
            "--height",
            "720",
            "--width",
            "1280",
            "--frames",
            "81",
            "--seed",
            "44",
            "--num_inference_steps",
            "40",
            "--time_shift",
            "5.0",
            "--enable_model_cpu_offload",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_wanvideo_t2v_720p():
    ckpt = "checkpoints/wan/Wan2.1-T2V-14B/"
    config = "configs/008_wanvideo/wan2_1_t2v_14B.yaml"
    prompt_file = "inputs/t2v/prompts.txt"
    savedir = "results/t2v/wanvideo/720P"
    result = subprocess.run(
        [
            "python3",
            "scripts/inference_new.py",
            "--ckpt_path",
            ckpt,
            "--config",
            config,
            "--prompt_file",
            prompt_file,
            "--savedir",
            savedir,
            "--height",
            "720",
            "--width",
            "1280",
            "--frames",
            "81",
            "--seed",
            "44",
            "--time_shift",
            "5.0",
            "--num_inference_steps",
            "50",
            "--enable_model_cpu_offload",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_hunyuan_i2v_720p():
    ckpt = "checkpoints/hunyuanvideo/HunyuanVideo-I2V"
    dit_weight = "checkpoints/hunyuanvideo/HunyuanVideo-I2V/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt"
    config = "configs/007_hunyuanvideo/hunyuanvideo_i2v.yaml"
    prompt_dir = "inputs/i2v/576x1024"
    savedir = "results/i2v/hunyuan"

    result = subprocess.run(
        [
            "python3",
            "scripts/inference_new.py",
            "--ckpt_path",
            ckpt,
            "--dit_weight",
            dit_weight,
            "--config",
            config,
            "--prompt_dir",
            prompt_dir,
            "--savedir",
            savedir,
            "--height",
            "720",
            "--width",
            "1280",
            "--i2v_resolution",
            "720p",
            "--frames",
            "129",
            "--seed",
            "44",
            "--num_inference_steps",
            "50",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_vc1_t2v_576x1024():
    ckpt = "checkpoints/videocrafter/t2v_v1_1024/model.ckpt"
    config = "configs/000_videocrafter/vc1_t2v_1024.yaml"
    prompt_file = "inputs/t2v/prompts.txt"
    res_dir = "results/t2v/videocrafter1-576x1024"
    result = subprocess.run(
        [
            "python3",
            "scripts/inference.py",
            "--ckpt_path",
            ckpt,
            "--config",
            config,
            "--prompt_file",
            prompt_file,
            "--savedir",
            res_dir,
            "--bs",
            "1",
            "--height",
            "576",
            "--width",
            "1024",
            "--fps",
            "28",
            "--seed",
            "123",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_vc2_t2v_320x512():
    # Dependencies
    ckpt = "checkpoints/videocrafter/t2v_v2_512_split"
    config = "configs/001_videocrafter2/vc2_t2v_320x512.yaml"
    prompt_file = "inputs/t2v/prompts.txt"
    result = subprocess.run(
        [
            "python3",
            "scripts/inference_new.py",
            "--ckpt_path",
            ckpt,
            "--config",
            config,
            "--prompt_file",
            prompt_file,
            "--savefps",
            "30",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def inference_vc2_t2v_320x512_lora():
    # Dependencies
    ckpt = "checkpoints/videocrafter/t2v_v2_512/model.ckpt"
    config = "configs/001_videocrafter2/vc2_t2v_lora.yaml"
    lorackpt = "YOUR_LORA_CKPT"
    prompt_file = "inputs/t2v/prompts.txt"
    res_dir = "results/train/003_vc2_lora_ft"
    result = subprocess.run(
        [
            "python3",
            "scripts/inference.py",
            "--seed",
            "123",
            "--mode",
            "t2v",
            "--ckpt_path",
            ckpt,
            "--lorackpt",
            lorackpt,
            "--config",
            config,
            "--savedir",
            res_dir,
            "--n_samples",
            "1",
            "--bs",
            "1",
            "--height",
            "320",
            "--width",
            "512",
            "--unconditional_guidance_scale",
            "12.0",
            "--ddim_steps",
            "50",
            "--ddim_eta",
            "1.0",
            "--prompt_file",
            prompt_file,
            "--fps",
            "28",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_cogvideox_i2v_lora():
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dependencies
    config = "configs/004_cogvideox/cogvideo5b-i2v.yaml"  # Experiment config

    # Experiment settings
    resroot = "results/train"  # Experiment saving directory
    expname = "cogvideox_i2v_5b"  # Experiment name
    datapath = "data/apply_lipstick/metadata.csv"

    result = subprocess.run(
        [
            "python",
            "scripts/train.py",
            "-t",
            "--base",
            config,
            "--logdir",
            resroot,
            "--name",
            f"{current_time}_{expname}",
            "--devices",
            "0,",
            "lightning.trainer.num_nodes=1",
            f"data.params.train.params.csv_path={datapath}",
            f"data.params.validation.params.csv_path={datapath}",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_cogvideox_i2v_fullft():
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dependencies
    config = "configs/004_cogvideox/cogvideo5b-i2v-fullft.yaml"  # Experiment config

    # Experiment settings
    resroot = "results/train"  # Experiment saving directory
    expname = "cogvideox_i2v_5b_fullft"  # Experiment name
    datapath = "data/apply_lipstick/metadata.csv"

    result = subprocess.run(
        [
            "python",
            "scripts/train.py",
            "-t",
            "--base",
            config,
            "--logdir",
            resroot,
            "--name",
            f"{current_time}_{expname}",
            "--devices",
            "0,1,2,3",
            "lightning.trainer.num_nodes=1",
            f"data.params.train.params.csv_path={datapath}",
            f"data.params.validation.params.csv_path={datapath}",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_cogvideox_t2v_lora():
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dependencies
    config = "configs/004_cogvideox/cogvideo5b.yaml"  # Experiment config
    datapath = "data/apply_lipstick/metadata.csv"

    # Experiment settings
    resroot = "results/train"  # Experiment saving directory
    expname = "cogvideox_t2v_5b"  # Experiment name
    result = subprocess.run(
        [
            "python",
            "scripts/train.py",
            "-t",
            "--base",
            config,
            "--logdir",
            resroot,
            "--name",
            f"{current_time}_{expname}",
            "--devices",
            "0,",
            "lightning.trainer.num_nodes=1",
            f"data.params.train.params.csv_path={datapath}",
            f"data.params.validation.params.csv_path={datapath}",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_cogvideox_t2v_fullft():
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dependencies
    config = "configs/004_cogvideox/cogvideo5b-t2v-fullft.yaml"  # Experiment config
    datapath = "data/apply_lipstick/metadata.csv"

    # Experiment settings
    resroot = "results/train"  # Experiment saving directory
    expname = "cogvideox_t2v_5b_fullft"  # Experiment name
    result = subprocess.run(
        [
            "python",
            "scripts/train.py",
            "-t",
            "--base",
            config,
            "--logdir",
            resroot,
            "--name",
            f"{current_time}_{expname}",
            "--devices",
            "0,1,2,3",
            "lightning.trainer.num_nodes=1",
            f"data.params.train.params.csv_path={datapath}",
            f"data.params.validation.params.csv_path={datapath}",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_dynamicrafter():
    # Dependencies
    sdckpt = "checkpoints/stablediffusion/v2-1_512-ema/model.ckpt"
    dcckpt = "checkpoints/dynamicrafter/i2v_576x1024/model_converted.ckpt"

    # Experiment settings
    expname = "002_dynamicrafterft_1024"  # Experiment name
    config = "configs/002_dynamicrafter/dc_i2v_1024.yaml"  # Experiment config
    resroot = "results/train"  # Experiment saving directory
    result = subprocess.run(
        [
            "python",
            "scripts/train.py",
            "-t",
            "--name",
            f"{current_time}_{expname}",
            "--base",
            config,
            "--logdir",
            resroot,
            "--sdckpt",
            sdckpt,
            "--ckpt",
            dcckpt,
            "--devices",
            "0,",
            "lightning.trainer.num_nodes=1",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_flux_lora():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CONFIG_PATH"] = "configs/006_flux/config"
    os.environ["DATACONFIG_PATH"] = "configs/006_flux/multidatabackend"
    os.environ["CONFIG_BACKEND"] = "json"
    result = subprocess.run(
        [
            "accelerate",
            "launch",
            "--mixed_precision=bf16",
            "--num_processes=1",
            "--num_machines=1",
            "scripts/train_flux_lora.py",
            "--config_path",
            f"{os.environ['CONFIG_PATH']}.{os.environ['CONFIG_BACKEND']}",
            "--data_config_path",
            f"{os.environ['DATACONFIG_PATH']}.{os.environ['CONFIG_BACKEND']}",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_opensorav10():
    # Experiment settings
    expname = "opensora"  # Experiment name
    config = "configs/003_opensora/opensorav10_256x256.yaml"  # Experiment config
    logdir = "results/train"  # Experiment saving directory
    result = subprocess.run(
        [
            "python",
            "scripts/train.py",
            "-t",
            "--devices",
            "0,",
            "lightning.trainer.num_nodes=1",
            "--base",
            config,
            "--name",
            f"{current_time}_{expname}",
            "--logdir",
            logdir,
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_videocrafter_lora():
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dependencies
    vc2_ckpt = "checkpoints/videocrafter/t2v_v2_512/model.ckpt"

    # Experiment settings
    expname = "videocrafter2_t2v_lora"  # Experiment name
    config = "configs/001_videocrafter2/vc2_t2v_lora.yaml"  # Experiment config
    resroot = "results/train"  # Experiment saving directory

    # Generate current time
    result = subprocess.run(
        [
            "python",
            "scripts/train.py",
            "-t",
            "--name",
            f"{current_time}_{expname}",
            "--base",
            config,
            "--logdir",
            resroot,
            "--ckpt",
            vc2_ckpt,
            "--devices",
            "0,",
            "lightning.trainer.num_nodes=1",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_videocrafter_v2():
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dependencies
    vc2_ckpt = "checkpoints/videocrafter/t2v_v2_512_split"  # pretrained checkpoint of videocrafter2
    config = "configs/001_videocrafter2/vc2_t2v_320x512.yaml"  # experiment config: model+data+training

    # Experiment saving directory and parameters
    resroot = "results/train"  # root directory for saving multiple experiments
    expname = "videocrafter2_320x512"  # experiment name
    result = subprocess.run(
        [
            "python",
            "scripts/train_new.py",
            "-t",
            "--ckpt",
            vc2_ckpt,
            "--base",
            config,
            "--logdir",
            resroot,
            "--name",
            f"{current_time}_{expname}",
            "--devices",
            "0,",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_hunyuan_t2v_lora():
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dependencies
    config = "configs/007_hunyuanvideo/hunyuanvideo_t2v_diffuser_lora.yaml"  # Experiment config

    # Experiment settings
    resroot = "results/train"  # Experiment saving directory
    expname = "hunyuanvideo_t2v_lora"  # Experiment name
    result = subprocess.run(
        [
            "python",
            "scripts/train.py",
            "-t",
            "--base",
            config,
            "--logdir",
            resroot,
            "--name",
            f"{current_time}_{expname}",
            "--devices",
            "0,1",
            "lightning.trainer.num_nodes=1",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_wan2_1_t2v_fullft():
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dependencies
    ckpt = "checkpoints/wan/Wan2.1-T2V-14B"
    config = "configs/008_wanvideo/wan2_1_t2v_14B_fullft.yaml"

    # Experiment saving directory and parameters
    resroot = "results/train"  # root directory for saving multiple experiments
    expname = "train_wanvideo_t2v_fullft"  # experiment name
    result = subprocess.run(
        [
            "python",
            "scripts/train_new.py",
            "-t",
            "--ckpt",
            ckpt,
            "--base",
            config,
            "--logdir",
            resroot,
            "--name",
            f"{expname}_{current_time}",
            "--devices",
            "0,",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_wan2_1_t2v_lora():
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dependencies
    ckpt = "checkpoints/wan/Wan2.1-T2V-14B"
    config = "configs/008_wanvideo/wan2_1_t2v_14B_lora.yaml"

    # Experiment saving directory and parameters
    resroot = "results/train"  # root directory for saving multiple experiments
    expname = "train_wanvideo_t2v_lora"  # experiment name
    result = subprocess.run(
        [
            "python",
            "scripts/train_new.py",
            "-t",
            "--ckpt",
            ckpt,
            "--base",
            config,
            "--logdir",
            resroot,
            "--name",
            f"{expname}_{current_time}",
            "--devices",
            "0,",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_wan2_1_i2v_fullft():
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dependencies
    ckpt = "checkpoints/wan/Wan2.1-I2V-14B-480P"
    config = "configs/008_wanvideo/wan2_1_i2v_14B_480P_fullft.yaml"

    # Experiment saving directory and parameters
    resroot = "results/train"  # root directory for saving multiple experiments
    expname = "train_wanvideo_i2v_fullft"  # experiment name
    result = subprocess.run(
        [
            "python",
            "scripts/train_new.py",
            "-t",
            "--ckpt",
            ckpt,
            "--base",
            config,
            "--logdir",
            resroot,
            "--name",
            f"{expname}_{current_time}",
            "--devices",
            "0,",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def train_wan2_1_i2v_lora():
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dependencies
    ckpt = "checkpoints/wan/Wan2.1-I2V-14B-480P"
    config = "configs/008_wanvideo/wan2_1_i2v_14B_480P_lora.yaml"

    # Experiment saving directory and parameters
    resroot = "results/train"  # root directory for saving multiple experiments
    expname = "train_wanvideo_i2v_lora"  # experiment name
    result = subprocess.run(
        [
            "python",
            "scripts/train_new.py",
            "-t",
            "--ckpt",
            ckpt,
            "--base",
            config,
            "--logdir",
            resroot,
            "--name",
            f"{expname}_{current_time}",
            "--devices",
            "0,",
            "--auto_resume",
        ]
        + sys.argv[1:],
        check=False,
    )
    exit(result.returncode)


def benchmark_attn_backends():
    """Benchmark eager vs sdpa vs flash on CogVideoX diffusers inference."""
    from scripts.benchmark_attn_backends import main

    raise SystemExit(main())

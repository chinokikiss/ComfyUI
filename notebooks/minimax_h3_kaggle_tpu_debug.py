# %% [markdown]
# MiniMax H3 TPU debug notebook for Kaggle
#
# This file is executable as a Python script and is split into notebook cells
# with `# %%` markers. It downloads the four public Comfy-Org/MiniMax-H3 files,
# checks the Kaggle TPU runtime, and runs the repository's isolated PyTorch/XLA
# and Keras/JAX diagnostics. Set `H3_PROMPT_JSON` to an API-format ComfyUI
# prompt to run the complete TE -> diffusion -> VAE workflow under XLA.
#
# The public model repository is licensed by Comfy-Org. Review its license
# before running or redistributing the downloaded files.

# %% [markdown]
# ## 1. Configuration
#
# Kaggle notebook inputs can be supplied as an API-format prompt JSON under
# `/kaggle/input/...`; set `H3_PROMPT_JSON` to that path before running this
# cell. Without a prompt, the notebook still downloads and audits all files
# and runs the standalone eight-device JAX diffusion probe.

# %%
from pathlib import Path
import json
import os
import platform
import shutil
import subprocess
import sys
import time


KAGGLE_ROOT = Path("/kaggle/working")
REPO_URL = "https://github.com/chinokikiss/ComfyUI.git"
REPO_DIR = KAGGLE_ROOT / "ComfyUI"
MODELS_DIR = REPO_DIR / "models"
REPORT_DIR = KAGGLE_ROOT / "minimax_h3_tpu_report"

PROMPT_JSON = Path(os.environ.get("H3_PROMPT_JSON", ""))
RUNS = int(os.environ.get("H3_RUNS", "2"))
RUN_FULL_WORKFLOW = PROMPT_JSON.is_file()

# The smoke shape compiles quickly. Set H3_REALISTIC_SHAPE=1 for a more useful
# v5e-8 probe: 13 video latent frames at 512x512, 80 audio latent frames and
# 256 text tokens. Context is still synthetic in the standalone JAX stage;
# the full ComfyUI workflow uses real Qwen3-VL embeddings.
REALISTIC_SHAPE = os.environ.get("H3_REALISTIC_SHAPE", "0") == "1"
if REALISTIC_SHAPE:
    VIDEO_T, VIDEO_H, VIDEO_W, AUDIO_T, CONTEXT_TOKENS = 13, 32, 32, 80, 256
else:
    VIDEO_T, VIDEO_H, VIDEO_W, AUDIO_T, CONTEXT_TOKENS = 1, 2, 2, 2, 64

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")
os.environ.setdefault("KERAS_BACKEND", "jax")

HF_REPO = "Comfy-Org/MiniMax-H3"
MODEL_FILES = {
    "diffusion": "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "text_encoder": "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "audio_vae": "vae/minimax_h3_audio_vae_fp32.safetensors",
    "video_vae": "vae/minimax_h3_video_vae_fp16.safetensors",
}

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
print({
    "python": sys.version,
    "platform": platform.platform(),
    "repo": str(REPO_DIR),
    "models": str(MODELS_DIR),
    "prompt": str(PROMPT_JSON) if PROMPT_JSON else None,
    "realistic_shape": REALISTIC_SHAPE,
})

# %% [markdown]
# ## 2. Install notebook-only dependencies and clone the exact code under test
#
# The notebook does not replace Kaggle's TPU-specific `torch_xla`, `jax` or
# `libtpu` wheels. Those packages must match the accelerator image. It only
# installs the downloader and safetensors metadata reader.

# %%
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "huggingface_hub>=0.27.0", "safetensors>=0.4.5",
])

if not (REPO_DIR / ".git").exists():
    subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
else:
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
sys.path.insert(0, str(REPO_DIR))

# %% [markdown]
# ## 3. Download and inspect the four H3 files
#
# `hf_hub_download` resumes through the Hugging Face cache and preserves the
# ComfyUI subdirectories under `/kaggle/working/ComfyUI/models`.

# %%
from huggingface_hub import hf_hub_download
from safetensors import safe_open


MODEL_PATHS = {}
for key, filename in MODEL_FILES.items():
    print(f"Downloading {key}: {filename}")
    MODEL_PATHS[key] = Path(hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        local_dir=str(MODELS_DIR),
        token=os.environ.get("HF_TOKEN"),
    ))

file_report = {}
for key, path in MODEL_PATHS.items():
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata() or {}
    file_report[key] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "tensor_count": len(keys),
        "first_keys": keys[:5],
        "metadata": metadata,
    }
print(json.dumps(file_report, indent=2, default=str))
(REPORT_DIR / "model_files.json").write_text(
    json.dumps(file_report, indent=2, default=str), encoding="utf-8"
)

# %% [markdown]
# ## 4. Check TPU visibility before loading any 20+ GB model
#
# A valid run must expose exactly eight TPU devices to both JAX and
# PyTorch/XLA. The notebook records versions and exits the corresponding stage
# cleanly when Kaggle has no TPU attached.

# %%
runtime_report = {"platform": platform.platform(), "python": sys.version}
try:
    import jax
    runtime_report["jax"] = jax.__version__
    runtime_report["jax_devices"] = [str(device) for device in jax.devices()]
    runtime_report["jax_tpu_devices"] = [str(device) for device in jax.devices("tpu")]
except Exception as error:
    runtime_report["jax_error"] = repr(error)

try:
    import torch
    import torch_xla
    import torch_xla.runtime as xr
    runtime_report["torch"] = torch.__version__
    runtime_report["torch_xla"] = getattr(torch_xla, "__version__", "unknown")
    runtime_report["torch_xla_device_type"] = xr.device_type()
    runtime_report["torch_xla_device_count"] = xr.global_runtime_device_count()
except Exception as error:
    runtime_report["torch_xla_error"] = repr(error)

print(json.dumps(runtime_report, indent=2, default=str))
(REPORT_DIR / "runtime.json").write_text(
    json.dumps(runtime_report, indent=2, default=str), encoding="utf-8"
)

# %% [markdown]
# ## 5. Run the full ComfyUI/XLA workflow or the standalone JAX probe
#
# With `H3_PROMPT_JSON` set, the repository verifier runs the real API prompt
# in a PyTorch/XLA child process and then runs JAX in a fresh child process.
# The prompt must reference these exact model filenames:
#
# - `minimax_h3_ref2va_pruned_int8_convrot.safetensors`
# - `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`
# - `minimax_h3_video_vae_fp16.safetensors`
# - `minimax_h3_audio_vae_fp32.safetensors`
#
# Without an API prompt, only the JAX diffusion probe is run. Its standalone
# context is synthetic; it validates TPU placement and graph execution but is
# not a quality-generation run.

# %%
def run_verify(stage=None):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prompt = PROMPT_JSON
    if not prompt.is_file():
        prompt = KAGGLE_ROOT / "h3_probe_prompt.json"
        prompt.write_text(json.dumps({
            "1": {"class_type": "SaveImage", "inputs": {}},
        }), encoding="utf-8")

    command = [
        sys.executable, "-m", "comfy.ldm.minimax.verify_tpu",
        "--prompt", str(prompt),
        "--diffusion", str(MODEL_PATHS["diffusion"]),
        "--report-dir", str(REPORT_DIR),
        "--runs", str(RUNS),
        "--video-t", str(VIDEO_T),
        "--video-h", str(VIDEO_H),
        "--video-w", str(VIDEO_W),
        "--audio-t", str(AUDIO_T),
        "--context-tokens", str(CONTEXT_TOKENS),
    ]
    if stage is not None:
        command += ["--stage", stage]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["PJRT_DEVICE"] = "TPU"
    env["KERAS_BACKEND"] = "jax"
    log_path = REPORT_DIR / (f"{stage or 'full'}.log")
    print("Running:", " ".join(command))
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=str(REPO_DIR), env=env,
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    print({"returncode": result.returncode,
           "seconds": round(time.perf_counter() - started, 2),
           "log": str(log_path)})
    return result.returncode


if RUN_FULL_WORKFLOW:
    verify_returncode = run_verify()
else:
    verify_returncode = run_verify(stage="jax")

# %% [markdown]
# ## 6. Summarize fallback and eight-chip checks

# %%
summary_path = REPORT_DIR / "summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
else:
    summary = {
        "stage": "jax",
        "returncode": verify_returncode,
        "report_dir": str(REPORT_DIR),
    }
print(json.dumps(summary, indent=2, default=str))

jax_report_path = REPORT_DIR / "jax.json"
if jax_report_path.exists():
    jax_report = json.loads(jax_report_path.read_text(encoding="utf-8"))
    print({
        "jax_devices": len(jax_report.get("devices", [])),
        "tp_sharded_large_weight_count": jax_report.get("tp_sharded_large_weight_count"),
        "large_weights_not_tp_sharded": jax_report.get("large_weights_not_tp_sharded"),
        "returns_video": jax_report.get("returns_video"),
    })

xla_report_path = REPORT_DIR / "xla.json"
if xla_report_path.exists():
    xla_report = json.loads(xla_report_path.read_text(encoding="utf-8"))
    print({
        "physical_devices": xla_report.get("physical_devices"),
        "runs_without_cpu_fallback": [
            run.get("no_cpu_fallback") for run in xla_report.get("runs", [])
        ],
        "fallback_counters": [
            run.get("cpu_fallback_counters") for run in xla_report.get("runs", [])
        ],
        "workflow_outputs": [
            run.get("workflow_outputs", {}) for run in xla_report.get("runs", [])
        ],
    })

# %% [markdown]
# ## 7. Package the reports for download

# %%
archive_path = shutil.make_archive(
    str(KAGGLE_ROOT / "minimax_h3_tpu_report"), "zip", REPORT_DIR
)
print(archive_path)

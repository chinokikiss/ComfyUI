# MiniMax H3 diffusion on Keras 3/JAX

`jax_diffusion.py` is a standalone H3 denoiser. It loads both ordinary H3
weights and the pruned `int8_tensorwise` ConvRot checkpoint. The existing
PyTorch nodes and sampler are unchanged. The text encoder, VAEs, and sampling
loop remain caller responsibilities.

In the `comfyui` conda environment on this Windows machine, `keras==3.15.1`
and `jax[cpu]==0.4.38` were used for validation. On a TPU VM, install the JAX
build appropriate for that VM as described in the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
Set `KERAS_BACKEND=jax` before importing Keras:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

from comfy.ldm.minimax.jax_diffusion import MiniMaxH3JAX

model = MiniMaxH3JAX.from_safetensors(
    "D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors"
)
video_velocity, audio_velocity = model.denoise(
    video_latent, audio_latent, timestep, qwen_hidden_states,
    payload=conditioning, sample_sigmas=sigma_schedule,
)
```

On TPU, the loader uses all visible TPU devices with tensor parallelism by
default. Pass `devices=jax.devices("tpu")[:8]` to select devices explicitly.
On CPU, the default remains one device. Checkpoint tensors
are read one at a time; the original checkpoint is not modified. The loader
accepts the target file's `L2P_bypass` trailer after its safetensors payload.

`video_latent` is `[1, 24, T, H, W]`, `audio_latent` is `[1, 32, 2, T40]`,
and `timestep` is the video sigma multiplied by 1000. `qwen_hidden_states`
is `[1, L, 5120]` or already refined `[1, L, hidden_size]`. The returned
arrays have the same stream shapes. Pass `prefix="model.diffusion_model."`
when those keys are prefixed in a combined checkpoint. `payload` uses the H3
conditioning keys from the PyTorch model. PDD checkpoints also require the
video `sample_sigmas` schedule.

The ConvRot path rotates each 256-feature activation group, quantizes each row
to int8, runs an int8 by int8 dot with int32 accumulation, then applies the
activation and per-output weight scales. It keeps the checkpoint's bfloat16
weights as bfloat16. One DiT block graph is JIT compiled and reused across
the stack; the first call includes compilation. Conditioning noise uses JAX's
seeded RNG, so noise samples differ from the PyTorch implementation when
augmentation is below 1.0.

Run an operator alignment and latency check against the actual checkpoint:

```powershell
conda run -n comfyui python -m comfy.ldm.minimax.bench_jax_int8 `
  "D:\minimaxH3\MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors" `
  --rows 512 --tokens 64 --repeats 5
```

The benchmark reads a slice of the first quantized QKV weight and compares
against Comfy Kitchen on CPU. A full 20.97 GB model load and TPU throughput
cannot be checked on this 31 GB RAM Windows machine. CPU latency is not a
prediction of TPU throughput; profile representative sequence lengths on the
v5e-8 VM before changing the sharding or quantized dot path.

On this machine, 64 tokens with 512 output rows matched exactly and took
3.02 ms in JAX versus 2.58 ms in Comfy Kitchen (median of 10). The complete
21,504-row QKV projection took 72.6 ms in JAX versus 126.7 ms in Comfy Kitchen
(median of 5); its maximum absolute difference was 0.03125 in bfloat16 output.

For one-forward tensor parallelism on a simulated CPU mesh, pass
`tensor_parallel=True, devices=jax.devices()[:8]`. Keras `ModelParallel`
partitions the text projection, QKV, and first MLP projections by output
features, and the attention output and second MLP projections by input
features. The eight-device layout and a small quantized forward were verified
with simulated CPU devices. XLA reported a full rematerialization during this
test; TPU execution and throughput have not yet been measured.

`xla_backend.py` provides a PyTorch/XLA SPMD mesh for the existing H3 text and
VAE modules. It marks large linear and convolution weights across eight TPU
chips and raises if XLA reports an `aten::` CPU fallback during a forward.
It requires a matching PyTorch/XLA TPU installation and cannot run in this
Windows CPU validation environment. The existing ComfyUI node workflow is not
yet wired to the JAX denoiser; the PyTorch/XLA and JAX stages require a
process boundary with host tensor transfer between them.

ComfyUI has an explicit `--xla` device mode. It requires a TPU runtime with
exactly eight visible chips, enables PyTorch/XLA SPMD, disables CPU offload and
dynamic VRAM, selects BF16 defaults, and applies the XLA placement checker to
models loaded by the normal patcher. Start ComfyUI on the TPU VM with:

```bash
python main.py --xla --bf16-unet --bf16-vae --bf16-text-enc --disable-api-nodes
```

For a single collection pass, export the API-format H3 prompt JSON from the
workflow and run:

```bash
python -m comfy.ldm.minimax.verify_tpu \
  --prompt /path/to/h3_api_prompt.json \
  --diffusion /path/to/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors \
  --models-dir /path/to/minimaxH3 \
  --report-dir /tmp/h3_tpu_report --runs 2
```

The script runs the ordinary ComfyUI prompt under PyTorch/XLA, then releases
that runtime and runs the eight-device JAX diffusion probe in a fresh process.
The XLA report includes the output-node file references produced by the prompt,
so a workflow containing a video output node can be checked without parsing the
console log. The JAX stage is deliberately a synthetic denoise probe: its
default zero latents and zero text context validate shapes, compilation and
sharding, but do not produce a playable video. It writes `summary.json`,
per-stage logs, the full PyTorch/XLA metrics report,
`aten::` fallback counters, transfer and compile metrics, per-node wall times,
large-weight sharding records, a JAX profiler trace, and a JAX device-memory
profile. `mixed_jax_xla_workflow_verified` remains false until a future sampler
adapter connects the JAX stage to the ComfyUI denoising loop.

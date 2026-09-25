import os
import json

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest
import torch
import comfy_kitchen
import jax
from safetensors.torch import save_file

pytest.importorskip("jax")
pytest.importorskip("keras")
import jax.numpy as jnp

from comfy.ldm.minimax.jax_diffusion import MiniMaxH3JAX, int8_convrot_linear
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.ops import disable_weight_init


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_int8_convrot_linear_matches_comfy_kitchen(dtype):
    rng = np.random.default_rng(11)
    x = torch.from_numpy(rng.standard_normal((7, 512)).astype(np.float32)).to(dtype)
    weight = torch.from_numpy(rng.integers(-127, 128, (48, 512), dtype=np.int8))
    scale = torch.from_numpy(rng.uniform(0.0001, 0.01, (48, 1)).astype(np.float32))
    bias = torch.from_numpy(rng.standard_normal(48).astype(np.float32)).to(dtype)
    expected = comfy_kitchen.int8_linear(x, weight, scale, bias, dtype,
                                          convrot=True, convrot_groupsize=256)
    actual = int8_convrot_linear(jnp.asarray(x.float().numpy(), dtype=str(dtype).removeprefix("torch.")),
                                 jnp.asarray(weight.numpy()), jnp.asarray(scale.numpy()),
                                 jnp.asarray(bias.float().numpy(), dtype=str(dtype).removeprefix("torch.")))
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), expected.float().numpy(),
                               rtol=2e-2 if dtype == torch.bfloat16 else 1e-5,
                               atol=2e-2 if dtype == torch.bfloat16 else 1e-5)


@pytest.mark.parametrize("tensor_parallel", [False, True])
def test_quantized_h3_blocks_run_with_jit(tmp_path, tensor_parallel):
    rng = np.random.default_rng(19)
    devices = jax.devices()[:8]
    model = MiniMaxH3JAX(hidden_size=256, num_layers=2, token_refiner_num_layers=0,
                          num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=512,
                          latents_dim=2, audio_latents_dim=2, text_dim=6,
                          time_embed_dim=8, adaln_curve_grid=3, rope_inv_freq_len=16,
                          quantized_blocks=True, weight_dtype="bfloat16", compute_dtype="bfloat16",
                          devices=devices, tensor_parallel=tensor_parallel)
    state = {}
    for key, var in model._weights_by_key().items():
        if key.endswith("weight_scale"):
            state[key] = rng.uniform(0.0001, 0.001, var.shape).astype(np.float32)
            state[key.removesuffix("weight_scale") + "comfy_quant"] = np.frombuffer(
                json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode(), dtype=np.uint8)
        elif var.dtype == "int8":
            state[key] = rng.integers(-8, 9, var.shape, dtype=np.int8)
        elif key.endswith("norm.weight") or ".norm" in key:
            state[key] = np.ones(var.shape, dtype=np.float32)
        else:
            state[key] = (rng.standard_normal(var.shape) * 0.01).astype(np.float32)
    model.load_state_dict(state)
    if tensor_parallel and len(devices) > 1:
        assert model.condition_proj.weight.value.sharding.spec == ("tp", None)
        assert model.condition_proj.bias.value.sharding.spec == ("tp",)
        assert model.blocks[0].attn.qkv_proj.weight.value.sharding.spec == ("tp", None)
        assert model.blocks[0].attn.out_proj.weight.value.sharding.spec == (None, "tp")
    inputs = (np.zeros((1, 2, 1, 2, 2), dtype=np.float32),
              np.zeros((1, 2, 2, 2), dtype=np.float32),
              np.array([500.0], dtype=np.float32),
              np.zeros((1, 3, 6), dtype=np.float32))
    options = dict(payload={"audio_scale": 2.0},
                   denoise_mask=np.full((1, 1, 1, 2, 2), 0.5, dtype=np.float32),
                   audio_denoise_mask=np.full((1, 1, 2, 2), 0.5, dtype=np.float32))
    video, audio = model.denoise(*inputs, **options)
    assert video.shape == (1, 2, 1, 2, 2)
    assert audio.shape == (1, 2, 2, 2)
    assert np.isfinite(np.asarray(video)).all()
    if tensor_parallel:
        reference = MiniMaxH3JAX(hidden_size=256, num_layers=2, token_refiner_num_layers=0,
                                  num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=512,
                                  latents_dim=2, audio_latents_dim=2, text_dim=6,
                                  time_embed_dim=8, adaln_curve_grid=3, rope_inv_freq_len=16,
                                  quantized_blocks=True, weight_dtype="bfloat16", compute_dtype="bfloat16")
        reference.load_state_dict(state)
        for got, want in zip((video, audio), reference.denoise(*inputs, **options)):
            np.testing.assert_allclose(np.asarray(got, dtype=np.float32), np.asarray(want, dtype=np.float32),
                                       rtol=2e-2, atol=2e-2)

    path = tmp_path / "int8_convrot.safetensors"
    save_file({key: torch.from_numpy(value.copy()) for key, value in state.items()}, path)
    with path.open("ab") as file:
        file.write(b"\nL2P_bypass_test_int8_convrot.safetensors_1785751384\n")
    streamed = MiniMaxH3JAX.from_safetensors(str(path), devices=devices,
                                               weight_dtype="bfloat16", compute_dtype="bfloat16",
                                               tensor_parallel=tensor_parallel)
    for got, want in zip(streamed.denoise(*inputs, **options), (video, audio)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize(("conditioned", "curve", "pdd"),
                         [(False, False, False), (True, False, False),
                          ("refs", False, False), (False, True, True)])
def test_jax_diffusion_matches_torch_forward(conditioned, curve, pdd, tmp_path):
    config = dict(hidden_size=8, num_layers=2, token_refiner_num_layers=1,
                  num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=16,
                  latents_dim=2, audio_latents_dim=2, text_dim=6,
                  timestep_input_dim=4, time_embed_hidden_size=8, time_embed_dim=4,
                  rope_inv_freq_len=1)
    if curve:
        config["adaln_curve_grid"] = 3
    torch_model = MiniMaxH3Model(**config, dtype=torch.float32, device="cpu", operations=torch.nn)
    if pdd:
        torch_model.final_layer.video_out = disable_weight_init.Linear(8, 16)
        torch_model.final_layer.audio_out = disable_weight_init.Linear(8, 4)
        torch_model.final_layer.video_out.out_features = 8
        torch_model.final_layer.audio_out.out_features = 2
    rng = np.random.default_rng(7)
    state = {}
    for key, value in torch_model.state_dict().items():
        if key.endswith(".weight") and ("norm" in key):
            array = 1.0 + rng.standard_normal(value.shape).astype(np.float32) * 0.02
        elif key == "rope.inv_freq":
            array = np.full(value.shape, 0.3, dtype=np.float32)
        else:
            array = rng.standard_normal(value.shape).astype(np.float32) * 0.02
        state[key] = torch.from_numpy(array)
    torch_model.load_state_dict(state)
    torch_model.requires_grad_(False)
    jax_model = MiniMaxH3JAX.from_state_dict(state)

    video = rng.standard_normal((1, 2, 2, 3, 4)).astype(np.float32)
    audio = rng.standard_normal((1, 2, 2, 3)).astype(np.float32)
    context = rng.standard_normal((1, 3, 6)).astype(np.float32)
    timestep = np.array([530.0], dtype=np.float32)
    payload = {}
    video_mask = audio_mask = None
    if conditioned is True:
        keyframe = {"resolved_frame_index": 0,
                    "latent": rng.standard_normal((1, 2, 1, 4, 4)).astype(np.float32),
                    "audio_latent": rng.standard_normal((1, 2, 2, 2)).astype(np.float32)}
        payload = {"keyframes": [keyframe], "cond_video_latents": [keyframe["latent"]],
                   "cond_audio_latents": [keyframe["audio_latent"]],
                   "visual_cond_noise_aug": 1.0, "audio_cond_noise_aug": 1.0,
                   "audio_scale": 2.0, "text_token_tags": np.array([[1, 0, 2]])}
        video_mask = np.ones((1, 1, 2, 3, 4), dtype=np.float32)
        video_mask[:, :, 0, :2, :2] = 0.5
        audio_mask = np.ones((1, 1, 2, 3), dtype=np.float32)
        audio_mask[:, :, :, 0] = 0.5
    elif conditioned == "refs":
        image = rng.standard_normal((1, 2, 1, 4, 4)).astype(np.float32)
        clip = rng.standard_normal((1, 2, 1, 4, 4)).astype(np.float32)
        soundtrack = rng.standard_normal((1, 2, 2, 2)).astype(np.float32)
        payload = {"refs": [{"kind": "image", "latent_h": 4, "latent_w": 4, "latent": image},
                            {"kind": "video_audio", "latent_t": 1, "latent_h": 4, "latent_w": 4,
                             "ref_audio_t": 2, "latent": clip, "audio_latent": soundtrack}],
                   "cond_video_latents": [image, clip], "cond_audio_latents": [soundtrack],
                   "visual_cond_noise_aug": 1.0, "audio_cond_noise_aug": 1.0}
    def as_torch(value):
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        if isinstance(value, list):
            return [as_torch(item) for item in value]
        if isinstance(value, dict):
            return {key: as_torch(item) for key, item in value.items()}
        return value
    torch_payload = as_torch(payload)
    sample_sigmas = np.array([0.53, 0.0], dtype=np.float32) if pdd else None
    transformer_options = {"sample_sigmas": torch.from_numpy(sample_sigmas)} if pdd else {}
    expected = torch_model([torch.from_numpy(video), torch.from_numpy(audio)],
                           torch.from_numpy(timestep), torch.from_numpy(context), transformer_options,
                           minimax_payload=torch_payload,
                           denoise_mask=torch.from_numpy(video_mask) if video_mask is not None else None,
                           audio_denoise_mask=torch.from_numpy(audio_mask) if audio_mask is not None else None)
    actual = jax_model.denoise(video, audio, timestep, context, payload,
                              denoise_mask=video_mask, audio_denoise_mask=audio_mask,
                              sample_sigmas=sample_sigmas)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(got), want.detach().numpy(), rtol=3e-4, atol=3e-4)
    if not conditioned:
        path = tmp_path / "h3.safetensors"
        save_file(state, path)
        streamed = MiniMaxH3JAX.from_safetensors(str(path))
        streamed_out = streamed.denoise(video, audio, timestep, context, sample_sigmas=sample_sigmas)
        for got, want in zip(streamed_out, expected):
            np.testing.assert_allclose(np.asarray(got), want.detach().numpy(), rtol=3e-4, atol=3e-4)
        if not curve:
            bf16_state = {key: value.to(torch.bfloat16) for key, value in state.items()}
            bf16_model = MiniMaxH3JAX.from_state_dict(bf16_state)
            assert bf16_model.blocks[0].attn.qkv_proj.weight.dtype == "bfloat16"
            bf16_out = bf16_model.denoise(video, audio, timestep, context)
            assert all(np.isfinite(np.asarray(value)).all() for value in bf16_out)

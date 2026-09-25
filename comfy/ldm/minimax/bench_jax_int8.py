"""CPU alignment and latency check for a real H3 ConvRot INT8 weight."""

import argparse
import os
import time

os.environ["KERAS_BACKEND"] = "jax"

import comfy_kitchen
import jax
import jax.numpy as jnp
import numpy as np
import torch

from comfy.ldm.minimax.jax_diffusion import TensorFile, int8_convrot_linear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    with TensorFile(args.checkpoint) as file:
        weight = file.get_tensor("blocks.0.attn.qkv_proj.weight")[:args.rows].copy()
        scale = file.get_tensor("blocks.0.attn.qkv_proj.weight_scale")[:args.rows].copy()

    rng = np.random.default_rng(25)
    x = rng.standard_normal((args.tokens, weight.shape[1])).astype(np.float32)
    torch_x = torch.from_numpy(x).to(torch.bfloat16)
    torch_weight = torch.from_numpy(weight)
    torch_scale = torch.from_numpy(scale)
    jax_x = jnp.asarray(x, dtype=jnp.bfloat16)
    jax_weight = jnp.asarray(weight)
    jax_scale = jnp.asarray(scale)
    fn = jax.jit(lambda value: int8_convrot_linear(value, jax_weight, jax_scale))

    reference = comfy_kitchen.int8_linear(torch_x, torch_weight, torch_scale, None,
                                           torch.bfloat16, convrot=True, convrot_groupsize=256)
    actual = fn(jax_x).block_until_ready()
    delta = np.asarray(actual, dtype=np.float32) - reference.float().numpy()
    print(f"shape=({args.tokens}, {weight.shape[1]}) x ({args.rows}, {weight.shape[1]})")
    print(f"max_abs={np.abs(delta).max():.6g} mean_abs={np.abs(delta).mean():.6g} "
          f"rms={np.sqrt(np.mean(delta * delta)):.6g}")

    torch_times, jax_times = [], []
    for _ in range(args.repeats):
        start = time.perf_counter()
        comfy_kitchen.int8_linear(torch_x, torch_weight, torch_scale, None,
                                  torch.bfloat16, convrot=True, convrot_groupsize=256)
        torch_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        fn(jax_x).block_until_ready()
        jax_times.append(time.perf_counter() - start)
    print(f"torch_cpu_median_ms={np.median(torch_times) * 1000:.3f} "
          f"jax_cpu_median_ms={np.median(jax_times) * 1000:.3f}")


if __name__ == "__main__":
    main()

"""Standalone Keras 3/JAX implementation of the MiniMax H3 diffusion network.

Set KERAS_BACKEND=jax before importing this module. The inputs, outputs and
state-dict keys match comfy.ldm.minimax.model.MiniMaxH3Model. ComfyUI's sampler,
patcher and offloading machinery are deliberately outside this module.
"""

import json
import math
import mmap
import re
import struct
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import keras
import numpy as np
from keras import ops

_SAFETENSOR_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": "bfloat16",
                      "I8": np.int8, "U8": np.uint8}


class TensorFile:
    """Read a safetensors payload, including the known L2P trailer variant."""

    def __init__(self, path):
        self.file = open(path, "rb")
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        if header_size > 100 * 1024 * 1024:
            raise ValueError("Invalid safetensors header size")
        self.header = json.loads(self.file.read(header_size))
        self.offset = 8 + header_size
        entries = [(key, value) for key, value in self.header.items() if key != "__metadata__"]
        cursor = 0
        for key, value in sorted(entries, key=lambda item: item[1]["data_offsets"][0]):
            start, end = value["data_offsets"]
            dtype = _SAFETENSOR_DTYPES.get(value["dtype"])
            if dtype is None or start != cursor or end - start != math.prod(value["shape"]) * np.dtype(dtype).itemsize:
                raise ValueError(f"Invalid safetensors tensor {key}")
            cursor = end
        self.file.seek(self.offset + cursor)
        trailer = self.file.read()
        if trailer and not re.fullmatch(rb"\nL2P_bypass_[A-Za-z0-9_.-]+_\d+\n", trailer):
            raise ValueError("Unexpected data after safetensors payload")
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def keys(self):
        return self.header.keys()

    def get_slice(self, key):
        item = self.header[key]
        return SimpleNamespace(shape=item["shape"], dtype=item["dtype"])

    def get_tensor(self, key):
        item = self.header[key]
        start = self.offset + item["data_offsets"][0]
        value = np.ndarray(item["shape"], dtype=_SAFETENSOR_DTYPES[item["dtype"]],
                           buffer=self.mapping, offset=start)
        return value.copy()

    def close(self):
        self.mapping.close()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


if keras.backend.backend() != "jax":
    raise RuntimeError("MiniMax H3 JAX diffusion requires KERAS_BACKEND=jax")


FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0


def time_shift_sigma(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def patchify_video(x, patch_size=(1, 2, 2)):
    b, c, full_t, full_h, full_w = x.shape
    pt, ph, pw = patch_size
    t, h, w = full_t // pt, full_h // ph, full_w // pw
    x = ops.reshape(x, (b, c, t, pt, h, ph, w, pw))
    return ops.reshape(ops.transpose(x, (0, 2, 4, 6, 1, 3, 5, 7)), (b * t * h * w, c * pt * ph * pw))


def unpatchify_video(rows, t, h, w, c, patch_size=(1, 2, 2)):
    pt, ph, pw = patch_size
    x = ops.reshape(rows, (-1, t, h, w, c, pt, ph, pw))
    return ops.reshape(ops.transpose(x, (0, 4, 1, 5, 2, 6, 3, 7)), (-1, c, t * pt, h * ph, w * pw))


def pack_audio(x):
    return ops.reshape(ops.transpose(x[0], (1, 2, 0)), (-1, x.shape[1]))


def unpack_audio(rows):
    return ops.expand_dims(ops.transpose(ops.reshape(rows, (2, -1, rows.shape[-1])), (2, 0, 1)), 0)


def _frame_grid(h, w):
    area = math.sqrt(h * w)
    def axis(dim):
        ratio = dim / area
        n = dim // 2
        return (np.arange(n, dtype=np.float64) * ratio / n + (1.0 - ratio) / 2.0) * 32.0
    hh, ww = np.meshgrid(axis(h), axis(w), indexing="ij")
    return np.stack((hh.ravel(), ww.ravel()), axis=-1), axis(w)


def _video_t_spans(n):
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]


def _audio_grid(cursor, t, w_low, w_high):
    g = np.zeros((t * 2, 3), dtype=np.float64)
    g[:, 0] = cursor + np.tile(np.arange(t), 2)
    g[:t, 2] = w_low
    g[t:, 2] = w_high
    return g


def _video_grid(vt, frame, cursor):
    times = cursor + np.r_[0.0, np.cumsum(_video_t_spans(vt)[:-1])]
    g = np.empty((vt, len(frame), 3), dtype=np.float64)
    g[:, :, 0] = times[:, None]
    g[:, :, 1:] = frame[None]
    return g.reshape(-1, 3)


def _regular_hadamard(size):
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1],
                   [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float32)
    h = h4
    while h.shape[0] < size:
        h = np.kron(h, h4)
    if h.shape[0] != size:
        raise ValueError(f"ConvRot group size must be a power of four, got {size}")
    return h / math.sqrt(size)


def int8_convrot_linear(x, weight, weight_scale, bias=None, group_size=256):
    """Comfy Kitchen's ConvRot W8A8 math with an INT32 JAX dot."""
    shape = x.shape
    k = shape[-1]
    if k % group_size:
        raise ValueError(f"ConvRot group size {group_size} does not divide {k}")
    h = jnp.asarray(_regular_hadamard(group_size), dtype=x.dtype)
    grouped = jnp.reshape(x, (-1, k // group_size, group_size))
    rotated = jnp.reshape(jnp.matmul(grouped, h), (-1, k))
    x_scale = jnp.maximum(jnp.max(jnp.abs(rotated), axis=-1, keepdims=True).astype(jnp.float32) / 127.0, 1e-30)
    scale_for_math = x_scale.astype(x.dtype)
    scale_for_math = jnp.where(scale_for_math == 0, jnp.finfo(x.dtype).tiny, scale_for_math)
    x_int8 = jnp.clip(jnp.round(rotated / scale_for_math), -128, 127).astype(jnp.int8)
    accum = jax.lax.dot_general(x_int8, weight,
                                dimension_numbers=(((1,), (1,)), ((), ())),
                                preferred_element_type=jnp.int32)
    y = (accum.astype(jnp.float32) * (x_scale * jnp.reshape(weight_scale, (1, -1)))).astype(x.dtype)
    if bias is not None:
        y = y + bias.astype(x.dtype)
    return jnp.reshape(y, (*shape[:-1], weight.shape[0]))


class PackedLayout:
    """CPU-only sequence plan, shared across denoising steps of the same shape."""

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None):
        frame, w_grid = _frame_grid(latent_h, latent_w)
        frame_rows = len(frame)
        segments = [("text", text_len)]
        g = np.zeros((text_len, 3), dtype=np.float64)
        g[:, 0] = np.arange(text_len)
        pos = [g]
        img_pos, img_update, audio_pos, audio_update = [], [], [], []
        row = text_len
        audio_width = (float(w_grid[0]), float(w_grid[-1]))
        cursor = float(text_len)
        for blk in refs or ():
            kind = blk["kind"]
            if kind == "image":
                cursor += 1.0
            elif kind == "audio":
                cursor += float(blk["ref_audio_t"])
            else:
                cursor += max(float(blk["ref_audio_t"]), sum(_video_t_spans(blk["latent_t"])))

        def add(kind, grid, stream, update):
            nonlocal row
            n = len(grid)
            segments.append((kind, n))
            pos.append(grid)
            if stream == "video":
                img_pos.extend(range(row, row + n))
                img_update.extend([update] * n)
            else:
                audio_pos.extend(range(row, row + n))
                audio_update.extend([update] * n)
            row += n

        for kf in keyframes or ():
            cond_t = cursor + FRAME_RESCALE * kf["resolved_frame_index"]
            if kf.get("latent") is not None:
                add("cond", _video_grid(kf["latent"].shape[2], frame, cond_t), "video", False)
            if kf.get("audio_latent") is not None:
                add("cond_audio", _audio_grid(cond_t, kf["audio_latent"].shape[-1], *audio_width), "audio", False)

        if refs:
            cursor = float(text_len)
            for blk in refs:
                kind = blk["kind"]
                if kind == "image":
                    r_frame, _ = _frame_grid(blk["latent_h"], blk["latent_w"])
                    g = np.empty((len(r_frame), 3), dtype=np.float64)
                    g[:, 0], g[:, 1:] = cursor, r_frame
                    add("ref_img", g, "video", False)
                    cursor += 1.0
                elif kind == "audio":
                    rt = blk["ref_audio_t"]
                    if rt:
                        add("ref_audio", _audio_grid(cursor, rt, *audio_width), "audio", False)
                    cursor += float(rt)
                elif kind in ("video", "video_audio"):
                    rt, vt = blk["ref_audio_t"], blk["latent_t"]
                    r_frame, r_width = _frame_grid(blk["latent_h"], blk["latent_w"])
                    if rt:
                        add("ref_audio", _audio_grid(cursor, rt, float(r_width[0]), float(r_width[-1])), "audio", False)
                    add("ref_img", _video_grid(vt, r_frame, cursor), "video", False)
                    cursor += max(float(rt), sum(_video_t_spans(vt)))

        add("audio", _audio_grid(cursor, audio_t, *audio_width), "audio", True)
        add("video", _video_grid(latent_t, frame, cursor), "video", True)
        self.seq_len = row
        self.position_ids = np.concatenate(pos).astype(np.float32)
        self.img_pos = np.asarray(img_pos, dtype=np.int32)
        self.img_update = np.asarray(img_update, dtype=bool)
        self.audio_pos = np.asarray(audio_pos, dtype=np.int32)
        self.audio_update = np.asarray(audio_update, dtype=bool)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        self.segments = []
        offset = 0
        for kind, n in segments:
            self.segments.append((offset, offset + n, kind))
            offset += n


class TorchLinear(keras.layers.Layer):
    def __init__(self, input_dim, output_dim, bias=True, weight_dtype="float32",
                 quantized=False, convrot_groupsize=256, **kwargs):
        super().__init__(dtype=weight_dtype, **kwargs)
        self.quantized = quantized
        self.convrot_groupsize = convrot_groupsize
        self.weight = self.add_weight(name="weight", shape=(output_dim, input_dim), initializer="zeros",
                                      dtype="int8" if quantized else weight_dtype, trainable=not quantized)
        self.weight_scale = self.add_weight(name="weight_scale", shape=(output_dim, 1), initializer="ones",
                                            dtype="float32", trainable=False) if quantized else None
        self.bias = self.add_weight(name="bias", shape=(output_dim,), initializer="zeros", dtype=weight_dtype) if bias else None
        self.built = True

    def call(self, x):
        if self.quantized:
            return int8_convrot_linear(x, self.weight.value, self.weight_scale.value,
                                       self.bias.value if self.bias is not None else None,
                                       self.convrot_groupsize)
        y = ops.matmul(x, ops.transpose(ops.cast(self.weight, x.dtype)))
        return y + ops.cast(self.bias, x.dtype) if self.bias is not None else y


class RMSNorm(keras.layers.Layer):
    def __init__(self, dim, eps, weight_dtype="float32", **kwargs):
        super().__init__(dtype=weight_dtype, **kwargs)
        self.weight = self.add_weight(name="weight", shape=(dim,), initializer="ones", dtype=weight_dtype)
        self.eps = eps
        self.built = True

    def call(self, x):
        variance = ops.mean(ops.square(ops.cast(x, "float32")), axis=-1, keepdims=True)
        return ops.cast(x * ops.rsqrt(variance + self.eps), x.dtype) * ops.cast(self.weight, x.dtype)


def _apply_rope(x, table):
    rot = table.shape[-1]
    half = rot // 2
    first, second = x[..., :half], x[..., half:rot]
    cos, sin = table[..., :half], table[..., half:]
    rotated = ops.concatenate((first * cos - second * sin, second * cos + first * sin), axis=-1)
    return ops.concatenate((rotated, x[..., rot:]), axis=-1)


class Attention(keras.layers.Layer):
    def __init__(self, hidden, heads, head_dim, eps, gate_compress=False, weight_dtype="float32",
                 quantized=False, **kwargs):
        super().__init__(dtype=weight_dtype, **kwargs)
        self.heads, self.head_dim = heads, head_dim
        inner = heads * head_dim
        self.qkv_proj = TorchLinear(hidden, inner * 3, bias=False, weight_dtype=weight_dtype, quantized=quantized, name="qkv_proj")
        self.q_norm = RMSNorm(head_dim, eps, weight_dtype=weight_dtype)
        self.k_norm = RMSNorm(head_dim, eps, weight_dtype=weight_dtype)
        self.out_proj = TorchLinear(inner, hidden, bias=False, weight_dtype=weight_dtype, quantized=quantized, name="out_proj")
        if gate_compress:
            self.to_gate_compress = TorchLinear(hidden, inner, bias=False, weight_dtype=weight_dtype)
        self.built = True

    def call(self, x, rope=None):
        s = x.shape[0]
        q, k, v = ops.split(self.qkv_proj(x), 3, axis=-1)
        q = self.q_norm(ops.reshape(q, (s, self.heads, self.head_dim)))
        k = self.k_norm(ops.reshape(k, (s, self.heads, self.head_dim)))
        v = ops.reshape(v, (s, self.heads, self.head_dim))
        if rope is not None:
            q, k = _apply_rope(q, rope), _apply_rope(k, rope)
        out = ops.dot_product_attention(q[None], k[None], v[None])
        return self.out_proj(ops.reshape(out, (s, self.heads * self.head_dim)))


class MLP(keras.layers.Layer):
    def __init__(self, hidden, ffn, weight_dtype="float32", quantized=False, **kwargs):
        super().__init__(dtype=weight_dtype, **kwargs)
        self.fc1 = TorchLinear(hidden, ffn * 2, bias=False, weight_dtype=weight_dtype, quantized=quantized, name="fc1")
        self.fc2 = TorchLinear(ffn, hidden, bias=False, weight_dtype=weight_dtype, quantized=quantized, name="fc2")
        self.built = True

    def call(self, x):
        gate, value = ops.split(self.fc1(x), 2, axis=-1)
        return self.fc2(ops.silu(gate) * value)


class RefinerBlock(keras.layers.Layer):
    def __init__(self, hidden, heads, head_dim, ffn, eps, qk_eps, weight_dtype="float32", **kwargs):
        super().__init__(dtype=weight_dtype, **kwargs)
        self.norm1 = RMSNorm(hidden, eps, weight_dtype=weight_dtype)
        self.norm2 = RMSNorm(hidden, eps, weight_dtype=weight_dtype)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, weight_dtype=weight_dtype)
        self.mlp = MLP(hidden, ffn, weight_dtype=weight_dtype)
        self.built = True

    def call(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TokenRefiner(keras.layers.Layer):
    def __init__(self, layers, hidden, heads, head_dim, ffn, eps, qk_eps, final_eps, weight_dtype="float32", **kwargs):
        super().__init__(dtype=weight_dtype, **kwargs)
        self.blocks = [RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps, weight_dtype=weight_dtype) for _ in range(layers)]
        self.final_norm = RMSNorm(hidden, final_eps, weight_dtype=weight_dtype)
        self.built = True

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class TimeEmbedder(keras.layers.Layer):
    def __init__(self, freq_dim, hidden, output, **kwargs):
        super().__init__(**kwargs)
        self.freq_dim = freq_dim
        self.proj_in = TorchLinear(freq_dim, hidden)
        self.proj_out = TorchLinear(hidden, output)
        self.built = True

    def call(self, t):
        half = self.freq_dim // 2
        freq = ops.exp(-math.log(10000.0) * ops.arange(half, dtype="float32") / half)
        angles = ops.cast(t[:, None], "float32") * freq[None]
        emb = ops.concatenate((ops.cos(angles), ops.sin(angles)), axis=-1)
        return self.proj_out(ops.silu(self.proj_in(emb)))


class AdalnProj(keras.layers.Layer):
    def __init__(self, t_dim, hidden, expand, modalities, apply_silu, weight_dtype="float32", **kwargs):
        super().__init__(dtype=weight_dtype, **kwargs)
        self.hidden, self.expand, self.modalities = hidden, expand, modalities
        self.apply_silu = apply_silu
        self.linear = TorchLinear(t_dim, expand * hidden * modalities, weight_dtype=weight_dtype)
        self.built = True

    def call(self, t_emb):
        x = self.linear(ops.silu(t_emb) if self.apply_silu else t_emb)
        x = ops.reshape(x, (-1, self.expand, self.hidden))
        return ops.unstack(x, axis=1)


class DiTBlock(keras.layers.Layer):
    def __init__(self, hidden, heads, head_dim, ffn, t_dim, eps, qk_eps, apply_silu, gate_compress,
                 weight_dtype="float32", quantized=False, **kwargs):
        super().__init__(dtype=weight_dtype, **kwargs)
        self.norm1 = RMSNorm(hidden, eps, weight_dtype=weight_dtype)
        self.norm2 = RMSNorm(hidden, eps, weight_dtype=weight_dtype)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, gate_compress=gate_compress,
                              weight_dtype=weight_dtype, quantized=quantized)
        self.mlp = MLP(hidden, ffn, weight_dtype=weight_dtype, quantized=quantized)
        self.adaln_proj = AdalnProj(t_dim, hidden, 6, 3, apply_silu,
                                    weight_dtype=weight_dtype if apply_silu else "float32")
        self.built = True

    def call(self, x, t_emb, mod_index, rope):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln_proj(t_emb)
        h = self.norm1(x)
        h = h * (1.0 + ops.cast(ops.take(scale1, mod_index, axis=0), h.dtype)) + ops.cast(ops.take(shift1, mod_index, axis=0), h.dtype)
        x = x + self.attn(h, rope) * ops.cast(ops.take(gate1, mod_index, axis=0), x.dtype)
        h = self.norm2(x)
        h = h * (1.0 + ops.cast(ops.take(scale2, mod_index, axis=0), h.dtype)) + ops.cast(ops.take(shift2, mod_index, axis=0), h.dtype)
        return x + self.mlp(h) * ops.cast(ops.take(gate2, mod_index, axis=0), x.dtype)


class FinalLayer(keras.layers.Layer):
    def __init__(self, hidden, t_dim, video_dim, audio_dim, eps, apply_silu, video_heads=1, audio_heads=1, weight_dtype="float32", **kwargs):
        super().__init__(dtype=weight_dtype, **kwargs)
        self.norm = RMSNorm(hidden, eps, weight_dtype=weight_dtype)
        self.adaln_proj = AdalnProj(t_dim, hidden, 2, 1, apply_silu,
                                    weight_dtype=weight_dtype if apply_silu else "float32")
        self.video_out = TorchLinear(hidden, video_dim * video_heads)
        self.audio_out = TorchLinear(hidden, audio_dim * audio_heads)
        self.built = True

    def predict_streams(self, x, t_emb, video_slice, audio_slice, video_t_index, audio_t_index, heads=None):
        shift, scale = self.adaln_proj(t_emb)
        def mod(span, indices):
            h = self.norm(x[span[0]:span[1]])
            return ops.cast(h * (1.0 + ops.take(scale, indices, axis=0)) + ops.take(shift, indices, axis=0), "float32")
        video, audio = mod(video_slice, video_t_index), mod(audio_slice, audio_t_index)
        if heads is None:
            return self.video_out(video), self.audio_out(audio)
        return (_pdd_head(self.video_out, video, *heads[0]),
                _pdd_head(self.audio_out, audio, *heads[1]))


def _pdd_head(head, x, n, start, stop, shift):
    grid = np.linspace(1.0, 0.0, n + 1)
    dt = np.diff(1.0 - shift * grid / (1.0 + (shift - 1.0) * grid))[start:stop]
    weights = ops.convert_to_tensor(dt / dt.sum(), dtype="float32")
    rows = ops.reshape(head.weight, (n, -1, head.weight.shape[1]))
    bias = ops.reshape(head.bias, (n, -1))
    first = max(start, 1)
    weight = rows[0] + ops.tensordot(weights[first - start:], rows[first:stop], axes=1)
    offset = bias[0] + ops.tensordot(weights[first - start:], bias[first:stop], axes=1)
    return ops.matmul(x, ops.transpose(weight)) + offset


class MiniMaxH3JAX(keras.Model):
    """H3 denoiser; call ``load_state_dict`` before ``denoise``."""

    def __init__(self, hidden_size=5376, num_layers=50, token_refiner_num_layers=2,
                 num_attention_heads=56, attention_head_dim=128, ffn_hidden_size=14336,
                 latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2), text_dim=5120,
                 timestep_input_dim=256, time_embed_hidden_size=5376, time_embed_dim=2688,
                 rope_inv_freq_len=16, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
                 sigma_shift_video=12.0, sigma_shift_audio=3.0, adaln_curve_grid=None,
                 gate_compress=False, video_heads=1, audio_heads=1,
                 weight_dtype="float32", compute_dtype="float32", jit_blocks=True,
                 quantized_blocks=False, devices=None, tensor_parallel=None, **kwargs):
        super().__init__(dtype=compute_dtype, **kwargs)
        if tensor_parallel is None:
            tensor_parallel = jax.default_backend() == "tpu"
        if devices is None:
            devices = jax.devices("tpu") if jax.default_backend() == "tpu" else jax.devices()[:1]
        self.stage_devices = tuple(devices)
        self.tensor_parallel = tensor_parallel
        if tensor_parallel:
            mesh = keras.distribution.DeviceMesh(shape=(len(self.stage_devices),), axis_names=("tp",), devices=self.stage_devices)
            layouts = keras.distribution.LayoutMap(mesh)
            for name in ("condition_proj", "qkv_proj", "fc1"):
                layouts[f"{name}/weight"] = ("tp", None)
                if name != "condition_proj":
                    layouts[f"{name}/weight_scale"] = ("tp", None)
            layouts["condition_proj/bias"] = ("tp",)
            for name in ("out_proj", "fc2"):
                layouts[f"{name}/weight"] = (None, "tp")
            self.distribution = keras.distribution.ModelParallel(layout_map=layouts)
        else:
            self.distribution = None
        self.hidden_size = hidden_size
        self.patch_size = tuple(patch_size)
        self.latents_dim = latents_dim
        self.sigma_shift_video = sigma_shift_video
        self.sigma_shift_audio = sigma_shift_audio
        self.video_heads, self.audio_heads = video_heads, audio_heads
        self.diffusion_compute_dtype = compute_dtype
        self.quantized_blocks = quantized_blocks
        self.jit_blocks = jit_blocks
        self._compiled_block = None
        with self.distribution.scope() if tensor_parallel else jax.default_device(self.stage_devices[0]):
            self.video_patch_proj = TorchLinear(latents_dim * math.prod(patch_size), hidden_size)
            self.audio_patch_proj = TorchLinear(audio_latents_dim, hidden_size)
            self.condition_proj = TorchLinear(text_dim, hidden_size, weight_dtype=weight_dtype, name="condition_proj")
            if adaln_curve_grid is None:
                self.time_embedder = TimeEmbedder(timestep_input_dim, time_embed_hidden_size, time_embed_dim)
            else:
                self.adaln_t_table = self.add_weight(name="adaln_t_table", shape=(adaln_curve_grid, time_embed_dim), initializer="zeros")
            self.rope_inv_freq = self.add_weight(name="inv_freq", shape=(rope_inv_freq_len,), initializer="zeros", trainable=False)
            self.token_refiner = TokenRefiner(token_refiner_num_layers, hidden_size, num_attention_heads,
                                              attention_head_dim, ffn_hidden_size, norm_eps, qk_norm_eps, final_norm_eps,
                                              weight_dtype=weight_dtype)
        self.block_devices = tuple(self.stage_devices[0] if tensor_parallel else
                                   self.stage_devices[min(i * len(self.stage_devices) // num_layers,
                                                          len(self.stage_devices) - 1)] for i in range(num_layers))
        blocks = []
        for device in self.block_devices:
            with self.distribution.scope() if tensor_parallel else jax.default_device(device):
                blocks.append(DiTBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size,
                                       time_embed_dim, norm_eps, qk_eps=qk_norm_eps,
                                       apply_silu=adaln_curve_grid is None, gate_compress=gate_compress,
                                       weight_dtype=weight_dtype, quantized=quantized_blocks))
        self.blocks = blocks
        with self.distribution.scope() if tensor_parallel else jax.default_device(self.block_devices[-1] if self.block_devices else self.stage_devices[0]):
            self.final_layer = FinalLayer(hidden_size, time_embed_dim, latents_dim * math.prod(patch_size),
                                          audio_latents_dim, final_norm_eps, adaln_curve_grid is None,
                                          video_heads=video_heads, audio_heads=audio_heads, weight_dtype=weight_dtype)
        self.built = True

    def _weights_by_key(self):
        out = {"rope.inv_freq": self.rope_inv_freq}
        if hasattr(self, "adaln_t_table"):
            out["adaln_t_table"] = self.adaln_t_table
        def visit(obj, prefix):
            for name, value in vars(obj).items():
                if name.startswith("_"):
                    continue
                if isinstance(value, TorchLinear):
                    out[prefix + name + ".weight"] = value.weight
                    if value.weight_scale is not None:
                        out[prefix + name + ".weight_scale"] = value.weight_scale
                    if value.bias is not None:
                        out[prefix + name + ".bias"] = value.bias
                elif isinstance(value, RMSNorm):
                    out[prefix + name + ".weight"] = value.weight
                elif isinstance(value, keras.layers.Layer):
                    visit(value, prefix + name + ".")
                elif isinstance(value, list):
                    for i, layer in enumerate(value):
                        if isinstance(layer, keras.layers.Layer):
                            visit(layer, prefix + name + f".{i}.")
        visit(self, "")
        return out

    def load_state_dict(self, state_dict):
        """Assign a PyTorch/NumPy H3 state dict, preserving its original keys."""
        expected = self._weights_by_key()
        missing = expected.keys() - state_dict.keys()
        if missing:
            raise KeyError("Missing MiniMax H3 weights: " + ", ".join(sorted(missing)[:5]))
        self._check_quant_metadata(state_dict)
        for key, var in expected.items():
            self._assign_weight(key, var, state_dict[key])

    def _check_quant_metadata(self, state_dict):
        for key in self._weights_by_key():
            if not key.endswith("weight_scale"):
                continue
            meta_key = key.removesuffix("weight_scale") + "comfy_quant"
            meta = state_dict.get(meta_key)
            if meta is None:
                raise KeyError(f"Missing MiniMax H3 quantization metadata: {meta_key}")
            if hasattr(meta, "detach"):
                meta = meta.detach().cpu().numpy()
            config = json.loads(np.asarray(meta, dtype=np.uint8).tobytes())
            if (config.get("format") != "int8_tensorwise" or not config.get("convrot")
                    or config.get("convrot_groupsize") != 256):
                raise ValueError(f"Unsupported MiniMax H3 quantization: {meta_key}")

    @staticmethod
    def _assign_weight(key, var, value):
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            if "bfloat16" in str(value.dtype):
                value = value.float()
            value = value.numpy()
        if tuple(value.shape) != tuple(var.shape):
            raise ValueError(f"{key}: expected {tuple(var.shape)}, got {tuple(value.shape)}")
        var.assign(jax.device_put(value, var.value.sharding))

    @classmethod
    def _config_from_state_dict(cls, state_dict):
        keys = state_dict.keys()
        count = lambda prefix: len({int(k[len(prefix):].split(".", 1)[0]) for k in keys if k.startswith(prefix)})
        video_in = state_dict["video_patch_proj.weight"]
        q_norm = state_dict["blocks.0.attn.q_norm.weight"]
        qkv = state_dict["blocks.0.attn.qkv_proj.weight"]
        cfg = dict(hidden_size=video_in.shape[0], num_layers=count("blocks."),
                   token_refiner_num_layers=count("token_refiner.blocks."),
                   num_attention_heads=qkv.shape[0] // (3 * q_norm.shape[0]),
                   attention_head_dim=q_norm.shape[0],
                   ffn_hidden_size=state_dict["blocks.0.mlp.fc1.weight"].shape[0] // 2,
                   latents_dim=video_in.shape[1] // 4,
                   audio_latents_dim=state_dict["audio_patch_proj.weight"].shape[1],
                   text_dim=state_dict["condition_proj.weight"].shape[1],
                   rope_inv_freq_len=state_dict["rope.inv_freq"].shape[0],
                   gate_compress="blocks.0.attn.to_gate_compress.weight" in keys,
                   video_heads=state_dict["final_layer.video_out.weight"].shape[0] // video_in.shape[1],
                   audio_heads=state_dict["final_layer.audio_out.weight"].shape[0] // state_dict["audio_patch_proj.weight"].shape[1])
        cfg["quantized_blocks"] = str(qkv.dtype).lower() in ("torch.int8", "int8", "i8")
        main_dtype = state_dict["blocks.0.norm1.weight"].dtype
        if "bfloat16" in str(main_dtype).lower() or "bf16" in str(main_dtype).lower():
            cfg["weight_dtype"] = cfg["compute_dtype"] = "bfloat16"
        if "adaln_t_table" in keys:
            cfg["adaln_curve_grid"], cfg["time_embed_dim"] = state_dict["adaln_t_table"].shape
        else:
            cfg["timestep_input_dim"] = state_dict["time_embedder.proj_in.weight"].shape[1]
            cfg["time_embed_hidden_size"] = state_dict["time_embedder.proj_in.weight"].shape[0]
            cfg["time_embed_dim"] = state_dict["time_embedder.proj_out.weight"].shape[0]
        return cfg

    @classmethod
    def from_state_dict(cls, state_dict, **overrides):
        """Construct and load from a ComfyUI H3 state dict."""
        cfg = cls._config_from_state_dict(state_dict)
        cfg.update(overrides)
        model = cls(**cfg)
        model.load_state_dict(state_dict)
        return model

    @classmethod
    def from_safetensors(cls, path, prefix="", **overrides):
        """Read a .safetensors checkpoint one weight at a time."""
        with TensorFile(path) as file:
            info = {key[len(prefix):]: file.get_slice(key)
                    for key in file.keys() if key.startswith(prefix) and key != "__metadata__"}
            cfg = cls._config_from_state_dict(info)
            cfg.update(overrides)
            model = cls(**cfg)
            expected = model._weights_by_key()
            missing = expected.keys() - info.keys()
            if missing:
                raise KeyError("Missing MiniMax H3 weights: " + ", ".join(sorted(missing)[:5]))
            metadata = {key: file.get_tensor(prefix + key) for key in info if key.endswith("comfy_quant")}
            model._check_quant_metadata(metadata)
            for key, var in expected.items():
                model._assign_weight(key, var, file.get_tensor(prefix + key))
        return model

    def preprocess_text_embeds(self, context):
        if context.shape[-1] == self.hidden_size:
            return context
        return self.token_refiner(self.condition_proj(context[0]))[None]

    def _rope(self, position_ids):
        angles = ops.convert_to_tensor(position_ids)[:, :, None] * self.rope_inv_freq[None, None, :]
        half = ops.reshape(angles, (len(position_ids), -1))
        half = ops.concatenate((half, half), axis=-1)
        return ops.cast(ops.concatenate((ops.cos(half[:, :half.shape[-1] // 2]),
                                          ops.sin(half[:, :half.shape[-1] // 2])), axis=-1), self.diffusion_compute_dtype)[:, None]

    def _time_emb(self, values):
        t = ops.convert_to_tensor(values, dtype="float32")
        if hasattr(self, "adaln_t_table"):
            table = self.adaln_t_table
            pos = ops.clip(t, 0.0, 1.0) * (table.shape[0] - 1)
            i0 = ops.minimum(ops.cast(ops.floor(pos), "int32"), table.shape[0] - 2)
            return ops.take(table, i0, axis=0) * (1.0 - (pos - i0)[:, None]) + ops.take(table, i0 + 1, axis=0) * (pos - i0)[:, None]
        return self.time_embedder(t)

    def call(self, inputs):
        h, t_emb, mod_index, rope = inputs
        for block, device in zip(self.blocks, self.block_devices):
            if self.tensor_parallel:
                h = block.call(h, t_emb, mod_index, rope)
            else:
                h = block.call(jax.device_put(h, device), jax.device_put(t_emb, device),
                               jax.device_put(mod_index, device), jax.device_put(rope, device))
        return h

    def _run_blocks(self, h, t_emb, mod_index, rope):
        if not self.jit_blocks or not self.blocks:
            return self.call((h, t_emb, mod_index, rope))
        if self._compiled_block is None:
            block = self.blocks[0]
            def apply(weights, state, x, times, indices, rotation):
                output, _ = block.stateless_call(weights, state, x, times, indices, rotation)
                return output
            self._compiled_block = jax.jit(apply)
        for block, device in zip(self.blocks, self.block_devices):
            if self.tensor_parallel:
                times, indices, rotation = t_emb, mod_index, rope
            else:
                h = jax.device_put(h, device)
                times = jax.device_put(t_emb, device)
                indices = jax.device_put(mod_index, device)
                rotation = jax.device_put(rope, device)
            weights = tuple(var.value for var in block.trainable_variables)
            state = tuple(var.value for var in block.non_trainable_variables)
            h = self._compiled_block(weights, state, h, times, indices, rotation)
        return h

    def denoise(self, video_x, audio_x, timestep, context, payload=None,
                denoise_mask=None, audio_denoise_mask=None, sample_sigmas=None):
        """Return H3 video/audio velocities for sigma * 1000 timestep."""
        payload = payload or {}
        video_x, audio_x, context = map(jnp.asarray, (video_x, audio_x, context))
        orig_t, orig_h, orig_w = video_x.shape[2:]
        pt, ph, pw = self.patch_size
        video_x = jnp.pad(video_x, ((0, 0), (0, 0), (0, -orig_t % pt), (0, -orig_h % ph), (0, -orig_w % pw)), mode="wrap")
        if video_x.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        latent_t, lat_h, lat_w = video_x.shape[2:]
        audio_t, text_len = audio_x.shape[-1], context.shape[1]
        layout = payload.get("layout")
        if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
            layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t, payload.get("keyframes"), payload.get("refs"))
        sigma_v = max(float(np.asarray(timestep).reshape(-1)[0]) / 1000.0, 1e-6)
        sigma_a = time_shift_sigma(sigma_v, self.sigma_shift_video, self.sigma_shift_audio)
        scale = float(payload.get("audio_scale", 1.0))
        audio_src = audio_x
        if scale != 1.0:
            audio_x = audio_x * (sigma_a / sigma_v)
        t_v, t_a = 1.0 - sigma_v, 1.0 - sigma_a
        vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
        aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
        seg_t = {"text": t_v, "video": t_v, "audio": t_a,
                 "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
                 "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug)}
        video_rows_t = audio_rows_t = None
        if denoise_mask is not None:
            m = np.asarray(denoise_mask)[0, 0]
            m = np.pad(m, ((0, 0), (0, lat_h - m.shape[-2]), (0, lat_w - m.shape[-1])), mode="edge")
            m = m.reshape(latent_t, lat_h // 2, 2, lat_w // 2, 2).max(axis=(2, 4)).ravel()
            if not np.all(m >= 1.0 - 1e-3):
                video_rows_t = np.minimum(1.0 - m * sigma_v, max(t_v, VISUAL_COND_TIMESTEP))
        if audio_denoise_mask is not None:
            m = np.asarray(audio_denoise_mask)[0, 0].ravel()
            if not np.all(m >= 1.0 - 1e-3):
                audio_rows_t = np.minimum(1.0 - m * sigma_a, max(t_a, AUDIO_COND_TIMESTEP))
        unique_t = sorted({t_v, t_a} | set(seg_t.values())
                          | (set(video_rows_t.tolist()) if video_rows_t is not None else set())
                          | (set(audio_rows_t.tolist()) if audio_rows_t is not None else set()))
        t_row = {t: i for i, t in enumerate(unique_t)}
        seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "cond_audio": 2, "ref_audio": 2}
        mod_index = np.empty(layout.seq_len, dtype=np.int32)
        text_tags = payload.get("text_token_tags")
        for a, b, kind in layout.segments:
            if kind == "text" and text_tags is not None:
                mod_index[a:b] = t_row[seg_t[kind]] * 3 + np.asarray(text_tags).reshape(-1)[:b-a]
            elif kind == "video" and video_rows_t is not None:
                mod_index[a:b] = np.asarray([t_row[t] * 3 for t in video_rows_t])
            elif kind == "audio" and audio_rows_t is not None:
                mod_index[a:b] = np.asarray([t_row[t] * 3 + 2 for t in audio_rows_t])
            else:
                mod_index[a:b] = t_row[seg_t[kind]] * 3 + seg_tag[kind]

        def condition_rows(name, pack, aug, seed):
            rows = []
            for z in payload.get(name, ()):
                r = pack(jnp.asarray(z, dtype=jnp.float32))
                if aug < 1.0:
                    noise = jax.random.normal(jax.random.PRNGKey(seed), r.shape, dtype=jnp.float32)
                    r = aug * r + (1.0 - aug) * noise
                rows.append(r)
            return ops.concatenate(rows, axis=0) if rows else None

        video_rows = patchify_video(ops.cast(video_x, "float32"), self.patch_size)
        audio_rows = pack_audio(ops.cast(audio_x, "float32"))
        cond_video = condition_rows("cond_video_latents", lambda z: patchify_video(z, self.patch_size), vis_aug, int(payload.get("seed", 0)))
        cond_audio = condition_rows("cond_audio_latents", pack_audio, aud_aug, int(payload.get("seed", 0)) + 1)
        if cond_video is not None:
            all_video = ops.zeros((len(layout.img_pos), video_rows.shape[-1]), dtype="float32")
            all_video = all_video.at[~layout.img_update].set(cond_video).at[layout.img_update].set(video_rows)
        else:
            all_video = video_rows
        if cond_audio is not None:
            all_audio = ops.zeros((len(layout.audio_pos), audio_rows.shape[-1]), dtype="float32")
            all_audio = all_audio.at[~layout.audio_update].set(cond_audio).at[layout.audio_update].set(audio_rows)
        else:
            all_audio = audio_rows
        video_embed = ops.cast(self.video_patch_proj(all_video), self.diffusion_compute_dtype)
        audio_embed = ops.cast(self.audio_patch_proj(all_audio), self.diffusion_compute_dtype)
        text_states = ops.cast(self.preprocess_text_embeds(context)[0], self.diffusion_compute_dtype)
        parts, voff, aoff = [], 0, 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                parts.append(text_states)
            elif kind in ("cond", "ref_img", "video"):
                parts.append(video_embed[voff:voff+n])
                voff += n
            else:
                parts.append(audio_embed[aoff:aoff+n])
                aoff += n
        h = ops.concatenate(parts, axis=0)
        t_emb = self._time_emb(unique_t)
        if not hasattr(self, "adaln_t_table"):
            t_emb = ops.cast(t_emb, self.diffusion_compute_dtype)
        rope = self._rope(layout.position_ids)
        video_slice = next((a, b) for a, b, kind in layout.segments if kind == "video")
        audio_slice = next((a, b) for a, b, kind in layout.segments if kind == "audio")
        video_t_index = jnp.asarray(mod_index[video_slice[0]:video_slice[1]] // 3, dtype=jnp.int32)
        audio_t_index = jnp.asarray(mod_index[audio_slice[0]:audio_slice[1]] // 3, dtype=jnp.int32)
        heads = None
        if self.video_heads > 1 or self.audio_heads > 1:
            if sample_sigmas is None:
                raise ValueError("MiniMax H3 PDD heads need the sampler's sigma schedule")
            schedule = np.asarray(sample_sigmas)
            i = int(np.argmin(np.abs(schedule - sigma_v)))
            next_sigma = schedule[min(i + 1, len(schedule) - 1)]
            def span(n, shift):
                start, stop = (round((1.0 - time_shift_sigma(s, self.sigma_shift_video, 1.0)) * n) for s in (sigma_v, next_sigma))
                return (n, min(start, n - 1), max(stop, min(start, n - 1) + 1), shift)
            heads = (span(self.video_heads, self.sigma_shift_video), span(self.audio_heads, self.sigma_shift_audio))
        h = self._run_blocks(h, t_emb, jnp.asarray(mod_index), rope)
        final_device = self.block_devices[-1] if self.block_devices else self.stage_devices[0]
        place = (lambda x: x) if self.tensor_parallel else (lambda x: jax.device_put(x, final_device))
        video_out, audio_out = self.final_layer.predict_streams(
            h, place(t_emb), video_slice, audio_slice,
            place(video_t_index), place(audio_t_index), heads)
        video_out = -unpatchify_video(video_out, latent_t // pt, lat_h // ph, lat_w // pw, self.latents_dim, self.patch_size)
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        audio_out = -unpack_audio(audio_out)
        if denoise_mask is not None:
            video_out = video_out * place(jnp.asarray(denoise_mask))
        if audio_denoise_mask is not None:
            audio_out = audio_out * place(jnp.asarray(audio_denoise_mask))
        if scale != 1.0:
            carry = sigma_a / sigma_v
            audio_out = (1.0 - scale) * place(audio_src) * carry + (1.0 + (scale - 1.0) * sigma_a) * audio_out
        return video_out.astype(video_x.dtype), audio_out.astype(audio_x.dtype)

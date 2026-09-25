"""Strict PyTorch/XLA placement for MiniMax H3 text and VAE modules."""

import numpy as np
import torch


_COLUMN_LINEAR = ("q_proj", "k_proj", "v_proj", "qkv_proj", "qkv", "to_qkv", "gate_proj",
                  "up_proj", "fc1", "linear_fc1", "w1", "proj_in", "condition_proj", "proj_out", "lm_head")
_ROW_LINEAR = ("o_proj", "out_proj", "to_out", "proj", "down_proj", "fc2", "linear_fc2", "w2")


def _weight_spec(name, module, chips):
    if isinstance(module, torch.nn.Embedding):
        return ("tp", None) if module.weight.shape[0] % chips == 0 else None
    if module.weight.ndim == 2:
        role = name.rsplit(".", 1)[-1]
        if role in _COLUMN_LINEAR and module.weight.shape[0] % chips == 0:
            return ("tp", None)
        if role in _ROW_LINEAR and module.weight.shape[1] % chips == 0:
            return (None, "tp")
    if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv3d)) and module.groups == 1:
        return ("tp",) + (None,) * (module.weight.ndim - 1) if module.weight.shape[0] % chips == 0 else None
    if isinstance(module, torch.nn.ConvTranspose1d) and module.groups == 1:
        return (None, "tp") + (None,) * (module.weight.ndim - 2) if module.weight.shape[1] % chips == 0 else None
    return None


class H3XLARuntime:
    """One process controlling all eight v5e devices via PyTorch/XLA SPMD."""

    def __init__(self, chips=8):
        import torch_xla
        import torch_xla.debug.metrics as metrics
        import torch_xla.distributed.spmd as spmd
        import torch_xla.runtime as runtime

        if runtime.device_type() != "TPU":
            raise RuntimeError("MiniMax H3 XLA execution requires TPU; CPU XLA is not accepted")
        runtime.use_spmd()
        count = runtime.global_runtime_device_count()
        if count != chips:
            raise RuntimeError(f"Expected {chips} TPU chips, found {count}")
        self.device = torch_xla.device()
        self.mesh = spmd.Mesh(np.arange(chips), (chips,), ("tp",))
        self.spmd = spmd
        self.metrics = metrics
        self.torch_xla = torch_xla
        self.chips = chips

    def place(self, model):
        model.to(self.device)
        for name, module in model.named_modules():
            if not hasattr(module, "weight") or module.weight is None:
                continue
            spec = _weight_spec(name, module, self.chips)
            weight = module.weight
            if spec is None:
                if weight.numel() >= 1_000_000:
                    raise RuntimeError(f"Large H3 layer has no TP layout: {name} {tuple(weight.shape)}")
                continue
            qdata = weight._qdata if hasattr(weight, "_qdata") else weight
            if qdata.device.type != "xla":
                raise RuntimeError(f"H3 weight remained on CPU: {name}")
            self.spmd.mark_sharding(qdata, self.mesh, spec)
            if hasattr(weight, "_params") and hasattr(weight._params, "scale"):
                scale = weight._params.scale
                if scale.device.type != "xla":
                    raise RuntimeError(f"H3 quantization scale remained on CPU: {name}")
                self.spmd.mark_sharding(scale, self.mesh,
                                        (spec[0],) + (None,) * (scale.ndim - 1))
            if getattr(module, "bias", None) is not None and spec[0] == "tp":
                self.spmd.mark_sharding(module.bias, self.mesh, ("tp",))
        for name, param in model.named_parameters():
            if param.device.type != "xla":
                raise RuntimeError(f"H3 parameter remained on CPU: {name}")
        return model

    def forward(self, model, *args, **kwargs):
        for value in torch.utils._pytree.tree_leaves((args, kwargs)):
            if isinstance(value, torch.Tensor) and value.device.type != "xla":
                raise RuntimeError("H3 XLA forward received a CPU tensor")
        before = {name: self.metrics.counter_value(name) for name in self.metrics.counter_names()
                  if name.startswith("aten::")}
        output = model(*args, **kwargs)
        self.torch_xla.sync()
        for value in torch.utils._pytree.tree_leaves(output):
            if isinstance(value, torch.Tensor) and value.device.type != "xla":
                raise RuntimeError("H3 XLA forward returned a CPU tensor")
        fallback = {name: self.metrics.counter_value(name) - before.get(name, 0)
                    for name in self.metrics.counter_names() if name.startswith("aten::")}
        fallback = {name: count for name, count in fallback.items() if count}
        if fallback:
            raise RuntimeError(f"H3 XLA forward used CPU fallback: {fallback}")
        return output

    def to_host(self, output):
        """Transfer completed XLA outputs for the authorized JAX process boundary."""
        self.torch_xla.sync()
        return torch.utils._pytree.tree_map(
            lambda value: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value,
            output)

import torch

from comfy.ldm.minimax.xla_backend import _weight_spec


def test_h3_xla_tp_weight_layouts():
    assert _weight_spec("condition_proj", torch.nn.Linear(5120, 5376, bias=False, device="meta"), 8) == ("tp", None)
    assert _weight_spec("time_embedder.proj_in", torch.nn.Linear(256, 5376, bias=False, device="meta"), 8) == ("tp", None)
    assert _weight_spec("model.layers.0.self_attn.q_proj", torch.nn.Linear(5120, 8192, bias=False, device="meta"), 8) == ("tp", None)
    assert _weight_spec("model.layers.0.self_attn.o_proj", torch.nn.Linear(8192, 5120, bias=False, device="meta"), 8) == (None, "tp")
    assert _weight_spec("model.layers.0.mlp.down_proj", torch.nn.Linear(25600, 5120, bias=False, device="meta"), 8) == (None, "tp")
    assert _weight_spec("visual.blocks.0.attn.qkv", torch.nn.Linear(1152, 3456, bias=False, device="meta"), 8) == ("tp", None)
    assert _weight_spec("decoder.transformer_blocks.0.ff.w2", torch.nn.Linear(8192, 2048, bias=False, device="meta"), 8) == (None, "tp")
    assert _weight_spec("decoder.proj_out", torch.nn.Linear(2048, 3072, bias=False, device="meta"), 8) == ("tp", None)

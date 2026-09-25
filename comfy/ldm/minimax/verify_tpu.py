"""Collect MiniMax H3 ComfyUI/XLA and JAX diagnostics in separate TPU processes."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


def read_prompt(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected a ComfyUI API-format prompt JSON")
    prompt = data.get("prompt", data)
    if not isinstance(prompt, dict) or not prompt or not all(
            isinstance(node, dict) and "class_type" in node and "inputs" in node
            for node in prompt.values()):
        raise ValueError("Expected a ComfyUI API-format prompt JSON")
    return prompt


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def xla_stage(args):
    import comfy.options

    comfy.options.enable_args_parsing()
    sys.argv = [sys.argv[0], "--xla", "--bf16-text-enc", "--bf16-vae", "--bf16-unet"]
    import folder_paths
    import comfy.model_management as mm
    import nodes
    import execution
    import torch
    import torch_xla
    import torch_xla.debug.metrics as metrics
    import torch_xla.debug.profiler as profiler
    import torch_xla.runtime as xr

    if args.models_dir:
        for kind in ("checkpoints", "diffusion_models", "text_encoders", "vae"):
            folder_paths.add_model_folder_path(kind, str(Path(args.models_dir).resolve()))
    asyncio.run(nodes.init_builtin_extra_nodes())
    prompt = read_prompt(args.prompt)
    outputs = [node_id for node_id, node in prompt.items()
               if getattr(nodes.NODE_CLASS_MAPPINGS.get(node["class_type"]), "OUTPUT_NODE", False)]
    if not outputs:
        raise ValueError("The API prompt has no registered output node")

    class Server:
        client_id = None
        last_node_id = None

        def __init__(self):
            self.events = []
            self.trace_started = False
            self.trace_error = None

        def send_sync(self, event, data, client_id):
            if event == "executing" and data.get("node") is not None:
                self.events.append((time.perf_counter(), data["node"]))
                node = prompt.get(data["node"], {})
                if not self.trace_started and "sampler" in node.get("class_type", "").lower():
                    try:
                        profiler.trace_detached("localhost:9012", str(Path(args.report_dir) / "xla_trace"),
                                                duration_ms=10000)
                        self.trace_started = True
                    except (RuntimeError, AttributeError) as error:
                        self.trace_error = str(error)

    profiler_error = None
    profiler_service = None
    try:
        profiler.start_server(9012)
        profiler_service = True
    except RuntimeError as error:
        profiler_error = str(error)

    class_sizes = []
    original_load = mm.LoadedModel.model_load

    def record_load(loaded, *call_args, **call_kwargs):
        result = original_load(loaded, *call_args, **call_kwargs)
        model = loaded.model.model
        large = []
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if weight is None or not hasattr(weight, "numel") or weight.numel() < 1_000_000:
                continue
            tensor = weight._qdata if hasattr(weight, "_qdata") else weight
            spec = torch_xla._XLAC._get_xla_sharding_spec(tensor) if tensor.device.type == "xla" else ""
            large.append({"name": name, "shape": list(weight.shape), "device": str(tensor.device),
                          "sharding": spec, "tp_sharded": "devices=" in spec and "8" in spec})
        class_sizes.append({"model": type(model).__name__, "device": str(loaded.device),
                            "large_weights": large,
                            "large_weight_count": len(large),
                            "tp_sharded_count": sum(w["tp_sharded"] for w in large)})
        return result

    mm.LoadedModel.model_load = record_load
    report = {"platform": platform.platform(), "python": sys.version, "torch": torch.__version__,
              "torch_xla": torch_xla.__version__, "device": str(mm.get_torch_device()),
              "device_type": xr.device_type(), "physical_devices": xr.global_runtime_device_count(),
              "workflow_nodes": {key: value["class_type"] for key, value in prompt.items()},
              "output_nodes": outputs, "runs": [], "loaded_models": class_sizes,
              "profiler_server_started": profiler_service is not None,
              "profiler_server_error": profiler_error}
    write_json(Path(args.report_dir) / "xla.json", report)
    server = Server()
    for run in range(args.runs):
        server.events = []
        metrics.clear_all()
        start = time.perf_counter()
        executor = execution.PromptExecutor(server, cache_type=False,
                                            cache_args={"ram": 0, "ram_inactive": 0})
        executor.execute(prompt, f"h3-xla-verify-{run}", execute_outputs=outputs)
        torch_xla.sync(wait=True)
        end = time.perf_counter()
        timings = []
        for index, (began, node_id) in enumerate(server.events):
            finished = server.events[index + 1][0] if index + 1 < len(server.events) else end
            timings.append({"node_id": node_id, "class_type": prompt.get(node_id, {}).get("class_type", "expanded node"),
                            "wall_seconds_to_next_node": round(finished - began, 4)})
        fallback = {name: metrics.counter_value(name) for name in metrics.counter_names()
                    if name.startswith("aten::") and metrics.counter_value(name)}
        metric_data = {}
        for name in ("CompileTime", "ExecuteTime", "ExecuteReplicatedTime",
                     "TransferToDeviceTime", "TransferFromDeviceTime"):
            if name in metrics.metric_names():
                count, total, samples = metrics.metric_data(name)
                metric_data[name] = {"samples": count, "total": str(total),
                                     "sample_values": [str(value) for _, value in samples]}
        (Path(args.report_dir) / f"xla_metrics_run_{run}.txt").write_text(
            metrics.metrics_report(), encoding="utf-8")
        errors = [{"node_id": data.get("node_id"), "node_type": data.get("node_type"),
                   "exception_type": data.get("exception_type"),
                   "exception_message": data.get("exception_message"), "traceback": data.get("traceback")}
                  for event, data in executor.status_messages if event == "execution_error"]
        history_result = getattr(executor, "history_result", {}) or {}
        report["runs"].append({"success": executor.success and not errors,
                               "wall_seconds": round(end - start, 4), "nodes": timings,
                               "cpu_fallback_counters": fallback, "no_cpu_fallback": not fallback,
                               "metrics": metric_data,
                               "errors": errors,
                               "workflow_outputs": history_result.get("outputs", {}),
                               "workflow_output_meta": history_result.get("meta", {}),
                               "memory": mm.get_free_memory(mm.get_torch_device())})
        report["trace_started"] = server.trace_started
        report["trace_error"] = server.trace_error
        write_json(Path(args.report_dir) / "xla.json", report)
        if errors:
            break
    return report


def jax_stage(args):
    os.environ["KERAS_BACKEND"] = "jax"
    import jax
    import jax.numpy as jnp
    import numpy as np
    import keras
    from comfy.ldm.minimax.jax_diffusion import MiniMaxH3JAX

    devices = jax.devices("tpu")
    if len(devices) != 8:
        raise RuntimeError(f"Expected eight JAX TPU devices, found {len(devices)}")
    report = {"jax": jax.__version__, "keras": keras.__version__,
              "devices": [str(device) for device in devices],
              "input_shape": {"video": [1, 24, args.video_t, args.video_h, args.video_w],
                              "audio": [1, 32, 2, args.audio_t],
                              "context_tokens": args.context_tokens},
              "input_kind": "synthetic_zero_latents_and_context",
              "returns_video": False, "returns": "video/audio denoise velocity",
              "runs": []}
    memory = {}
    for device in devices:
        try:
            memory[str(device)] = device.memory_stats()
        except RuntimeError as error:
            memory[str(device)] = {"error": str(error)}
    report["memory_before_load"] = memory
    start = time.perf_counter()
    model = MiniMaxH3JAX.from_safetensors(args.diffusion, devices=devices, tensor_parallel=True)
    report["load_seconds"] = round(time.perf_counter() - start, 4)
    memory = {}
    for device in devices:
        try:
            memory[str(device)] = device.memory_stats()
        except RuntimeError as error:
            memory[str(device)] = {"error": str(error)}
    report["memory_after_load"] = memory
    weights = model._weights_by_key()
    large = [{"name": key, "shape": list(var.shape), "sharding": str(var.value.sharding),
              "tp_sharded": "tp" in str(var.value.sharding.spec),
              "shard_count": len(var.value.addressable_shards),
              "shard_shapes": [list(shard.data.shape) for shard in var.value.addressable_shards],
              "shard_bytes": [int(shard.data.nbytes) for shard in var.value.addressable_shards]}
             for key, var in weights.items() if len(var.shape) == 2 and var.shape[0] * var.shape[1] >= 1_000_000]
    report["large_weights"] = large
    report["large_weight_count"] = len(large)
    report["tp_sharded_large_weight_count"] = sum(w["tp_sharded"] for w in large)
    report["large_weights_not_tp_sharded"] = [w["name"] for w in large if not w["tp_sharded"]]
    write_json(Path(args.report_dir) / "jax.json", report)

    video = jnp.zeros((1, model.latents_dim, args.video_t, args.video_h, args.video_w), dtype=jnp.bfloat16)
    audio = jnp.zeros((1, model.audio_latents_dim, 2, args.audio_t), dtype=jnp.bfloat16)
    context = jnp.zeros((1, args.context_tokens, model.condition_proj.weight.shape[1]), dtype=jnp.bfloat16)
    timestep = np.asarray([500.0], dtype=np.float32)
    sample_sigmas = np.linspace(1.0, 0.0, 17, dtype=np.float32)
    for run in range(args.runs):
        start = time.perf_counter()
        tracing = False
        if run == args.runs - 1:
            try:
                jax.profiler.start_trace(str(Path(args.report_dir) / "jax_trace"))
                tracing = True
            except RuntimeError as error:
                report["trace_error"] = str(error)
        try:
            output = model.denoise(video, audio, timestep, context, sample_sigmas=sample_sigmas)
            jax.block_until_ready(output)
        finally:
            if tracing:
                jax.profiler.stop_trace()
        report["runs"].append({"wall_seconds": round(time.perf_counter() - start, 4),
                               "output_shapes": [list(x.shape) for x in output],
                               "output_sharding": [str(x.sharding) for x in output]})
        write_json(Path(args.report_dir) / "jax.json", report)
    try:
        jax.profiler.save_device_memory_profile(str(Path(args.report_dir) / "jax_memory.pprof"))
    except RuntimeError as error:
        report["memory_profile_error"] = str(error)
        write_json(Path(args.report_dir) / "jax.json", report)
    return report


def child(args):
    path = Path(args.report_dir) / f"{args.stage}.json"
    try:
        report = xla_stage(args) if args.stage == "xla" else jax_stage(args)
        write_json(path, report)
    except Exception:
        prior = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        prior["fatal_error"] = traceback.format_exc()
        write_json(path, prior)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="ComfyUI API-format H3 workflow JSON")
    parser.add_argument("--diffusion", required=True, help="H3 diffusion safetensors path")
    parser.add_argument("--models-dir", help="Directory containing TE and VAE safetensors files")
    parser.add_argument("--report-dir", required=True, help="Directory for all diagnostic artifacts")
    parser.add_argument("--runs", type=int, default=2, help="Cold and repeated runs per stage")
    parser.add_argument("--video-t", type=int, default=1)
    parser.add_argument("--video-h", type=int, default=2)
    parser.add_argument("--video-w", type=int, default=2)
    parser.add_argument("--audio-t", type=int, default=2)
    parser.add_argument("--context-tokens", type=int, default=64)
    parser.add_argument("--inspect-only", action="store_true", help="Validate input files without starting TPU runtimes")
    parser.add_argument("--stage", choices=("xla", "jax"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    prompt = read_prompt(args.prompt)
    if not Path(args.diffusion).is_file():
        parser.error(f"Missing diffusion checkpoint: {args.diffusion}")
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.stage:
        child(args)
        return
    summary = {"workflow_nodes": {key: value["class_type"] for key, value in prompt.items()},
               "diffusion_bytes": Path(args.diffusion).stat().st_size,
               "mixed_jax_xla_workflow_verified": False, "stages": {}}
    if not args.inspect_only:
        for stage in ("xla", "jax"):
            command = [sys.executable, "-m", "comfy.ldm.minimax.verify_tpu",
                       "--stage", stage, "--prompt", args.prompt, "--diffusion", args.diffusion,
                       "--report-dir", str(report_dir), "--runs", str(args.runs),
                       "--video-t", str(args.video_t), "--video-h", str(args.video_h),
                       "--video-w", str(args.video_w), "--audio-t", str(args.audio_t),
                       "--context-tokens", str(args.context_tokens)]
            if args.models_dir:
                command += ["--models-dir", args.models_dir]
            env = os.environ.copy()
            if stage == "xla":
                env["XLA_METRICS_FILE"] = str(report_dir / "xla_step_metrics.txt")
            with (report_dir / f"{stage}.log").open("w", encoding="utf-8") as log:
                status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
            stage_report_path = report_dir / f"{stage}.json"
            stage_report = json.loads(stage_report_path.read_text(encoding="utf-8")) if stage_report_path.exists() else {}
            stage_summary = {"exit_code": status.returncode,
                             "report": str(stage_report_path),
                             "log": str(report_dir / f"{stage}.log"),
                             "fatal_error": stage_report.get("fatal_error")}
            if stage == "xla":
                stage_summary.update({
                    "physical_devices": stage_report.get("physical_devices"),
                    "runs_without_cpu_fallback": [run.get("no_cpu_fallback") for run in stage_report.get("runs", [])],
                    "fallback_counters": [run.get("cpu_fallback_counters") for run in stage_report.get("runs", [])],
                    "compile_and_transfer_metrics": [run.get("metrics") for run in stage_report.get("runs", [])],
                    "workflow_outputs": [run.get("workflow_outputs", {}) for run in stage_report.get("runs", [])],
                })
            else:
                stage_summary.update({
                    "jax_devices": len(stage_report.get("devices", [])),
                    "tp_sharded_large_weight_count": stage_report.get("tp_sharded_large_weight_count"),
                    "large_weights_not_tp_sharded": stage_report.get("large_weights_not_tp_sharded"),
                    "returns_video": stage_report.get("returns_video", False),
                    "runs": stage_report.get("runs", []),
                })
            summary["stages"][stage] = stage_summary
            write_json(report_dir / "summary.json", summary)
    write_json(report_dir / "summary.json", summary)
    print(report_dir / "summary.json")


if __name__ == "__main__":
    main()

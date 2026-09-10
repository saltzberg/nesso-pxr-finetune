#!/usr/bin/env python3
"""Bounded, one-record installed-Nesso affinity LoRA audit.

Run inside the pinned Nesso image selected externally by NESSO_IMAGE. No install,
source patch, bulk extraction, endpoint fitting, or production queue is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import signal
import time
import traceback
from pathlib import Path

# Set before numerical imports, including in dataloader subprocesses.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "4"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def tree_map(value, fn):
    import torch

    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: tree_map(v, fn) for k, v in value.items()}
    return value


def tensor_schema(value):
    import torch

    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {k: tensor_schema(v) for k, v in value.items()}
    return value


def capture_one(args):
    """Replay one cached processed record, capturing exact affinity pre-hook kwargs."""
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from nesso.data.inference import NessoInferenceDataModule
    from nesso.data.types import Manifest, Record
    from nesso.model.models.nesso1 import Nesso1

    if args.device != "cuda" or not torch.cuda.is_available():
        raise ValueError(
            "one-record upstream capture requires --device cuda and one free GPU"
        )
    if torch.cuda.device_count() != 1:
        raise ValueError("expose exactly one GPU for the pilot")
    # Validate membership without reading any challenge outcomes into a fit.
    with args.development_manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    matching = [r for r in rows if r.get("record_id") == args.record_id]
    if len(matching) != 1 or matching[0].get("modeling_role") != "development":
        raise ValueError("record must match one current development row")
    root = args.processed_root
    record_path = root / "records" / f"{args.record_id}.json"
    record = Record.load(record_path)
    inputs = [record_path, root / "structures" / f"{args.record_id}.npz", args.ccd]
    inputs += sorted((root / "rdkit_conformers").glob(f"{args.record_id}*.pkl"))
    inputs += sorted((root / "esm_embeddings").glob("*.safetensors"))
    if not inputs or any(not p.is_file() for p in inputs):
        raise FileNotFoundError(
            "processed record/structure/ESM/conformer/CCD cache incomplete"
        )
    datamodule = NessoInferenceDataModule(
        manifest=Manifest(records=[record]),
        target_dir=root,
        esm_emb_dir=root / "esm_embeddings",
        ligand_dir=root / "rdkit_conformers",
        ccd_pkl=args.ccd,
        num_workers=1,
        use_esm_all_layers=False,
        esm_emb_dim=1280,
        esm_num_layers=33,
    )
    seed_everything(42, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    model = Nesso1.from_pretrained(args.checkpoint)
    protocol = {
        "pose_protein_cutoff": 15.0,
        "recycling_steps": 5,
        "affinity_protein_cutoff": 15.0,
        "refine_protein_inference": True,
        "refine_protein_cutoff": 22.0,
        "refine_protein_tokens_budget": 256,
        "save_metadata": False,
    }
    model.predict_args.update(protocol)
    model.eval()
    captured = {}

    def pre_hook(module, positional, kwargs):
        if captured:
            raise RuntimeError("expected exactly one affinity invocation")
        # Record objects are writer metadata, not affinity inputs. Retain every
        # tensor feature unchanged; exclude non-tensor batch bookkeeping.
        tensor_kwargs = {**kwargs, "feats": {
            k: v for k, v in kwargs["feats"].items()
            if isinstance(v, torch.Tensor)
        }}
        captured["kwargs"] = tree_map(tensor_kwargs, lambda t: t.detach().cpu().clone())
        captured["cpu_rng"] = torch.get_rng_state()
        captured["cuda_rng"] = torch.cuda.get_rng_state()

    def post_hook(module, positional, output):
        captured["native_output"] = tree_map(output, lambda t: t.detach().cpu().clone())

    hooks = [
        model.affinity_module.register_forward_pre_hook(pre_hook, with_kwargs=True),
        model.affinity_module.register_forward_hook(post_hook),
    ]
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    started = time.perf_counter()
    try:
        result = trainer.predict(model, datamodule=datamodule, return_predictions=True)
    finally:
        for hook in hooks:
            hook.remove()
    if (
        len(result or []) != 1
        or result[0].get("exception")
        or "native_output" not in captured
    ):
        raise RuntimeError("one-record capture failed")
    captured["provenance"] = {
        "kind": "real_processed_record_affinity_pre_hook",
        "record_id": args.record_id,
        "challenge_label_training": False,
        "member": "affinity_module",
        "protocol": protocol,
        "seed": 42,
        "precision": "bf16-mixed",
        "num_workers": 1,
        "order": [args.record_id],
        "restart_boundary": "new single-record RNG stream",
        "input_hashes": {str(p): sha256(p) for p in inputs},
        "checkpoint_sha256": sha256(args.checkpoint / "model.safetensors"),
        "seconds": time.perf_counter() - started,
    }
    path = args.output_dir / "real_affinity_inputs.pt"
    torch.save(captured, path)
    write_json(
        args.output_dir / "capture.json",
        {
            **captured["provenance"],
            "input_schema": tensor_schema(captured["kwargs"]),
            "capture_sha256": sha256(path),
        },
    )
    del model, trainer, result
    torch.cuda.empty_cache()
    return path


def load_affinity(checkpoint, device):
    from nesso.model.modules.affinity import AffinityModule
    from safetensors import safe_open

    hparams = json.loads((checkpoint / "hparams.json").read_text())
    model = AffinityModule(
        token_s=hparams["token_s"],
        token_z=hparams["token_z"],
        **hparams["affinity_model_args"],
    )
    with safe_open(
        checkpoint / "model.safetensors", framework="pt", device="cpu"
    ) as source:
        state = {
            k.removeprefix("affinity_module."): source.get_tensor(k)
            for k in source.keys()
            if k.startswith("affinity_module.")
        }
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def pilot(args, input_path):
    import torch

    from nesso_pxr.low_data_adapter import (
        adapter_checkpoint,
        base_fingerprint,
        inject_lora,
        load_adapter_checkpoint,
    )

    data = torch.load(input_path, map_location="cpu", weights_only=True)
    if data["provenance"]["kind"] != "real_processed_record_affinity_pre_hook":
        raise ValueError(
            "real pilot requires captured real upstream inputs, never random tensors"
        )
    if data["provenance"]["checkpoint_sha256"] != sha256(
        args.checkpoint / "model.safetensors"
    ):
        raise ValueError("capture/checkpoint mismatch")
    inputs = tree_map(data["kwargs"], lambda t: t.to(args.device))
    # Differentiable PyTorch implementation; retain all crop/features/tensor values.
    # Explicitly quantify native-kernel vs differentiable-replay drift below.
    inputs["use_kernels"] = False
    model = load_affinity(args.checkpoint, args.device)
    source_path = Path(inspect.getfile(type(model)))
    write_json(
        args.output_dir / "architecture.json",
        {
            "source_path": str(source_path),
            "source_sha256": sha256(source_path),
            "linear_modules": {
                n: [m.in_features, m.out_features]
                for n, m in model.named_modules()
                if isinstance(m, torch.nn.Linear)
            },
            "model_parameters": sum(p.numel() for p in model.parameters()),
        },
    )
    frozen_hash = base_fingerprint(model)
    targets = ["esm_proj.1", "esm_proj.3", "pairformer_stack.layers.0.transition_z.fc1"]

    def forward():
        # Released eval dropout still consumes RNG: reset for paired parity only.
        torch.set_rng_state(data["cpu_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state(data["cuda_rng"])
        return model(**inputs)

    with torch.no_grad():
        baseline = forward()
    inject_lora(model, targets, rank=4, alpha=4)
    with torch.no_grad():
        zero = forward()
    zero_error = max(float((zero[k] - baseline[k]).abs().max()) for k in baseline)
    if zero_error != 0:
        raise AssertionError(f"zero-update parity failed: {zero_error}")
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    initial = {n: p.detach().clone() for n, p in parameters.items()}
    optimizer = torch.optim.SGD(parameters.values(), lr=1e-3)
    # Non-scientific engineering loss: one-unit shift of detached native output.
    # No assay outcome is used; these are not fitted endpoint predictions.
    target = baseline["affinity_pred_value"].detach() + 1.0
    history = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = forward()
        loss = torch.nn.functional.smooth_l1_loss(output["affinity_pred_value"], target)
        loss.backward()
        norms = {}
        for name, p in parameters.items():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise AssertionError(f"missing/nonfinite gradient: {name}")
            norms[name] = float(p.grad.norm())
            if (step or name.endswith("lora_B")) and norms[name] <= 0:
                raise AssertionError(f"zero intended gradient: {name}")
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise AssertionError("frozen gradient")
        optimizer.step()
        history.append(
            {"step": step, "loss": float(loss.detach()), "gradient_norms": norms}
        )
    changes = {n: float((p - initial[n]).abs().max()) for n, p in parameters.items()}
    if any(v <= 0 for v in changes.values()) or base_fingerprint(model) != frozen_hash:
        raise AssertionError("adapter update or frozen invariance failed")
    with torch.no_grad():
        final = forward()
    checkpoint = adapter_checkpoint(model)
    adapter_path = args.output_dir / "adapter.pt"
    torch.save(checkpoint, adapter_path)
    model = load_affinity(args.checkpoint, args.device)
    load_adapter_checkpoint(
        model, torch.load(adapter_path, map_location="cpu", weights_only=True)
    )
    with torch.no_grad():
        reloaded = forward()
    reload_error = max(float((reloaded[k] - final[k]).abs().max()) for k in final)
    if reload_error != 0:
        raise AssertionError(f"reload parity failed: {reload_error}")
    torch.save(
        tree_map(
            {"baseline": baseline, "final": final, "reloaded": reloaded},
            lambda t: t.detach().cpu(),
        ),
        args.output_dir / "outputs.pt",
    )
    return {
        "status": "pass",
        "classification": "real_affinity_module_engineering_feasibility",
        "challenge_label_training": False,
        "scientific_fit": False,
        "objective": (
            "smooth-L1(native continuous output, detached baseline + 1), two SGD steps"
        ),
        "record_id": data["provenance"]["record_id"],
        "member": "affinity_module",
        "targets": targets,
        "rank": 4,
        "alpha": 4,
        "trainable_parameters": sum(p.numel() for p in parameters.values()),
        "zero_update_max_abs": zero_error,
        "reload_max_abs": reload_error,
        "frozen_unchanged": True,
        "frozen_base_sha256": frozen_hash,
        "gradient_history": history,
        "adapter_max_abs_updates": changes,
        "output_change_max_abs": float(
            (final["affinity_pred_value"] - baseline["affinity_pred_value"]).abs().max()
        ),
        "native_to_differentiable_max_abs": {
            k: float((baseline[k].cpu() - data["native_output"][k]).abs().max())
            for k in baseline
        },
        "native_use_kernels": data["kwargs"]["use_kernels"],
        "replay_use_kernels": False,
        "mode": "eval (gradients enabled; no crop or dropout policy change)",
        "device": args.device,
        "threads": torch.get_num_threads(),
        "input_sha256": sha256(input_path),
        "adapter_sha256": sha256(adapter_path),
        "torch_version": str(torch.__version__),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/nesso-1/v1.0.0")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--inputs", type=Path, help="Previously captured real_affinity_inputs.pt"
    )
    parser.add_argument(
        "--capture-one",
        action="store_true",
        help="Replay exactly one processed record first",
    )
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--record-id")
    parser.add_argument("--ccd", type=Path, default=Path("models/nesso-1/ccd.pkl"))
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=Path("data/published/modeling_manifest.csv"),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 600:
        parser.error("timeout must be between 1 and 600 seconds")
    if args.capture_one and (
        not args.processed_root or not args.record_id or args.inputs
    ):
        parser.error(
            "--capture-one requires --processed-root/--record-id and excludes --inputs"
        )
    if not args.capture_one and not args.inputs:
        parser.error("provide --inputs or --capture-one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "pilot.json").exists():
        parser.error("pilot.json already exists: use a new attempt output directory")
    started = time.perf_counter()

    def timeout_handler(signum, frame):
        raise TimeoutError("ten-minute pilot budget exceeded")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout_seconds)
    try:
        import torch

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        path = capture_one(args) if args.capture_one else args.inputs
        summary = pilot(args, path)
    except Exception as error:
        summary = {
            "status": "blocked",
            "error": str(error),
            "traceback": traceback.format_exc(),
            "classification": "no_real_feasibility_success",
        }
        write_json(
            args.output_dir / "pilot.json",
            {**summary, "seconds": time.perf_counter() - started},
        )
        raise
    finally:
        signal.alarm(0)
    summary["seconds"] = time.perf_counter() - started
    write_json(args.output_dir / "pilot.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

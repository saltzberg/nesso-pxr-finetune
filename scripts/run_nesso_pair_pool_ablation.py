#!/usr/bin/env python3
"""Extract Nesso representations with direct pair-pooling ablations."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from types import MethodType
from typing import Any

from nesso_pxr.pairwise_ablation import (
    PAIR_POOL_VARIANTS,
    register_pair_pool_hooks,
)


def _prediction_failed(prediction: dict[str, Any], torch: Any) -> bool:
    exception = prediction.get("exception")
    return exception is True or (
        isinstance(exception, torch.Tensor)
        and exception.numel() > 0
        and bool(exception.any().item())
    )


def extract_pair_pool_ablations(
    selection_path: Path,
    processed_root: Path,
    checkpoint: Path,
    ccd: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    recycling_steps: int = 5,
    progress: bool = False,
) -> dict[str, Any]:
    """Replay upstream Nesso once and capture three final pair-pool variants."""

    import torch
    from lightning.pytorch import Trainer, seed_everything
    from nesso.data.inference import NessoInferenceDataModule
    from nesso.data.types import Manifest, Record
    from nesso.model.models.nesso1 import Nesso1
    from safetensors.torch import save_file

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Nesso feature extraction")

    with selection_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        selection_fields = list(reader.fieldnames or [])
        selection = sorted(reader, key=lambda row: int(row["feature_row"]))
    record_ids = [row["record_id"] for row in selection]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("selection record IDs must be unique")
    records = [
        Record.load(processed_root / "records" / f"{record_id}.json")
        for record_id in record_ids
    ]
    datamodule = NessoInferenceDataModule(
        manifest=Manifest(records=records),
        target_dir=processed_root,
        esm_emb_dir=processed_root / "esm_embeddings",
        ligand_dir=processed_root / "rdkit_conformers",
        ccd_pkl=ccd,
        num_workers=1,
        use_esm_all_layers=False,
        esm_emb_dim=1280,
        esm_num_layers=33,
    )

    seed_everything(seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    model = Nesso1.from_pretrained(checkpoint)
    model.predict_args.update(
        {
            "pose_protein_cutoff": 15.0,
            "recycling_steps": recycling_steps,
            "affinity_protein_cutoff": 15.0,
            "refine_protein_inference": True,
            "refine_protein_cutoff": 22.0,
            "refine_protein_tokens_budget": 256,
            "save_metadata": False,
        }
    )
    model.eval()

    captured: dict[str, dict[str, Any]] = {}
    hooks = register_pair_pool_hooks(model, captured)
    original_predict_step = model.predict_step

    def predict_step_with_ablations(
        self: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        captured.clear()
        prediction = Nesso1.predict_step(self, batch, batch_idx, dataloader_idx)
        if not _prediction_failed(prediction, torch):
            if set(captured) != {"member1", "member2"}:
                raise RuntimeError(
                    f"expected two captured members, got {sorted(captured)}"
                )
            with torch.autocast("cuda", enabled=False):
                for member, variants in captured.items():
                    head = (
                        self.affinity_module.affinity_heads
                        if member == "member1"
                        else self.affinity_module2.affinity_heads
                    )
                    for variant, representation in variants.items():
                        prediction[f"_{member}_{variant}_repr"] = representation
                        prediction[f"_{member}_{variant}_pred_value"] = (
                            head.to_affinity_pred_value(representation).reshape(-1)
                        )
        retained = {
            "exception",
            "record_id",
            "affinity_pred_value",
            "affinity_pred_value1",
            "affinity_pred_value2",
        }
        for member in ("member1", "member2"):
            for variant in PAIR_POOL_VARIANTS:
                retained.add(f"_{member}_{variant}_repr")
                retained.add(f"_{member}_{variant}_pred_value")
        return {key: value for key, value in prediction.items() if key in retained}

    model.predict_step = MethodType(predict_step_with_ablations, model)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=progress,
    )

    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    torch.set_grad_enabled(False)
    torch.cuda.reset_peak_memory_stats()
    try:
        result = trainer.predict(model, datamodule=datamodule, return_predictions=True)
        predictions = list(result or [])
    finally:
        model.predict_step = original_predict_step
        for handle in hooks:
            handle.remove()
    wall_seconds = time.perf_counter() - started

    tensor_rows: dict[str, list[Any]] = {}
    metadata_rows: list[dict[str, Any]] = []
    for batch_index, prediction in enumerate(predictions):
        row = selection[batch_index]
        if _prediction_failed(prediction, torch):
            failures.append(
                {
                    "feature_row": int(row["feature_row"]),
                    "record_id": str(row["record_id"]),
                    "message": "Nesso predict_step returned exception=True",
                }
            )
            continue
        if str(prediction.get("record_id")) != str(row["record_id"]):
            raise RuntimeError("prediction order does not match the selection")
        metadata_rows.append(row)
        for member in ("member1", "member2"):
            for variant in PAIR_POOL_VARIANTS:
                for suffix in ("repr", "pred_value"):
                    source = f"_{member}_{variant}_{suffix}"
                    destination = f"{member}_{variant}_{suffix}"
                    value = prediction[source].detach().float().squeeze(0).cpu()
                    tensor_rows.setdefault(destination, []).append(value)
        for source in (
            "affinity_pred_value1",
            "affinity_pred_value2",
            "affinity_pred_value",
        ):
            tensor_rows.setdefault(source, []).append(
                prediction[source].detach().float().reshape(-1)[0].cpu()
            )

    if failures or len(metadata_rows) != len(selection):
        raise RuntimeError(
            f"pair-pool extraction failed for {len(failures)} of {len(selection)} rows"
        )
    tensors = {
        name: torch.stack(values, dim=0).contiguous()
        for name, values in tensor_rows.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_dir / "features.safetensors"))
    with (output_dir / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selection_fields)
        writer.writeheader()
        writer.writerows(metadata_rows)
    with (output_dir / "failures.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("feature_row", "record_id", "message")
        )
        writer.writeheader()
        writer.writerows(failures)

    parity = max(
        float(
            torch.max(
                torch.abs(
                    tensors["member1_all_pairs_pred_value"]
                    - tensors["affinity_pred_value1"]
                )
            )
        ),
        float(
            torch.max(
                torch.abs(
                    tensors["member2_all_pairs_pred_value"]
                    - tensors["affinity_pred_value2"]
                )
            )
        ),
    )
    summary = {
        "status": "pass",
        "selected": int(len(selection)),
        "succeeded": int(len(metadata_rows)),
        "failed": int(len(failures)),
        "seed": seed,
        "recycling_steps": recycling_steps,
        "variants": list(PAIR_POOL_VARIANTS),
        "scope": (
            "Direct final pair-pooling ablation; upstream protein information "
            "may remain in ligand-ligand z cells."
        ),
        "baseline_head_parity_max_abs": parity,
        "wall_seconds": wall_seconds,
        "mean_wall_seconds": wall_seconds / len(selection),
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ccd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recycling-steps", type=int, default=5)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = extract_pair_pool_ablations(
        args.selection,
        args.processed_root,
        args.checkpoint,
        args.ccd,
        args.output_dir,
        seed=args.seed,
        recycling_steps=args.recycling_steps,
        progress=args.progress,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

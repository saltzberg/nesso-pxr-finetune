"""Lightning-exact Nesso feature extraction for the bounded GPU smoke test."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import MethodType
from typing import Any

import pandas as pd

from nesso_pxr.smoke import (
    FEATURE_TENSOR_NAMES,
    _feature_schema,
    _register_representation_hooks,
)


def _prediction_failed(prediction: dict[str, Any], torch: Any) -> bool:
    exception = prediction.get("exception")
    return exception is True or (
        isinstance(exception, torch.Tensor)
        and exception.numel() > 0
        and bool(exception.any().item())
    )


def extract_smoke_features_lightning(
    selection_path: Path,
    processed_root: Path,
    checkpoint: Path,
    ccd: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    recycling_steps: int = 5,
    reference_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Run the upstream Lightning prediction path and capture both representations."""

    import torch
    from lightning.pytorch import Trainer, seed_everything
    from nesso.data.inference import NessoInferenceDataModule
    from nesso.data.types import Manifest, Record
    from nesso.model.models.nesso1 import Nesso1
    from safetensors.torch import save_file

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU smoke test")

    selection = pd.read_csv(selection_path).sort_values("feature_row")
    if selection["feature_row"].tolist() != list(range(len(selection))):
        raise ValueError("selection feature_row must be contiguous and zero-based")

    seed_everything(seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")

    records = [
        Record.load(processed_root / "records" / f"{record_id}.json")
        for record_id in selection["record_id"]
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

    captured: dict[str, Any] = {}
    hooks = _register_representation_hooks(model, captured)
    original_predict_step = model.predict_step

    def predict_step_with_features(
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
                    f"expected two captured representations, got {sorted(captured)}"
                )
            prediction["_member1_affinity_repr"] = captured["member1"].detach().clone()
            prediction["_member2_affinity_repr"] = captured["member2"].detach().clone()
        return prediction

    model.predict_step = MethodType(predict_step_with_features, model)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    failures: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    torch.set_grad_enabled(False)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    predict_started = time.perf_counter()
    try:
        result = trainer.predict(
            model,
            datamodule=datamodule,
            return_predictions=True,
        )
        predictions = list(result or [])
    except Exception as error:
        failures.extend(
            {
                "feature_row": int(row.feature_row),
                "record_id": str(row.record_id),
                "stage": "lightning_predict",
                "error_type": type(error).__name__,
                "message": str(error),
            }
            for row in selection.itertuples(index=False)
        )
    finally:
        torch.cuda.synchronize()
        predict_wall_seconds = time.perf_counter() - predict_started
        peak_memory = int(torch.cuda.max_memory_allocated())
        model.predict_step = original_predict_step
        for hook in hooks:
            hook.remove()

    tensor_rows: dict[str, list[Any]] = {name: [] for name in FEATURE_TENSOR_NAMES}
    metadata_rows: list[dict[str, Any]] = []
    mean_wall_seconds = predict_wall_seconds / len(predictions) if predictions else None

    for batch_index, prediction in enumerate(predictions):
        selection_row = selection.iloc[batch_index].to_dict()
        record_id = str(selection_row["record_id"])
        if _prediction_failed(prediction, torch):
            failures.append(
                {
                    "feature_row": int(selection_row["feature_row"]),
                    "record_id": record_id,
                    "stage": "lightning_predict",
                    "error_type": "NessoPredictionError",
                    "message": "Nesso predict_step returned exception=True",
                }
            )
            continue

        try:
            observed_record_id = str(prediction.get("record_id"))
            if observed_record_id != record_id:
                raise RuntimeError(
                    f"prediction order mismatch: {observed_record_id} != {record_id}"
                )
            member1 = prediction["_member1_affinity_repr"].float()
            member2 = prediction["_member2_affinity_repr"].float()
            if tuple(member1.shape) != (1, 384) or tuple(member2.shape) != (1, 384):
                raise RuntimeError(
                    f"unexpected representation shapes {member1.shape}, {member2.shape}"
                )

            with torch.inference_mode():
                reconstructed1 = (
                    model.affinity_module.affinity_heads.to_affinity_pred_value(member1)
                    .reshape(-1)
                    .float()
                )
                reconstructed2 = (
                    model.affinity_module2.affinity_heads.to_affinity_pred_value(
                        member2
                    )
                    .reshape(-1)
                    .float()
                )
                score1 = prediction["affinity_pred_value1"].reshape(-1).float()
                score2 = prediction["affinity_pred_value2"].reshape(-1).float()
                ensemble = prediction["affinity_pred_value"].reshape(-1).float()

            values = {
                "member1_affinity_repr": member1.squeeze(0).cpu(),
                "member2_affinity_repr": member2.squeeze(0).cpu(),
                "affinity_pred_value1": score1.squeeze(0).cpu(),
                "affinity_pred_value2": score2.squeeze(0).cpu(),
                "affinity_pred_value": ensemble.squeeze(0).cpu(),
                "reconstructed_affinity_pred_value1": (reconstructed1.squeeze(0).cpu()),
                "reconstructed_affinity_pred_value2": (reconstructed2.squeeze(0).cpu()),
            }
            for name, value in values.items():
                tensor_rows[name].append(value)

            internal_error1 = float((score1 - reconstructed1).abs().max().item())
            internal_error2 = float((score2 - reconstructed2).abs().max().item())
            ensemble_error = float(
                (ensemble - ((score1 + score2) / 2.0)).abs().max().item()
            )
            reference_error1 = abs(
                float(score1.item())
                - float(selection_row["reference_affinity_pred_value1"])
            )
            reference_error2 = abs(
                float(score2.item())
                - float(selection_row["reference_affinity_pred_value2"])
            )
            reference_ensemble_error = abs(
                float(ensemble.item())
                - float(selection_row["reference_affinity_pred_value"])
            )
            metadata_rows.append(
                {
                    **selection_row,
                    "wall_seconds": mean_wall_seconds,
                    "peak_cuda_memory_mb": peak_memory / (1024**2),
                    "observed_affinity_pred_value1": float(score1.item()),
                    "observed_affinity_pred_value2": float(score2.item()),
                    "observed_affinity_pred_value": float(ensemble.item()),
                    "internal_reconstruction_error1": internal_error1,
                    "internal_reconstruction_error2": internal_error2,
                    "ensemble_mean_error": ensemble_error,
                    "reference_error1": reference_error1,
                    "reference_error2": reference_error2,
                    "reference_ensemble_error": reference_ensemble_error,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "feature_row": int(selection_row["feature_row"]),
                    "record_id": record_id,
                    "stage": "feature_audit",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame.from_records(metadata_rows)
    failure_columns = [
        "feature_row",
        "record_id",
        "stage",
        "error_type",
        "message",
    ]
    failure_frame = pd.DataFrame.from_records(
        failures,
        columns=failure_columns,
    )
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    failure_frame.to_csv(output_dir / "failures.csv", index=False)

    tensors = {
        name: torch.stack(rows).contiguous()
        for name, rows in tensor_rows.items()
        if rows
    }
    if tensors:
        save_file(tensors, output_dir / "features.safetensors")

    internal_columns = [
        "internal_reconstruction_error1",
        "internal_reconstruction_error2",
        "ensemble_mean_error",
    ]
    reference_columns = [
        "reference_error1",
        "reference_error2",
        "reference_ensemble_error",
    ]
    internal_max = (
        float(metadata[internal_columns].to_numpy().max())
        if not metadata.empty
        else math.inf
    )
    reference_max = (
        float(metadata[reference_columns].to_numpy().max())
        if not metadata.empty
        else math.inf
    )
    summary = {
        "status": (
            "pass"
            if len(metadata) == len(selection)
            and not failures
            and internal_max <= 1e-6
            and reference_max <= reference_tolerance
            else "fail"
        ),
        "execution_path": "lightning.Trainer.predict",
        "selected": int(len(selection)),
        "succeeded": int(len(metadata)),
        "failed": int(len(failures)),
        "representation_dimension": 384,
        "internal_parity_max_abs": internal_max,
        "standard_reference_max_abs": reference_max,
        "reference_tolerance": reference_tolerance,
        "wall_seconds_total": time.perf_counter() - total_start,
        "wall_seconds_predict": predict_wall_seconds,
        "wall_seconds_mean": mean_wall_seconds,
        "peak_cuda_memory_mb": peak_memory / (1024**2),
        "seed": seed,
        "recycling_steps": recycling_steps,
        "num_workers": 1,
        "cuda_device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    schema = _feature_schema(
        selection_path=selection_path,
        checkpoint=checkpoint,
        recycling_steps=recycling_steps,
        reference_tolerance=reference_tolerance,
    )
    schema["execution_path"] = "lightning.Trainer.predict with temporary head hooks"
    (output_dir / "feature_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n"
    )
    return summary

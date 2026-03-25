#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import re
import sys
import types
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger(__name__)


def _install_ldm_stub() -> None:
    """Install a minimal `ldm` stub so external evaluator modules can import."""
    if "ldm" in sys.modules:
        return

    ldm = types.ModuleType("ldm")
    models = types.ModuleType("ldm.models")
    diffusion = types.ModuleType("ldm.models.diffusion")
    ddim = types.ModuleType("ldm.models.diffusion.ddim")

    class DDIMSampler:
        pass

    ddim.DDIMSampler = DDIMSampler
    diffusion.ddim = ddim
    models.diffusion = diffusion
    ldm.models = models
    sys.modules["ldm"] = ldm
    sys.modules["ldm.models"] = models
    sys.modules["ldm.models.diffusion"] = diffusion
    sys.modules["ldm.models.diffusion.ddim"] = ddim


_install_ldm_stub()


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SRC_ROOT_CANDIDATES: tuple[Path, ...] = ()
GRID_TABLE_COLUMNS = ["step", "class_instance", "instance", "pdir", "prompt", "grid"]
MANIFEST_REQUIRED_COLUMNS = frozenset(
    {"class_instance", "instance", "pdir", "prompt", "gen_dir", "step"}
)
CLI_REQUIRED_COLUMNS = frozenset(
    {"class_instance", "instance", "pdir", "prompt", "gen_dir", "src_dir"}
)
PATH_LIKE = Union[str, bytes, os.PathLike, Path]
RESAMPLE_BICUBIC = (
    Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
)


def sanitize_optional_path(value: object) -> Optional[Path]:
    """Normalize a manifest path field while preserving blanks as `None`."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    text = os.fsdecode(value).strip().strip("'").strip('"').strip()
    if not text or text.lower() == "nan":
        return None
    return Path(text)


def sanitize_path(value: PATH_LIKE) -> Path:
    """Normalize a required path field."""
    path = sanitize_optional_path(value)
    if path is None:
        raise ValueError("Expected a non-empty path value.")
    return path


def resolve_device(requested_device: Optional[str] = None) -> torch.device:
    """Return the effective evaluation device."""
    device_name = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available; falling back to CPU.")
        device_name = "cpu"
    return torch.device(device_name)


def validate_manifest_columns(
    manifest_df: pd.DataFrame, required_columns: Sequence[str]
) -> None:
    """Validate that the manifest contains the columns needed by a workflow."""
    missing = sorted(set(required_columns) - set(manifest_df.columns))
    if missing:
        raise SystemExit(f"Manifest is missing columns: {missing}")


def load_manifest(
    manifest_csv: PATH_LIKE, required_columns: Sequence[str]
) -> pd.DataFrame:
    """Load a manifest CSV and validate the expected schema."""
    manifest_path = sanitize_path(manifest_csv)
    try:
        manifest_df = pd.read_csv(manifest_path)
    except FileNotFoundError as exc:
        raise SystemExit(f"Manifest CSV not found: {manifest_path}") from exc
    except Exception as exc:
        raise SystemExit(f"Failed to read manifest CSV {manifest_path}: {exc}") from exc

    if manifest_df.empty:
        raise SystemExit(f"Empty manifest CSV: {manifest_path}")

    validate_manifest_columns(manifest_df, required_columns)
    return manifest_df


def list_images(directory: Optional[PATH_LIKE]) -> list[Path]:
    """List supported image files in a directory."""
    dir_path = sanitize_optional_path(directory)
    if dir_path is None or not dir_path.is_dir():
        return []

    try:
        return sorted(
            path
            for path in dir_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMG_EXTS
        )
    except OSError as exc:
        LOGGER.warning("Failed to list images in %s: %s", dir_path, exc)
        return []


def load_imgs_minus1_1(
    paths: Sequence[PATH_LIKE],
    cap: Optional[int] = None,
    uniform_size: int = 224,
) -> torch.Tensor:
    """
    Load images as BCHW float tensors in [-1, 1].

    Images are center-cropped to a square, resized to `uniform_size`, and stacked.
    """
    selected_paths = paths[:cap] if cap is not None else paths
    tensors: list[torch.Tensor] = []

    for raw_path in selected_paths:
        image_path = sanitize_path(raw_path)
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size
                side = min(width, height)
                left = (width - side) // 2
                top = (height - side) // 2
                image = image.crop((left, top, left + side, top + side))
                image = image.resize((uniform_size, uniform_size), RESAMPLE_BICUBIC)
                image_np = np.asarray(image, dtype=np.float32) / 255.0
        except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError) as exc:
            LOGGER.warning("Skipping unreadable image %s: %s", image_path, exc)
            continue

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        tensors.append(image_tensor * 2.0 - 1.0)

    if not tensors:
        return torch.empty(0, 3, uniform_size, uniform_size)
    return torch.stack(tensors, dim=0)


def clean_prompt(prompt: str) -> str:
    """Remove obvious formatting artifacts from a prompt string."""
    text = str(prompt)
    text = re.sub(r"\.format\([^)]*\)\s*$", "", text)
    text = re.sub(r"\bsks\b", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def import_user_evaluator(evaluator_py: PATH_LIKE) -> Any:
    """Dynamically import a user-provided evaluator module."""
    evaluator_path = sanitize_path(evaluator_py)
    spec = importlib.util.spec_from_file_location("user_eval", str(evaluator_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator module from {evaluator_path}")

    user_eval = importlib.util.module_from_spec(spec)
    sys.modules["user_eval"] = user_eval
    spec.loader.exec_module(user_eval)

    if not hasattr(user_eval, "ImageDirEvaluator"):
        raise AttributeError(
            f"Evaluator module {evaluator_path} does not define ImageDirEvaluator."
        )
    return user_eval


def build_evaluator(
    device: torch.device,
    clip_model: str,
    evaluator_py: Optional[PATH_LIKE] = None,
) -> Any:
    """Build the evaluator used for CLIP/DINO metrics."""
    if evaluator_py is None:
        from evaluation.evaluation import ImageDirEvaluator as EvaluatorClass
    else:
        user_eval = import_user_evaluator(evaluator_py)
        EvaluatorClass = user_eval.ImageDirEvaluator

    return EvaluatorClass(device=str(device), clip_model=clip_model)


def extract_label_from_record(record: Any) -> Optional[int]:
    """
    Normalize class-label access across the LMDB schema variants used in this repo.
    """
    if isinstance(record, Mapping):
        for key in ("class_labels", "label", "labels"):
            if key not in record or record[key] is None:
                continue
            try:
                return int(record[key])
            except (TypeError, ValueError):
                continue

    if isinstance(record, (list, tuple)) and len(record) >= 2:
        try:
            return int(record[1])
        except (TypeError, ValueError):
            return None

    return None


def extract_image_from_record(record: Any) -> torch.Tensor:
    """Extract the image tensor from a dataset item."""
    if isinstance(record, Mapping) and "images" in record:
        image = record["images"]
    elif isinstance(record, (list, tuple)) and record:
        image = record[0]
    else:
        raise KeyError("Dataset record does not expose an image tensor.")

    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    return image


def load_lmdb_dataset(
    lmdb_path: Optional[PATH_LIKE],
    manifest_df: Optional[pd.DataFrame] = None,
) -> tuple[Optional[Any], Optional[torch.Tensor]]:
    """Open the LMDB dataset and collect a label tensor for source sampling."""
    dataset_path = sanitize_optional_path(lmdb_path)
    if dataset_path is None:
        return None, None

    from main.data.sd_image_dataset import SDImageDatasetLMDB

    try:
        dataset = SDImageDatasetLMDB(str(dataset_path), tokenizer_one=None)
    except Exception as exc:
        LOGGER.warning("Failed to open LMDB dataset %s: %s", dataset_path, exc)
        return None, None

    labels: list[int] = []
    try:
        for index in range(len(dataset)):
            record = dataset[index]
            label = extract_label_from_record(record)
            if label is None:
                LOGGER.warning(
                    "LMDB dataset item %s has no label; CLIP-I/DINO-I will be skipped.",
                    index,
                )
                return None, None
            labels.append(label)
    except Exception as exc:
        LOGGER.warning("Failed while reading LMDB labels from %s: %s", dataset_path, exc)
        return None, None

    if not labels:
        LOGGER.warning("LMDB dataset %s is empty; CLIP-I/DINO-I will be skipped.", dataset_path)
        return None, None

    labels_all = torch.tensor(labels, dtype=torch.long)
    unique_labels, counts = torch.unique(labels_all, return_counts=True)
    LOGGER.info(
        "LMDB labels: %s",
        dict(zip(unique_labels.tolist(), counts.tolist())),
    )

    if manifest_df is not None and "src_label" in manifest_df.columns:
        try:
            manifest_labels = pd.unique(manifest_df["src_label"].astype("Int64"))[:10]
            LOGGER.info("First manifest src_label values: %s", manifest_labels.tolist())
        except Exception:
            LOGGER.debug("Unable to summarize manifest src_label values.", exc_info=True)

    return dataset, labels_all


def empty_image_batch(size: int, device: torch.device) -> torch.Tensor:
    """Create an empty BCHW batch on the target device."""
    return torch.empty(0, 3, size, size, device=device)


def _src_batch_from_lmdb(
    dataset: Any,
    labels_all: torch.Tensor,
    label_id: int,
    k: int,
    device: torch.device,
    size: int = 224,
) -> torch.Tensor:
    """Load up to `k` reference images for a label from the LMDB dataset."""
    matching_indices = torch.nonzero(labels_all == int(label_id), as_tuple=False).flatten()
    if matching_indices.numel() == 0 or k <= 0:
        return empty_image_batch(size, device)

    images: list[torch.Tensor] = []
    for index in matching_indices[:k].tolist():
        try:
            record = dataset[index]
            image = extract_image_from_record(record)
        except Exception as exc:
            LOGGER.warning(
                "Skipping LMDB source sample %s for label %s: %s",
                index,
                label_id,
                exc,
            )
            continue

        image = image.to(device=device, dtype=torch.float32, non_blocking=True)
        image_01 = (image + 1.0) / 2.0
        image_01 = torch.nn.functional.interpolate(
            image_01,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )
        images.append(image_01 * 2.0 - 1.0)

    if not images:
        return empty_image_batch(size, device)
    return torch.cat(images, dim=0)


def resolve_src_dir(
    instance: str,
    roots: Sequence[PATH_LIKE] = SRC_ROOT_CANDIDATES,
    fallback_names: Sequence[str] = (),
    must_contain_exts: Sequence[str] = tuple(IMG_EXTS),
) -> Optional[Path]:
    """
    Resolve a source directory from an instance name and candidate roots.

    The search tries a few stable name normalizations before giving up.
    """
    root_paths = [
        root_path
        for root in roots
        if (root_path := sanitize_optional_path(root)) is not None
    ]

    candidate_names = {
        instance,
        instance.lower(),
        instance.replace(" ", "_"),
        instance.replace("-", "_"),
    }
    candidate_names.update(name for name in fallback_names if name)

    for root in root_paths:
        for name in candidate_names:
            candidate = root / name
            if not candidate.is_dir():
                continue
            if must_contain_exts and not list_images(candidate):
                continue
            return candidate
    return None


def to_float(value: Any) -> float:
    """Convert tensors or scalars to a Python float."""
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def compute_group_metrics(
    evaluator: Any,
    prompt: str,
    src_batch: torch.Tensor,
    gen_batch: torch.Tensor,
) -> tuple[float, float, float]:
    """Compute CLIP-I, CLIP-T, and DINO-I for a source/generated batch pair."""
    with torch.no_grad():
        clip_t = to_float(evaluator.txt_to_img_similarity(prompt, gen_batch))

        if src_batch.shape[0] == 0:
            return float("nan"), clip_t, float("nan")

        clip_i = to_float(evaluator.img_to_img_similarity(src_batch, gen_batch))

        if hasattr(evaluator, "dino_img_to_img_similarity"):
            dino_i = to_float(evaluator.dino_img_to_img_similarity(src_batch, gen_batch))
        elif hasattr(evaluator, "evaluate"):
            _, _, dino_value = evaluator.evaluate(gen_batch, src_batch, prompt)
            dino_i = to_float(dino_value)
        else:
            dino_i = float("nan")

    return clip_i, clip_t, dino_i


def evaluate_group(
    evaluator: Any,
    prompt: str,
    src_batch: torch.Tensor,
    gen_dir: Path,
    max_gen_per_group: int,
    uniform_size: int,
    device: torch.device,
) -> Optional[tuple[torch.Tensor, tuple[float, float, float]]]:
    """Load generated images for a directory and compute metrics for the group."""
    gen_paths = list_images(gen_dir)
    if not gen_paths:
        LOGGER.warning("No generated images found in %s; skipping group.", gen_dir)
        return None

    gen_batch = load_imgs_minus1_1(
        gen_paths,
        cap=max_gen_per_group,
        uniform_size=uniform_size,
    ).to(device=device, non_blocking=True)
    if gen_batch.shape[0] == 0:
        LOGGER.warning("No readable generated images found in %s; skipping group.", gen_dir)
        return None

    metrics = compute_group_metrics(evaluator, prompt, src_batch, gen_batch)
    return gen_batch, metrics


def build_instance_summary(
    group_df: pd.DataFrame, include_counts: bool = False
) -> pd.DataFrame:
    """Aggregate per-group metrics to the per-instance level."""
    summary = (
        group_df.groupby(["class_instance", "instance"], as_index=False)
        .agg(
            clip_i_mean=("clip_i", "mean"),
            clip_t_mean=("clip_t", "mean"),
            dino_i_mean=("dino_i", "mean"),
        )
        .sort_values(["class_instance", "instance"])
    )

    if include_counts:
        extras = (
            group_df.groupby(["class_instance", "instance"], as_index=False)
            .agg(num_groups=("pdir", "size"), n_src_used=("n_src", "first"))
            .sort_values(["class_instance", "instance"])
        )
        summary = summary.merge(extras, on=["class_instance", "instance"], how="left")

    return summary


def save_manifest_results(
    manifest_csv: PATH_LIKE,
    group_df: pd.DataFrame,
    instance_df: pd.DataFrame,
) -> tuple[Path, Path]:
    """Save manifest-based evaluation CSVs beside the manifest file."""
    manifest_path = sanitize_path(manifest_csv)
    out_dir = manifest_path.parent
    group_path = out_dir / f"{manifest_path.stem}_groups.csv"
    instance_path = out_dir / f"{manifest_path.stem}_instances.csv"
    group_df.to_csv(group_path, index=False)
    instance_df.to_csv(instance_path, index=False)
    return group_path, instance_path


def save_cli_results(
    save_prefix: PATH_LIKE,
    group_df: pd.DataFrame,
    instance_df: pd.DataFrame,
) -> tuple[Path, Path]:
    """Save CLI evaluation CSVs using the requested prefix."""
    out_prefix = sanitize_path(save_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    group_path = out_prefix.with_name(f"{out_prefix.stem}_groups.csv")
    instance_path = out_prefix.with_name(f"{out_prefix.stem}_instances.csv")
    group_df.to_csv(group_path, index=False)
    instance_df.to_csv(instance_path, index=False)
    return group_path, instance_path


def log_manifest_results_to_wandb(
    group_df: pd.DataFrame,
    instance_df: pd.DataFrame,
    grid_rows: Sequence[Mapping[str, Any]],
    wandb_prefix: str,
) -> None:
    """Log manifest-based aggregate metrics and tables to W&B."""
    if wandb.run is None:
        LOGGER.warning("W&B run is not initialized; skipping manifest logging.")
        return

    class_instances = group_df["class_instance"].astype(str).unique().tolist()
    class_suffix = class_instances[0] if len(class_instances) == 1 else "ALL"
    table_rows = []
    for row in grid_rows:
        grid_path = row.get("grid_path")
        grid_image = (
            wandb.Image(str(grid_path), caption="grid")
            if isinstance(grid_path, Path) and grid_path.exists()
            else None
        )
        table_rows.append(
            [
                int(row["step"]),
                str(row["class_instance"]),
                str(row["instance"]),
                str(row["pdir"]),
                str(row["prompt"]),
                grid_image,
            ]
        )

    wandb.log(
        {
            f"{wandb_prefix}/{class_suffix}/overall_clip_i": float(
                instance_df["clip_i_mean"].mean()
            )
            if not instance_df.empty
            else float("nan"),
            f"{wandb_prefix}/{class_suffix}/overall_clip_t": float(
                instance_df["clip_t_mean"].mean()
            )
            if not instance_df.empty
            else float("nan"),
            f"{wandb_prefix}/{class_suffix}/overall_dino_i": float(
                instance_df["dino_i_mean"].mean()
            )
            if not instance_df.empty
            else float("nan"),
            f"{wandb_prefix}/{class_suffix}/per_instance_table": wandb.Table(
                dataframe=instance_df
            ),
            f"{wandb_prefix}/{class_suffix}/per_group_table": wandb.Table(
                columns=GRID_TABLE_COLUMNS,
                data=table_rows,
            ),
        }
    )


def setup_wandb(entity: str, project: str, run_name: str) -> None:
    """Initialize a W&B run for the CLI workflow."""
    if wandb.run is not None:
        LOGGER.info("Reusing existing W&B run: %s", wandb.run.name)
        return
    wandb.init(entity=entity, project=project, name=run_name)


def resolve_src_dirs(
    manifest_df: pd.DataFrame,
    max_src: int,
    uniform_size: int,
    device: torch.device,
    src_root_candidates: Sequence[PATH_LIKE] = SRC_ROOT_CANDIDATES,
) -> dict[str, torch.Tensor]:
    """Load source image batches per instance for the CLI workflow."""
    src_cache: dict[str, torch.Tensor] = {}

    for instance, sub_df in manifest_df.groupby("instance", sort=False):
        explicit_src_dir = sanitize_optional_path(
            sub_df["src_dir"].iloc[0] if "src_dir" in sub_df.columns else None
        )

        src_dir = explicit_src_dir
        if src_dir is None or not src_dir.is_dir():
            fallback_names = [
                str(value)
                for value in sub_df.get("class_instance", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ]
            src_dir = resolve_src_dir(
                instance=str(instance),
                roots=src_root_candidates,
                fallback_names=fallback_names,
            )
            if src_dir is None:
                LOGGER.warning(
                    "No source directory found for instance '%s'; skipping.",
                    instance,
                )
                continue
            LOGGER.info("Using resolved source directory for '%s': %s", instance, src_dir)

        src_paths = list_images(src_dir)
        if not src_paths:
            LOGGER.warning(
                "No source images found in %s for instance '%s'; skipping.",
                src_dir,
                instance,
            )
            continue

        src_cache[str(instance)] = load_imgs_minus1_1(
            src_paths,
            cap=max_src,
            uniform_size=uniform_size,
        ).to(device=device, non_blocking=True)

    return src_cache


def evaluate_all_groups(
    manifest_df: pd.DataFrame,
    evaluator: Any,
    src_cache: Mapping[str, torch.Tensor],
    max_gen_per_group: int,
    uniform_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate every manifest group for the CLI workflow."""
    group_rows: list[dict[str, Any]] = []

    for (class_instance, instance, pdir), sub_df in manifest_df.groupby(
        ["class_instance", "instance", "pdir"], sort=False
    ):
        instance_key = str(instance)
        if instance_key not in src_cache:
            continue

        row = sub_df.iloc[0].to_dict()
        gen_dir = sanitize_path(row["gen_dir"])
        prompt = clean_prompt(str(row.get("prompt", "")))
        src_batch = src_cache[instance_key]

        evaluated_group = evaluate_group(
            evaluator=evaluator,
            prompt=prompt,
            src_batch=src_batch,
            gen_dir=gen_dir,
            max_gen_per_group=max_gen_per_group,
            uniform_size=uniform_size,
            device=device,
        )
        if evaluated_group is None:
            continue

        gen_batch, (clip_i, clip_t, dino_i) = evaluated_group
        group_rows.append(
            {
                "class_instance": str(class_instance),
                "instance": instance_key,
                "pdir": str(pdir),
                "prompt": prompt,
                "n_src": int(src_batch.shape[0]),
                "n_gen": int(gen_batch.shape[0]),
                "clip_i": clip_i,
                "clip_t": clip_t,
                "dino_i": dino_i,
                "gen_dir": str(gen_dir),
            }
        )

    return group_rows


def log_cli_results_to_wandb(group_df: pd.DataFrame, instance_df: pd.DataFrame) -> None:
    """Log the CLI workflow outputs to W&B."""
    if wandb.run is None:
        LOGGER.warning("W&B run is not initialized; skipping CLI logging.")
        return

    for row in group_df.to_dict("records"):
        wandb.log(
            {
                "group/clip_i": float(row["clip_i"]),
                "group/clip_t": float(row["clip_t"]),
                "group/dino_i": float(row["dino_i"]),
                "group/n_src": int(row["n_src"]),
                "group/n_gen": int(row["n_gen"]),
                "group/class_instance": str(row["class_instance"]),
                "group/instance": str(row["instance"]),
                "group/pdir": str(row["pdir"]),
            }
        )

    wandb.log(
        {
            "overall/clip_i_mean": float(instance_df["clip_i_mean"].mean())
            if not instance_df.empty
            else float("nan"),
            "overall/clip_t_mean": float(instance_df["clip_t_mean"].mean())
            if not instance_df.empty
            else float("nan"),
            "overall/dino_i_mean": float(instance_df["dino_i_mean"].mean())
            if not instance_df.empty
            else float("nan"),
            "per_instance_table": wandb.Table(dataframe=instance_df),
            "per_group_table": wandb.Table(dataframe=group_df),
        }
    )


def run_clip_dino_on_manifest(
    manifest_csv: str,
    lmdb_path: Optional[str],
    clip_model: str = "ViT-B/32",
    max_src: int = 4,
    max_gen_per_group: int = 4,
    uniform_size: int = 224,
    wandb_prefix: str = "eval/step",
) -> None:
    """
    Run CLIP/DINO metrics for a manifest produced during training.

    This is the entrypoint used by `train_sd.py`, so its public signature stays
    intentionally small and stable.
    """
    manifest_df = load_manifest(manifest_csv, MANIFEST_REQUIRED_COLUMNS)
    device = resolve_device()
    evaluator = build_evaluator(device=device, clip_model=clip_model)
    dataset, labels_all = load_lmdb_dataset(lmdb_path, manifest_df=manifest_df)

    group_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []

    for row in manifest_df.to_dict("records"):
        gen_dir = sanitize_path(row["gen_dir"])
        prompt = str(row.get("prompt", ""))
        src_batch = empty_image_batch(uniform_size, device)

        src_label = row.get("src_label")
        use_lmdb = dataset is not None and labels_all is not None
        if use_lmdb and src_label not in ("", None) and not pd.isna(src_label):
            try:
                src_batch = _src_batch_from_lmdb(
                    dataset=dataset,
                    labels_all=labels_all,
                    label_id=int(src_label),
                    k=max_src,
                    device=device,
                    size=uniform_size,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Failed to load LMDB reference batch for label %s: %s",
                    src_label,
                    exc,
                )

        evaluated_group = evaluate_group(
            evaluator=evaluator,
            prompt=prompt,
            src_batch=src_batch,
            gen_dir=gen_dir,
            max_gen_per_group=max_gen_per_group,
            uniform_size=uniform_size,
            device=device,
        )
        if evaluated_group is None:
            continue

        gen_batch, (clip_i, clip_t, dino_i) = evaluated_group
        group_rows.append(
            {
                "step": int(row["step"]),
                "class_instance": str(row["class_instance"]),
                "instance": str(row["instance"]),
                "pdir": str(row["pdir"]),
                "prompt": prompt,
                "n_src": int(src_batch.shape[0]),
                "n_gen": int(gen_batch.shape[0]),
                "clip_i": clip_i,
                "clip_t": clip_t,
                "dino_i": dino_i,
            }
        )
        grid_rows.append(
            {
                "step": int(row["step"]),
                "class_instance": str(row["class_instance"]),
                "instance": str(row["instance"]),
                "pdir": str(row["pdir"]),
                "prompt": prompt,
                "grid_path": gen_dir / "grid.png",
            }
        )

    if not group_rows:
        raise SystemExit("No evaluated groups. Check manifest gen_dir, image extensions, and caps.")

    group_df = pd.DataFrame(group_rows).sort_values(["class_instance", "pdir"])
    instance_df = build_instance_summary(group_df, include_counts=False)
    save_manifest_results(manifest_csv, group_df, instance_df)
    log_manifest_results_to_wandb(group_df, instance_df, grid_rows, wandb_prefix)


def main() -> None:
    """CLI entrypoint for standalone manifest evaluation."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        "Run CLIP/DINO metrics on an evaluation manifest."
    )
    parser.add_argument(
        "--manifest_csv",
        type=Path,
        required=True,
        help=(
            "CSV with columns: class_instance,instance,pdir,prompt,gen_dir,src_dir "
            "(src_dir can be blank)"
        ),
    )
    parser.add_argument(
        "--evaluator_py",
        type=Path,
        required=True,
        help="Path to the evaluation.py module that defines ImageDirEvaluator.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--max_src", type=int, default=16)
    parser.add_argument("--max_gen_per_group", type=int, default=4)
    parser.add_argument("--uniform_size", type=int, default=224)
    parser.add_argument("--save_prefix", type=Path, default=Path("clip_eval_results"))
    parser.add_argument("--wandb_entity", default="meluxis-ets")
    parser.add_argument("--wandb_project", default="dreambooth")
    parser.add_argument(
        "--wandb_run_name",
        default="clip_eval_all_classes_instances",
    )
    parser.add_argument(
        "--src_root_candidate",
        action="append",
        type=Path,
        default=None,
        help=(
            "Optional source-image root to search when src_dir is blank in the manifest. "
            "Can be provided multiple times."
        ),
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    setup_wandb(args.wandb_entity, args.wandb_project, args.wandb_run_name)
    evaluator = build_evaluator(
        device=device,
        clip_model=args.clip_model,
        evaluator_py=args.evaluator_py,
    )
    manifest_df = load_manifest(args.manifest_csv, CLI_REQUIRED_COLUMNS)
    src_cache = resolve_src_dirs(
        manifest_df=manifest_df,
        max_src=args.max_src,
        uniform_size=args.uniform_size,
        device=device,
        src_root_candidates=tuple(args.src_root_candidate or SRC_ROOT_CANDIDATES),
    )

    group_rows = evaluate_all_groups(
        manifest_df=manifest_df,
        evaluator=evaluator,
        src_cache=src_cache,
        max_gen_per_group=args.max_gen_per_group,
        uniform_size=args.uniform_size,
        device=device,
    )
    if not group_rows:
        raise SystemExit("No evaluated groups. Check paths and that images exist.")

    group_df = pd.DataFrame(group_rows).sort_values(["class_instance", "pdir"])
    instance_df = build_instance_summary(group_df, include_counts=True)

    LOGGER.info("CLIP-I mean: %.4f", float(instance_df["clip_i_mean"].mean()))
    LOGGER.info("CLIP-T mean: %.4f", float(instance_df["clip_t_mean"].mean()))
    LOGGER.info("DINO-I mean: %.4f", float(instance_df["dino_i_mean"].mean()))

    save_cli_results(args.save_prefix, group_df, instance_df)
    log_cli_results_to_wandb(group_df, instance_df)
    wandb.finish()


if __name__ == "__main__":
    main()

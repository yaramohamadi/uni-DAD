#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-runwayml/stable-diffusion-v1-5}"
TARGET_CKPT_PATH="${TARGET_CKPT_PATH:-${PROJECT_ROOT}/checkpoints/backpack_backpack}"
OUTDIR="${OUTDIR:-${PROJECT_ROOT}/test_outputs}"
PROMPT="${PROMPT:-a prt backpack on a wooden table}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-}"
CLASS_NAME="${CLASS_NAME:-}"
RARE_TOKEN="${RARE_TOKEN:-prt}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
INSTANCE_ID="${INSTANCE_ID:-}"
PROMPT_ID="${PROMPT_ID:-0}"
SEEDS="${SEEDS:-0}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
DEVICE="${DEVICE:-cuda}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

export MODEL_ID TARGET_CKPT_PATH OUTDIR PROMPT PROMPT_TEMPLATE CLASS_NAME RARE_TOKEN
export PROMPTS_FILE INSTANCE_ID PROMPT_ID SEEDS HEIGHT WIDTH DEVICE VALIDATE_ONLY

mkdir -p "${OUTDIR}"

cat <<EOF
[test.sh] PROJECT_ROOT=${PROJECT_ROOT}
[test.sh] MODEL_ID=${MODEL_ID}
[test.sh] TARGET_CKPT_PATH=${TARGET_CKPT_PATH}
[test.sh] OUTDIR=${OUTDIR}
[test.sh] DEVICE=${DEVICE}
[test.sh] SEEDS=${SEEDS}
[test.sh] VALIDATE_ONLY=${VALIDATE_ONLY}
EOF

python - <<'PYCODE'
import json
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

sys.path.insert(0, os.environ["PROJECT_ROOT"])
from main.pipeline.paused_generation import (
    LIVE_INSTANCES,
    expand_prompt,
    extract_prompt_blocks,
    read_instances,
)

if TYPE_CHECKING:
    from diffusers import AutoencoderKL

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
MODEL_ID = os.environ["MODEL_ID"]
TARGET_CKPT_PATH = Path(os.environ["TARGET_CKPT_PATH"])
OUTDIR = Path(os.environ["OUTDIR"])
PROMPT = os.environ.get("PROMPT", "")
PROMPT_TEMPLATE = os.environ.get("PROMPT_TEMPLATE", "")
CLASS_NAME = os.environ.get("CLASS_NAME", "")
RARE_TOKEN = os.environ.get("RARE_TOKEN", "prt")
PROMPTS_FILE = os.environ.get("PROMPTS_FILE", "")
INSTANCE_ID_RAW = os.environ.get("INSTANCE_ID", "")
PROMPT_ID = int(os.environ.get("PROMPT_ID", "0"))
HEIGHT = int(os.environ.get("HEIGHT", "512"))
WIDTH = int(os.environ.get("WIDTH", "512"))
VALIDATE_ONLY = os.environ.get("VALIDATE_ONLY", "0") in {"1", "true", "True"}
REQUESTED_DEVICE = os.environ.get("DEVICE", "cuda")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",") if x.strip()]

if not TARGET_CKPT_PATH.exists():
    raise SystemExit(f"TARGET_CKPT_PATH does not exist: {TARGET_CKPT_PATH}")
if not SEEDS:
    raise SystemExit("SEEDS must contain at least one integer.")
if HEIGHT % 8 != 0 or WIDTH % 8 != 0:
    raise SystemExit("HEIGHT and WIDTH must be divisible by 8.")

if REQUESTED_DEVICE.startswith("cuda") and torch.cuda.is_available():
    device = torch.device(REQUESTED_DEVICE)
else:
    if REQUESTED_DEVICE.startswith("cuda"):
        print("[test.sh] CUDA requested but unavailable; falling back to CPU.", flush=True)
    device = torch.device("cpu")

dtype = torch.float16 if device.type == "cuda" else torch.float32


def resolve_prompt() -> tuple[str, str | None, str | None, str | None]:
    if PROMPTS_FILE and INSTANCE_ID_RAW != "":
        pairs = read_instances(PROMPTS_FILE)
        instance_id = int(INSTANCE_ID_RAW)
        if instance_id < 0 or instance_id >= len(pairs):
            raise SystemExit(
                f"INSTANCE_ID={instance_id} is out of range for {PROMPTS_FILE} (n={len(pairs)})."
            )
        inst_name, class_name = pairs[instance_id]
        object_prompts, live_prompts = extract_prompt_blocks(PROMPTS_FILE)
        prompt_bank = live_prompts if (inst_name, class_name) in LIVE_INSTANCES else object_prompts
        if not prompt_bank:
            raise SystemExit(f"No prompt bank could be parsed from {PROMPTS_FILE}.")
        template = prompt_bank[PROMPT_ID % len(prompt_bank)]
        return expand_prompt(template, RARE_TOKEN, class_name), inst_name, class_name, template

    if PROMPT_TEMPLATE:
        if not CLASS_NAME:
            raise SystemExit("CLASS_NAME is required when PROMPT_TEMPLATE is used.")
        return expand_prompt(PROMPT_TEMPLATE, RARE_TOKEN, CLASS_NAME), None, CLASS_NAME, PROMPT_TEMPLATE

    if not PROMPT.strip():
        raise SystemExit(
            "Provide PROMPT, or PROMPTS_FILE+INSTANCE_ID, or PROMPT_TEMPLATE+CLASS_NAME."
        )
    return PROMPT.strip(), None, CLASS_NAME or None, None


def _unwrap_state_dict(state_dict: dict) -> dict:
    for key in ("state_dict", "model_state_dict", "model", "module"):
        value = state_dict.get(key)
        if isinstance(value, dict):
            state_dict = value
    return state_dict


def _extract_generator_state_dict(state_dict: dict) -> dict:
    prefixes = (
        "feedforward_model.",
        "module.feedforward_model.",
        "model.feedforward_model.",
    )
    for prefix in prefixes:
        if any(key.startswith(prefix) for key in state_dict):
            return {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
    return state_dict


def _resolve_checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("pytorch_model.bin", "model_ema.pt", "model.pt"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raise SystemExit(
        "Could not resolve a generator checkpoint file from TARGET_CKPT_PATH. "
        "Expected a Diffusers UNet folder, a training checkpoint dir with pytorch_model.bin, "
        "or a model.pt/model_ema.pt file."
    )


def load_generation_modules(
    model_id: str,
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Any, Any, Any, Any]:
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device).eval()
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype).to(device).eval()

    if checkpoint_path.is_dir() and (checkpoint_path / "unet" / "config.json").exists():
        print(f"[test.sh] Loading Diffusers UNet from root folder: {checkpoint_path}", flush=True)
        unet = UNet2DConditionModel.from_pretrained(
            str(checkpoint_path),
            subfolder="unet",
            torch_dtype=dtype,
        )
        return scheduler, tokenizer, text_encoder, vae, unet.to(device).eval()

    if checkpoint_path.is_dir() and (checkpoint_path / "config.json").exists():
        print(f"[test.sh] Loading Diffusers UNet from folder: {checkpoint_path}", flush=True)
        unet = UNet2DConditionModel.from_pretrained(
            str(checkpoint_path),
            torch_dtype=dtype,
        )
        return scheduler, tokenizer, text_encoder, vae, unet.to(device).eval()

    checkpoint_file = _resolve_checkpoint_file(checkpoint_path)
    print(f"[test.sh] Loading generator weights from: {checkpoint_file}", flush=True)
    state_dict = torch.load(checkpoint_file, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise SystemExit(f"Unsupported checkpoint payload type: {type(state_dict)}")
    state_dict = _unwrap_state_dict(state_dict)
    state_dict = _extract_generator_state_dict(state_dict)

    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=dtype,
    )
    missing, unexpected = unet.load_state_dict(state_dict, strict=False)
    print(
        f"[test.sh] Loaded generator state_dict. missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if missing:
        print(f"[test.sh] Missing keys sample: {missing[:8]}", flush=True)
    if unexpected:
        print(f"[test.sh] Unexpected keys sample: {unexpected[:8]}", flush=True)
    return scheduler, tokenizer, text_encoder, vae, unet.to(device).eval()


def decode_to_pil(vae: "AutoencoderKL", latents: torch.Tensor) -> Image.Image:
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    latents = (latents / vae.config.scaling_factor).to(device=vae_device, dtype=vae_dtype)
    with torch.no_grad():
        image = vae.decode(latents).sample.float().clamp(-1, 1)
    image = ((image[0] + 1.0) / 2.0).clamp(0, 1)
    image = image.permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
    return Image.fromarray(image)


def save_grid(images: list[Image.Image], out_path: Path) -> None:
    if not images:
        return
    cols = min(4, len(images))
    rows = math.ceil(len(images) / cols)
    width, height = images[0].size
    grid = Image.new("RGB", (cols * width, rows * height))
    for idx, image in enumerate(images):
        x = (idx % cols) * width
        y = (idx // cols) * height
        grid.paste(image, (x, y))
    grid.save(out_path)


prompt_text, instance_name, class_name, prompt_template_used = resolve_prompt()
out_dir = OUTDIR / (instance_name or "custom_prompt")
out_dir.mkdir(parents=True, exist_ok=True)

meta = {
    "model_id": MODEL_ID,
    "target_ckpt_path": str(TARGET_CKPT_PATH),
    "prompt": prompt_text,
    "prompt_template": prompt_template_used,
    "class_name": class_name,
    "instance_name": instance_name,
    "height": HEIGHT,
    "width": WIDTH,
    "device": str(device),
    "dtype": str(dtype),
    "seeds": SEEDS,
}
print("[test.sh] Resolved generation config:")
print(json.dumps(meta, indent=2), flush=True)

if VALIDATE_ONLY:
    print("[test.sh] VALIDATE_ONLY=1 -> skipping model load and generation.", flush=True)
    raise SystemExit(0)

scheduler, tokenizer, text_encoder, vae, unet = load_generation_modules(
    MODEL_ID,
    TARGET_CKPT_PATH,
    device,
    dtype,
)

for module in (text_encoder, vae, unet):
    module.requires_grad_(False)

text_inputs = tokenizer(
    [prompt_text],
    max_length=tokenizer.model_max_length,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
)
input_ids = text_inputs.input_ids.to(device)
with torch.no_grad():
    text_emb = text_encoder(input_ids)[0].to(device=device, dtype=dtype)

num_train_timesteps = int(scheduler.config.num_train_timesteps)
t_last = torch.tensor([num_train_timesteps - 1], device=device, dtype=torch.long)
alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=dtype)
a_bar = alphas_cumprod[t_last].view(-1, 1, 1, 1)

saved_images: list[Image.Image] = []
for seed in SEEDS:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    noise = torch.randn(
        1,
        int(unet.config.in_channels),
        HEIGHT // 8,
        WIDTH // 8,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        eps = unet(noise, t_last, text_emb).sample
        x0 = (noise - torch.sqrt(1.0 - a_bar) * eps) / torch.sqrt(a_bar)
        image = decode_to_pil(vae, x0)
    out_path = out_dir / f"seed_{seed:03d}.png"
    image.save(out_path)
    saved_images.append(image)
    print(f"[test.sh] Wrote {out_path}", flush=True)

save_grid(saved_images, out_dir / "grid.png")
(out_dir / "generation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"[test.sh] Done. Outputs saved under {out_dir}", flush=True)
PYCODE

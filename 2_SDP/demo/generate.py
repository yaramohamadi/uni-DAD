# How this file is used in the current repo:
# 1) The main generation path is pause-time generation inside main/pipeline/train_sd.py.
#    That path imports the prompt helpers from this file via paused_generation.py.
# 2) To trigger generation in the normal training workflow, launch the trainer
#    (see scripts/run_train.sh) with flags such as:
#    --gen_pause_steps --gen_prompts_file --gen_outputs_root --gen_training_name
#    --gen_samples_per_prompt --gen_seeds
# 3) This file does not currently expose its own argparse/__main__ entrypoint, so
#    it is better thought of as a shared helper module plus an offline image writer.
import os
import re
import json
import argparse
import subprocess
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import List, Tuple, Union
import ast, re
# The defaults below mirror the repo's DreamBooth-style prompt file and output
# directory conventions used by pause-time generation and offline exports.
# ------------------------------ CONFIG --------------------------------

# Instances that should use LIVE SUBJECT prompts (others use OBJECT prompts)
LIVE_INSTANCES = {
    ("dog", "dog"), ("dog2", "dog"), ("dog3", "dog"), ("dog5", "dog"),
    ("dog6", "dog"), ("dog7", "dog"), ("dog8", "dog"),
    ("cat", "cat"), ("cat2", "cat"),
}

# Default paths relative to repo root. Adjust if your project uses different paths.
DEFAULT_PROMPTS_FILE = "data/prompts_and_classes.txt"
OUTPUTS_DIR          = Path("/projets/Mdesbos/Dreambooth_outputs/")  # Path("outputs")
GEN_SUBDIR           = "gen"
N_SAMPLES_PER_PROMPT = 4
DEFAULT_SEEDS        = [0, 1, 2, 3]
DEFAULT_DTYPE        = "float16"  # 'float16' or 'float32'

# These prompt-file parsers are shared with main/pipeline/paused_generation.py so
# training-time paused generation and any offline use interpret prompts identically.
def read_instances(prompts_file: Path) -> List[Tuple[str,str]]:
    """
    Parse (instance, class) pairs from the 'Classes' section at the top of prompts_and_classes.txt.
    Expected format:
        Classes
        subject_name,class
        backpack,backpack
        dog2,dog
        ...
    Stops when it reaches a blank line or the 'Prompts' section.
    """
    lines = prompts_file.read_text(encoding="utf-8").splitlines()
    pairs: List[Tuple[str,str]] = []
    in_classes = False
    for ln in lines:
        s = ln.strip()
        if not s:
            # blank line ends the section if we already collected some
            if in_classes and pairs:
                break
            continue
        if s.lower().startswith("classes"):
            in_classes = True
            continue
        if s.lower().startswith("prompts"):
            break
        if in_classes:
            if s.lower().startswith("subject_name"):
                continue
            if "," in s:
                inst, cls = [x.strip() for x in s.split(",", 1)]
                if inst and cls and inst != "..." and cls != "...":
                    pairs.append((inst, cls))
    return pairs

def extract_prompt_blocks(prompts_file: Union[str, Path]) -> Tuple[List[str], List[str]]:
    """
    Robust parser for OBJECT and LIVE prompt sections.
    Supports:
      - Plain lines under headers (OBJECT PROMPTS / LIVE PROMPTS)
      - Bulleted lists (- "a {0} {1} ...", * '...')
      - Python-like list literals ([ "a {0} {1} ...", ... ])
      - Fallback: quoted strings followed by .format(unique_token, class_token)
    Returns (object_prompts, live_prompts), each normalized to exactly 25 prompts.
    """
    p = Path(prompts_file)
    txt_lines = p.read_text(encoding="utf-8").splitlines()

    def _clean_line(s: str) -> str:
        s = s.strip()
        if not s:
            return ""
        # strip common bullets / numbering
        s = re.sub(r'^\s*(?:[-*•]\s+|\d+\.\s+|\[\s*)', '', s)
        # strip trailing commas/brackets
        s = s.rstrip(",")
        s = re.sub(r'\s*\]\s*,?\s*$', '', s)
        # strip quotes if line is a single quoted item
        if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in "\"'"):
            s = s[1:-1]
        return s.strip()

    def _collect_after(header_keywords: List[str]) -> List[str]:
        out, take = [], False
        for raw in txt_lines:
            line = raw.strip()
            upper = line.upper()
            # start collection when header keywords all appear
            if all(k.upper() in upper for k in header_keywords):
                take = True
                continue
            if not take:
                continue
            # stop on next apparent header
            if upper and upper.isupper() and len(line) < 80 and any(
                k in upper for k in ["PROMPTS", "OBJECT", "LIVE", "CLASSES", "SUBJECT"]
            ):
                break
            if not line or line.startswith("#"):
                continue
            # handle Python list blocks by trying a literal_eval
            if line.startswith("["):
                try:
                    arr = ast.literal_eval(line)
                    if isinstance(arr, list):
                        for it in arr:
                            s = _clean_line(str(it))
                            if s:
                                out.append(s)
                        continue
                except Exception:
                    pass
            s = _clean_line(line)
            if s:
                out.append(s)
        return out

    object_prompts = _collect_after(["OBJECT", "PROMPTS"])
    live_prompts   = _collect_after(["LIVE", "PROMPTS"])

    # Fallbacks:
    if not object_prompts and not live_prompts:
        raw = p.read_text(encoding="utf-8")
        # Grab quoted strings followed by .format(…unique…, …class…)
        candidates = re.findall(
            r"""["']([^"']+)["']\s*\.\s*format\s*\(\s*[^,]*unique[^,]*\s*,\s*[^)]*class[^)]*\)""",
            raw, flags=re.IGNORECASE
        )
        object_prompts = candidates[:]
        live_prompts   = candidates[:]

    # If one is missing, mirror the other
    if object_prompts and not live_prompts:
        live_prompts = object_prompts[:]
    if live_prompts and not object_prompts:
        object_prompts = live_prompts[:]

    # Dedup while preserving order
    def _dedup(seq: List[str]) -> List[str]:
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    object_prompts = _dedup(object_prompts)
    live_prompts   = _dedup(live_prompts)

    # Normalize to exactly 25 prompts each (pad by cycling; trim if longer)
    def _to_25(seq: List[str]) -> List[str]:
        if not seq:
            return []
        if len(seq) >= 25:
            return seq[:25]
        k, out = len(seq), []
        i = 0
        while len(out) < 25:
            out.append(seq[i % k])
            i += 1
        return out

    object_prompts = _to_25(object_prompts)
    live_prompts   = _to_25(live_prompts)

    return object_prompts, live_prompts


def expand_prompt(template: str, class_name: str):
    """
    Support both explicit format placeholders:
      '{0} {1}'.format(unique_token, class_token)
    and named placeholders:
      '{unique_token} {class_token}'
    We replace with training convention: unique_token = 'sks', class_token = class_name.
    """
    unique_token = f"prt"
    class_token  = class_name
    s = template
    # Named placeholders
    s = s.replace("{unique_token}", unique_token).replace("{class_token}", class_token)
    # Format-style placeholders
    s = s.replace("{0}", unique_token).replace("{1}", class_token)
    return s

# generate_images() is the offline helper in this file. It assumes the caller has
# already constructed a Diffusers pipeline object (`pipe`) and only handles prompt
# expansion, output folder layout, seed control, and optional W&B logging.
# In the current repo, the closest end-to-end launch example is the paused-generation
# path from train_sd.py rather than a direct `python scripts/generate.py ...` call.
def generate_images(pipe, prompts, out_dir: Path, n_per_prompt=4, seeds=None, device="cuda",
                    wandb_run=None, instance=None, class_name=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if seeds is None:
        seeds = list(range(n_per_prompt))

    # Save the expanded prompts for traceability
    (out_dir / "prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")

    import torch
    for p_idx, prompt in enumerate(prompts):
        p_folder = out_dir / f"p_{p_idx:03d}"
        p_folder.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            g = torch.Generator(device=device).manual_seed(seed)
            image = pipe(prompt, guidance_scale=7.5, generator=g).images[0]
            img_path = p_folder / f"seed_{seed:03d}.png"
            image.save(img_path)
            if wandb_run is not None:
                try:
                    import wandb
                    wandb_run.log({
                        "prompt_idx": p_idx,
                        "seed": seed,
                        "instance": instance,
                        "class": class_name,
                        "prompt": prompt,
                        "image": wandb.Image(str(img_path), caption=f"{prompt} | seed={seed}")
                    })
                except Exception as e:
                    warn(f"W&B log failed for {img_path}: {e}")

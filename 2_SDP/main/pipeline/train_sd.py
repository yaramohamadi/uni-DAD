import os

DEBUG_RUNTIME = os.environ.get("DEBUG_RUNTIME", "0") in ("1", "true", "True")

if DEBUG_RUNTIME:
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("PYTORCH_SDP_KERNEL", "math")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")

import torch

if DEBUG_RUNTIME:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

import matplotlib
matplotlib.use('Agg')
from main.utils import prepare_images_for_saving, draw_valued_array, draw_probability_histogram
from main.prepare_data.sd_image_dataset import SDImageDatasetLMDB    
from transformers import CLIPTokenizer
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from main.utils import SDTextDataset, cycle 
from main.models.sd_unified_model import SDUniModel
from main.pipeline.paused_generation import (
    LIVE_INSTANCES,
    extract_prompt_blocks,
    expand_prompt,
    read_instances,
)
from main.prepare_data.sd_text_dataset import SDTextDatasetLMDB 
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from accelerate import Accelerator
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType
)
import torch
import torch.distributed as dist
import argparse 
import wandb 
import time 
import os
import re
import lmdb
import numpy as np
from torch.utils.data import Subset
import re
import math
from torchvision.utils import make_grid
import sys
print(">>> argv:", " ".join(sys.argv), flush=True)

generator_grad_norm = None
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

def _dist_ready():
    return dist.is_available() and dist.is_initialized()

from PIL import Image
import numpy as np
import torch

print("[ENV] torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
print("[ENV] device", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0), flush=True)
from torch.backends.cuda import sdp_kernel
print("[ENV] SDPA", sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True), flush=True)

# This module is the end-to-end SD1.5 training entrypoint for the pipeline.
# The main lifecycle is:
# 1) build runtime / model / dataloaders in Trainer.__init__,
# 2) alternate generator, guidance, and optional target-teacher updates in train_one_step,
# 3) emit pause-time generations and evaluation manifests from _paused_generate.
def _to_pil(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if t.ndim == 4:  # [B,C,H,W] -> take first
            t = t[0]
        if t.ndim == 3 and t.shape[0] in (1, 3):  # CHW -> HWC
            t = t.permute(1, 2, 0)
        if t.max() <= 1.0:
            t = (t * 255.0).round()
        return Image.fromarray(t.numpy().astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported type for _to_pil: {type(x)}")


# assumes ds[i] returns dict with 'class_labels'
def _collect_labels(ds, max_scan=10_000_000):
    labels = []
    N = len(ds)
    for i in range(min(N, max_scan)):
        y = ds[i]['class_labels']
        # y could be a scalar tensor
        labels.append(int(y.item() if hasattr(y, "item") else int(y)))
    return np.array(labels, dtype=np.int64)

# if ds[i] doesnt include labels :
def read_all_labels_from_lmdb(lmdb_path, max_scan=10_000_000):
    """
    Returns a numpy array of int64 labels by reading keys:
      labels_{idx:06d}_data
    If no labels are present, raises a KeyError with a helpful message.
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    labels = []
    with env.begin(write=False) as txn:
        i = 0
        while i < max_scan:
            key = f"labels_{i:06d}_data".encode("utf-8")
            b = txn.get(key)
            if b is None:
                break
            labels.append(int(np.frombuffer(b, dtype=np.int64)[0]))
            i += 1
    if not labels:
        raise KeyError(
            f"No label keys found in LMDB at {lmdb_path}. "
            f"Expected keys like labels_000000_data. "
            f"Rebuild the LMDB with labels or provide guidance_include_labels=None to use ALL."
        )
    return np.asarray(labels, dtype=np.int64)

# parse comma-separated ids of gen_include_labels from the training call in run_train.sh
def _parse_ids(csv_or_none):
    if not csv_or_none:
        return None
    return sorted({int(x) for x in csv_or_none.split(",") if x.strip()})

def _to_int_list(val, name: str, sep: str = ","):
    """
    Normalize CLI args that can be str like "1,2,3", a list/tuple of strs or ints,
    or already a list of ints. Returns a flat list[int] with order preserved, no dupes.
    """
    if val is None:
        return []

    # start with a list of tokens (as strings)
    tokens = []
    if isinstance(val, (list, tuple, set)):
        for x in val:
            if isinstance(x, str):
                tokens.extend([t for t in x.split(sep) if t.strip() != ""])
            else:
                tokens.append(x)  # likely an int already
    elif isinstance(val, str):
        tokens = [t for t in val.split(sep) if t.strip() != ""]
    else:
        # single int or something else castable
        tokens = [val]

    # cast to int, preserving order and removing duplicates
    out, seen = [], set()
    for t in tokens:
        v = int(t)  # int("5") or int(5)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def _require_equal(a_len: int, b_len: int, a_name: str, b_name: str):
    if a_len != b_len:
        raise ValueError(f"--{a_name} ({a_len}) must equal --{b_name} ({b_len}).")

def infer_num_labels_from_lmdb(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    # Fast & memory-light: sample some labels to estimate max id
    max_id = -1
    with env.begin() as txn:
        # Try to read a recorded shape first
        shp = txn.get(b"labels_shape")
        if shp:
            # optional sanity only; not used for count
            pass
        # Sample all or first N (change 100000 if you have many)
        i = 0
        while True:
            key = f"labels_{i:06d}_data".encode()
            val = txn.get(key)
            if val is None:
                break
            cur = np.frombuffer(val, dtype=np.int64)
            max_id = max(max_id, int(cur[0]))
            i += 1
    return max_id + 1

# remove the rare token when it appears as a word, e.g., " a prt cat" -> " a cat"
def make_teacher_prompt(s: str, rare_token: str) -> str:
    # collapse duplicate whitespace as well
    s2 = re.sub(rf"\b{re.escape(rare_token)}\b", "", s)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


class Trainer:
    def __init__(self, args):
        self.args = args
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True 

        # Initialization is intentionally front-loaded here so every rank sees the
        # same run layout before any model wrapping or data iteration begins.
        # --- Accelerate / W&B init ---
        accelerator_project_config = ProjectConfiguration(logging_dir=args.log_path)
        kwargs_handlers = [ddp_kwargs]
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            kwargs_handlers=kwargs_handlers,
            mixed_precision="bf16",
            log_with="wandb",
            project_config=accelerator_project_config,
            split_batches=False
        )
        set_seed(args.seed + accelerator.process_index)

        print(accelerator.state)

        # ---- single timestamp shared across ranks ----
        ts = int(time.time())
        if _dist_ready():
            t = torch.tensor([ts], device=accelerator.device)
            if not accelerator.is_main_process:
                t.zero_()
            dist.broadcast(t, src=0)
            ts = int(t.item())

        # Save to self for later use (on all ranks)
        self.output_path = os.path.join(args.output_path, f"time_{ts}_seed{args.seed}")
        self.cache_dir   = os.path.join( (args.cache_dir or
                                  os.environ.get("DMD2_CACHE") or
                                  os.path.join(os.path.expanduser("~"), ".cache", "dmd2")),
                                 f"time_{ts}_seed{args.seed}" )

        # ---- create dirs only on main process; avoid races with exist_ok=True ----
        if accelerator.is_main_process:
            os.makedirs(args.log_path, exist_ok=True)
            os.makedirs(self.output_path, exist_ok=True)
            os.makedirs(self.cache_dir, exist_ok=True)

        # sync before anyone uses the folders
        accelerator.wait_for_everyone()

        # ---- W&B: init only on main process ----
        if accelerator.is_main_process:

            # honor WANDB_DIR if provided; otherwise use args.log_path
            log_dir = os.environ.get("WANDB_DIR", getattr(args, "log_path", "."))
            os.makedirs(log_dir, exist_ok=True)

            run = wandb.init(
                config=args,
                dir=log_dir,
                mode=os.environ.get("WANDB_MODE", "online"),
                entity=args.wandb_entity,
                project=args.wandb_project,
            )

            # only snapshot code if not explicitly disabled
            if os.environ.get("WANDB_DISABLE_CODE", "").lower() not in ("1", "true", "yes"):
                try:
                    wandb.run.log_code(
                        ".",
                        include_fn=lambda p: p.endswith((".py", ".sh", ".yaml", ".yml", ".json")),
                        exclude_fn=lambda p: any(seg in p for seg in ["venv/", "data/", "output_dmd2/", "checkpoints/", ".git/"]),
                    )
                except Exception as _e:
                    print(f"[W&B] log_code skipped: {_e}", flush=True)
            else:
                print("[W&B] Code snapshot disabled by WANDB_DISABLE_CODE", flush=True)

            wandb.run.name = args.wandb_name
            print(f"run dir: {run.dir}")

            # unify x-axis across all charts
            # wandb.define_metric("*", step_metric="trainer/step")

            self.wandb_folder = run.dir
            os.makedirs(self.wandb_folder, exist_ok=True)


        if args.num_classes is None:
            lmdb_for_labels = args.train_lmdb_path or args.real_image_path
            args.num_classes = infer_num_labels_from_lmdb(lmdb_for_labels)
            print(f"[init] inferred num_classes (instances) = {args.num_classes}")
        # sanity check
        if (args.cls_on_clean_image or args.gen_cls_loss) and args.num_classes < 2:
            raise ValueError(
                f"num_classes={args.num_classes} invalid for CrossEntropy; need >=2."
            )
        # ---- model ----
        self.model = SDUniModel(args, accelerator)
        # ---- Optional: override the SOURCE-TEACHER UNet with a DreamBooth UNet ----
       
        if getattr(args, "source_unet_path", None):
            from diffusers import UNet2DConditionModel
            src_path = args.source_unet_path

            # Build a temporary UNet directly from the DreamBooth folder
            db_unet = UNet2DConditionModel.from_pretrained(src_path)

            # Find the "source teacher" UNet inside your composite model.
            # We check several common names to avoid hard assumptions.
            candidate_attrs = [
                # typical names seen in DMD2-style codebases:
                "source_teacher_unet", "real_unet", "source_unet",
                # sometimes the teacher lives under guidance_model:
                ("guidance_model", "source_teacher_unet"),
                ("guidance_model", "real_unet"),
                ("guidance_model", "source_unet"),
            ]
            target_unet = None
            for attr in candidate_attrs:
                if isinstance(attr, tuple):
                    parent, child = attr
                    mod = getattr(self.model, parent, None)
                    if mod is not None and hasattr(mod, child):
                        target_unet = getattr(mod, child)
                        break
                else:
                    if hasattr(self.model, attr):
                        target_unet = getattr(self.model, attr)
                        break

            if target_unet is None:
                raise RuntimeError(
                    "Could not locate a source-teacher UNet on self.model. "
                    "Look for an attribute like source_teacher_unet/real_unet/source_unet."
                )

            # Load weights (allow missing keys for safety; SD1.5 DreamBooth UNet should match)
            missing, unexpected = target_unet.load_state_dict(db_unet.state_dict(), strict=False)
            print(f"[DreamBooth->Source] missing={len(missing)} unexpected={len(unexpected)}")

            # Keep the source teacher frozen
            for p in target_unet.parameters():
                p.requires_grad = False

            # (Optional) also initialize the generator UNet from the same weights
            if getattr(args, "init_generator_from_source", False):
                # try typical generator names
                gen_unet = None
                for name in ["feedforward_model", "generator_unet", "student_unet"]:
                    if hasattr(self.model, name):
                        gen_unet = getattr(self.model, name)
                        break
                if gen_unet is not None:
                    mg, ug = gen_unet.load_state_dict(db_unet.state_dict(), strict=False)
                    print(f"[DreamBooth->Generator] missing={len(mg)} unexpected={len(ug)}")
                else:
                    print("[DreamBooth] generator UNet not found; skipped init_generator_from_source")


        # --- Freeze any VAE so DDP doesn't expect its grads ---
        def _freeze_module(m):
            if m is None:
                return
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

        # Try common attribute names on top model and its submodules
        for holder in (
            self.model,
            getattr(self.model, "feedforward_model", None),
            getattr(self.model, "guidance_model", None),
        ):
            if holder is None:
                continue
            for name in ("vae", "first_stage_model", "autoencoder", "autoencoder_kl"):
                _freeze_module(getattr(holder, name, None))

        # Sanity: ensure no VAE params still trainable
        _leaking = [n for n, p in self.model.named_parameters()
                    if p.requires_grad and ("vae" in n or "first_stage_model" in n)]
        assert not _leaking, f"VAE params still trainable: {_leaking[:5]}"

        self.max_grad_norm = args.max_grad_norm
        self.denoising = args.denoising
        self.step = 0 

        if args.ckpt_only_path is not None:
            print("[INIT CHECKPOINTs] - starting from given checkpoints")
            if accelerator.is_main_process:
                print(f"loading ckpt only from {args.ckpt_only_path}")
            generator_path = os.path.join(args.ckpt_only_path, "pytorch_model.bin")
            guidance_path = os.path.join(args.ckpt_only_path, "pytorch_model_1.bin")
            print(self.model.feedforward_model.load_state_dict(torch.load(generator_path, map_location="cpu"), strict=False))
            print(self.model.guidance_model.load_state_dict(torch.load(guidance_path, map_location="cpu"), strict=False))

            # Initialize weights only; do NOT adopt the old step
            self.step = 0
            if accelerator.is_main_process:
                print(f"Initialized from {args.ckpt_only_path}; starting fresh at step {self.step}.")

        if args.targetteacher_ckpt_path is not None:
            accelerator.print("[INIT CHECKPOINTs] - starting from given target teacher checkpoints")
            
            # Find / load UNet weights from a diffusers-style folder
            unet_dir = os.path.join(args.targetteacher_ckpt_path, "unet")
            if os.path.isdir(unet_dir):
                # Typical diffusers checkpoint: root has "unet/" with diffusion_pytorch_model.bin
                tmp_unet = UNet2DConditionModel.from_pretrained(
                    args.targetteacher_ckpt_path,
                    subfolder="unet"
                )
                sd_tt = tmp_unet.state_dict()
                del tmp_unet
                accelerator.print(f"[TT] Loaded UNet weights from {unet_dir}")
            else:
                # Fallback: user gave a direct .bin path
                accelerator.print(f"[TT] Loading raw state_dict from {args.targetteacher_ckpt_path}")
                sd_tt = torch.load(args.targetteacher_ckpt_path, map_location="cpu")


            tt_mod = getattr(self.model.guidance_model, "target_teacher_unet", None) or \
                    getattr(self.model.guidance_model, "target_unet", None)
            assert tt_mod is not None, "enable_target_teacher=True but no target_teacher_unet/target_unet found in guidance_model"

            accelerator.print(tt_mod.load_state_dict(sd_tt, strict=False))
            accelerator.print(f"Initialized target teacher from {args.targetteacher_ckpt_path}; starting fresh at step {self.step}.")

            # Freeze TT params when a specific TT checkpoint is provided
            for p in tt_mod.parameters():
                p.requires_grad = False
            tt_mod.eval()

            accelerator.print(
                f"Initialized target teacher from {args.targetteacher_ckpt_path}; "
                f"FROZEN for training (offline teacher mode)."
            )
            
            tt_params = [p for p in tt_mod.parameters() if p.requires_grad]
            if len(tt_params) > 0:
                self.tt_optimizer = torch.optim.AdamW(tt_params, lr=args.guidance_lr, betas=(0.9, 0.999), weight_decay=0.01)
            else:
                self.tt_optimizer = None  # No optimizer for TT since it's frozen

        if args.generator_ckpt_path is not None:
            if accelerator.is_main_process:
                print(f"loading generator ckpt from {args.generator_ckpt_path}")
            print(self.model.feedforward_model.load_state_dict(torch.load(args.generator_ckpt_path, map_location="cpu"), strict=True))
      
        self.tokenizer = CLIPTokenizer.from_pretrained(
            args.model_id, subfolder="tokenizer"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        uncond_input_ids = self.tokenizer(
            [""], max_length=self.tokenizer.model_max_length,  ###
            return_tensors="pt", padding="max_length", truncation=True
        ).input_ids.to(accelerator.device)

        # ---- TRAIN TEXT DATASET (prompts) ----
        if args.train_lmdb_path is not None:
            dataset = SDTextDatasetLMDB(
                args.train_lmdb_path,
                tokenizer_one=self.tokenizer
            )
        else:
            dataset = SDTextDataset(args.train_prompt_path, self.tokenizer)

        self.uncond_embedding = self.model.text_encoder(uncond_input_ids)[0]

        # ---- REAL IMAGE DATASET (for GAN / denoising) ----
        real_dataset = SDImageDatasetLMDB(
            args.real_image_path, 
            tokenizer_one=self.tokenizer
        )

        # filter datasets by instance ids if requested
        # 1) gather all labels once from the IMAGE dataset
        # robust label loading: prefer dataset if it exposes 'class_labels', else read from LMDB
        try:
            labels_all = _collect_labels(real_dataset)
        except Exception:
            lmdb_for_labels = args.real_image_path or args.train_lmdb_path
            assert lmdb_for_labels, "Need a path to LMDB to read labels from."
            labels_all = read_all_labels_from_lmdb(lmdb_for_labels)

        all_idx = np.arange(len(labels_all))

        # 2) parse target set for generator; guidance left as ALL when None (Option B)
        gen_keep = _parse_ids(getattr(args, "gen_include_labels", None))
        
        # indices for generator: target-only if provided; else all
        if gen_keep:
            gen_mask = np.isin(labels_all, gen_keep)
            idx_gen  = all_idx[gen_mask]
            assert len(idx_gen) > 0, f"[gen] No samples match labels {gen_keep}"
        else:
            idx_gen = all_idx
            print("[gen] using ALL samples")

        # 3) build two aligned subsets (text + image)
        # Keeping the same filtered indices for both datasets preserves prompt/image
        # alignment when we later draw from text-only and image-only dataloaders.
        text_ds   = Subset(dataset,      idx_gen.tolist())
        image_ds  = Subset(real_dataset, idx_gen.tolist())

        # (optional) batch-size sanity
        assert len(text_ds)  >= args.batch_size, f"[gen] only {len(text_ds)} samples remain (< batch={args.batch_size})"
        
        print(f"[split] generator: {len(text_ds)} samples (labels {gen_keep if gen_keep else 'ALL'})")
        

        # ---- DATALOADERS ----
        #GRADIENT CHECKPOINTING DISABLE 
        gen_unet = getattr(self.model, "feedforward_model", None)
        guid     = getattr(self.model, "guidance_model", None)
        fake_unet = getattr(guid, "fake_unet", None) if guid is not None else None

        def _disable_ckpt(m):
            if m is None: return
            # diffusers-style
            fn_disable = getattr(m, "disable_gradient_checkpointing", None)
            if callable(fn_disable):
                try:
                    fn_disable()
                except Exception:
                    pass
            else:
                # Fallback: some versions accept enable_gradient_checkpointing(False)
                fn_enable = getattr(m, "enable_gradient_checkpointing", None)
                if callable(fn_enable):
                    import inspect
                    try:
                        # Only pass False if the method actually accepts an arg
                        if len(inspect.signature(fn_enable).parameters) > 1:
                            fn_enable(False)
                        else:
                            # no-arg version: calling it would ENABLE, so skip
                            pass
                    except Exception:
                        pass

            # Hard-stop: ensure every block has the flag off
            for sm in m.modules():
                if hasattr(sm, "gradient_checkpointing"):
                    sm.gradient_checkpointing = False
                

        if hasattr(self.model.guidance_model, "vae"):
            for p in self.model.guidance_model.vae.parameters():
                p.requires_grad = False
            self.model.guidance_model.vae.eval()
        
        dataloader = torch.utils.data.DataLoader(text_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        dataloader = accelerator.prepare(dataloader)
        self.dataloader = cycle(dataloader)

        real_dataloader = torch.utils.data.DataLoader(
            image_ds, num_workers=args.num_workers, 
            batch_size=args.batch_size, shuffle=True, 
            drop_last=True
        )
        real_dataloader = accelerator.prepare(real_dataloader)
        self.real_dataloader = cycle(real_dataloader)

        # use two dataloader 
        # as the generator and guidance model are trained at different paces 
        # even when they read from the same subset. Separate iterators avoid one
        # branch exhausting or reshuffling the other's view of the epoch.
        guidance_dataloader = torch.utils.data.DataLoader(text_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        guidance_dataloader = accelerator.prepare(guidance_dataloader)
        self.guidance_dataloader = cycle(guidance_dataloader)

        self.guidance_cls_loss_weight = args.guidance_cls_loss_weight 

        self.cls_on_clean_image = args.cls_on_clean_image 
        self.gen_cls_loss = args.gen_cls_loss 
        self.gen_cls_loss_weight = args.gen_cls_loss_weight 
        self.previous_time = None 
        self.tt_run_count = 0  # how many times TT actually ran

        # how many samples to show in W&B image grids
        self.num_visuals = getattr(args, "num_visuals", 16)  # default 16; override via --num_visuals

        if self.denoising:
            denoising_dataloader = torch.utils.data.DataLoader(
                real_dataset, num_workers=args.num_workers, 
                batch_size=args.batch_size, shuffle=True, 
                drop_last=True
            )
            denoising_dataloader = accelerator.prepare(denoising_dataloader)
            self.denoising_dataloader = cycle(denoising_dataloader)

        self.fsdp = args.fsdp 

        # --- FSDP pre-sync for from-scratch distillation (no ckpt_only_path) ---
        if self.fsdp and (args.ckpt_only_path is None):
            # In fsdp hybrid_shard case, parameters initialized on different nodes may have different values.
            # To fix this, we save the initial model on rank 0 and reload it on all ranks.
            accelerator.print("[INIT CHECKPOINTs] - distilling from scratch (FSDP sync)")

            gen_tmp   = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}", "pytorch_model.bin")
            guid_tmp  = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}", "pytorch_model_1.bin")
            tmp_dir   = os.path.dirname(gen_tmp)

            if accelerator.is_main_process:
                accelerator.print(f"Saving current model to {tmp_dir} to fix FSDP hybrid sharding parameter mismatch")
                os.makedirs(tmp_dir, exist_ok=True)
                torch.save(self.model.feedforward_model.state_dict(), gen_tmp)
                torch.save(self.model.guidance_model.state_dict(),  guid_tmp)

            accelerator.wait_for_everyone()
            accelerator.print(f"Reloading initial weights from {tmp_dir}")
            self.model.feedforward_model.load_state_dict(torch.load(gen_tmp,  map_location="cpu"), strict=True)
            self.model.guidance_model.load_state_dict( torch.load(guid_tmp, map_location="cpu"), strict=True)
            accelerator.print("FSDP sync reload done")

        # --- Wrap models with FSDP if requested (TT lives inside guidance_model) ---
        if self.fsdp:
            # Only feedforward_model and guidance_model are wrapped; no separate target_teacher module.
            self.model.feedforward_model, self.model.guidance_model = accelerator.prepare(
                self.model.feedforward_model, self.model.guidance_model
            )

        # --- Optimizers (already in your code, keep as-is) ---
        self.optimizer_generator = torch.optim.AdamW(
            [param for param in self.model.feedforward_model.parameters() if param.requires_grad], 
            lr=args.generator_lr, 
            betas=(0.9, 0.999),  # pytorch's default 
            weight_decay=0.01     # pytorch's default 
        )
        print("lr optimiser generator :", args.generator_lr)

        # ---- Guidance optimizer/scheduler (excludes TT params) ----
        tt_mod = getattr(self.model.guidance_model, "target_teacher_unet", None) or \
                 getattr(self.model.guidance_model, "target_unet", None)
        tt_param_ids = set(id(p) for p in (tt_mod.parameters() if tt_mod else []))
            
        self.guidance_params = [
            p for p in self.model.guidance_model.parameters()
            if p.requires_grad and id(p) not in tt_param_ids
        ]

        # ---- Sanity: guidance trainables (excludes TT) ----
        n_guid = sum(p.numel() for p in self.guidance_params)
        n_tt_pre = sum(p.numel() for p in (tt_mod.parameters() if tt_mod else []))
        accelerator.print(f"[init] guidance trainable params={n_guid:,}  (TT params excluded={n_tt_pre:,})")
        assert n_guid > 0, "[init] No trainable params in guidance (after excluding TT)!"

        self.optimizer_guidance = torch.optim.AdamW(
            self.guidance_params, 
            lr=args.guidance_lr, 
            betas=(0.9, 0.999),  # pytorch's default 
            weight_decay=0.01     # pytorch's default 
        )
        print("lr optimiser guidance :", args.guidance_lr)

        # Schedulers
        self.scheduler_generator = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_generator,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters
        )
        self.scheduler_guidance = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_guidance,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters 
        )

        # --- set up for no DMD / teacher flags ---
        # These booleans define which training actors are active:
        # - generator/source-teacher distillation,
        # - guidance/fake-UNet training,
        # - optional target-teacher online updates.
        self.gan_alone         = bool(getattr(args, "gan_alone", False))
        self.use_source_teacher = bool(getattr(args, "use_source_teacher", True))

        # enforce “teacher first when both scheduled”
        args.gan_alone          = self.gan_alone
        self.enable_target_teacher = (not self.gan_alone) and bool(getattr(args, "enable_target_teacher", True))
        args.use_source_teacher = self.use_source_teacher  

        # ---- Target-Teacher (TT) optimizer/scheduler ----
        self.tt_params    = []
        self.tt_optimizer = None
        self.tt_scheduler = None

        if self.enable_target_teacher and (args.targetteacher_ckpt_path is None):
            tt_mod = getattr(self.model.guidance_model, "target_teacher_unet", None) or \
                     getattr(self.model.guidance_model, "target_unet", None)
            assert tt_mod is not None, "enable_target_teacher=True but no target_teacher_unet/target_unet in guidance_model"

            self.tt_params = [p for p in tt_mod.parameters() if p.requires_grad]

            self.tt_optimizer = torch.optim.AdamW(
                self.tt_params,
                lr=args.guidance_lr,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )
            self.tt_scheduler = get_scheduler(
                "constant_with_warmup",
                optimizer=self.tt_optimizer,
                num_warmup_steps=args.warmup_step,
                num_training_steps=args.train_iters,
            )
            self.tt_max_grad_norm = self.max_grad_norm

        # --- FSDP: prepare optimizers / schedulers ---
        if self.fsdp:
            to_prep = [
                self.optimizer_generator, self.optimizer_guidance,
                self.scheduler_generator, self.scheduler_guidance,
            ]
            if self.enable_target_teacher:
                to_prep += [self.tt_optimizer, self.tt_scheduler]

            prepped = accelerator.prepare(*to_prep)

            i = 0
            self.optimizer_generator = prepped[i]; i += 1
            self.optimizer_guidance  = prepped[i]; i += 1
            self.scheduler_generator = prepped[i]; i += 1
            self.scheduler_guidance  = prepped[i]; i += 1
            if self.enable_target_teacher:
                self.tt_optimizer    = prepped[i]; i += 1
                self.tt_scheduler    = prepped[i]; i += 1

        else:
            # non-FSDP: prepare models + opts + schedulers together
            to_prep = [
                self.model.feedforward_model, self.model.guidance_model,
                self.optimizer_generator, self.optimizer_guidance,
                self.scheduler_generator, self.scheduler_guidance,
            ]
            if self.enable_target_teacher:
                to_prep += [self.tt_optimizer, self.tt_scheduler]

            prepped = accelerator.prepare(*to_prep)

            i = 0
            self.model.feedforward_model = prepped[i]; i += 1
            self.model.guidance_model    = prepped[i]; i += 1
            self.optimizer_generator     = prepped[i]; i += 1
            self.optimizer_guidance      = prepped[i]; i += 1
            self.scheduler_generator     = prepped[i]; i += 1
            self.scheduler_guidance      = prepped[i]; i += 1
            if self.enable_target_teacher:
                self.tt_optimizer        = prepped[i]; i += 1
                self.tt_scheduler        = prepped[i]; i += 1

        # --- After prepare: refresh TT + guidance param lists for clipping/logging ---
        if self.enable_target_teacher:
            tt_mod_wrapped = getattr(self.model.guidance_model, "target_teacher_unet", None) or \
                             getattr(self.model.guidance_model, "target_unet", None)
            if tt_mod_wrapped is not None:
                self.tt_params = [p for p in tt_mod_wrapped.parameters() if p.requires_grad]
            else:
                self.tt_params = []

        tt_mod = getattr(self.model.guidance_model, "target_teacher_unet", None) or \
                 getattr(self.model.guidance_model, "target_unet", None)
        tt_param_ids = set(id(p) for p in (tt_mod.parameters() if tt_mod else []))

        self.guidance_params = [
            p for p in self.model.guidance_model.parameters()
            if p.requires_grad and id(p) not in tt_param_ids
        ]

        tt_mod = getattr(self.model.guidance_model, "target_teacher_unet", None) or \
                 getattr(self.model.guidance_model, "target_unet", None)
        tt_param_ids = set(id(p) for p in (tt_mod.parameters() if tt_mod else []))

        self.guidance_params = [p for p in self.model.guidance_model.parameters()
                        if p.requires_grad and id(p) not in tt_param_ids]
                
        self.accelerator = accelerator
        self.train_iters = args.train_iters
        self.batch_size = args.batch_size
        self.resolution = args.resolution

        # track whether GAN heads have been added to the guidance optimizer
        self._gan_heads_added = False

        # Logging cadence
        self.log_iters = args.log_iters
        self.wandb_iters = args.wandb_iters
        self.log_loss = bool(getattr(args, "log_loss", False))
        self.output_path  = getattr(args, "output_path", "./outputs")
        self.grid_size = int(getattr(args, "grid_size", 4))
        # Diffusion config
        self.latent_resolution = args.latent_resolution
        self.latent_channel = 4  # SD1.5 latents are 4-channels

        # Book-keeping
        self.previous_time = None

        # ---- args normalization for pause-gen ----
        # (safe even if flags weren’t passed)
        # Normalize once here so downstream helpers can assume lists/strings are in
        # canonical form instead of handling CLI edge cases repeatedly.
        args.gen_pause_steps = _to_int_list(getattr(args, "gen_pause_steps", None), "gen_pause_steps")
        args.gen_training_name = getattr(args, "gen_training_name", None) or args.wandb_name
        seeds_val = getattr(args, "gen_seeds", None)
        args.gen_seeds = _to_int_list(seeds_val if seeds_val is not None else "0,1,2,3", "gen_seeds")

        if getattr(args, "gen_samples_per_prompt", None) is None:
            args.gen_samples_per_prompt = len(args.gen_seeds)
        _require_equal(args.gen_samples_per_prompt, len(args.gen_seeds), "gen_samples_per_prompt", "gen_seeds")

        # ---- gradient clipping config ----
        self.max_grad_norm = args.max_grad_norm

        self.no_save = args.no_save
        self.max_checkpoint = args.max_checkpoint

        self.dfake_gen_update_ratio = args.dfake_gen_update_ratio
        self.dteacher_update_ratio  = args.dteacher_update_ratio

        if args.checkpoint_path is not None:
            self.load(args.checkpoint_path)
        
        print(f"[cfg] gen_outputs_root={args.gen_outputs_root}")
        print(f"[cfg] gen_training_name={args.gen_training_name}")
        print(f"[cfg] output_path={args.output_path}")
   
    def fsdp_state_dict(self, model):
        fsdp_fullstate_save_policy = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
        ):
            checkpoint = model.state_dict()

        return checkpoint 

    def load(self, checkpoint_path):
        # this is used for non-fsdp models.
        self.step = int(checkpoint_path.replace("/", "").split("_")[-1])
        print(self.accelerator.load_state(checkpoint_path, strict=False))
        self.accelerator.print(f"Loaded checkpoint from {checkpoint_path}")

    @torch.no_grad()
    def _paused_generate(self):
        """
        Pause-time generation (SD-1.5):
        - 25 prompts per bucket (object/live), 4 seeds per prompt
        - single UNet eval at t = num_train_timesteps - 1
        - eps→x0 reconstruction; decode to [-1,1]
        - saves to: {gen_outputs_root}/{class}_{instance}/{gen_training_name}/gen_{step:06d}/p_XXX/seed_YYY.png
        - logs W&B table with 2×2 grid per prompt
        """
        from pathlib import Path
        import torch
        import os 
        
        # Ensure model is in eval mode and synced across processes
        self.accelerator.wait_for_everyone()
        self.model.eval()

        acc = self.accelerator
        if not acc.is_main_process:
            return

        args = self.args
        step = self.step
        dev  = acc.device

        # Data
        pairs = read_instances(args.gen_prompts_file)
        obj_prompts, live_prompts = extract_prompt_blocks(args.gen_prompts_file)

        acc.print(f"[prompt list] obj: {obj_prompts} and [prompt list] obj: {live_prompts}")
        acc.print(f"[pause@{step}] pairs={len(pairs)} : {pairs} | obj_prompts={len(obj_prompts)} | live_prompts={len(live_prompts)}")
        if not obj_prompts and not live_prompts:
            acc.print(f"[pause@{step}] No prompts parsed; skipping.")
            return

        target_ids = _parse_ids(getattr(args, "gen_include_labels", None)) or [0]

        # IO
        # Pause-time generation is intentionally main-process only because it writes
        # files, logs W&B media, and optionally triggers evaluation side effects.
        run_name = args.gen_training_name or args.wandb_name
        out_root = Path(args.gen_outputs_root)
        out_root.mkdir(parents=True, exist_ok=True)

        # Diffusion config
        T_last  = int(self.model.num_train_timesteps - 1)
        t_batch = torch.tensor([T_last], device=dev, dtype=torch.long)
        C, H, W = 4, self.latent_resolution, self.latent_resolution

        def x0_from_eps(x_t: torch.Tensor, eps: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
            if hasattr(self.model, "alphas_cumprod") and self.model.alphas_cumprod is not None:
                a_bar = self.model.alphas_cumprod[t_tensor].view(-1,1,1,1).float()
            else:
                a_bar = (torch.ones_like(t_tensor, dtype=torch.float32)*0.0047).view(-1,1,1,1)
            b_bar = 1.0 - a_bar
            return (x_t - b_bar.sqrt()*eps) / a_bar.sqrt()

        # W&B table
        TABLE_KEY = "Paused_generations"
        columns   = ["step","instance","class","prompt_id","prompt","images","debug"]
        rows_data = []
        table = wandb.Table(columns=["step", "instance", "class", "prompt_id", "prompt", "images"])

        for inst_id in target_ids:
            if inst_id >= len(pairs):
                acc.print(f"[pause@{step}] WARNING: inst_id {inst_id} >= {len(pairs)}; skipping.")
                continue

            inst_name, class_token = pairs[inst_id]
            # The prompt template file keeps separate prompt banks for living vs.
            # object instances, so we pick the bank dynamically per instance id.
            is_live = (inst_name, class_token) in LIVE_INSTANCES
            prompts_list = live_prompts if is_live else obj_prompts
            print(f"inst_name : {inst_name} and class token {class_token} and is live {is_live}")

            class_inst = f"{class_token}_{inst_name}"
            base_dir   = out_root / class_inst / run_name / f"gen_{step:06d}"
            acc.print(f"[pause@{step}] {class_inst}: {len(prompts_list)} prompts → {base_dir}")
       
            for p_idx, tmpl in enumerate(prompts_list):
                prompt_text = expand_prompt(tmpl, args.rare_token, class_token)
                acc.print(f"[cleaned prompt list] {prompt_text} ")
                p_dir = base_dir / f"p_{p_idx:03d}"; p_dir.mkdir(parents=True, exist_ok=True)
                self.accelerator.print(f"[pause] P_dir={p_dir}  ")

                # text encode (CLIPTextModel)
                
                tok = self.tokenizer(
                    [prompt_text],
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True
                ).input_ids.to(dev)
                text_emb = self.model.text_encoder(tok)[0]  # [1, L, 768]
                
                imgs_for_grid = []
                debug_pil = None  # first single image for the row

                for i, s in enumerate(args.gen_seeds):
                    g = torch.Generator(device=dev); g.manual_seed(int(s))
                    noise = torch.randn(1, C, H, W, generator=g, device=dev, dtype=torch.float32)
                    # This preview path uses the current feedforward model as a
                    # single-step epsilon predictor, then reconstructs x0 directly.
                    eps = self.model.feedforward_model(noise, t_batch.expand(1), text_emb).sample
                    x0  = x0_from_eps(noise, eps, t_batch.expand(1))

                    # decode to [-1,1] → CPU
                    if hasattr(self.model, "decode_image"):
                        img = self.model.decode_image(x0).clamp(-1, 1)
                    else:
                        dec = self.vae.decode(x0 * (1.0/0.18215)).sample.float()
                        img = dec.clamp(-1, 1)
                    img_cpu = img.detach().cpu()

                    # save single
                    single_arr = prepare_images_for_saving(img_cpu, resolution=self.resolution, grid_size=1)
                    single_pil = _to_pil(single_arr).convert("RGB")
                    single_path = p_dir / f"seed_{int(s):03d}.png"
                    single_pil.save(single_path)

                    if i == 0:
                        debug_pil = single_pil

                    imgs_for_grid.append(img_cpu[i:i+1])

                """ FOR FEW DENOISING STEPS VERSION
                imgs_for_grid = []
                debug_pil = None  # first single image for the row

                # Batched multi-step generation for this prompt
                seeds_list = [int(s) for s in args.gen_seeds]
                B = len(seeds_list)

                imgs = self.model.sample_ddim_sd15(
                    prompt_texts=[prompt_text] * B,
                    num_inference_steps=int(args.num_denoising_step),        
                    guidance_scale=float(getattr(args, "gen_guidance_scale", 7.5)),
                    height=self.resolution,
                    width=self.resolution,
                    seeds=seeds_list,
                )

                imgs_cpu = imgs.detach().cpu()
                for i, s in enumerate(seeds_list):
                    single_arr = prepare_images_for_saving(imgs_cpu[i:i+1], resolution=self.resolution, grid_size=1)
                    single_pil = _to_pil(single_arr).convert("RGB")
                    single_path = p_dir / f"seed_{int(s):03d}.png"
                    single_pil.save(single_path)

                    if i == 0:
                        debug_pil = single_pil

                    imgs_for_grid.append(imgs_cpu[i:i+1])
                """
                # Optionally, you can log other intermediate results as well:
                generated_img_grid = prepare_images_for_saving(img_cpu, resolution=self.resolution, grid_size=4)
                wandb.log({
                    f"generated_images_step_{self.step}": wandb.Image(generated_img_grid),
                })


                # ---- add one row to table (grid + debug) ----
                if imgs_for_grid:
                    grid = torch.cat(imgs_for_grid, dim=0)  # CPU
                    grid_arr = prepare_images_for_saving(grid, resolution=self.resolution, grid_size=2)
                    grid_pil = _to_pil(grid_arr).convert("RGB")
                    grid_path = p_dir / "grid.png"
                    grid_pil.save(grid_path)

                    # path for debug image (first seed)
                    if debug_pil is not None:
                        debug_path = p_dir / f"seed_{int(args.gen_seeds[0]):03d}.png"
                        # if somehow missing, fallback to any seed or to grid
                        if not os.path.exists(str(debug_path)):
                            seeds = sorted(p_dir.glob("seed_*.png"))
                            debug_path = seeds[0] if seeds else grid_path
                    else:
                        debug_path = grid_path
                    
                    # only main process appends & counts 
                    rows_data.append([
                        int(step),
                        str(inst_name),
                        str(class_token),
                        int(p_idx),
                        str(prompt_text),
                        wandb.Image(str(grid_path),  caption=f"{inst_name}/{class_token} • grid p{p_idx:03d}"),
                        wandb.Image(str(debug_path), caption=f"{inst_name}/{class_token} • seed {args.gen_seeds[0]}"),
                    ])
                    wandb.log({
                        f"Paused_generations_step_{self.step}": wandb.Image(str(grid_path)),
                    })
                    

            # --- EVALUATION for THIS instance only ---
            if self.accelerator.is_main_process:   # guard against multi-process duplication
                src_label = (args.eval_inst2label_map.get(inst_name, int(inst_id))
                            if getattr(args, "eval_inst2label_map", None) else int(inst_id))

                # The evaluation pipeline is file-based on purpose: training writes a
                # manifest for each pause step, and metric code consumes it later
                # without depending on in-memory trainer state.
                eval_rows = []
                for p_idx, tmpl in enumerate(prompts_list):
                    prompt_text = expand_prompt(tmpl, args.rare_token, class_token)
                    p_dir = base_dir / f"p_{p_idx:03d}"
                    eval_rows.append({
                        "class_instance": class_inst,
                        "instance": inst_name,
                        "pdir": f"p_{p_idx:03d}",
                        "prompt": prompt_text,
                        "gen_dir": str(p_dir),
                        "src_dir": "",
                        "src_label": src_label,
                        "run_dir": str(out_root / class_inst / run_name),
                        "step": int(step),
                    })

                if eval_rows:
                    import pandas as pd
                    df = pd.DataFrame(eval_rows)
                    eval_dir = out_root / class_inst / run_name / "eval"
                    eval_dir.mkdir(parents=True, exist_ok=True)
                    out_csv = eval_dir / f"eval_manifest_step{step:06d}.csv"
                    df.to_csv(out_csv, index=False)
                    self.accelerator.print(f"[eval] wrote manifest → {out_csv}")

                    if args.eval_enable:
                        from evaluation.run_metrics import run_clip_dino_on_manifest
                        run_clip_dino_on_manifest(
                            manifest_csv=str(out_csv),
                            lmdb_path=getattr(args, "real_image_path", None),
                            clip_model=getattr(args, "eval_clip_model", "ViT-B/32"),
                            max_src=getattr(args, "eval_max_src", 4),
                            max_gen_per_group=getattr(args, "eval_max_gen_per_group", 4),
                            wandb_prefix=f"eval/step"
                        )
                
                    #wandb.log({eval_rows[0]["class_instance"] + f"/eval_manifest_step{step:06d}": wandb.Table(dataframe=df)}, step=self.step)
                    table = wandb.Table(columns=columns, data=rows_data)
                    wandb.log({TABLE_KEY: table})
        #--------------end of csv creation file for related run----------------------------------------------------------------------------
        # needs one per run as we redirect to the generated images (gen dir) for each run to a different folder 
        #--------------------------------

        # If this step is the last pause step, produce the run-wide final summary now
        if getattr(args, "eval_finalize_on_last_pause", False):
            try:
                last_pause = max(getattr(args, "gen_pause_steps", []) or [step])
            except Exception:
                last_pause = step
            if int(step) == int(last_pause) and self.accelerator.is_main_process:
                from dmd2_eval import summarize_instance_eval
                # summarize EACH instance you touched in this run
                touched_instances = []
                for inst_id in target_ids:
                    if inst_id >= len(pairs): continue
                    inst_name, class_token = pairs[inst_id]
                    class_inst = f"{class_token}_{inst_name}"
                    run_dir = out_root / class_inst / run_name
                    res = summarize_instance_eval(run_dir, policy=args.eval_finalize_policy,
                                                wandb_prefix="eval/final")
                    touched_instances.append((inst_name, res))
                self.accelerator.print(f"[eval/final] summarized {len(touched_instances)} instances at step {step}")
       
        if self.accelerator.is_main_process:
            if len(rows_data) > 0:
                table = wandb.Table(columns=columns, data=rows_data)
                print(f"[pause@{step}] logging table key='{TABLE_KEY}' with {len(rows_data)} rows")
                wandb.log({TABLE_KEY: table})
            else:
                print(f"[pause@{step}] no rows_data; not logging table")

        torch.cuda.empty_cache()




    def save(self):
        # NOTE: we save the checkpoints to two places 
        # 1. output_path: save the latest one, this is assumed to be a permanent storage
        # 2. cache_dir: save all checkpoints, this is assumed to be a temporary storage
        # training states 
        # If FSDP is used, we only save the model parameter as I haven't figured out how to save the optimizer state without oom yet, help is appreciated.
        # Otherwise, we use the default accelerate save_state function 
        
        os.makedirs(self.output_path, exist_ok=True)

        # If you add the CLI flag in Patch 3, honor it here:
        if getattr(self.args, "skip_accelerate_state", False):
            self.accelerator.print(f"[ckpt] Skipping accelerate.save_state at step {self.step} per flag.")
            self._save_model_only_fallback(self.output_path)
            return

        # Full state (optimizer, schedulers, etc.)
        self.accelerator.save_state(self.output_path)  # may raise if disk/quota issues

        # If you already save EMA / model separately, keep that too.
        if hasattr(self, "ema") and self.ema is not None:
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())
            ckpt_file = os.path.join(self.output_path, "model_ema.pt")
            self.accelerator.save(self.model.state_dict(), ckpt_file)
            self.ema.restore(self.model.parameters())
        else:
            ckpt_file = os.path.join(self.output_path, "model.pt")
            self.accelerator.save(self.model.state_dict(), ckpt_file)
        print("done saving")
        torch.cuda.empty_cache()

    def _save_model_only_fallback(self, output_path: str = None):
        """Minimal checkpoint that avoids optimizer state; resilient on tight filesystems."""
        output_path = output_path or os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}_modelonly")
        os.makedirs(output_path, exist_ok=True)
        # Save EMA if you use it; otherwise raw
        if hasattr(self, "ema") and self.ema is not None:
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())
            ckpt_file = os.path.join(output_path, "model_ema.pt")
            self.accelerator.save(self.model.state_dict(), ckpt_file)
            self.ema.restore(self.model.parameters())
        else:
            ckpt_file = os.path.join(output_path, "model.pt")
            self.accelerator.save(self.model.state_dict(), ckpt_file)
        self.accelerator.print(f"[ckpt] Wrote model-only checkpoint to: {ckpt_file}")


    def train_one_step(self):
        self.model.train()

        accelerator = self.accelerator    
        # One outer iteration may perform up to three optimizations on the same
        # sampled noise batch: generator on its own cadence, guidance every step,
        # and target-teacher on its own cadence.
        DO_TEACHER   = self.enable_target_teacher and (self.step % self.dteacher_update_ratio == 0)
        DO_GENERATOR = (self.step % self.dfake_gen_update_ratio == 0)
        generator_grad_norm = None
        guidance_grad_norm  = None
        last_tt_loss = None
        tt_log = None

        # ---------- add near top of train_one_step (once) ----------
        def _debug_stats(tag, x):
            x_ = x.detach().float()
            print(f"[{tag}] shape={tuple(x_.shape)} "
                f"min={x_.min().item():.3f} max={x_.max().item():.3f} mean={x_.mean().item():.3f}",flush=True)
        def _finite(x):
            import torch
            if x is None: return True
            if isinstance(x, (float, int)): 
                return math.isfinite(float(x))
            if isinstance(x, torch.Tensor):
                return torch.isfinite(x).all().item()
            return True

        def _check_tensors(d, tag, max_items=6):
            import torch
            if d is None: return True
            ok = True
            report = []
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    fv = torch.isfinite(v)
                    if not fv.all():
                        ok = False
                        try:
                            report.append(f"{k}: min={v.nan_to_num().min().item():.3e} "
                                        f"max={v.nan_to_num().max().item():.3e} "
                                        f"has_nan={(~torch.isfinite(v)).any().item()}")
                        except Exception:
                            report.append(f"{k}: non-finite")
                        if len(report) >= max_items: break
            if not ok:
                print(f"[NAN] {tag} non-finite -> " + "; ".join(report), flush=True)
            return ok

        def _check_params(module, tag, max_items=6):
            import torch
            bad = []
            for n, p in module.named_parameters(recurse=True):
                if p is not None and p.data is not None and p.requires_grad:
                    if not torch.isfinite(p.data).all():
                        bad.append(n)
                        if len(bad) >= max_items: break
            if bad:
                print(f"[NAN PARAM] {tag}: {len(bad)} params non-finite, e.g. {bad}", flush=True)
                return False
            return True

        def _check_grads(module, tag, max_items=6):
            import torch
            bad = []
            for n, p in module.named_parameters(recurse=True):
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        bad.append(n)
                        if len(bad) >= max_items: break
            if bad:
                print(f"[NAN GRAD] {tag}: {len(bad)} grads non-finite, e.g. {bad}", flush=True)
                return False
            return True
        # -----------------------------------------------------------


        # 4 channel for SD-VAE, please adapt for other autoencoders 
        noise = torch.randn(self.batch_size, self.latent_channel, self.latent_resolution, self.latent_resolution, device=accelerator.device)
        visual = self.step % self.wandb_iters == 0
        COMPUTE_GENERATOR_GRADIENT = (self.step % self.dfake_gen_update_ratio == 0)
        
        # pick which stream to draw from this step
        if COMPUTE_GENERATOR_GRADIENT:
            batch = next(self.dataloader)
        else:
            batch = next(self.guidance_dataloader)

        student_ids = batch["text_input_ids_one"] 
        # Accept [B,L], [B,1,L], or [B,1,1,L] (legacy LMDB)
        if student_ids.dim() == 4 and student_ids.size(1) == 1 and student_ids.size(2) == 1:
            student_ids = student_ids.squeeze(2).squeeze(1)
        
        if student_ids.dim() == 3 and student_ids.size(1) == 1:
            student_ids = student_ids.squeeze(1)
        student_ids = student_ids.to(accelerator.device)

        raw_prompts = batch.get("raw_prompt", [""] * student_ids.size(0))

        # Build both teacher prompt variants explicitly:
        # Student and target teacher keep the rare token because they model the
        # personalized concept, while the source teacher sees the cleaned prompt.
        prompts_with_rare = raw_prompts[:]  # as-is, contains rare token
        prompts_no_rare   = [make_teacher_prompt(p, self.args.rare_token) for p in raw_prompts]

        # Use dataset-provided teacher ids if available; otherwise tokenize both variants
        t_ids_no_rare = batch.get("teacher_input_ids_one", None)
        if t_ids_no_rare is not None:
            # dataset is assumed to have NO-RARE teacher ids in 'teacher_input_ids_one'
            if t_ids_no_rare.dim() == 3 and t_ids_no_rare.size(1) == 1:
                t_ids_no_rare = t_ids_no_rare.squeeze(1)
            t_ids_no_rare = t_ids_no_rare.to(accelerator.device)
        else:
            # fallback: tokenize NO-RARE
            t_ids_no_rare = self.tokenizer(
                prompts_no_rare,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).input_ids.to(accelerator.device)

        # ALWAYS tokenize WITH-RARE for target-teacher/guidance/student alignment
        t_ids_with_rare = self.tokenizer(
            prompts_with_rare,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).input_ids.to(accelerator.device)

        # Canonical PromptPack (names are explicit; keep legacy keys for back-compat)
        text_embedding = {
            # student (with rare)
            "student_input_ids_one": student_ids,
            # target teacher (with rare)
            "target_teacher_input_ids_one": t_ids_with_rare,
            # source teacher / real_unet (no rare)
            "source_teacher_input_ids_one": t_ids_no_rare,

            # legacy: many models expect 'teacher_input_ids_one'. Route it to WITH-RARE
            # so that a target teacher that still reads this key gets the correct policy.
            # Source teacher must be updated to read 'source_teacher_input_ids_one'.
            "teacher_input_ids_one": t_ids_with_rare,

            "raw_prompt": raw_prompts,
        }

        # Uncond unchanged
        uncond_embedding = self.uncond_embedding.repeat(student_ids.size(0), 1, 1)

      
        if self.denoising:
            denoising_dict = next(self.denoising_dataloader)
        else:
            denoising_dict = None

        # need real data if classifier is on OR multi-head GAN is enabled
        # We always need real images now (fake UNet ε-loss uses target images)
        real_train_dict = next(self.real_dataloader)

        
        # generate images and optionaly compute the generator gradient
        # The first forward produces the fake batch and any generator-side losses.
        # Its logged guidance payload is then reused by the guidance branch so both
        # updates refer to the same generated samples.
        generator_loss_dict, generator_log_dict = self.model(
            noise, text_embedding, uncond_embedding, 
            visual=visual, denoising_dict=denoising_dict,
            compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
            real_train_dict=real_train_dict,
            generator_turn=True,
            guidance_turn=False
        )
        print("[PROMPT GENERATOR t,s:]", text_embedding.items(), text_embedding.items())
        if COMPUTE_GENERATOR_GRADIENT and self.use_source_teacher :
            if "dmtrain_pred_real_image" not in generator_log_dict or \
            "dmtrain_pred_fake_image" not in generator_log_dict:
                # one-shot noisy but invaluable
                self.accelerator.print(
                    "[DBG] DM keys missing from generator_log_dict. "
                    f"Available keys: {sorted(list(generator_log_dict.keys()))}"
                )

        # After generator forward (before any backward)
        _check_tensors(generator_log_dict, "gen/log")
        gen_loss_ok = True
        for key in ("loss_dm", "gen_cls_loss", "gan_bce_g"):
            if key in generator_loss_dict:
                if not _finite(generator_loss_dict[key]):
                    print(f"[NAN] gen loss component {key} is non-finite", flush=True)
                    gen_loss_ok = False

        # first update the generator if the current step is a multiple of dfake_gen_update_ratio
        generator_loss = 0.0 

        if COMPUTE_GENERATOR_GRADIENT and gen_loss_ok:
            if not self.args.gan_alone:
                generator_loss += generator_loss_dict["loss_dm"] * self.args.dm_loss_weight

            if self.cls_on_clean_image and ("gen_cls_loss" in generator_loss_dict):
                generator_loss += generator_loss_dict["gen_cls_loss"] * self.gen_cls_loss_weight

            #  GAN G-loss BEFORE backward 
            if self.args.multihead_gan and ("gan_bce_g" in generator_loss_dict):
                generator_loss += generator_loss_dict["gan_bce_g"]

            if not _finite(generator_loss):
                print("[NAN] generator_loss is non-finite — skipping gen step", flush=True)

            
            else:
                with torch.no_grad():
                    print(f"[DBG] generator_loss: {generator_loss.item() if torch.isfinite(generator_loss) else 'NaN/Inf'}", flush=True)
                    print(f"[DBG] requires_grad: {getattr(generator_loss, 'requires_grad', None)}", flush=True)

                # If generator_loss is an aggregate of components, check them too:
                try:
                    comps = {
                        "dm": locals().get("dm_loss", None),
                        "gan": locals().get("gan_loss", None),
                        "guidance": locals().get("guidance_loss", None),
                        "lpips": locals().get("lpips_loss", None),
                    }
                    for k,v in list(comps.items()):
                        if v is not None:
                            v_detached = v.detach()
                            print(f"[DBG] {k}_loss:", float(v_detached) if torch.isfinite(v_detached) else "NaN/Inf", flush=True)
                except Exception as _e:
                    pass

                assert getattr(generator_loss, "requires_grad", True), \
                    "[ERR] generator_loss has no grad — earlier detach?"

                if not torch.isfinite(generator_loss):
                    print("[SKIP] Non-finite generator_loss — skipping backward", flush=True)
                    return
                self.accelerator.backward(generator_loss)
                _check_grads(self.model.feedforward_model, "gen/grad")
                #generator_grad_norm = accelerator.clip_grad_norm_(self.model.feedforward_model.parameters(), self.max_grad_norm)
                generator_grad_norm = self.accelerator.clip_grad_norm_( self.model.feedforward_model.parameters(), self.max_grad_norm)

            self.optimizer_generator.step()
            # if we also compute gan loss, the classifier may also receive gradient 
            # zero out guidance model's gradient avoids undesired gradient accumulation
            _check_params(self.model.feedforward_model, "gen/params")
            self.optimizer_generator.zero_grad() 
            self.optimizer_guidance.zero_grad()
            self.scheduler_generator.step()

        # update the guidance model (dfake and classifier)
       
        guidance_loss_dict, guidance_log_dict = self.model(
            noise, text_embedding, uncond_embedding, 
            visual=visual, denoising_dict=denoising_dict,
            real_train_dict=real_train_dict,
            compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
            generator_turn=False,
            guidance_turn=True,
            guidance_data_dict=generator_log_dict['guidance_data_dict']
        )
        print("[PROMPT GUIDANCE t,s:]", text_embedding.items(), text_embedding.items())
        _check_tensors(guidance_log_dict, "guid/log")
        guid_loss_ok = True
        for key in ("loss_fake_mean", "guidance_cls_loss", "gan_bce_d"):
            if key in guidance_loss_dict and (guidance_loss_dict[key] is not None):
                if not _finite(guidance_loss_dict[key]):
                    print(f"[NAN] guidance loss component {key} is non-finite", flush=True)
                    guid_loss_ok = False
       
        if guid_loss_ok:
            guidance_loss = 0

            # ---- Backprop & step for guidance (fake UNet ε, cls, GAN-D) ----
            guidance_loss += guidance_loss_dict["loss_fake_mean"]

            # optional realism classifier on real/fake
            if self.cls_on_clean_image and ("guidance_cls_loss" in guidance_loss_dict):
                guidance_loss = guidance_loss + self.guidance_cls_loss_weight * guidance_loss_dict["guidance_cls_loss"]

            # optional GAN discriminator loss
            if self.args.multihead_gan and ("gan_bce_d" in guidance_loss_dict):
                guidance_loss += guidance_loss_dict["gan_bce_d"]


            if not _finite(guidance_loss):
                print("[NAN] guidance_loss is non-finite — skipping guid step", flush=True)
            else:
                self.accelerator.backward(guidance_loss)
        
                _check_grads(self.model.guidance_model, "guid/grad")
                
                guidance_grad_norm = self.accelerator.clip_grad_norm_(self.guidance_params, self.max_grad_norm)

                print("guidance_grad_norm", guidance_grad_norm)  
                self.optimizer_guidance.step()
                _check_params(self.model.guidance_model, "guid/params")
                self.optimizer_guidance.zero_grad()
        self.scheduler_guidance.step()


        # combine the two dictionaries 
        loss_dict = {**generator_loss_dict, **guidance_loss_dict}
        log_dict = {**generator_log_dict, **guidance_log_dict}

        generated_image_mean = log_dict["guidance_data_dict"]['image'].mean()
        generated_image_std = log_dict["guidance_data_dict"]['image'].std()

        generated_image_mean = accelerator.gather(generated_image_mean).mean()
        generated_image_std = accelerator.gather(generated_image_std).mean()

        if COMPUTE_GENERATOR_GRADIENT:
            if not self.args.gan_alone and self.args.use_source_teacher:
                dmtrain_pred_real_image_mean = log_dict['dmtrain_pred_real_image'].mean()
                dmtrain_pred_real_image_std = log_dict['dmtrain_pred_real_image'].std()

                dmtrain_pred_real_image_std = log_dict['dmtrain_pred_real_image'].std()
                dmtrain_pred_real_image_std = accelerator.gather(dmtrain_pred_real_image_std).mean()

                dmtrain_pred_fake_image_mean = log_dict['dmtrain_pred_fake_image'].mean()
                dmtrain_pred_fake_image_std = log_dict['dmtrain_pred_fake_image'].std()

                dmtrain_pred_fake_image_mean = accelerator.gather(dmtrain_pred_fake_image_mean).mean()
                dmtrain_pred_fake_image_std = accelerator.gather(dmtrain_pred_fake_image_std).mean()

        # ===== Target-Teacher step (must run BEFORE generator when both scheduled) =====
        # --- Initialize TT logging placeholders (once per iter) ---
        last_tt_loss = None
        last_tt_pred_x0 = None

        if DO_TEACHER:
            # The target teacher uses real images plus text conditioning from the
            # current real batch, but it is stepped with its own optimizer/scheduler.
            # --- Build teacher-side conditioning for SD-1.5 ---
            # SD-1.5: single stream; make sure shape is [B, L]
            ids = real_train_dict["text_input_ids_one"]
            print("[PROMPT INPUT IN TT :]", type(ids), getattr(ids, "shape", None), getattr(ids, "dtype", None))
            if ids.dim() == 3 and ids.size(1) == 1:
                ids = ids.squeeze(1)
            ids = ids.to(self.accelerator.device)
            real_text_embedding = self.model.text_encoder(ids)[0] 
            print("[PROMPT INPUT IN TT real_t_emb :]", real_text_embedding)   
            real_uncond_embedding = None
            real_added_conds = None

            # --- Unified TT loss for SD-1.5 ---
            tt_loss_dict, tt_log = self.model(
                noise, text_embedding, uncond_embedding,
                teacher_turn=True,
                teacher_data_dict={
                    "real_image": real_train_dict["images"],        # RGB in [0,1]
                    "text_embedding": real_text_embedding,          # teacher text (SD1.5)
                    "uncond_embedding": real_uncond_embedding,      # None for SD1.5 unless you add it
                    "unet_added_conditions": real_added_conds,      # None for SD1.5
                }
            )

            loss_tt = tt_loss_dict["loss_target_teacher_mean"]
            self.accelerator.backward(loss_tt)

            if self.tt_optimizer is not None:
                # ---- IMPORTANT: step the TEACHER optimizer (NOT the guidance optimizer) ----
                _ = self.accelerator.clip_grad_norm_(self.tt_params, self.max_grad_norm)
                self.tt_optimizer.step()
                self.tt_optimizer.zero_grad(set_to_none=True)
                self.tt_scheduler.step()
            else:
                print("[ALERT] tt_optimizer is None; skipping TT step")

            # Count this TT update and decide whether to preview
            if not hasattr(self, "tt_run_count"):
                self.tt_run_count = 0
            self.tt_run_count += 1
            do_tt_preview = (self.tt_run_count % 200 == 0)

            # Log snapshot for visuals
            last_tt_loss = loss_tt.detach()
            last_tt_pred_x0 = tt_log.get("tt_pred_x0", None)

            # TEACHER IMAGE VISU - Iterative Denoising Process
            if do_tt_preview:
                with torch.no_grad():
                    # -- resolve teacher UNet --
                    tt_unet = getattr(self.model.guidance_model, "target_teacher_unet", None) \
                            or getattr(self.model.guidance_model, "target_unet", None)
                    if tt_unet is None:
                        print("[ALERT] TT Unet not found for generation")
                        pass
                    else:
                        tt_unet.eval()

                        acc = self.accelerator
                        dev = acc.device

                        # --- conditioning (already built earlier in DO_TEACHER) ---
                        real_text_embedding = real_text_embedding
                        real_added_conds  = None

                        # --- shapes ---
                        # Use your existing latent shape variables (same as elsewhere in your code).
                        B = real_train_dict["text_input_ids_one"].shape[0]                       # preview same batch size as current teacher texts
                        C = self.latent_channel                       # e.g., 4 for SD-v1.5
                        H = self.latent_resolution                    # latent H (e.g., 64 for 512px)
                        W = self.latent_resolution                    # latent W (e.g., 64 for 512px)

                        # --- schedule (DDIM 50 steps by default) ---
                        T_total = int(self.model.num_train_timesteps)     # e.g., 1000
                        K       = getattr(self.args, "tt_preview_steps", 200)
                        # indices: t_0 > t_1 > ... > t_{K-1}=0
                        t_idx   = torch.linspace(T_total - 1, 0, K, device=dev).long()
                        # ensure strictly descending & unique
                        t_idx   = torch.unique_consecutive(t_idx, dim=0)

                        # precompute ᾱ, sqrt(ᾱ), sqrt(1-ᾱ) as tensors for fast indexing
                        a_bar = self.model.alphas_cumprod.to(dev).float()                 # [T]
                        sqrt_a_bar = torch.sqrt(a_bar)                                    # [T]
                        sqrt_one_minus = torch.sqrt(1.0 - a_bar)                          # [T]

                        # --- start from pure noise at t_0 ---
                        x = torch.randn(B, C, H, W, device=dev, dtype=torch.float32)

                        # --- iterative DDIM (η=0): x_{t_next} = sqrt(ᾱ_{t_next}) x0 + sqrt(1-ᾱ_{t_next}) ε_pred ---
                        for k in range(len(t_idx) - 1):
                            t      = t_idx[k].view(-1)                    # scalar long
                            t_next = t_idx[k + 1].view(-1)

                            # current ᾱ and factors
                            sa_t  = sqrt_a_bar.index_select(0, t).view(1,1,1,1)           # sqrt(ᾱ_t)
                            so_t  = sqrt_one_minus.index_select(0, t).view(1,1,1,1)       # sqrt(1-ᾱ_t)
                            sa_tp = sqrt_a_bar.index_select(0, t_next).view(1,1,1,1)      # sqrt(ᾱ_{t_next})
                            so_tp = sqrt_one_minus.index_select(0, t_next).view(1,1,1,1)  # sqrt(1-ᾱ_{t_next})

                            # UNet predicts ε at timestep t
                            out = tt_unet(x, t.expand(B), encoder_hidden_states=real_text_embedding)

                            eps = out.sample if hasattr(out, "sample") else out

                            # derive x0 from (x_t, ε, ᾱ_t)
                            x0 = (x - so_t * eps) / sa_t

                            # deterministic DDIM step to t_next
                            x  = sa_tp * x0 + so_tp * eps

                        # final decode at t=0 (last element of t_idx is 0 by construction)
                        imgs = self.model.decode_image(x).clamp(-1, 1)  # [-1,1] range

                        # make a compact grid and log
                        nrow = int(math.sqrt(B)) if int(math.sqrt(B))**2 == B else min(B, 8)
                        grid = make_grid(imgs, nrow=nrow, normalize=True, value_range=(-1, 1))
                        if acc.is_main_process and wandb.run is not None:
                            wandb.log({"tt/iterative_denoise_preview": wandb.Image(grid)}, step=self.step)
        
        if self.denoising:
            original_image_mean = denoising_dict["images"].mean()
            original_image_std = denoising_dict["images"].std()

            original_image_mean = accelerator.gather(original_image_mean).mean()
            original_image_std = accelerator.gather(original_image_std).mean()


        # --- compute realism means every step (if available) ---
        preds_fake = log_dict.get("pred_realism_on_fake")
        preds_real = log_dict.get("pred_realism_on_real")
        probs_fake = log_dict.get("pred_prob_on_fake") 
        probs_real = log_dict.get("pred_prob_on_real")

        def _mean_safe(x):
            import numpy as np, torch
            if x is None: return None
            if isinstance(x, torch.Tensor):
                try:
                    x = self.accelerator.gather(x)
                except Exception:
                    pass
                x = x.detach().float().view(-1).cpu().numpy()
            else:
                x = np.array(x).reshape(-1)
            if x.size == 0 or not np.isfinite(x).all():
                return None
            return float(np.mean(x))

        fake_mean = _mean_safe(preds_fake)
        real_mean = _mean_safe(preds_real)
        fake_prob_mean = _mean_safe(probs_fake)  # probability mean on fake
        real_prob_mean = _mean_safe(probs_real)  # probability mean on real

        if self.accelerator.is_main_process and self.log_loss and (not visual):
            wandb_loss_dict = {
                "loss_fake_mean": guidance_loss_dict['loss_fake_mean'].item(),
                "generated_image_mean": generated_image_mean.item(),
                "generated_image_std": generated_image_std.item(),
                "batch_size": len(noise)
            }
            if guidance_grad_norm is not None:
                 wandb_loss_dict["guidance_grad_norm"] = guidance_grad_norm.item()
            
            #if fake_mean is not None:
            #    wandb_loss_dict["realism/fake_mean"] = fake_mean
            #if real_mean is not None:
            #    wandb_loss_dict["realism/real_mean"] = real_mean
            if fake_prob_mean is not None and real_prob_mean is not None:
                wandb_loss_dict["realism/fake_prob_mean"] = fake_prob_mean
                wandb_loss_dict["realism/real_prob_mean"] = real_prob_mean
            
            if self.args.multihead_gan:
                if "gan_bce_g" in loss_dict:
                    wandb_loss_dict["gan_bce_g"] = loss_dict["gan_bce_g"].item()
                if "gan_bce_d" in loss_dict:
                    wandb_loss_dict["gan_bce_d"] = loss_dict["gan_bce_d"].item()

            if COMPUTE_GENERATOR_GRADIENT and (not self.args.gan_alone) and self.args.use_source_teacher:
                wandb_loss_dict.update(
                    {
                        "dmtrain_pred_real_image_mean": dmtrain_pred_real_image_mean.item(),
                        "dmtrain_pred_real_image_std": dmtrain_pred_real_image_std.item(),
                        "dmtrain_pred_fake_image_mean": dmtrain_pred_fake_image_mean.item(),
                        "dmtrain_pred_fake_image_std": dmtrain_pred_fake_image_std.item()
                    }
                )

            if self.denoising:
                wandb_loss_dict.update({
                    "original_image_mean": original_image_mean.item(),
                    "original_image_std": original_image_std.item()
                })

            if COMPUTE_GENERATOR_GRADIENT and (generator_grad_norm is not None):
                if generator_grad_norm is not None:
                    wandb_loss_dict["generator_grad_norm"] = generator_grad_norm.item()

                if not self.args.gan_alone:
                    wandb_loss_dict["loss_dm"] = loss_dict['loss_dm'].item()
                    if "dmtrain_gradient_norm" in log_dict:
                        wandb_loss_dict["dmtrain_gradient_norm"] = log_dict["dmtrain_gradient_norm"]
                if self.gen_cls_loss:
                    wandb_loss_dict.update({
                        "gen_cls_loss": loss_dict['gen_cls_loss'].item()
                    })

            if self.cls_on_clean_image:
                wandb_loss_dict.update({
                    "guidance_cls_loss": loss_dict['guidance_cls_loss'].item()
                })
            
            if last_tt_loss is not None:
                wandb_loss_dict["loss_target_teacher_mean"] = last_tt_loss.item()

            wandb.log(
                wandb_loss_dict,
                step=self.step, 
                commit=True
            )

        if visual:
            #data_dict = {}
            if not self.args.gan_alone and self.args.use_source_teacher:

                with torch.no_grad():

                    print(log_dict.keys())
                    # If decoded keys are missing, decode the latent x0 to image [-1,1]
                    # if "dmtrain_pred_real_image_decoded" not in log_dict and "dmtrain_pred_real_image" in log_dict:
                    try:
                        r_lat = log_dict["dmtrain_pred_real_image"].detach()
                        f_lat = log_dict["dmtrain_pred_fake_image"].detach()
                        t_lat = log_dict["dmtrain_pred_target_image"].detach()

                        log_dict["dmtrain_pred_real_image_decoded"]   = self.model.decode_image(r_lat)
                        log_dict["dmtrain_pred_fake_image_decoded"]   = self.model.decode_image(f_lat)
                        log_dict["dmtrain_pred_target_image_decoded"] = self.model.decode_image(t_lat)
                    
                    except Exception:
                        # Fallback: assume already in image space
                        log_dict["dmtrain_pred_real_image_decoded"] = log_dict["dmtrain_pred_real_image"]
                        log_dict["dmtrain_pred_fake_image_decoded"] = log_dict["dmtrain_pred_fake_image"]
                        log_dict["dmtrain_pred_target_image_decoded"] = log_dict["dmtrain_pred_target_image"]

                    print(log_dict.keys())
                    log_dict['dmtrain_pred_real_image_decoded'] = accelerator.gather(log_dict['dmtrain_pred_real_image_decoded'])
                    log_dict['dmtrain_pred_fake_image_decoded'] = accelerator.gather(log_dict['dmtrain_pred_fake_image_decoded'])
                    log_dict['dmtrain_pred_target_image_decoded'] = accelerator.gather(log_dict['dmtrain_pred_target_image_decoded'])

            log_dict['generated_image'] = accelerator.gather(log_dict['generated_image'])

            # TT visuals (if TT ran)
            if tt_log is not None:
                tt_noisy_latents = accelerator.gather(tt_log["target_noisy_latents"])
                tt_x0_latents    = accelerator.gather(tt_log["target_latents"])
                n_vis = min(self.num_visuals, tt_x0_latents.shape[0])          

            if self.denoising:
                log_dict["original_clean_image"] = accelerator.gather(log_dict["original_clean_image"])
                log_dict['denoising_timestep']   = accelerator.gather(log_dict['denoising_timestep'])

            # if self.cls_on_clean_image:
            #     log_dict['real_image'] = accelerator.gather(real_train_dict['images'])

        if accelerator.is_main_process and visual:
            with torch.no_grad():
                if not self.args.gan_alone and self.args.use_source_teacher:
                    (
                        dmtrain_pred_real_image, dmtrain_pred_fake_image, dmtrain_pred_target_image
                    ) = (
                        log_dict['dmtrain_pred_real_image_decoded'], log_dict['dmtrain_pred_fake_image_decoded'] , log_dict['dmtrain_pred_target_image_decoded']
                    )

                    # only add dmtrain_gradient_norm if present
                    if "dmtrain_gradient_norm" in log_dict:
                        data_dict["dmtrain_gradient_norm"] = log_dict["dmtrain_gradient_norm"]


                    dmtrain_pred_real_image_grid = prepare_images_for_saving(dmtrain_pred_real_image, resolution=self.resolution, grid_size=self.grid_size)
                    dmtrain_pred_fake_image_grid = prepare_images_for_saving(dmtrain_pred_fake_image, resolution=self.resolution, grid_size=self.grid_size)
                    dmtrain_pred_target_image_grid = prepare_images_for_saving(dmtrain_pred_target_image, resolution=self.resolution, grid_size=self.grid_size)

                    difference_scale_grid = draw_valued_array(
                        (dmtrain_pred_real_image - dmtrain_pred_fake_image).abs().mean(dim=[1, 2, 3]).cpu().numpy(), 
                        output_dir=self.wandb_folder, grid_size=self.grid_size
                    )

                    difference_fk_rl = (dmtrain_pred_real_image - dmtrain_pred_fake_image)
                    difference_fk_rl = (difference_fk_rl - difference_fk_rl.min()) / (difference_fk_rl.max() - difference_fk_rl.min())
                    difference_fk_rl = (difference_fk_rl - 0.5)/0.5
                    difference_fk_rl = prepare_images_for_saving(difference_fk_rl, resolution=self.resolution, grid_size=self.grid_size)

                    differencefk_trg = (dmtrain_pred_target_image - dmtrain_pred_fake_image)
                    differencefk_trg = (differencefk_trg - differencefk_trg.min()) / (differencefk_trg.max() - differencefk_trg.min())
                    differencefk_trg = (differencefk_trg - 0.5)/0.5
                    differencefk_trg = prepare_images_for_saving(differencefk_trg, resolution=self.resolution, grid_size=self.grid_size)

                    differencerl_trg = (dmtrain_pred_target_image - dmtrain_pred_real_image)
                    differencerl_trg = (differencerl_trg - differencerl_trg.min()) / (differencerl_trg.max() - differencerl_trg.min())
                    differencerl_trg = (differencerl_trg - 0.5)/0.5
                    differencerl_trg = prepare_images_for_saving(differencerl_trg, resolution=self.resolution, grid_size=self.grid_size)

                    if last_tt_pred_x0 is not None:
                        # decode and log a small grid
                        tt_imgs = last_tt_pred_x0[:self.num_visuals]
                        tt_imgs = self.model.decode_image(tt_imgs)  # returns [-1,1] fp32
                        tt_grid = prepare_images_for_saving(tt_imgs, resolution=self.resolution, grid_size=self.grid_size)

                        
                    data_dict = {
                        "dmtrain_pred_real_image": wandb.Image(dmtrain_pred_real_image_grid),
                        "dmtrain_pred_fake_image": wandb.Image(dmtrain_pred_fake_image_grid),
                        "dmtrain_pred_target_image": wandb.Image(dmtrain_pred_target_image_grid),
                        "loss_dm": loss_dict['loss_dm'].item(),
                        "difference_fk_rl": wandb.Image(difference_fk_rl),
                        "difference_fk_trg": wandb.Image(differencefk_trg),
                        "difference_real_trg": wandb.Image(differencerl_trg),
                        "difference_norm_grid": wandb.Image(difference_scale_grid),
                    }
                    if tt_grid is not None:
                        data_dict["tt/loss_target_teacher_preview"] = wandb.Image(tt_grid)
                else:
                    data_dict = {} 

                generated_image = log_dict['generated_image']
                generated_image_grid = prepare_images_for_saving(generated_image, resolution=self.resolution, grid_size=self.grid_size)

                
                generated_image_mean = generated_image.mean()
                generated_image_std = generated_image.std()
               
                # ---- TT visuals (if available) ----
                if tt_log is not None:
                    # decode a small grid from both x0 and x_t
                    tt_x0_img = self.model.decode_image(tt_x0_latents[:n_vis])
                    tt_xt_img = self.model.decode_image(tt_noisy_latents[:n_vis])

                    tt_x0_grid = prepare_images_for_saving(tt_x0_img, resolution=self.resolution, grid_size=self.grid_size)
                    tt_xt_grid = prepare_images_for_saving(tt_xt_img, resolution=self.resolution, grid_size=self.grid_size)

                    data_dict.update({
                        "tt/target_x0": wandb.Image(tt_x0_grid),
                        "tt/target_x_t": wandb.Image(tt_xt_grid),
                    })

                    if last_tt_loss is not None:
                        data_dict["tt/loss_target_teacher_mean"] = last_tt_loss.item()
            
                if generator_grad_norm is not None:
                    data_dict["generator_grad_norm"] = generator_grad_norm.item()
                
                if guidance_grad_norm is not None:
                    data_dict["guidance_grad_norm"] = guidance_grad_norm.item()

                data_dict.update({
                    "generated_image": wandb.Image(generated_image_grid),
                    "loss_fake_mean": loss_dict['loss_fake_mean'].item(),
                    "generator_grad_norm": generator_grad_norm.item(),
                    "guidance_grad_norm": guidance_grad_norm.item(),
                })


                if self.denoising:
                    origianl_clean_image = log_dict["original_clean_image"]
                    origianl_clean_image_grid = prepare_images_for_saving(origianl_clean_image, resolution=self.resolution, grid_size=self.grid_size)

                    denoising_timestep = log_dict["denoising_timestep"]
                    denoising_timestep_grid = draw_valued_array(denoising_timestep.cpu().numpy(), output_dir=self.wandb_folder, grid_size=self.grid_size)

                    data_dict.update(
                        {
                            "original_clean_image": wandb.Image(origianl_clean_image_grid),
                            "original_image_mean": original_image_mean.item(),
                            "original_image_std": original_image_std.item(),
                            "denoising_timestep": wandb.Image(denoising_timestep_grid)
                        }
                    )

                if self.cls_on_clean_image:
                    data_dict['guidance_cls_loss'] = loss_dict['guidance_cls_loss'].item()

                    if self.gen_cls_loss:
                        data_dict['gen_cls_loss'] = loss_dict['gen_cls_loss'].item()

                #pred_realism_on_fake = log_dict["pred_realism_on_fake"]
                #pred_realism_on_real = log_dict["pred_realism_on_real"]

                #hist_pred_realism_on_fake = draw_probability_histogram(pred_realism_on_fake.cpu().numpy())
                #hist_pred_realism_on_real = draw_probability_histogram(pred_realism_on_real.cpu().numpy())

                #real_image = log_dict['real_image']
                #real_image_grid = prepare_images_for_saving(real_image, resolution=self.resolution, grid_size=self.grid_size)

                #data_dict.update(
                #    {
                #        "hist_pred_realism_on_fake": wandb.Image(hist_pred_realism_on_fake),
                #        "hist_pred_realism_on_real": wandb.Image(hist_pred_realism_on_real),
                #        "real_image": wandb.Image(real_image_grid)
                #    }
                #)


                def _maybe_debug(tag, x):
                    try:
                        if x is not None:
                            _debug_stats(tag, x)
                        else:
                            print(f"[{tag}] <none>", flush=True)
                    except Exception as e:
                        print(f"[{tag}] <skip: {type(x).__name__}> ({e})", flush=True)

                # --- Debug stats (safe) ---
                # Always (you already gather this above when visual=True)
                _maybe_debug("gen/raw", log_dict.get("generated_image", None))

                # DM predictions (latents + decoded), when DM is enabled (not gan_alone)
                if not self.args.gan_alone:
                    for k, tag in [
                        ("dmtrain_pred_real_image",          "dm/real_lat"),
                        ("dmtrain_pred_fake_image",          "dm/fake_lat"),
                        ("dmtrain_pred_real_image_decoded",  "dm/real_dec"),
                        ("dmtrain_pred_fake_image_decoded",  "dm/fake_dec"),
                    ]:
                        if k in log_dict:
                            _maybe_debug(tag, log_dict[k])

                # Guidance-side image fed to the critic (comes via guidance_data_dict)
                gd = log_dict.get("guidance_data_dict", None)
                if isinstance(gd, dict) and ("image" in gd):
                    _maybe_debug("guid/image", gd["image"])

                # Optional realism heads (if your model logs them)
                for k in ("pred_realism_on_fake", "pred_realism_on_real",
                        "pred_prob_on_fake",    "pred_prob_on_real"):
                    if k in log_dict:
                        _maybe_debug(f"cls/{k}", log_dict[k])

                # Target-Teacher tensors (if TT ran this step)
                if tt_log is not None:
                    # These were already created above when visual=True
                    try:
                        _maybe_debug("tt/target_latents",     tt_x0_latents)
                        _maybe_debug("tt/target_noisy_latents", tt_noisy_latents)
                    except NameError:
                        pass
                    if last_tt_pred_x0 is not None:
                        _maybe_debug("tt/pred_x0", last_tt_pred_x0)
           
                # --- GAN losses (multi-head) ---
                if self.args.multihead_gan:
                    # optional: print stats in console
                    _maybe_debug("gan/bce_g", loss_dict.get("gan_bce_g"))
                    _maybe_debug("gan/bce_d", loss_dict.get("gan_bce_d"))

                    v = loss_dict.get("gan_bce_g", None)
                    if isinstance(v, torch.Tensor) and torch.isfinite(v).all():
                        loss_dict["gan_bce_g"] = v.item()
                    else:
                        print("[skip] gan_bce_g missing or non-finite", flush=True)

                    v = loss_dict.get("gan_bce_d", None)
                    if isinstance(v, torch.Tensor) and torch.isfinite(v).all():
                        loss_dict["gan_bce_d"] = v.item()
                    else:
                        print("[skip] gan_bce_d missing or non-finite", flush=True)

                # Denoising branch (if enabled)
                if self.denoising:
                    if "original_clean_image" in log_dict:
                        _maybe_debug("denoise/orig_clean", log_dict["original_clean_image"])
                    if "denoising_timestep" in log_dict:
                        _maybe_debug("denoise/timestep", log_dict["denoising_timestep"])

                # Log current LRs to catch scheduler/LR explosions
                try:
                    print(f"[lr] gen={self.optimizer_generator.param_groups[0]['lr']:.3e} "
                        f"guid={self.optimizer_guidance.param_groups[0]['lr']:.3e}", flush=True)
                except Exception:
                    pass
                                
              
                wandb.log(data_dict, step=self.step, commit=True)

        self.accelerator.wait_for_everyone()

    def train(self):
        for step in range(0, self.train_iters):               
            self.step = step 
            self.train_one_step()

            # Pause-time generation happens inside the main loop so snapshots line up
            # exactly with the current weights and scheduler state at this step.
           # before saving (if configured), pause and generate
            if self.step in self.args.gen_pause_steps:
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    print(f"[pause] step {self.step}: paused generation")
                self._paused_generate()
                self.accelerator.wait_for_everyone()

            # checkpoint space safe
            if (not getattr(self, "no_save", False)) and (self.step % self.log_iters == 0):
                try:
                    self.save()
                except Exception as e:
                    self.accelerator.print(f"[WARN] save() threw: {e} — continuing.")
                    try:
                        self._save_model_only_fallback()
                    except Exception as e2:
                        self.accelerator.print(f"[WARN] model-only fallback save also failed: {e2}")

            if self.accelerator.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    wandb.log({"per iteration time": current_time-self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.step += 1

def parse_args():
    parser = argparse.ArgumentParser()
    # Core runtime / checkpoint locations.
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.environ.get("DMD2_OUT") or os.path.join(os.path.expanduser("~"), "dmd2_checkpoints"),
        help="Directory to save checkpoints (default: $DMD2_OUT or ~/dmd2_checkpoints)",
        )
    parser.add_argument(
        "--log_path",
        type=str,
        default=os.environ.get("DMD2_LOG") or os.path.join(os.path.expanduser("~"), "dmd2_logs"),
        help="Directory for logs/W&B files (default: $DMD2_LOG or ~/dmd2_logs)"
        )
    parser.add_argument(
    "--output_path_targetteacher",
    type=str,
    default=os.environ.get("TT_OUT") or os.path.join(os.path.expanduser("~"), "targetteacher_checkpoints"),
    help="Directory to save checkpoints (default: $TT_OUT or ~/targetteacher_checkpoints)",
    )
    parser.add_argument("--train_iters", type=int, default=1000000)
    parser.add_argument("--log_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--initialie_generator", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_iters", type=int, default=100)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="max grad norm for network")
    parser.add_argument("--warmup_step", type=int, default=500, help="warmup step for network")
    parser.add_argument("--min_step_percent", type=float, default=0.02, help="minimum step percent for training")
    parser.add_argument("--max_step_percent", type=float, default=0.98, help="maximum step percent for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--ckpt_only_path", type=str, default=None, help="checkpoint (no optimizer state) only path")
    parser.add_argument("--targetteacher_ckpt_path", type=str, default=None, help=" for target teacher init checkpoint (no optimizer state) only path if none init from dmd2")
    parser.add_argument("--train_prompt_path", type=str)
    parser.add_argument("--rare_token", type=str, default="prt",
                    help="Rare token used for student & generator prompts (will be removed for teacher prompts)")

    # Adapt-then-distill initialization knobs.
    # Adapt then distill pipeline - loading weights Dreambooth
    parser.add_argument(
    "--source_unet_path",
    type=str,
    default=None,
    help="Path to a Diffusers UNet folder (e.g., .../checkpoint-800/unet) to initialize the frozen source teacher UNet."
    )
    parser.add_argument(
        "--init_generator_from_source",
        action="store_true",
        help="Also initialize the generator UNet from the same DreamBooth UNet (optional)."
    )


    # Dataset selection and loss/head configuration.
    parser.add_argument("--train_lmdb_path", type=str, default=None, help="LMDB with 'prompts' keys; if set, overrides --train_prompt_path")
    parser.add_argument("--num_classes", type=int, default=None,
    help="Output size of classifier head(s). Use number of instances when training per-instance.")
    parser.add_argument("--gen_include_labels", type=str, default=None,
    help="Comma-separated instance ids for the generator stream (e.g., '17'). If unset, uses all.")
    parser.add_argument("--guidance_include_labels", type=str, default=None,
    help="Comma-separated instance ids for the guidance stream. If unset, uses ALL instances (recommended).")

    parser.add_argument("--multihead_gan", action="store_true",
        help="Enable multi-head GAN heads on UNet encoder+mid (BCE).")
    parser.add_argument("--gan_bce_weight_d", type=float, default=1.0,
        help="Weight for D's BCE loss.")
    parser.add_argument("--gan_bce_weight_g", type=float, default=1.0,
        help="Weight for G's BCE loss.")
    parser.add_argument("--gan_bce_random_t", action="store_true",
        help="If set, score at a random timestep per batch; else use t=0.")
    parser.add_argument("--gan_bce_timestep", type=int, default=0,
        help="Fixed timestep for BCE critic when gan_bce_random_t is False.")

    parser.add_argument("--latent_resolution", type=int, default=64)
    parser.add_argument("--real_guidance_scale", type=float, default=6.0)
    parser.add_argument("--fake_guidance_scale", type=float, default=1.0)
    parser.add_argument("--grid_size", type=int, default=2)
    parser.add_argument("--no_save", action="store_true", help="don't save ckpt for debugging only")
    parser.add_argument("--cache_dir", type=str, default=None,
        help="Writable cache directory (overrides default /mnt/localssd)")
    parser.add_argument("--log_loss", action="store_true", help="log loss at every iteration")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--latent_channel", type=int, default=4)
    parser.add_argument("--max_checkpoint", type=int, default=150)
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=1)
    parser.add_argument("--dteacher_update_ratio", type=int, default=1,  # default == gen ratio
                        help="How often to update target-teacher (TT). Default: same as dfake_gen_update_ratio.")
    parser.add_argument("--dtarget_dm_update_ratio", type=int, default=10,
                        help="(Reserved) How often to apply Target-DM. For now, tied to generator ratio.")
    parser.add_argument("--generator_lr", type=float)
    parser.add_argument("--guidance_lr", type=float)
    parser.add_argument("--cls_on_clean_image", action="store_true")
    parser.add_argument("--gen_cls_loss", action="store_true")
    parser.add_argument("--gen_cls_loss_weight", type=float, default=0)
    parser.add_argument("--guidance_cls_loss_weight", type=float, default=0)
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--generator_ckpt_path", type=str)
    parser.add_argument("--conditioning_timestep", type=int, default=999)
    parser.add_argument("--tiny_vae", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="apply gradient checkpointing for dfake and generator. this might be a better option than FSDP")
    parser.add_argument("--dm_loss_weight", type=float, default=1.0)

    parser.add_argument("--denoising", action="store_true", help="train the generator for denoising")
    parser.add_argument("--denoising_timestep", type=int, default=1000)
    parser.add_argument("--num_denoising_step", type=int, default=1)
    parser.add_argument("--denoising_loss_weight", type=float, default=1.0)

    # For few step denoising
    parser.add_argument("--denoising_sigma_end", type=float, default=0.5,
        help="Final (smallest) sigma for the unroll")

    parser.add_argument("--diffusion_gan", action="store_true")
    parser.add_argument("--diffusion_gan_max_timestep", type=int, default=0)
    parser.add_argument("--revision", type=str)

    parser.add_argument("--real_image_path", type=str)
    parser.add_argument("--gan_alone", action="store_true", help="only use the gan loss without dmd")
    parser.add_argument("--backward_simulation", action="store_true")

    parser.add_argument("--generator_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # Freeze UNet encoder features only during the D-step of Multi-Head GAN
    parser.add_argument(
        "--mhgan-freeze-encoder-dstep",
        dest="mhgan_freeze_encoder_dstep",
        action="store_true",
        default=False,
        help="Detach UNet encoder features during the discriminator (D) step so D-loss does not backprop into UNet. Default: on.",
    )
    parser.add_argument(
        "--no-mhgan-freeze-encoder-dstep",
        dest="mhgan_freeze_encoder_dstep",
        action="store_false",
        help="Disable freezing; allow D-loss to backprop into UNet encoder.",
    )
    
    # Pause-time generation and file-based evaluation controls.
        # --- Generation-at-pause controls ---
    parser.add_argument("--gen_pause_steps", type=str, default=None,
        help="Comma-separated steps to pause and generate at (e.g. '500,1000,2000').")
    parser.add_argument("--gen_prompts_file", type=str, default="data/prompts_and_classes.txt",
        help="Path to prompts_and_classes.txt.")
    parser.add_argument("--gen_outputs_root", type=str, default="output_dmd2",
        help="Root folder to save paused generations.")
    parser.add_argument("--gen_training_name", type=str, default=None,
        help="Folder name used under outputs_root. Defaults to --wandb_name.")
    parser.add_argument("--gen_samples_per_prompt", type=int, default=4,
        help="Number of images to sample per prompt (default: 4).")
    parser.add_argument("--gen_seeds", type=str, default="0,1,2,3",
        help="Comma-separated seeds for per-prompt images; length must match --gen_samples_per_prompt.")
    parser.add_argument("--gen_delete_ckpt", action="store_true",
        help="After generation at a pause step, delete the checkpoint folder that was just saved.")
    parser.add_argument("--gen_guidance_scale", type=float, default=7.5,
        help="Guidance scale to use during paused generation (for SD1.5).")
    parser.add_argument("--gen_batch_prompts", type=int, default=4,
        help="How many prompts to batch together during paused generation.")
    
    parser.add_argument("--skip_accelerate_state",
        action="store_true",
        help="Skip accelerate.save_state (optimizer/scheduler). Saves model-only to reduce I/O."
        )
    
    #For evaluation
    # --- Pause-eval controls (CLIP/DINO) ---
    parser.add_argument("--eval_enable", action="store_true",
        help="Run CLIP-I/CLIP-T/DINO-I evaluation for each paused generation step.")
    parser.add_argument("--eval_clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--eval_max_src", type=int, default=4)
    parser.add_argument("--eval_max_gen_per_group", type=int, default=4)
    parser.add_argument("--eval_uniform_size", type=int, default=224)
    parser.add_argument("--eval_src_root", type=str, default=None,
        help="If set, take source instance images from this root/<instance_name>/; otherwise sample from LMDB.")
    parser.add_argument("--eval_finalize_policy", type=str, default="last",
                    choices=["last","mean","best_clip_i","best_clip_t","best_dino"],
                    help="How to pick the final run-wide score for an instance.")
    parser.add_argument("--eval_finalize_on_last_pause", action="store_true",
                    help="Also compute the final instance summary when the last pause step finishes.")
    
    # Teacher/source-target feature toggles.
    # Switch on/off for teacher target model  or teacher source model
    parser.add_argument("--use_source_teacher", action="store_true", default=None)
    parser.add_argument("--enable_target_teacher", action="store_true", default=None)
    parser.add_argument("--dm_weight_source", type=float, default=1.0,
                    help="Weight for Source-DM (frozen source UNet) direction.")
    parser.add_argument("--dm_weight_target", type=float, default=1.0,
                    help="Weight for Target-DM (target-teacher UNet) direction.")


    args = parser.parse_args()

    # Normalize CLI strings once so the rest of the trainer can treat them as
    # structured lists/ints instead of re-parsing user input on each call site.
    # Pause-gen args normalization
    args.gen_pause_steps = sorted({int(s) for s in (getattr(args, "gen_pause_steps", "") or "").split(",") if s.strip()})
    args.gen_training_name = getattr(args, "gen_training_name", None) or args.wandb_name
    args.gen_seeds = [int(s) for s in (getattr(args, "gen_seeds", "0,1,2,3") or "").split(",") if s.strip()]
    if getattr(args, "gen_samples_per_prompt", None) is None:
        args.gen_samples_per_prompt = len(args.gen_seeds)
    if args.gen_samples_per_prompt != len(args.gen_seeds):
        raise ValueError(f"--gen_samples_per_prompt ({args.gen_samples_per_prompt}) must equal number of --gen_seeds ({len(args.gen_seeds)}).")

    print(f"[args] pauses={args.gen_pause_steps} seeds={args.gen_seeds} prompts_file={args.gen_prompts_file}")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    assert args.gradient_accumulation_steps == 1, "grad accumulation not supported yet"

    assert not (args.fsdp and args.gradient_checkpointing), "currently, we don't support both options. open an issue for details."

    assert args.wandb_iters % args.dfake_gen_update_ratio == 0, "wandb_iters should be a multiple of dfake_gen_update_ratio"

    return args 

if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)

    trainer.train()
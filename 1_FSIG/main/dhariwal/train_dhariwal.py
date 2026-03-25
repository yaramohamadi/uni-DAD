"""
Trainer for Dhariwal-style unified diffusion + GAN with DMD losses.

This module:
  - Parses CLI arguments for model, optimization, and logging hyperparameters.
  - Builds a dhariwalUniModel with separate feedforward (generator) and guidance
    networks, optionally including a target-teacher branch.
  - Loads real images from an LMDB dataset (for GAN / DMD supervision) with
    simple tensor-based augmentations.
  - Configures optimizers and schedulers for:
        * generator (feedforward_model)
        * guidance model (excluding target_unet parameters)
        * optional target teacher (target_unet) with its own optimizer/scheduler
  - Supports:
        * DMD loss with source/target teacher weights
        * optional GAN classifier (single or multi-head) with several GAN losses
        * few-step denoising unroll (K-step generator updates)
        * optional label dropout and null-conditioning
  - Implements a training loop with:
        * gradient accumulation via `Accelerator.accumulate`
        * optional target-teacher updates on a configurable cadence
        * weights-only checkpoint loading and robust checkpoint saving
        * micro-batch aggregation for metrics and visualizations
        * rich Weights & Biases logging (images, histograms, scalars).

Entry point:
  - `parse_args()` parses all configuration flags.
  - `Trainer(args)` sets up models, data, optimizers, and logging.
  - `Trainer.train()` runs the main training loop.
"""


import matplotlib
matplotlib.use('Agg')

from main.utils import (
    prepare_images_for_saving, draw_valued_array, cycle,
    draw_probability_histogram, RandomHorizontalFlipTensor,
    _is_locked_or_not_ready, _safe_rmtree,
)
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs
from diffusers.optimization import get_scheduler
from main.data.lmdb_dataset import LMDBDataset
from main.dhariwal.dhariwal_unified_model import dhariwalUniModel
from accelerate import Accelerator
import argparse 
import shutil
import wandb 
import torch 
import time 
import math
from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext
import os, glob, torch
from collections import OrderedDict


def _strip_module_prefix(sd):
    if any(k.startswith("module.") for k in sd.keys()):
        return OrderedDict((k[len("module."):], v) for k, v in sd.items())
    return sd

def _parse_shard_index(filename: str) -> int:
    """
    Returns -1 for the base file (e.g., pytorch_model.bin / model.safetensors),
    otherwise the numeric suffix for shards (e.g., pytorch_model_1.bin -> 1).
    """
    base = os.path.basename(filename)
    stem, _ext = os.path.splitext(base)         # e.g., "pytorch_model_1", ".bin"
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    return -1  # base

def remap_key(k: str) -> str:
    # Handle source → target mapping
    if k.startswith("real_unet."):
        return "guidance_model.module.real_unet." + k[len("real_unet."):]
    if k.startswith("fake_unet."):
        return "guidance_model.module.fake_unet." + k[len("fake_unet."):]
    if k.startswith("unet."):
        return "feedforward_model.module.unet." + k[len("unet."):]
    if k.startswith("model."):
        return "feedforward_model.module.model." + k[len("model."):]
    if k.startswith("karras_sigmas"):
        return "guidance_model.module.karras_sigmas" + k[len("karras_sigmas"):]
    return k  # unchanged

def load_weights_only(checkpoint_dir, model, accelerator=None, strict=False):
    # collect shards
    shards = []
    shards += glob.glob(os.path.join(checkpoint_dir, "pytorch_model*.bin"))
    shards += glob.glob(os.path.join(checkpoint_dir, "model*.bin"))
    shards += glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))
    assert shards, f"No model weight files found in {checkpoint_dir}"

    # sort: base first (index -1), then 0..N (if present)
    shards = sorted(shards, key=lambda p: (_parse_shard_index(p) != -1, _parse_shard_index(p)))

    from contextlib import nullcontext
    ctx = accelerator.main_process_first() if accelerator is not None else nullcontext()

    merged = OrderedDict()
    with ctx:
        for f in shards:
            if f.endswith(".safetensors"):
                from safetensors.torch import load_file
                part = load_file(f)
            else:
                part = torch.load(f, map_location="cpu")
            if isinstance(part, dict) and "state_dict" in part and len(part) == 1:
                part = part["state_dict"]
            part = _strip_module_prefix(part)
            # apply remap
            remapped = OrderedDict((remap_key(k), v) for k, v in part.items())
            merged.update(remapped)

    target = accelerator.unwrap_model(model) if accelerator is not None else model
    
    missing, unexpected = target.load_state_dict(merged, strict=strict)
    msg = (f"[weights-only] Loaded {len(shards)} shard(s) from {checkpoint_dir} "
           f"| missing={len(missing)} unexpected={len(unexpected)}")
    (accelerator.print if accelerator is not None else print)(msg)

    # Optional: dump a few of the missing/unexpected for debugging
    if missing: print("Missing sample:", missing)
    if unexpected: print("Unexpected sample:", unexpected)

    return missing, unexpected



class Trainer:
    def __init__(self, args):

        self.args = args

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True 

        accelerator_project_config = ProjectConfiguration(logging_dir=args.output_path)

        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=True,          # <-- important
            gradient_as_bucket_view=True          # small perf win; optional
            # static_graph=False is default; keep it False since your graph changes
        )

        accelerator = Accelerator(
            gradient_accumulation_steps=args.grad_accum_steps,
            mixed_precision="bf16" if args.use_bf16 else "no",
            log_with="wandb",
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs],         # <-- add this
        )

        set_seed(args.seed + accelerator.process_index)

        # TODO I removed seed and time from checkpoint path
        # Enable checkpoint resuming and saving in the same wandb run
        if accelerator.is_main_process:
            #if args.checkpoint_path is not None:
            #    # Resume into the SAME run folder (parent of the checkpoint directory)
            #    resume_run_dir = os.path.dirname(args.checkpoint_path.rstrip("/"))
            #    self.output_path = resume_run_dir
            #    os.makedirs(self.output_path, exist_ok=True)
            #else:
                # fresh run
            output_path = os.path.join(args.output_path)
            os.makedirs(output_path, exist_ok=True)
            self.output_path = output_path

            if args.cache_dir != "":
                if args.checkpoint_path is not None:
                    # mirror cache dir name to the resumed run_id/seed pattern if you use it
                    parent = os.path.basename(os.path.dirname(self.output_path))  # e.g., time_1756313057_seed10
                    self.cache_dir = os.path.join(args.cache_dir, parent)
                else:
                    self.cache_dir = args.cache_dir
                os.makedirs(self.cache_dir, exist_ok=True)
                

        self.model = dhariwalUniModel(args, accelerator)
        self.dataset_name = args.dataset_name
        self.real_image_path = args.real_image_path

        self.dfake_gen_update_ratio = args.dfake_gen_update_ratio
        self.num_train_timesteps = args.num_train_timesteps 

        self.cls_loss_weight = args.cls_loss_weight 

        self.gan_classifier = args.gan_classifier 
        self.gan_adv_loss = args.gan_adv_loss
        self.gen_cls_loss_weight = args.gen_cls_loss_weight 
        self.dmd_loss_weight = args.dmd_loss_weight # Added by me
        self.no_save = args.no_save
        self.previous_time = None 
        self.step = 0 
        self.global_step = 0
        self.cache_checkpoints = (args.cache_dir != "")
        self.max_checkpoint = args.max_checkpoint

        self.label_dropout_p = args.label_dropout_p

        if args.ckpt_only_path is not None:
            if accelerator.is_main_process:
                print(f"loading checkpoints without optimizer states from {args.ckpt_only_path}")
            # state_dict = torch.load(args.ckpt_only_path, map_location="cpu")
            # print(self.model.load_state_dict(state_dict, strict=False))
            generator_path = os.path.join(args.ckpt_only_path, "pytorch_model.bin")
            guidance_path = os.path.join(args.ckpt_only_path, "pytorch_model_1.bin")

            generator_state_dict = torch.load(generator_path, map_location="cpu")
            guidance_state_dict = torch.load(guidance_path, map_location="cpu")

            print(self.model.feedforward_model.load_state_dict(generator_state_dict, strict=False))
            print(self.model.guidance_model.load_state_dict(guidance_state_dict, strict=False))

            self.step = int(args.ckpt_only_path.replace("/", "").split("_")[-1])

        if args.generator_ckpt_path is not None:
            if accelerator.is_main_process:
                print(f"loading generator checkpoints from {args.generator_ckpt_path}")
            generator_path = os.path.join(args.generator_ckpt_path, "pytorch_model.bin")
            print(self.model.feedforward_model.load_state_dict(torch.load(generator_path, map_location="cpu"), strict=True))

        # also load the training dataset images, this will be useful for GAN loss 
        real_transform = RandomHorizontalFlipTensor(p=0.5)
        real_dataset = LMDBDataset(args.real_image_path, transform=real_transform)

        real_image_dataloader = torch.utils.data.DataLoader(
            real_dataset, batch_size=args.batch_size, shuffle=True, 
            drop_last=True, num_workers=0
        )

        real_image_dataloader = accelerator.prepare(real_image_dataloader)
        self.real_image_dataloader = cycle(real_image_dataloader)
        
        print("loading learning rate schedulers and optimizers")
        print(f"Generator lr: {args.generator_lr}, Guidance lr: {args.guidance_lr}")
        # ---- Guidance optimizer: exclude target_unet params ----
        guidance_params = []
        for n, p in self.model.guidance_model.named_parameters():
            if not p.requires_grad:
                continue
            # Keep guidance grads off the target teacher
            if n.startswith("target_unet."):
                continue
            guidance_params.append(p)

        self.optimizer_guidance = torch.optim.AdamW(
            guidance_params, lr=args.guidance_lr, betas=(0.9, 0.999), weight_decay=0.01
        )
        self.optimizer_generator = torch.optim.AdamW(
            [param for param in self.model.feedforward_model.parameters() if param.requires_grad], 
            lr=args.generator_lr, 
            betas=(0.9, 0.999),  # pytorch's default 
            weight_decay=0.01  # pytorch's default 
        )

        # actually this scheduler is not very useful (it warms up from 0 to max_lr in 500 / num_gpu steps), but we keep it here for consistency
        self.scheduler_guidance = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_guidance,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters 
        )

        self.scheduler_generator = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_generator,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters 
        )

        # ---- Optional Target-Teacher optimizer (separate step/schedule) ----
        self.optimizer_target_teacher = None
        self.scheduler_target_teacher = None
        if args.use_target_teacher and args.train_target_teacher:
            tt_params = [p for n, p in self.model.guidance_model.named_parameters()
                        if n.startswith("target_unet.") and p.requires_grad]
            self.optimizer_target_teacher = torch.optim.AdamW(
                tt_params,
                lr=args.guidance_lr,
                betas=(0.9, 0.999),
                weight_decay=0.01
            )
            self.scheduler_target_teacher = get_scheduler(
                "constant_with_warmup",
                optimizer=self.optimizer_target_teacher,
                num_warmup_steps=args.warmup_step,
                num_training_steps=args.train_iters
            )

        # the self.model is not wrapped in ddp, only its two subnetworks are wrapped 
        prepare_args = [
            self.model.feedforward_model,
            self.model.guidance_model,
            self.optimizer_guidance,
            self.optimizer_generator,
            self.scheduler_guidance,
            self.scheduler_generator,
        ]

        if self.optimizer_target_teacher is not None:
            prepare_args += [self.optimizer_target_teacher, self.scheduler_target_teacher]

        prepared = accelerator.prepare(*prepare_args)

        (self.model.feedforward_model,
        self.model.guidance_model,
        self.optimizer_guidance,
        self.optimizer_generator,
        self.scheduler_guidance,
        self.scheduler_generator, *maybe_tt) = prepared

        if self.optimizer_target_teacher is not None:
            (self.optimizer_target_teacher, self.scheduler_target_teacher) = maybe_tt

        self.accelerator = accelerator
        self.train_iters = args.train_iters
        self.batch_size = args.batch_size
        self.resolution = args.resolution 
        self.log_iters = args.log_iters
        self.wandb_iters = args.wandb_iters
        self.conditioning_sigma = args.conditioning_sigma 

        self.label_dim = args.label_dim
        self.eye_matrix = torch.eye(self.label_dim, device=accelerator.device)
        self.null_index = self.label_dim - 1 if self.label_dim > 0 else None
        self.delete_ckpts = args.delete_ckpts
        self.max_grad_norm = args.max_grad_norm

        if args.checkpoint_path is not None:
            print("Attempting to resume from intermediate checkpoint....")
            self.load(args.checkpoint_path)
            args.checkpoint_path = None  # create a new directory for continued training

        
        # set a single new LR
        for g in self.optimizer_generator.param_groups:
            g["lr"] = args.generator_lr

        # set a single new LR
        for g in self.optimizer_guidance.param_groups:
            g["lr"] = args.guidance_lr


        if self.accelerator.is_main_process:
            run = wandb.init(config=args, dir=self.output_path, **{"mode": "online", "entity": args.wandb_entity, "project": args.wandb_project})
            wandb.run.log_code(".")
            wandb.run.name = args.wandb_name
            print(f"run dir: {run.dir}")
            self.wandb_folder = run.dir
            os.makedirs(self.wandb_folder, exist_ok=True)

        self._mb_clear()

    # Grad accumulation wandb
    # ---- micro-batch accumulator (per grad-accum window) ----
    def _mb_clear(self):
        self._mb_tensors = defaultdict(list)  # per-key list of tensors
        self._mb_scalars = defaultdict(float) # sums for scalars we want to average
        self._mb_count   = 0                  # how many micro-batches in the window

    def _mb_put(self, log_dict, loss_dict):
        # Tensors we want to aggregate across micro-batches (extend as needed)
        tkeys = [
            'generated_image',
            'dmtrain_noisy_latents',
            'dmtrain_pred_real_image',
            'dmtrain_pred_fake_image',
            'dmtrain_grad',
            'dmtrain_timesteps',
            'faketrain_latents',
            'faketrain_noisy_latents',
            'faketrain_x0_pred',
            'pred_realism_on_fake',
            'pred_realism_on_real',
            'critic_fake',
            'critic_real',
            'tt_pred_x0',
            'dmtrain_pred_real_image_source',
            'dmtrain_pred_real_image_target',
            'dmtrain_grad_source',
            'dmtrain_grad_target',
        ]

        for k in tkeys:
            v = log_dict.get(k, None)
            if isinstance(v, torch.Tensor):
                # keep on device; cheaper to gather once at the end
                self._mb_tensors[k].append(v.detach())

        # Scalars we want to average over the window (extend as needed)
        for k in ['loss_dm', 'loss_fake_mean', 'guidance_cls_loss', 'gen_cls_loss', 'loss_target_teacher']:
            if k in loss_dict:
                v = loss_dict[k]
                self._mb_scalars[k] += (float(v.detach().item()) if torch.is_tensor(v) else float(v))

        self._mb_count += 1

    def _mb_concat_and_gather(self):
        """Concatenate micro-batches and gather across processes for logging/metrics."""
        cat = {}

        if self.accelerator.sync_gradients:
            n_mb = len(self._mb_tensors.get("generated_image", []))
            print(f"[DEBUG] micro-batches with generated_image this window: {n_mb} (world_size={self.accelerator.num_processes})")

        for k, lst in self._mb_tensors.items():
            if len(lst) > 0:
                cat[k] = torch.cat(lst, dim=0)                    # [sum_B, ...] across micro-batches
                cat[k] = self.accelerator.gather_for_metrics(cat[k])  # gather across processes (-> CPU)
            else:
                cat[k] = None

        means = {}
        denom = max(1, self._mb_count)
        for k, s in self._mb_scalars.items():
            means[k] = s / denom

        return cat, means


    def load(self, checkpoint_path):
        # Expecting directories like .../checkpoint_model_000123
        self.global_step = 0# int(checkpoint_path.rstrip("/").split("_")[-1])
        accum = self.accelerator.gradient_accumulation_steps
        
        print("Loading model weights only (no optimizer/scheduler state)")
        # print(self.accelerator.load_state(checkpoint_path, strict=False))
        load_weights_only(checkpoint_path, self.model, accelerator=self.accelerator, strict=False)
        self.accelerator.print(f"Loaded checkpoint from {checkpoint_path}")
        self.step = self.global_step * max(1, accum)  # micro-step counter aligned to optimizer step

        # weights-only resume
        # load_weights_only(checkpoint_path, self.model, accelerator=self.accelerator, strict=False)
        # self.global_step = int(checkpoint_path.rstrip("/").split("_")[-1]) + 1
        # accum = self.accelerator.gradient_accumulation_steps
        # self.step = self.global_step * max(1, accum)  # micro-step counter aligned to optimizer step
        # self.accelerator.print("Resumed weights-only; optimizer/scheduler reset.")

    def save(self):
        run_root = Path(self.output_path)
        gs = self.global_step  # snapshot once
        final_dir = run_root / f"checkpoint_model_{gs:06d}"
        tmp_dir   = run_root / f".checkpoint_model_{gs:06d}.tmp"

        print(f"start saving checkpoint to {final_dir}", flush=True)

        if final_dir.exists():
            msg = "already exists (.READY) -> skip" if (final_dir / ".READY").exists() and (final_dir / "pytorch_model.bin").exists() else "exists (incomplete?) -> skip"
            print(f"[save] {final_dir} {msg}.", flush=True)
            if tmp_dir.exists():
                _safe_rmtree(tmp_dir)
        else:
            if tmp_dir.exists():
                _safe_rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            self.accelerator.save_state(str(tmp_dir), safe_serialization=False)

            try:
                os.replace(str(tmp_dir), str(final_dir))
            except OSError as e:
                if final_dir.exists():
                    print(f"[save] {final_dir} appeared during save; skipping this step. ({e})", flush=True)
                    _safe_rmtree(tmp_dir)
                else:
                    raise
            else:
                (final_dir / ".READY").touch()

                if self.delete_ckpts:
                    for folder in os.listdir(self.output_path):
                        if folder.startswith("checkpoint_model") and folder != f"checkpoint_model_{gs:06d}":
                            d = run_root / folder
                            if (d / ".READY").exists() and not (d / ".EVAL_LOCK").exists():
                                _safe_rmtree(d)

                if self.cache_checkpoints:
                    cache_ckpt = Path(self.cache_dir) / f"checkpoint_model_{gs:06d}"
                    if cache_ckpt.exists():
                        _safe_rmtree(cache_ckpt)
                    shutil.copytree(str(final_dir), str(cache_ckpt), dirs_exist_ok=False)

                    checkpoints = sorted(
                        p for p in Path(self.cache_dir).iterdir()
                        if p.is_dir() and p.name.startswith("checkpoint_model")
                    )
                    if len(checkpoints) > self.max_checkpoint:
                        for p in checkpoints[:-self.max_checkpoint]:
                            _safe_rmtree(p)

                print("done saving", flush=True)

    # Clipping label embeddings for conditional learning
    def _clip_label_embedding(self, module, max_norm: float):
        label_params = [p for p in self._iter_label_emb_params(module) if p.grad is not None]
        if label_params:
            torch.nn.utils.clip_grad_norm_(label_params, max_norm)

    def _iter_label_emb_params(self, module):
        for n, p in module.named_parameters():
            if n.endswith("label_emb.weight"):
                yield p


    def train_one_step(self):

        def _strip_module_prefix_name(name: str) -> str:
            return name[7:] if name.startswith("module.") else name


        self.model.train()
        accelerator = self.accelerator
        accum = accelerator.gradient_accumulation_steps

        # Optimizer-step index for this window (clean step numbering for logging)
        self.global_step = (self.step) // max(1, accum)

        # ----- batch prep (unchanged) -----
        real_dict = next(self.real_image_dataloader)
        real_image = real_dict["images"] * 2.0 - 1.0
        if self.label_dim > 0:
            real_label = self.eye_matrix[real_dict["class_labels"].squeeze(dim=1)]
            labels = torch.randint(0, self.label_dim - 1, (self.batch_size,), device=accelerator.device)
            labels = self.eye_matrix[labels]
            if self.label_dropout_p > 0.0:
                if torch.rand(1, device=accelerator.device) < self.label_dropout_p:
                    labels = self.eye_matrix[self.null_index].expand(self.batch_size, -1)
                    real_label = self.eye_matrix[self.null_index].expand(self.batch_size, -1)
        else:
            real_label = None
            labels = None
        real_train_dict = {"real_image": real_image, "real_label": real_label}

        scaled_noise = torch.randn(
            self.batch_size, 3, self.resolution, self.resolution, device=accelerator.device
        ) * self.conditioning_sigma
        timestep_sigma = torch.ones(self.batch_size, device=accelerator.device) * self.conditioning_sigma

        # Stable within the window
        COMPUTE_GENERATOR_GRADIENT = (self.global_step % self.dfake_gen_update_ratio == 0)

        tt_out = None

        generator_grad_norm = torch.tensor(0.0, device=accelerator.device)
        guidance_grad_norm  = torch.tensor(0.0, device=accelerator.device)

        # -------------------------------
        # Single accumulation gate (GEN)
        # -------------------------------
        with accelerator.accumulate(self.model.feedforward_model):
        # ------------------- Generator turn -------------------
            gen_loss_dict, gen_log_dict = self.model(
                scaled_noise, timestep_sigma, labels,
                real_train_dict=real_train_dict,
                compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
                generator_turn=True, guidance_turn=False
            )
            if COMPUTE_GENERATOR_GRADIENT:
                generator_loss = self.dmd_loss_weight * gen_loss_dict["loss_dm"]
                if self.gan_classifier:
                    generator_loss = generator_loss + gen_loss_dict["gen_cls_loss"] * self.gen_cls_loss_weight
                accelerator.backward(generator_loss)

            # ------------------- Guidance turn --------------------
            guid_sync_ctx = nullcontext() if accelerator.sync_gradients else accelerator.no_sync(self.model.guidance_model)
            with guid_sync_ctx:
                guid_loss_dict, guid_log_dict = self.model(
                    scaled_noise, timestep_sigma, labels,
                    real_train_dict=real_train_dict,
                    compute_generator_gradient=False,
                    generator_turn=False, guidance_turn=True,
                    guidance_data_dict=gen_log_dict.get('guidance_data_dict', None)
                )
                guidance_loss = guid_loss_dict["loss_fake_mean"]
                if self.gan_classifier:
                    guidance_loss = guidance_loss + guid_loss_dict["guidance_cls_loss"] * self.cls_loss_weight
                accelerator.backward(guidance_loss)

            # ------------- Step G and Guidance (sync edge) -------------
            if accelerator.sync_gradients:
                if self.label_dim > 0:
                    pass

                # clip & step generator
                generator_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.feedforward_model.parameters(), self.max_grad_norm
                )
                # clip & step guidance (exclude TT params)
                guidance_params_no_tt = [
                    p for n, p in self.model.guidance_model.named_parameters()
                    if (not _strip_module_prefix_name(n).startswith("target_unet.")) and p.grad is not None
                ]
                guidance_grad_norm = accelerator.clip_grad_norm_(guidance_params_no_tt, self.max_grad_norm)

                self.optimizer_generator.step()
                self.optimizer_generator.zero_grad(set_to_none=True)
                self.scheduler_generator.step()

                self.optimizer_guidance.step()
                self.optimizer_guidance.zero_grad(set_to_none=True)
                self.scheduler_guidance.step()

            # ==========================================================
            # NEW: Target-Teacher update AFTER guidance step
            # Run once per optimizer step, outside the generator accumulate block
            # ==========================================================
            tt_cadence_hits = bool(self.args.tt_match_guidance) or bool(COMPUTE_GENERATOR_GRADIENT)
            if accelerator.sync_gradients and tt_cadence_hits and getattr(self, "optimizer_target_teacher", None) is not None:
                gm = self.model.guidance_model
                inner_gm = gm.module if hasattr(gm, "module") else gm  # unwrap DDP

                # Fresh TT forward (optionally use the same generated image, detached)
                with torch.enable_grad():
                    tt_out = inner_gm.compute_loss_target_teacher(
                        real_image=real_train_dict["real_image"],
                        real_label=real_train_dict["real_label"],
                        maybe_fake_image=(gen_log_dict.get("generated_image", None).detach()
                                        if ('gen_log_dict' in locals() and gen_log_dict.get("generated_image", None) is not None)
                                        else None),
                        train_on_real=getattr(self.args, "target_teacher_train_on_real", True),
                        train_on_fake=getattr(self.args, "target_teacher_train_on_fake", False),
                    )
                    tt_loss = tt_out["loss_target_teacher"] * getattr(self.args, "target_teacher_loss_weight", 1.0)

                # This backward is the ONLY one happening now; guidance already reduced
                accelerator.backward(tt_loss)
                if accelerator.sync_gradients:
                    # clip only TT params and step TT optimizer
                    tt_params_for_clip = [
                        p for n, p in self.model.guidance_model.named_parameters()
                        if _strip_module_prefix_name(n).startswith("target_unet.") and p.grad is not None
                    ]
                    if self.max_grad_norm and self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(tt_params_for_clip, max_norm=self.max_grad_norm)

                    self.optimizer_target_teacher.step()
                    self.optimizer_target_teacher.zero_grad(set_to_none=True)
                    if getattr(self, "scheduler_target_teacher", None) is not None:
                        self.scheduler_target_teacher.step()


        # ---- safe-merge logs for W&B
        log_dict = gen_log_dict.copy()
        for k, v in guid_log_dict.items():
            if (k not in log_dict) or isinstance(v, torch.Tensor):
                log_dict[k] = v
        if tt_out is not None:
            if "tt_pred_x0" in tt_out: log_dict["tt_pred_x0"] = tt_out["tt_pred_x0"]
            # add the scalar in a consistent key to be averaged
            log_dict["tt/loss"] = tt_out["loss_target_teacher"].detach()
        loss_dict = {**gen_loss_dict, **guid_loss_dict}

        # ALSO add a scalar entry so _mb_put can average it
        if tt_out is not None:
            loss_dict["loss_target_teacher"] = float(tt_out["loss_target_teacher"].detach().item())

        self.log_everything(loss_dict, log_dict, generator_grad_norm, guidance_grad_norm, accum)


    def train(self):
        accum = self.accelerator.gradient_accumulation_steps

        for _ in range(self.global_step, self.train_iters):

            self.train_one_step()
            # We just finished one micro-step; did we close an accumulation window?
            did_sync = ((self.step + 1) % max(1, accum) == 0)

            if did_sync and self.accelerator.is_main_process:
                global_step = (self.step + 1) // max(1, accum)

                if (not self.no_save) and global_step % self.log_iters == 0:
                    # train_one_step() already computed/used self.global_step, but ensure consistency:
                    self.global_step = global_step
                    self.save()

                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    wandb.log({"per iteration time": current_time - self.previous_time}, step=global_step)
                    self.previous_time = current_time

            self.accelerator.wait_for_everyone()
            self.step += 1  # micro-step counter


    def log_everything(self, loss_dict, log_dict, generator_grad_norm, guidance_grad_norm, accum=None):
        # Feed THIS micro-batch into the accumulation buffers
        self._mb_put(log_dict, loss_dict)

        # ===== Logging once per optimizer step =====
        if self.accelerator.sync_gradients and (self.global_step % self.wandb_iters == 0):
            # IMPORTANT: do aggregation/gather on ALL ranks to avoid deadlock
            batched, scalar_means = self._mb_concat_and_gather()

            if self.accelerator.is_main_process:
                with torch.no_grad():
                    # Safe accessor: returns gathered tensor if available, else current-batch, else None
                    def agg_or_none(key):
                        v = batched.get(key, None)
                        if v is not None:
                            return v
                        return log_dict.get(key, None)

                    # -------- tensors (aggregated if available) --------
                    generated_image         = agg_or_none('generated_image')                 # [-1,1]
                    dmtrain_noisy_latents   = agg_or_none('dmtrain_noisy_latents')          # [-1,1] or [0,1]
                    dmtrain_pred_real_image = agg_or_none('dmtrain_pred_real_image')        # may be None when both teachers active
                    dmtrain_pred_fake_image = agg_or_none('dmtrain_pred_fake_image')        # [-1,1]
                    dmtrain_grad            = agg_or_none('dmtrain_grad')                   # may be None when both teachers active
                    dmtrain_timesteps       = agg_or_none('dmtrain_timesteps')              # [B]
                    faketrain_latents       = agg_or_none('faketrain_latents')
                    faketrain_noisy_latents = agg_or_none('faketrain_noisy_latents')
                    faketrain_x0_pred       = agg_or_none('faketrain_x0_pred')

                    # New per-teacher logs
                    src_pred = agg_or_none('dmtrain_pred_real_image_source')
                    tgt_pred = agg_or_none('dmtrain_pred_real_image_target')
                    src_grad = agg_or_none('dmtrain_grad_source')
                    tgt_grad = agg_or_none('dmtrain_grad_target')

                    # TT preview (already gathered)
                    tt_pred_x0 = agg_or_none('tt_pred_x0')

                    # Build a single payload
                    data_dict = {}

                    # ---- per-teacher previews ----
                    if src_pred is not None:
                        src_pred_grid = prepare_images_for_saving(src_pred, resolution=self.resolution)
                        data_dict["dm/src_pred_grid"] = wandb.Image(src_pred_grid)
                    if tgt_pred is not None:
                        tgt_pred_grid = prepare_images_for_saving(tgt_pred, resolution=self.resolution)
                        data_dict["dm/tgt_pred_grid"] = wandb.Image(tgt_pred_grid)

                    if src_grad is not None:
                        eps = 1e-12
                        gmin, gmax = src_grad.min(), src_grad.max()
                        src_grad_viz = (src_grad - gmin) / (gmax - gmin + eps)
                        src_grad_viz = (src_grad_viz - 0.5) / 0.5
                        data_dict["dm/src_grad_grid"] = wandb.Image(prepare_images_for_saving(src_grad_viz, resolution=self.resolution))
                    if tgt_grad is not None:
                        eps = 1e-12
                        gmin, gmax = tgt_grad.min(), tgt_grad.max()
                        tgt_grad_viz = (tgt_grad - gmin) / (gmax - gmin + eps)
                        tgt_grad_viz = (tgt_grad_viz - 0.5) / 0.5
                        data_dict["dm/tgt_grad_grid"] = wandb.Image(prepare_images_for_saving(tgt_grad_viz, resolution=self.resolution))

                    if tt_pred_x0 is not None:
                        tt_pred_grid = prepare_images_for_saving(tt_pred_x0, resolution=self.resolution)
                        data_dict["tt/preview_pred"] = wandb.Image(tt_pred_grid)

                    # -------- visuals & simple stats (legacy panels guarded) --------
                    if generated_image is not None:
                        gen_img_vis = (generated_image * 0.5 + 0.5).clamp(0, 1)
                        generated_image_brightness = float(gen_img_vis.mean())
                        generated_image_std        = float(gen_img_vis.std())
                        generated_image_grid       = prepare_images_for_saving(generated_image, resolution=self.resolution)
                        data_dict["generated_image"]            = wandb.Image(generated_image_grid)
                        data_dict["generated_image_brightness"] = generated_image_brightness
                        data_dict["generated_image_std"]        = generated_image_std

                    # pick a reference for legacy plots (prefer source → target → legacy)
                    def first_non_none(*xs):
                        for x in xs:
                            if x is not None:
                                return x
                        return None

                    dm_ref_real = first_non_none(dmtrain_pred_real_image, src_pred, tgt_pred)
                    dm_ref_grad = first_non_none(dmtrain_grad,            src_grad, tgt_grad)
                    dm_fake     = dmtrain_pred_fake_image

                    if dm_ref_grad is not None:
                        eps = 1e-12
                        gmin, gmax = dm_ref_grad.min(), dm_ref_grad.max()
                        grad_norm = (dm_ref_grad - gmin) / (gmax - gmin + eps)
                        grad_norm = (grad_norm - 0.5) / 0.5
                        data_dict["gradient"] = wandb.Image(prepare_images_for_saving(grad_norm, resolution=self.resolution))
                        data_dict["gradient_brightness"] = float(dm_ref_grad.mean())
                        data_dict["gradient_std"]        = float(dm_ref_grad.std(dim=[1, 2, 3]).mean())

                    if dmtrain_noisy_latents is not None:
                        data_dict["dmtrain_noisy_latents_grid"] = wandb.Image(prepare_images_for_saving(dmtrain_noisy_latents, resolution=self.resolution))

                    if dm_ref_real is not None:
                        dm_ref_real_grid = prepare_images_for_saving(dm_ref_real, resolution=self.resolution)
                        data_dict["dmtrain_pred_real_image_grid"] = wandb.Image(dm_ref_real_grid)
                        data_dict["dmtrain_pred_real_image_mean"] = float(((dm_ref_real*0.5+0.5).clamp(0,1)).mean())
                        data_dict["dmtrain_pred_real_image_std"]  = float(((dm_ref_real*0.5+0.5).clamp(0,1)).std())

                    if dm_fake is not None:
                        dm_fake_grid = prepare_images_for_saving(dm_fake, resolution=self.resolution)
                        data_dict["dmtrain_pred_fake_image_grid"] = wandb.Image(dm_fake_grid)
                        data_dict["dmtrain_pred_fake_image_mean"] = float(((dm_fake*0.5+0.5).clamp(0,1)).mean())
                        data_dict["dmtrain_pred_fake_image_std"]  = float(((dm_fake*0.5+0.5).clamp(0,1)).std())

                    if (dm_ref_real is not None) and (dm_fake is not None):
                        diff = dm_fake - dm_ref_real
                        eps = 1e-12
                        dmin, dmax = diff.min(), diff.max()
                        diff_vis = (diff - dmin) / (dmax - dmin + eps)
                        diff_vis = (diff_vis - 0.5) / 0.5
                        data_dict["difference"]            = wandb.Image(prepare_images_for_saving(diff_vis, resolution=self.resolution))
                        data_dict["difference_brightness"] = float(diff.mean())

                    if dmtrain_timesteps is not None:
                        dmtrain_timesteps_grid = draw_valued_array(
                            dmtrain_timesteps.squeeze().detach().cpu().numpy(),
                            output_dir=self.wandb_folder
                        )
                        data_dict["dmtrain_timesteps_grid"] = wandb.Image(dmtrain_timesteps_grid)

                    if dm_ref_grad is not None and dm_ref_real is not None and dm_fake is not None:
                        gradient_scale_grid = draw_valued_array(
                            dm_ref_grad.abs().mean(dim=[1,2,3]).detach().cpu().numpy(),
                            output_dir=self.wandb_folder
                        )
                        difference_scale_grid = draw_valued_array(
                            (dm_ref_real - dm_fake).abs().mean(dim=[1,2,3]).detach().cpu().numpy(),
                            output_dir=self.wandb_folder
                        )
                        data_dict["gradient_scale_grid"]  = wandb.Image(gradient_scale_grid)
                        data_dict["difference_norm_grid"] = wandb.Image(difference_scale_grid)

                    # averaged losses over the accumulation window (fall back to current batch if missing)
                    loss_dm_mean        = float(scalar_means.get('loss_dm',        loss_dict.get('loss_dm', 0.0)))
                    loss_fake_mean_mean = float(scalar_means.get('loss_fake_mean', loss_dict.get('loss_fake_mean', 0.0)))
                    tt_loss_mean        = float(scalar_means.get('loss_target_teacher', loss_dict.get('loss_target_teacher', 0.0)))

                    data_dict["loss_dm"]        = loss_dm_mean
                    data_dict["loss_fake_mean"] = loss_fake_mean_mean
                    data_dict["tt/loss"]        = tt_loss_mean

                    # grad norms coming from the training step
                    gen_gn = float(generator_grad_norm.item() if torch.is_tensor(generator_grad_norm) else generator_grad_norm)
                    gui_gn = float(guidance_grad_norm.item() if torch.is_tensor(guidance_grad_norm) else guidance_grad_norm)
                    data_dict["generator_grad_norm"] = gen_gn
                    data_dict["guidance_grad_norm"]  = gui_gn

                    # ---- also log the fake-train trio (you had these before) ----
                    if faketrain_latents is not None:
                        data_dict["faketrain_latents"] = wandb.Image(prepare_images_for_saving(faketrain_latents, resolution=self.resolution))
                    if faketrain_noisy_latents is not None:
                        data_dict["faketrain_noisy_latents"] = wandb.Image(prepare_images_for_saving(faketrain_noisy_latents, resolution=self.resolution))
                    if faketrain_x0_pred is not None:
                        data_dict["faketrain_x0_pred"] = wandb.Image(prepare_images_for_saving(faketrain_x0_pred, resolution=self.resolution))

                    # ---- GAN extras (with safe CPU numpy) ----
                    if self.gan_classifier:
                        if 'guidance_cls_loss' in scalar_means:
                            data_dict['guidance_cls_loss'] = float(scalar_means['guidance_cls_loss'])
                        if 'gen_cls_loss' in scalar_means:
                            data_dict['gen_cls_loss'] = float(scalar_means['gen_cls_loss'])

                        prf = batched.get("pred_realism_on_fake", None)
                        prr = batched.get("pred_realism_on_real", None)
                        if (prf is not None) and (prr is not None):
                            prf_np = prf.detach().flatten().float().cpu().numpy()
                            prr_np = prr.detach().flatten().float().cpu().numpy()
                            data_dict["hist/pred_realism_on_fake"] = wandb.Histogram(prf_np, num_bins=50)
                            data_dict["hist/pred_realism_on_real"] = wandb.Histogram(prr_np, num_bins=50)
                            data_dict["pred_realism_on_fake_mean"] = float(prf_np.mean())
                            data_dict["pred_realism_on_real_mean"] = float(prr_np.mean())

                            hist_fake_img = draw_probability_histogram(prf_np)
                            hist_real_img = draw_probability_histogram(prr_np)
                            data_dict["hist_img/pred_realism_on_fake"] = wandb.Image(hist_fake_img)
                            data_dict["hist_img/pred_realism_on_real"] = wandb.Image(hist_real_img)

                        cf = batched.get("critic_fake", None)
                        cr = batched.get("critic_real", None)
                        if (cf is not None) and (cr is not None):
                            cf_sig_np = torch.sigmoid(cf).detach().flatten().cpu().numpy()
                            cr_sig_np = torch.sigmoid(cr).detach().flatten().cpu().numpy()
                            data_dict["hist/critic_fake_sigmoid"] = wandb.Histogram(cf_sig_np, num_bins=50)
                            data_dict["hist/critic_real_sigmoid"] = wandb.Histogram(cr_sig_np, num_bins=50)
                            if 'wgan_gp' in scalar_means:
                                data_dict['wgan_gp'] = float(scalar_means['wgan_gp'])

                    # Effective batch and step
                    data_dict["effective_batch_size"] = int(self.batch_size * max(1, accum) * self.accelerator.num_processes)
                    data_dict["optimizer_step"]       = int(self.global_step)

                    # Log once
                    wandb.log(data_dict, step=self.global_step)

            # either way, after sync we’re ready for the next window
            if self.accelerator.sync_gradients:
                self._mb_clear()
        self.accelerator.wait_for_everyone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--train_iters", type=int, default=1000000)
    parser.add_argument("--log_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--initialie_generator", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_iters", type=int, default=100)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--label_dim", type=int, default=0)
    parser.add_argument("--warmup_step", type=int, default=500, help="warmup step for network")
    parser.add_argument("--min_step_percent", type=float, default=0.02, help="minimum step percent for training")
    parser.add_argument("--max_step_percent", type=float, default=0.98, help="maximum step percent for training")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--conditioning_sigma", type=float, default=80.0)

    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--dataset_name", type=str, default='ffhq')
    parser.add_argument("--ckpt_only_path", type=str, default=None, help="checkpoint (no optimizer state) only path")
    parser.add_argument("--delete_ckpts", action="store_true")
    parser.add_argument("--max_checkpoint", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--max_grad_norm", type=int, default=10)
    parser.add_argument("--real_image_path", type=str)
    parser.add_argument("--generator_lr", type=float)
    parser.add_argument("--guidance_lr", type=float)
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=1)

    parser.add_argument("--cls_loss_weight", type=float, default=1.0)
    parser.add_argument("--gan_classifier", action="store_true")
    parser.add_argument("--gen_cls_loss_weight", type=float, default=0)
    parser.add_argument("--diffusion_gan", action="store_true")
    parser.add_argument("--diffusion_gan_max_timestep", type=int, default=0)

    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--generator_ckpt_path", type=str)

    # ---------------------- my additions -----------------------
    parser.add_argument("--dmd_loss_weight", type=float, default=1, 
        help="DMD loss weight, 0 means no DMD loss")
    parser.add_argument('--gan_multihead', action='store_true',
        help='Enable multi-scale discriminator heads (Sushko §3.2).')
    parser.add_argument('--gan_head_type', default='patch', choices=['patch','global'],
        help='patch = 1x1 conv → HxW map; global = 1x1 conv + GAP → scalar.')
    parser.add_argument('--gan_head_layers', type=str, default='all',
        help='Comma-separated layer names/indices to attach heads, or "all".')

    # GAN stuff
    parser.add_argument('--gan_adv_loss', default='hinge',
    choices=['hinge','bce','wgan','lsgan'],
    help='Adversarial loss.')
    parser.add_argument('--hinge_margin', type=float, default=1.0,
        help='Margin m in hinge loss (max(0, m - D(x)), max(0, m + D(G(z))).')
    parser.add_argument('--bce_smooth', type=float, default=0.0,
        help='Label smoothing ε for BCE (real=1-ε, fake/gen=ε).')
    parser.add_argument('--ls_target_real', type=float, default=1.0)
    parser.add_argument('--ls_target_fake', type=float, default=0.0)
    parser.add_argument('--ls_target_gen',  type=float, default=1.0)
    parser.add_argument('--r1_gamma', type=float, default=0.0,
        help='R1 regularization weight (applies to non-WGAN losses).')
    parser.add_argument('--wgan_gp_lambda', type=float, default=10.0,
        help='>0 enables WGAN-GP with this lambda (e.g., 10.0).')
    parser.add_argument("--grad_accum_steps", type=int, default=1,
        help="Gradient accumulation steps for larger effective batch sizes.")
    parser.add_argument("--denoising", action="store_true",
        help="Enable few-step K-step generator unroll")
    parser.add_argument("--num_denoising_step", type=int, default=1,
        help="K steps for the generator (e.g., 2–4)")
    parser.add_argument("--denoising_sigma_end", type=float, default=0.5,
        help="Final (smallest) sigma for the unroll")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--label_dropout_p", type=float, default=0.30, 
        help="Probability to drop labels for the entire micro-batch (CFG-style).")
    parser.add_argument("--train_fake_on_real", action="store_true")
    parser.add_argument("--reverse_dmd", action="store_true",)

    # ---------------- Adding target teacher --------------------
    parser.add_argument("--use_source_teacher", type=float, default=1.0,
        help="Use a source teacher model for DMD loss")
    parser.add_argument("--use_target_teacher", type=float, default=0.0,)
    parser.add_argument("--train_target_teacher", type=float, default=1.0,
        help="Train the target teacher model along with the main model")
    parser.add_argument(
        "--tt_match_guidance",
        action="store_true",
        help="If set, update Target Teacher every optimizer step (same cadence as guidance). "
            "If not set, TT updates only on generator steps (gated by dfake_gen_update_ratio)."
    )
    parser.add_argument("--dmd_source_weight", type=float, default=1.0,
        help="Weight for DMD w.r.t. SOURCE (frozen) teacher.")
    parser.add_argument("--dmd_target_weight", type=float, default=1.0,
        help="Weight for DMD w.r.t. TARGET (trainable) teacher.")
    parser.add_argument("--target_teacher_ckpt_path", type=str, default=None,
        help="Path to a .pt file to initialize the Target Teacher (overrides --model_id for TT only).")
    # -----------------------------------------------------------

    args = parser.parse_args()

    if args.checkpoint_path == "":
        print("No checkpoint path specified, training from scratch.")
        args.checkpoint_path = None
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    assert args.wandb_iters % args.dfake_gen_update_ratio == 0, "wandb_iters should be a multiple of dfake_gen_update_ratio"

    return args 

if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)
    trainer.train()
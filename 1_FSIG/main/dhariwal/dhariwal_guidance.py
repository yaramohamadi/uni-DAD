"""
Guidance module for Dhariwal-style EDM generators with DMD2-style distillation
and GAN losses.

This file defines:
  - dhariwalGuidance: a wrapper around one or more Dhariwal/ADM UNets that
    provides:
      * Distribution Matching Distillation (DMD) losses using:
          - a frozen source teacher (real_unet)
          - an optional target teacher (target_unet)
          - a trainable student/guidance network (fake_unet)
      * GAN-based realism supervision via:
          - a single bottleneck classifier head (Dhariwal & Nichol style), or
          - optional multi-head patch / global discriminators over multiple
            UNet feature scales.
      * Support for different adversarial losses:
          - WGAN(-GP), hinge, LSGAN, and BCE with label smoothing.
      * Optional R1 and WGAN-GP regularization.

  - Karras sigma scheduling (get_sigmas_karras) and Karras-style EDM weighting
    for training losses.
  - Utility helpers for critic scores, gradient penalties, and multi-scale
    feature selection for the GAN heads.

The module is used by dhariwalUniModel to provide the guidance / discriminator
side in the unified generator–discriminator training loop.
"""

from main.dhariwal.dhariwal_network import get_edm_network, load_pt_with_logs
import torch.nn.functional as F
import torch.nn as nn
import dnnlib 
import pickle 
import torch
import copy 
from main.dhariwal.dhariwal_network import _map_sigma_to_t, _onehot_to_class_index
from typing import Tuple
from typing import Optional
import os

# utils
def _avg_spatial(x):
    return x.mean(dim=(2,3), keepdim=False) if x.ndim == 4 else x  # [B,1,H,W]→[B,1]

def _gan_losses(
    logits_real, logits_fake, 
    mode='hinge', 
    hinge_margin: float = 1.0,
    bce_smooth: float = 0.0,
    ls_targets: Tuple[float,float,float] = (1.0, 0.0, 1.0)  # (real, fake, gen)
):
    # Flatten so it works for [B,1], [B,HW], etc.
    logits_fake = logits_fake.view(logits_fake.size(0), -1)
    logits_real_flat = None if logits_real is None else logits_real.view(logits_real.size(0), -1)

    if mode == 'wgan':
        d_loss_real = 0.0 if logits_real_flat is None else -logits_real_flat.mean()
        d_loss_fake = logits_fake.mean()
        d_loss = d_loss_real + d_loss_fake
        g_loss = -logits_fake.mean()
        return d_loss, g_loss

    if mode == 'hinge':
        m = hinge_margin
        d_loss_real = 0.0 if logits_real_flat is None else torch.relu(m - logits_real_flat).mean()
        d_loss_fake = torch.relu(m + logits_fake).mean()
        d_loss = d_loss_real + d_loss_fake
        g_loss = (-logits_fake).mean()
        return d_loss, g_loss

    if mode == 'lsgan':
        # Least-squares GAN: real→a, fake→b, gen→c
        a, b, c = ls_targets
        mse = nn.MSELoss()
        if logits_real_flat is None:
            d_loss = mse(torch.sigmoid(logits_fake), torch.full_like(logits_fake, b))
        else:
            d_loss = mse(torch.sigmoid(logits_real_flat), torch.full_like(logits_real_flat, a)) \
                   + mse(torch.sigmoid(logits_fake),      torch.full_like(logits_fake,      b))
        g_loss = mse(torch.sigmoid(logits_fake), torch.full_like(logits_fake, c))
        return d_loss, g_loss

    # 'bce'
    bce = nn.BCEWithLogitsLoss()
    eps = bce_smooth
    if logits_real_flat is None:
        d_loss = bce(logits_fake, torch.zeros_like(logits_fake))
    else:
        d_loss = bce(logits_real_flat, torch.full_like(logits_real_flat, 1.0 - eps)) \
               + bce(logits_fake,      torch.full_like(logits_fake,      0.0 + eps))
    g_loss = bce(logits_fake, torch.full_like(logits_fake, 1.0 - eps))
    return d_loss, g_loss


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0):
    # from https://github.com/crowsonkb/k-diffusion
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas
#------------------------------------------------------------


class dhariwalGuidance(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args 
        self.accelerator = accelerator 

        # GAN multi-head options
        self.gan_multihead = getattr(args, 'gan_multihead', False)
        self.gan_head_type = getattr(args, 'gan_head_type', 'patch')
        self.gan_head_layers = getattr(args, 'gan_head_layers', 'all')
        self.gan_adv_loss = getattr(args, 'gan_adv_loss', 'hinge')
        self.wgan_gp_lambda = getattr(args, 'wgan_gp_lambda', 10.0)  # set >0 to enable GP (e.g., 10.0)

        self.hinge_margin = float(getattr(args, 'hinge_margin', 1.0))
        self.bce_smooth   = float(getattr(args, 'bce_smooth', 0.0))
        self.ls_real      = float(getattr(args, 'ls_target_real', 1.0))
        self.ls_fake      = float(getattr(args, 'ls_target_fake', 0.0))
        self.ls_gen       = float(getattr(args, 'ls_target_gen', 1.0))
        self.r1_gamma     = float(getattr(args, 'r1_gamma', 0.0))

        self.use_source_teacher  = getattr(args, "use_source_teacher", True)
        self.use_target_teacher  = getattr(args, "use_target_teacher", False)
        self.train_target_teacher = getattr(args, "train_target_teacher", False)
        self.train_fake_on_real = getattr(args, 'train_fake_on_real', False)

        self.dmd_source_weight = float(getattr(args, "dmd_source_weight", 1.0))
        self.dmd_target_weight = float(getattr(args, "dmd_target_weight", 1.0))

        if self.train_fake_on_real:
            print("Fake-score training: REAL images")
        else:
            print("Fake-score training: GENERATOR outputs")
                

        # with dnnlib.util.open_url(args.model_id) as f:
        #    temp_edm = pickle.load(f)['ema']

        # initialize the real unet 
        args_teacher = copy.deepcopy(args)
        args_teacher.label_dim = 0  # unconditional teacher (Only for face dataset it's like this)
        self.real_unet = get_edm_network(args_teacher).to(accelerator.device)
        self.real_unet = load_pt_with_logs(self.real_unet, args.model_id)  # load the .pt file here
        # self.real_unet.load_state_dict(temp_edm.state_dict(), strict=True)
        self.real_unet.requires_grad_(False)
        del self.real_unet.model.map_augment
        self.real_unet.model.map_augment = None

        self.fake_unet = get_edm_network(args).to(accelerator.device)      # label_dim = K+1 for conditional or 0 for unconditional
        # Load the SAME teacher weights; label-embedding layers will be missing -> filled randomly (forgiving load).
        self.fake_unet = load_pt_with_logs(self.fake_unet, args.model_id)
        self.fake_unet.requires_grad_(True)

        param = next(self.fake_unet.parameters())

        if self.use_target_teacher:
            args_ttar = copy.deepcopy(args)
            self.target_unet = get_edm_network(args_ttar).to(accelerator.device)

            # 1) Init from the same base as source teacher (current behavior)
            self.target_unet = load_pt_with_logs(self.target_unet, args.model_id)

            # 2) If a finetuned TT checkpoint is provided, load it on top (new behavior)
            tt_ckpt = getattr(args, "target_teacher_ckpt_path", None)
            if tt_ckpt is not None and os.path.exists(tt_ckpt):
                print(f"[TargetTeacher] Loading finetuned weights from: {tt_ckpt}")
                self.target_unet = load_pt_with_logs(self.target_unet, tt_ckpt)
            else:
                if tt_ckpt:
                    print(f"[TargetTeacher] WARNING: target_teacher_ckpt_path not found: {tt_ckpt} (using {args.model_id})")

            # 3) Freeze or unfreeze based on train_target_teacher
            self.target_unet.requires_grad_(bool(self.train_target_teacher))

            # 4) Match the map_augment cleanup only when frozen (like real_unet)
            if self.train_target_teacher == False:
                try:
                    del self.target_unet.model.map_augment
                    self.target_unet.model.map_augment = None
                except Exception:
                    pass
        else:
            self.target_unet = None

        # some training hyper-parameters 
        self.sigma_data = args.sigma_data
        self.sigma_max = args.sigma_max
        self.sigma_min = args.sigma_min
        self.rho = args.rho

        self.reverse_dmd = args.reverse_dmd

        self.gan_classifier = args.gan_classifier
        self.diffusion_gan = args.diffusion_gan 
        self.diffusion_gan_max_timestep = args.diffusion_gan_max_timestep

        

        
        # Figure out bottleneck channels dynamically (works for 256x256 ADM)
        with torch.no_grad():
            dummy_x = torch.zeros(1, 3, args.resolution, args.resolution, device=accelerator.device)
            dummy_sigma = torch.ones(1, device=accelerator.device) * self.sigma_min  # any valid sigma
            # unconditional => label None; conditional => pass a 1-hot of shape [1, label_dim]
            dummy_label = None if args.label_dim == 0 else torch.zeros(1, args.label_dim, device=accelerator.device)

            feat = self.fake_unet(dummy_x, dummy_sigma, dummy_label, return_bottleneck=True)

            bottleneck_c = feat.shape[1]

        # Initialize head for single-head GAN (Dhariwal & Nichol)
        if self.gan_classifier and not self.gan_multihead:
            # ----- ORIGINAL single bottleneck head (unchanged) -----
            self.cls_pred_branch = nn.Sequential(
                nn.Conv2d(kernel_size=4, in_channels=bottleneck_c, out_channels=bottleneck_c, stride=2, padding=1),  # 8x8 -> 4x4
                nn.GroupNorm(num_groups=32, num_channels=bottleneck_c),
                nn.SiLU(),
                nn.Conv2d(kernel_size=4, in_channels=bottleneck_c, out_channels=bottleneck_c, stride=4, padding=0),  # 4x4 -> 1x1
                nn.GroupNorm(num_groups=32, num_channels=bottleneck_c),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(kernel_size=1, in_channels=bottleneck_c, out_channels=1, stride=1, padding=0),
            )
            self.cls_pred_branch.requires_grad_(True)

        # Initialize heads for multi-head GAN (Sushko §3.2)
        elif self.gan_classifier and self.gan_multihead:
            # ----- MULTI-HEAD init with selection spec -----
            with torch.no_grad():
                dummy_x = torch.zeros(1, 3, args.resolution, args.resolution, device=accelerator.device)
                dummy_sigma = torch.ones(1, device=accelerator.device) * self.sigma_min

                # Map EDM → DDPM state for UNet hook pass
                cfac = 1.0 / torch.sqrt(1.0 + dummy_sigma.view(1,1,1,1)**2)
                x_t = cfac * dummy_x
                t0 = torch.zeros(1, dtype=torch.long, device=accelerator.device)
                y0 = None if args.label_dim == 0 else torch.zeros(1, args.label_dim, device=accelerator.device)
                y = _onehot_to_class_index(y0)

                # Get *all* feats once to know names and channels
                feats_all = self.fake_unet.extract_multi_scale_features(x_t, t0, y, 'all')

                # Resolve which ones to actually use based on args.gan_head_layers
                # Examples of accepted values:
                #   'all', 'last', 'last_k:3', 'last_pct:0.7', ['enc_3','mid','dec_2'], [8,9,10]
                selected_names = self._resolve_head_names(feats_all, self.gan_head_layers)
                self.gan_selected_names = selected_names  # keep for later

            heads = nn.ModuleDict()
            for name in selected_names:
                ch = feats_all[name].shape[1]
                if self.gan_head_type == 'patch':
                    heads[name] = nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0)
                else:  # 'global'
                    heads[name] = nn.Sequential(
                        nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0),
                        nn.AdaptiveAvgPool2d(1),
                    )
            self.multi_heads = heads.to(accelerator.device)
            self.multi_heads.requires_grad_(True)
            # ------------------------------------------------
            
        self.num_train_timesteps = args.num_train_timesteps  
        # small sigma first, large sigma later
        karras_sigmas = torch.flip(
            get_sigmas_karras(self.num_train_timesteps, sigma_max=self.sigma_max, sigma_min=self.sigma_min, 
                rho=self.rho
            ),
            dims=[0]
        )    
        self.register_buffer("karras_sigmas", karras_sigmas)

        self.min_step = int(args.min_step_percent * self.num_train_timesteps)
        self.max_step = int(args.max_step_percent * self.num_train_timesteps)
        # del temp_edm

    # For some GAN-heads we need R1 (Hinge/BCE/LSGAN)
    def _r1_penalty(self, real_img, real_label):
        if self.r1_gamma <= 0.0:
            return torch.tensor(0.0, device=real_img.device, dtype=real_img.dtype)
        real_img.requires_grad_(True)
        scores_real = self._critic_score(real_img, real_label)  # [B]
        grad_real = torch.autograd.grad(
            outputs=scores_real.sum(),
            inputs=real_img,
            create_graph=True, retain_graph=True
        )[0]
        penalty = (grad_real.view(grad_real.size(0), -1).pow(2).sum(dim=1)).mean()
        return 0.5 * self.r1_gamma * penalty


    def compute_distribution_matching_loss(self, latents, labels):
        """
        DMD loss for the GENERATOR turn.
        - Grad MUST flow to `latents` (-> generator).
        - Teachers are frozen (no grad).
        - Guidance (fake_unet) is detached so generator step never backprops into it.
        """
        original_latents = latents                     # keep grad
        B = latents.shape[0]

        # Sample noise/sigmas WITHOUT disabling grad on latents
        timesteps = torch.randint(
            self.min_step, min(self.max_step + 1, self.num_train_timesteps),
            (B, 1, 1, 1), device=latents.device, dtype=torch.long
        )
        timestep_sigma = self.karras_sigmas[timesteps]
        noise = torch.randn_like(latents)
        noisy_latents = latents + timestep_sigma.reshape(-1, 1, 1, 1) * noise  # grad to latents

        # --- student (fake score) prediction: detach for generator stability
        pred_fake_image = self.fake_unet(noisy_latents, timestep_sigma, labels).detach()

        # --- teacher predictions: run under no_grad (frozen)
        pred_src = None
        pred_tgt = None
        if self.use_source_teacher:
            with torch.no_grad():
                pred_src = self.real_unet(noisy_latents, timestep_sigma, None)
        if self.use_target_teacher and (self.target_unet is not None):
            with torch.no_grad():
                pred_tgt = self.target_unet(noisy_latents, timestep_sigma, labels)

        # Per-teacher DMD pieces (grad flows only via `latents`)
        losses = []
        log_extras = {}

        if (pred_src is not None) and (self.dmd_source_weight != 0.0):
            p_real_s = (latents - pred_src)
            p_fake_s = (latents - pred_fake_image)
            w_s = torch.abs(p_real_s).mean(dim=[1,2,3], keepdim=True).clamp_min(1e-8)
            grad_s = (p_real_s - p_fake_s) / w_s if not self.reverse_dmd else (p_fake_s - p_real_s) / w_s
            grad_s = torch.nan_to_num(grad_s)
            loss_s = 0.5 * F.mse_loss(original_latents, (original_latents - grad_s).detach(), reduction="mean")
            losses.append(self.dmd_source_weight * loss_s)
            log_extras["dmtrain_pred_real_image_source"] = pred_src.detach()
            log_extras["dmtrain_grad_source"]            = grad_s.detach()

        if (pred_tgt is not None): #  and (self.dmd_target_weight != 0.0)
            p_real_t = (latents - pred_tgt)
            p_fake_t = (latents - pred_fake_image)
            w_t = torch.abs(p_real_t).mean(dim=[1,2,3], keepdim=True).clamp_min(1e-8)
            grad_t = (p_real_t - p_fake_t) / w_t if not self.reverse_dmd else (p_fake_t - p_real_t) / w_t
            grad_t = torch.nan_to_num(grad_t)
            loss_t = 0.5 * F.mse_loss(original_latents, (original_latents - grad_t).detach(), reduction="mean")
            losses.append(self.dmd_target_weight * loss_t)
            log_extras["dmtrain_pred_real_image_target"] = pred_tgt.detach()
            log_extras["dmtrain_grad_target"]            = grad_t.detach()

        if len(losses) == 0:

            raise RuntimeError("DMD requested, but neither teacher active or both weights are 0.")

        total_dmd = sum(losses)  # <-- this requires grad via `original_latents`

        loss_dict = {"loss_dm": total_dmd}
        dm_log_dict = {
            "dmtrain_noisy_latents": noisy_latents.detach(),
            "dmtrain_pred_fake_image": pred_fake_image,   # already detached
            "dmtrain_timesteps": timesteps.detach(),
        }
        dm_log_dict.update(log_extras)

        # Fill legacy single-teacher keys only when exactly one teacher is active
        only_src = ("dmtrain_pred_real_image_source" in dm_log_dict) and ("dmtrain_pred_real_image_target" not in dm_log_dict)
        only_tgt = ("dmtrain_pred_real_image_target" in dm_log_dict) and ("dmtrain_pred_real_image_source" not in dm_log_dict)
        if only_src:
            dm_log_dict["dmtrain_pred_real_image"] = dm_log_dict["dmtrain_pred_real_image_source"]
            dm_log_dict["dmtrain_grad"]            = dm_log_dict["dmtrain_grad_source"]
        elif only_tgt:
            dm_log_dict["dmtrain_pred_real_image"] = dm_log_dict["dmtrain_pred_real_image_target"]
            dm_log_dict["dmtrain_grad"]            = dm_log_dict["dmtrain_grad_target"]

        return loss_dict, dm_log_dict


    def compute_loss_fake(
        self,
        latents,
        labels,
    ):
        batch_size = latents.shape[0]

        latents = latents.detach() # no gradient to generator 
    
        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            [batch_size, 1, 1, 1], 
            device=latents.device,
            dtype=torch.long
        )
        timestep_sigma = self.karras_sigmas[timesteps]
        noisy_latents = latents + timestep_sigma.reshape(-1, 1, 1, 1) * noise

        fake_x0_pred = self.fake_unet(
            noisy_latents, timestep_sigma, labels
        )

        snrs = timestep_sigma**-2

        # weight_schedule karras 
        weights = snrs + 1.0 / self.sigma_data**2

        target = latents 

        loss_fake = torch.mean(
            weights * (fake_x0_pred - target)**2
        )

        loss_dict = {
            "loss_fake_mean": loss_fake
        }

        fake_log_dict = {
            "faketrain_latents": latents.detach(),
            "faketrain_noisy_latents": noisy_latents.detach(),
            "faketrain_x0_pred": fake_x0_pred.detach()
        }
        return loss_dict, fake_log_dict


    def compute_loss_target_teacher(
        self,
        real_image: torch.Tensor,
        real_label: Optional[torch.Tensor],
        maybe_fake_image: Optional[torch.Tensor] = None,
        train_on_real: bool = True,
        train_on_fake: bool = False,
    ):
        """
        Standard EDM-style denoising loss for the Target Teacher (TT).
        Mirrors compute_loss_fake, but uses self.target_unet and real images by default.
        """
        assert self.target_unet is not None, "Target teacher is not initialized"

        losses = []

        if train_on_real:
            latents = real_image.detach()
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, self.num_train_timesteps,
                [latents.shape[0], 1, 1, 1],
                device=latents.device, dtype=torch.long
            )
            timestep_sigma = self.karras_sigmas[timesteps]
            noisy_latents = latents + timestep_sigma.reshape(-1,1,1,1) * noise
            x0_pred = self.target_unet(noisy_latents, timestep_sigma, real_label)
            snrs = timestep_sigma**-2
            weights = snrs + 1.0 / self.sigma_data**2
            losses.append(torch.mean(weights * (x0_pred - latents) ** 2))

            tt_pred_x0 = x0_pred.detach()

        if train_on_fake and (maybe_fake_image is not None):
            latents_f = maybe_fake_image.detach()
            noise_f = torch.randn_like(latents_f)
            t_f = torch.randint(
                0, self.num_train_timesteps,
                [latents_f.shape[0], 1, 1, 1],
                device=latents_f.device, dtype=torch.long
            )
            sigma_f = self.karras_sigmas[t_f]
            noisy_latents_f = latents_f + sigma_f.reshape(-1,1,1,1) * noise_f
            x0_pred_f = self.target_unet(noisy_latents_f, sigma_f, real_label)
            snrs_f = sigma_f**-2
            weights_f = snrs_f + 1.0 / self.sigma_data**2
            losses.append(torch.mean(weights_f * (x0_pred_f - latents_f) ** 2))

            tt_pred_x0 = x0_pred_f.detach()

        total = sum(losses) / max(1, len(losses))
        return {"loss_target_teacher": total,
                "tt_pred_x0": tt_pred_x0
                }



    def compute_cls_logits(self, image, label):
        if self.diffusion_gan:
            timesteps = torch.randint(
                0, self.diffusion_gan_max_timestep, [image.shape[0]], device=image.device, dtype=torch.long
            )
            timestep_sigma = self.karras_sigmas[timesteps]
            image = image + timestep_sigma.reshape(-1, 1, 1, 1) * torch.randn_like(image)
        else:
            timesteps = torch.zeros([image.shape[0]], dtype=torch.long, device=image.device)
            timestep_sigma = self.karras_sigmas[timesteps]

        rep = self.fake_unet(
            image, timestep_sigma, label, return_bottleneck=True
        ).float() 

        logits = self.cls_pred_branch(rep).squeeze(dim=[2, 3])
        return logits

    # ---------- feature extraction hooks (for multi-head GAN) ----------
    def _resolve_head_names(self, example_feats, spec):
        """
        Resolve which multi-scale feature keys should have GAN heads.
        Supports:
        - 'all'
        - 'last' (== last_k:1)
        - 'last_k:N' (e.g., 'last_k:3')
        - 'last_pct:P' (P in (0,1], e.g., 'last_pct:0.7')
        - list[str] of exact names in example_feats.keys()
        - list[int] indices into list(example_feats.keys()) (ordered)
        """
        keys = list(example_feats.keys())  # ordered
        n = len(keys)

        # Direct passthrough if user already provided a list of names
        if isinstance(spec, (list, tuple)):
            # allow indices or names
            selected = []
            for item in spec:
                if isinstance(item, int):
                    idx = item % n
                    selected.append(keys[idx])
                else:
                    if item not in example_feats:
                        raise ValueError(f"Unknown feature name '{item}'. Available: {keys}")
                    selected.append(item)
            return selected

        if not isinstance(spec, str):
            raise ValueError(f"gan_head_layers must be str or list, got {type(spec)}")

        spec = spec.strip().lower()
        if spec == 'all':
            return keys
        if spec == 'last':
            return keys[-1:]

        if spec.startswith('last_k:'):
            k = int(spec.split(':', 1)[1])
            k = max(1, min(n, k))
            return keys[-k:]

        if spec.startswith('last_pct:'):
            pct = float(spec.split(':', 1)[1])
            pct = max(0.0, min(1.0, pct))
            k = max(1, int(round(n * pct)))
            return keys[-k:]

        # Back-compat: if user passed a single exact key name
        if spec in example_feats:
            return [spec]

        raise ValueError(
            f"Unrecognized gan_head_layers spec '{spec}'. "
            f"Use 'all', 'last', 'last_k:<int>', 'last_pct:<float>', "
            f"a list of names, or a list of indices. Available keys: {keys}"
        )

    def _subset_feats(self, feats_dict, selected_names):
        """Return an OrderedDict-like subset preserving order of selected_names."""
        return {k: feats_dict[k] for k in selected_names}


    def _extract_head_features(self, image, label):
        # optional diffusion noise for GAN
        if self.diffusion_gan:
            timesteps = torch.randint(0, self.diffusion_gan_max_timestep, [image.shape[0]],
                                    device=image.device, dtype=torch.long)
        else:
            timesteps = torch.zeros([image.shape[0]], dtype=torch.long, device=image.device)
        timestep_sigma = self.karras_sigmas[timesteps]
        x = image + timestep_sigma.view(-1,1,1,1) * torch.randn_like(image)

        # EDM → DDPM mapping for UNet hooks
        cfac = 1.0 / torch.sqrt(1.0 + timestep_sigma.view(-1,1,1,1)**2)
        x_t = cfac * x
        t = _map_sigma_to_t(timestep_sigma, self.fake_unet.alphas_cumprod)
        y = _onehot_to_class_index(label)

        # Always extract 'all', then subset with the same names we built heads for
        feats_all = self.fake_unet.extract_multi_scale_features(x_t, t, y, 'all')
        feats = self._subset_feats(feats_all, getattr(self, 'gan_selected_names', list(feats_all.keys())))
        return feats


    
    def _critic_score(self, image, label):
        """Unified scalar critic score per sample, works for single-head and multi-head."""
        if self.gan_classifier and self.gan_multihead and getattr(self, 'multi_heads', None) is not None:
            feats = self._extract_head_features(image, label)
            outs = []
            for name, head in self.multi_heads.items():
                out = head(feats[name])      # [B,1,H,W] or [B,1,1,1]
                out = _avg_spatial(out)      # [B,1]
                outs.append(out)
            score = torch.stack(outs, dim=0).mean(dim=0)  # [B,1]
            return score.view(score.size(0))              # [B]
        else:
            score = self.compute_cls_logits(image, label)  # [B,1]
            return score.view(score.size(0))               # [B]


    def _wgan_gradient_penalty(self, real_img, fake_img, label):
        """WGAN-GP (Gulrajani et al.) on interpolates between real and fake."""
        if self.wgan_gp_lambda <= 0.0:
            return torch.tensor(0.0, device=real_img.device, dtype=real_img.dtype)

        batch_size = real_img.size(0)
        eps = torch.rand(batch_size, 1, 1, 1, device=real_img.device, dtype=real_img.dtype)
        interp = eps * real_img + (1 - eps) * fake_img
        interp.requires_grad_(True)

        # critic score per sample
        scores = self._critic_score(interp, label)  # [B]
        grad_outputs = torch.ones_like(scores)

        grads = torch.autograd.grad(
            outputs=scores, inputs=interp,
            grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]  # [B, C, H, W]
        grads = grads.view(grads.size(0), -1)
        gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean() * self.wgan_gp_lambda
        return gp


    def compute_generator_clean_cls_loss(self, fake_image, fake_labels):
        # Get unified critic score for fake
        scores_fake = self._critic_score(fake_image, fake_labels)  # [B]
        if self.gan_adv_loss in ('wgan','hinge'):
            g_loss = (-scores_fake).mean()
            return {"gen_cls_loss": g_loss}
        # bce or lsgan: reuse helper, no real logits for G
        _, g_loss = _gan_losses(
            logits_real=None, logits_fake=scores_fake.view(-1,1),
            mode=self.gan_adv_loss,
            hinge_margin=self.hinge_margin,
            bce_smooth=self.bce_smooth,
            ls_targets=(self.ls_real, self.ls_fake, self.ls_gen)
        )
        return {"gen_cls_loss": g_loss}


    def compute_guidance_clean_cls_loss(self, real_image, fake_image, real_label, fake_label):
        # Get unified critic scores
        scores_real = self._critic_score(real_image.detach(), real_label)  # [B]
        scores_fake = self._critic_score(fake_image.detach(), fake_label)  # [B]

        if self.gan_adv_loss == 'wgan':
            d_loss = (scores_fake - scores_real).mean()
            gp = self._wgan_gradient_penalty(real_image.detach(), fake_image.detach(), real_label)
            d_loss = d_loss + gp
            log_dict = {
                "critic_real": scores_real.detach(),
                "critic_fake": scores_fake.detach(),
                "wgan_gp": torch.as_tensor(gp).detach()
            }
            return {"guidance_cls_loss": d_loss}, log_dict

        # hinge/bce/lsgan path
        d_loss, _ = _gan_losses(
            scores_real.view(-1,1), scores_fake.view(-1,1),
            mode=self.gan_adv_loss,
            hinge_margin=self.hinge_margin,
            bce_smooth=self.bce_smooth,
            ls_targets=(self.ls_real, self.ls_fake, self.ls_gen)
        )
        r1 = torch.tensor(0.0, device=real_image.device)

        log_dict = {
            "pred_realism_on_real": torch.sigmoid(scores_real.view(-1,1)).squeeze(1).detach(),
            "pred_realism_on_fake": torch.sigmoid(scores_fake.view(-1,1)).squeeze(1).detach(),
            "r1": torch.as_tensor(r1).detach(),
        }
        return {"guidance_cls_loss": d_loss}, log_dict


    def generator_forward(
        self,
        image,
        labels
    ):
        loss_dict = {} 
        log_dict = {}

        # image.requires_grad_(True)
        if self.use_target_teacher or self.use_source_teacher: 
            dm_dict, dm_log_dict = self.compute_distribution_matching_loss(image, labels)
            loss_dict.update(dm_dict)
            log_dict.update(dm_log_dict)
        else:
            print("No teachers active for DMD loss computation!")

        

        if self.gan_classifier:
            clean_cls_loss_dict = self.compute_generator_clean_cls_loss(image, labels)
            loss_dict.update(clean_cls_loss_dict)

        # loss_dm = loss_dict["loss_dm"]
        # gen_cls_loss = loss_dict["gen_cls_loss"]

        # grad_dm = torch.autograd.grad(loss_dm, image, retain_graph=True)[0]
        # grad_cls = torch.autograd.grad(gen_cls_loss, image, retain_graph=True)[0]

        # print(f"dm {grad_dm.abs().mean()} cls {grad_cls.abs().mean()}")

        return loss_dict, log_dict 

    def guidance_forward(
        self,
        image,
        labels,
        real_train_dict=None
    ):


        if self.train_fake_on_real:
            fake_dict, fake_log_dict = self.compute_loss_fake(
                real_train_dict['real_image'],
                real_train_dict['real_label']
            )
        else:
            fake_dict, fake_log_dict = self.compute_loss_fake(
                image, labels
            )

        loss_dict = fake_dict 
        log_dict = fake_log_dict

        if self.gan_classifier:
            clean_cls_loss_dict, clean_cls_log_dict = self.compute_guidance_clean_cls_loss(
                real_image=real_train_dict['real_image'], 
                fake_image=image,
                real_label=real_train_dict['real_label'],
                fake_label=labels
            )
            loss_dict.update(clean_cls_loss_dict)
            log_dict.update(clean_cls_log_dict)
        return loss_dict, log_dict 

    def forward(
        self,
        generator_turn=False,
        guidance_turn=False,
        generator_data_dict=None, 
        guidance_data_dict=None
    ):          
        if generator_turn:
            loss_dict, log_dict = self.generator_forward(
                image=generator_data_dict['image'],
                labels=generator_data_dict['label']
            )
        elif guidance_turn:
            loss_dict, log_dict = self.guidance_forward(
                image=guidance_data_dict['image'],
                labels=guidance_data_dict['label'],
                real_train_dict=guidance_data_dict['real_train_dict']
            ) 
        else:
            raise NotImplementedError 

        return loss_dict, log_dict 
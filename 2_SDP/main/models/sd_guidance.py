from main.utils import get_x0_from_noise
from diffusers import UNet2DConditionModel, DDIMScheduler
from main.models.sd_unet_forward import classify_forward
from main.models.multihead_gan_bce import MultiHeadGlobalBCEGan
import torch.nn.functional as F
import torch.nn as nn
import torch
import types 

# PATCH: robust CFG handling when uncond is None
# PATCH: robust CFG + accept both cfg_scale and guidance_scale
# This module owns the teacher/critic side of the SD pipeline. train_sd.py sends
# generated latents/images here, and this file returns whichever losses belong to
# the current turn: generator DM/GAN terms, guidance denoising terms, or TT loss.
# predict_noise centralizes CFG-style UNet calls so source teacher, fake UNet, and
# target teacher all follow the same conditioning and dtype/device handling rules.
def predict_noise(unet, noisy_latents, text_embeddings, uncond_embedding, timesteps, 
    guidance_scale=1.0, unet_added_conditions=None, uncond_unet_added_conditions=None
):
    # --- ensure inputs match UNet's dtype & device ---
    udev   = next(unet.parameters()).device
    udtype = next(unet.parameters()).dtype

    noisy_latents    = noisy_latents.to(device=udev, dtype=udtype)
    text_embeddings  = text_embeddings.to(device=udev, dtype=udtype)
    if uncond_embedding is not None:
        uncond_embedding = uncond_embedding.to(device=udev, dtype=udtype)
    timesteps        = timesteps.to(device=udev)  # long is fine

    def _cast_cond(d):
        if d is None: return None
        out = {}
        for k, v in d.items():
            # time_ids sometimes are float; text_embeds are float; cast safely
            out[k] = v.to(device=udev, dtype=udtype)
        return out

    unet_added_conditions         = _cast_cond(unet_added_conditions)
    uncond_unet_added_conditions  = _cast_cond(uncond_unet_added_conditions)
    # --------------------------------------------------

    CFG_GUIDANCE = guidance_scale > 1.0
    if CFG_GUIDANCE:
        model_input = torch.cat([noisy_latents] * 2, dim=0)
        embeddings  = torch.cat([uncond_embedding, text_embeddings], dim=0)
        tsteps      = torch.cat([timesteps] * 2, dim=0)

        if unet_added_conditions is not None:
            assert uncond_unet_added_conditions is not None
            condition_input = {k: torch.cat(
                [uncond_unet_added_conditions[k], unet_added_conditions[k]], dim=0
            ) for k in unet_added_conditions.keys()}
        else:
            condition_input = None

        noise_pred = unet(model_input, tsteps, embeddings, added_cond_kwargs=condition_input).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    else:
        noise_pred = unet(
            noisy_latents, timesteps, text_embeddings, added_cond_kwargs=unet_added_conditions
        ).sample

    return noise_pred


class SDGuidance(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args 
        self.accelerator = accelerator
        self.device = accelerator.device

        # There are up to three UNets in play here:
        # - real_unet: frozen source teacher for source-DM,
        # - fake_unet: trainable guidance/critic backbone,
        # - target_teacher_unet: optional online teacher trained on real images.
        # Initialize real unet Source Teacher
        self.real_unet = UNet2DConditionModel.from_pretrained(
            args.model_id,
            subfolder="unet"
        ).float()
        self.real_unet.requires_grad_(False) #frozen
        self.use_source_teacher = bool(getattr(args, "use_source_teacher", True))
        self.dm_weight_source = float(getattr(args, "dm_weight_source", 1.0))
        
        self.gan_alone = args.gan_alone 

        if self.gan_alone or (not self.use_source_teacher):
            self.real_unet = None

        # Keep the frozen source teacher strictly fp32 and guard None
        if self.real_unet is not None:
            with torch.cuda.amp.autocast(enabled=False):
                self.real_unet.to(self.device)
                self.real_unet.eval()
                for p in self.real_unet.parameters():
                    p.requires_grad = False

        # fake_unet is the model updated during the guidance turn. It supplies the
        # fake denoising loss, optional classifier features, and the GAN critic backbone.
        # Initialize Online fake unet - fake teacher 
        self.fake_unet = UNet2DConditionModel.from_pretrained(
            args.model_id,
            subfolder="unet"
        ).float()
        self.fake_unet.requires_grad_(True) #for the dummy pass
        self.fake_unet.to(self.device)   

        # The optional target teacher gives the project a second teacher signal that
        # can track the target instance distribution instead of staying frozen.
        # Initialize Online Target-Teacher UNet (trained on real images) 
        self.enable_target_teacher = bool(getattr(args, "enable_target_teacher", True))
        if self.enable_target_teacher:
            self.target_teacher_unet = UNet2DConditionModel.from_pretrained(
                args.model_id, subfolder="unet"
            ).float()
            self.target_teacher_unet.requires_grad_(True)
        else:
            self.target_teacher_unet = None
        
        self.dm_weight_target = float(getattr(args, "dm_weight_target", 1.0))

        # MultiHeadGlobalBCEGan reuses fake_unet encoder features as a lightweight
        # discriminator instead of introducing a separate image-space backbone.
        # Initialize Multihead GAN 
        self.use_multihead_gan = bool(getattr(args, "multihead_gan", False))
        self.gan_bce_weight_d  = float(getattr(args, "gan_bce_weight_d", 1.0))
        self.gan_bce_weight_g  = float(getattr(args, "gan_bce_weight_g", 1.0))
        self.gan_bce_random_t  = bool(getattr(args, "gan_bce_random_t", False))
        self.gan_bce_timestep  = int(getattr(args, "gan_bce_timestep", 0))
        self.mhgan_freeze_encoder_dstep = bool(getattr(args, "mhgan_freeze_encoder_dstep", False))

        # Initialize one head
        self.diffusion_gan = args.diffusion_gan 
        self.diffusion_gan_max_timestep = args.diffusion_gan_max_timestep
        
        self.cls_on_clean_image = args.cls_on_clean_image 
        self.gen_cls_loss = args.gen_cls_loss 
        
    
        if self.cls_on_clean_image and not self.use_multihead_gan:
            self._classify_forward = types.MethodType(
                classify_forward, self.fake_unet
            )

            if accelerator.is_local_main_process:
                print("Note that we randomly initialized a bunch of parameters. FSDP mode 4 hybrid_shard will have non-synced parameters across nodes which would lead to training problems. The current solution is to save the checkpoint 0 and resume")

            # SDv1.5 
            self.cls_pred_branch = nn.Sequential(
                nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1), # 8x8 -> 4x4 
                nn.GroupNorm(num_groups=32, num_channels=1280),
                nn.SiLU(),
                nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=4, padding=0), # 4x4 -> 1x1
                nn.GroupNorm(num_groups=32, num_channels=1280),
                nn.SiLU(),
                nn.Conv2d(kernel_size=1, in_channels=1280, out_channels=1, stride=1, padding=0), # 1x1 -> 1x1
            )

            self.cls_pred_branch.requires_grad_(True)

        elif self.use_multihead_gan:
            # 1) instantiate the critic on the *trainable* fake_unet
            self.multihead_gan = MultiHeadGlobalBCEGan(self.fake_unet)

            # 2) force a one-time dummy pass to *build heads now* (so optimizer sees them)
            with torch.no_grad():
                # latent size: H = W = resolution/8 for SD (both SD1.5 )
                in_ch = int(getattr(self.fake_unet.config, "in_channels", 4))
                H = W = int(self.args.resolution // 8)

                lat = torch.zeros(1, in_ch, H, W, device=self.device, dtype=self.fake_unet.dtype)
                t0  = torch.zeros(1, dtype=torch.long, device=self.device)

                # cross-attn dim comes from UNet config; token length 77 is fine
                ca_dim = getattr(self.fake_unet.config, "cross_attention_dim", None)
                cond = None if ca_dim is None else torch.zeros(1, 77, int(ca_dim), device=self.device, dtype=self.fake_unet.dtype)

                # SDXL needs added cond kwargs; SD1.5 can pass None
                added = None

                # don’t let this dummy pass create UNet grads
                self.multihead_gan.freeze_encoder(True)
                _ = self.multihead_gan.populate_taps(lat, t0, cond, added)
                _ = self.multihead_gan.score_from_cached()  # builds heads & clears cache
                self.multihead_gan.freeze_encoder(False)

                # ensure critic heads are trainable
                self.multihead_gan.heads.requires_grad_(True)

            #self.multihead_gan = heads.to(accelerator.device)
            #self.multihead_gan.requires_grad_(True)
    
        # somehow FSDP requires at least one network with dense parameters (models from diffuser are lazy initialized so their parameters are empty in fsdp mode)
        #self.dummy_network = DummyNetwork() 
        #self.dummy_network.requires_grad_(False)


        use_fp16 = getattr(args, "use_fp16", False)
        if use_fp16 and self.device.type == "cuda":
            self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            class _NoOpCtx:
                def __enter__(self): return None
                def __exit__(self, *exc): return False
            self.network_context_manager = _NoOpCtx()


        self.scheduler = DDIMScheduler.from_pretrained(
            args.model_id,
            subfolder="scheduler"
        )
        
        alphas_cumprod = self.scheduler.alphas_cumprod
        self.register_buffer(
            "alphas_cumprod",
            alphas_cumprod
        )
        
        self.real_guidance_scale = args.real_guidance_scale 
        self.fake_guidance_scale = args.fake_guidance_scale
        assert self.fake_guidance_scale == 1, "no guidance for fake"

        self.gradient_checkpointing = args.gradient_checkpointing 
        if self.gradient_checkpointing and not (self.cls_on_clean_image and not self.use_multihead_gan):
            self.fake_unet.enable_gradient_checkpointing()

        self.num_train_timesteps = args.num_train_timesteps 
        self.min_step = int(args.min_step_percent * self.scheduler.num_train_timesteps)
        self.max_step = int(args.max_step_percent * self.scheduler.num_train_timesteps)
        



                
    #ensure UNet input is latents 
    # Most losses in this module operate in SD latent space. This helper lets the
    # caller pass either RGB images or already-encoded latents without duplicating VAE logic.
    def _as_latents_for_unet(self, image: torch.Tensor) -> torch.Tensor:
        """
        Accepts pixels [B,3,H,W] in [0,1] or latents [B,4,h,w]; returns latents on fake_unet device/dtype.
        """
        if image.dim() == 4 and image.shape[1] == 3:
            vae = self.vae
            imgs = image.to(device=vae.device, dtype=vae.dtype)
            imgs = imgs * 2.0 - 1.0
            with torch.no_grad():
                latents = vae.encode(imgs).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            return latents.to(device=self.fake_unet.device, dtype=self.fake_unet.dtype)
        # assume already latents
        return image.to(device=self.fake_unet.device, dtype=self.fake_unet.dtype)
    
    # choose timesteps
    def _gan_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.gan_bce_random_t:
            return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device, dtype=torch.long)
        return torch.full((batch_size,), int(self.gan_bce_timestep), device=device, dtype=torch.long)

    # The clean-image classifier path reuses fake_unet encoder features via
    # classify_forward() and then applies a shallow prediction head on the bottleneck.
    def compute_cls_logits(self, image, text_embedding, unet_added_conditions):
        
        # sample can be pixels [B,3,512,512] or latents [B,4,64,64]
        if image.dim() == 4 and image.shape[1] == 3:
            # convert pixel images (0..1) to VAE latent space
            vae = self.vae
            vae_device = next(vae.parameters()).device
            vae_dtype  = next(vae.parameters()).dtype

            imgs = image.to(device=vae_device, dtype=vae_dtype) 
            imgs = imgs * 2.0 - 1.0  # [-1,1] expected by SD VAE

            with torch.no_grad():
                latents = vae.encode(imgs).latent_dist.sample()
                latents = latents * vae.config.scaling_factor  # [B,4,64,64]

            # match UNet dtype/device
            image = latents.to(device=self.fake_unet.device, dtype=self.fake_unet.dtype)

        # we are operating on the VAE latent space, no further normalization needed for now 
        if self.diffusion_gan:
            timesteps = torch.randint(
                0, self.diffusion_gan_max_timestep, [image.shape[0]], device=image.device, dtype=torch.long
            )
            image = self.scheduler.add_noise(image, torch.randn_like(image), timesteps)
        else:
            timesteps = torch.zeros([image.shape[0]], dtype=torch.long, device=image.device)
        
        #classify through UNet feature path
        with self.network_context_manager:
            rep = self._classify_forward(
                image, timesteps, text_embedding,
                added_cond_kwargs=unet_added_conditions,
                classify_mode=True
            )

        # we only use the bottleneck layer 
        rep = rep[-1].float()
        logits = self.cls_pred_branch(rep).squeeze(dim=[2, 3])
        return logits


    # Distribution matching is the generator-side distillation objective. It compares
    # the generator's current x_t/x0 behavior against the source and/or target teacher
    # on the same noisy latents, then turns those gaps into a gradient target on x0.
    def compute_distribution_matching_loss(
        self, 
        latents,
        text_embedding,
        unet_added_conditions=None,
        uncond_embedding=None,
        teacher_unet_added_conditions=None,   
        teacher_uncond_embedding=None,        
        uncond_unet_added_conditions=None,
        teacher_text_embedding=None,         
        ):
        """
        DMD loss for the GENERATOR turn.
        - Grad MUST flow to `latents` (-> generator).
        - Teachers are frozen (no grad).
        - Guidance (fake_unet) is detached so generator step never backprops into it.
        """
    
        original_latents = latents 
        batch_size = latents.shape[0]

        # placeholders created OUTSIDE no_grad so we can use them for loss with grad
        grad_s = None
        grad_t = None
        losses = []  # moved outside no_grad
        log_extras = {}

        uncond_unet_added_conditions = (
            uncond_unet_added_conditions if 'uncond_unet_added_conditions' in locals() else None
        )

        with torch.no_grad():
            # Original behavior: sample t and re-noise original_latents (x0)
            timesteps = torch.randint(
                self.min_step, 
                min(self.max_step+1, self.num_train_timesteps),
                [batch_size], 
                device=latents.device,
                dtype=torch.long
            )
            noise = torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
    
            # run at full precision as autocast and no_grad doesn't work well together 
            # --- Fake unet branch  ---
            pred_fake_noise = predict_noise(
                unet=self.fake_unet,
                noisy_latents=noisy_latents,
                text_embeddings=text_embedding,
                uncond_embedding=uncond_embedding,
                timesteps=timesteps,
                guidance_scale=self.fake_guidance_scale,
                unet_added_conditions=unet_added_conditions,
                uncond_unet_added_conditions=uncond_unet_added_conditions,
            )
                
            pred_fake_image = get_x0_from_noise(
                noisy_latents.double(), pred_fake_noise.double(),
                self.alphas_cumprod.double(), timesteps
            )
            p_fake = (noisy_latents - pred_fake_image)

            log_extras["dmtrain_pred_fake_image"] = pred_fake_image.detach().float()
            
            #pred_real_image_any = None

            # --- Real Unet branch - Source-DM (optional) ---
            if self.use_source_teacher and (self.real_unet is not None):
                src_text = teacher_text_embedding if teacher_text_embedding is not None else text_embedding
                src_added_conds = teacher_unet_added_conditions if teacher_unet_added_conditions is not None else unet_added_conditions
                src_uncond = teacher_uncond_embedding if teacher_uncond_embedding is not None else uncond_embedding

                pred_real_noise = predict_noise(
                    unet=self.real_unet,
                    noisy_latents=noisy_latents,
                    text_embeddings=src_text,
                    uncond_embedding=src_uncond,
                    timesteps=timesteps,
                    guidance_scale=self.real_guidance_scale,
                    unet_added_conditions=src_added_conds,
                    uncond_unet_added_conditions=uncond_unet_added_conditions,
                )
                pred_real_image = get_x0_from_noise(
                    noisy_latents.double(), pred_real_noise.double(),
                    self.alphas_cumprod.double(), timesteps
                )
                log_extras["dmtrain_pred_real_image"] = pred_real_image.detach().float()

                p_real_s = (noisy_latents - pred_real_image)
                w_s = torch.abs(p_real_s).mean(dim=[1, 2, 3], keepdim=True).clamp_min(1e-8)
                grad_s = (p_real_s - p_fake) / w_s
                grad_s = torch.nan_to_num(grad_s)

            # --- Target-Teacher branch (optional) ---
            if getattr(self, "target_teacher_unet", None) is not None:
                pred_tt_noise = predict_noise(
                    unet=self.target_teacher_unet,
                    noisy_latents=noisy_latents,
                    text_embeddings=text_embedding,
                    uncond_embedding=uncond_embedding,
                    timesteps=timesteps,
                    guidance_scale=self.real_guidance_scale,
                    unet_added_conditions=unet_added_conditions,
                    uncond_unet_added_conditions=uncond_unet_added_conditions,
                )
                pred_tt_image = get_x0_from_noise(
                    noisy_latents.double(), pred_tt_noise.double(),
                    self.alphas_cumprod.double(), timesteps
                )
                p_real_t = (noisy_latents - pred_tt_image)
                w_t = torch.abs(p_real_t).mean(dim=[1, 2, 3], keepdim=True).clamp_min(1e-8)
                grad_t = (p_real_t - p_fake) / w_t
                grad_t = torch.nan_to_num(grad_t)

                # LOG: separate eps vs x0, and mark as target teacher
                log_extras["dmtrain_pred_target_image"] = pred_tt_image.detach().float()

            # Magnitudes / gaps (computed safely inside no_grad)
            def _mean_abs(x): return x.abs().mean(dim=[1, 2, 3]).mean().item()
            log_extras["dm/p_fake_mag"] = _mean_abs(p_fake)
            if 'p_real_s' in locals(): log_extras["dm/p_real_s_mag"] = _mean_abs(p_real_s)
            if 'p_real_t' in locals(): log_extras["dm/p_real_t_mag"] = _mean_abs(p_real_t)
            if 'p_real_s' in locals(): log_extras["dm/gap_s_fake"] = _mean_abs(p_real_s - p_fake)
            if 'p_real_t' in locals(): log_extras["dm/gap_t_fake"] = _mean_abs(p_real_t - p_fake)

        # --- compute DM losses OUTSIDE no_grad so grad flows to generator latents ---
       # with torch.autocast(device_type="cuda", enabled=False):

        if grad_s is not None:
            loss_s = 0.5 * F.mse_loss(
                original_latents.float(),
                (original_latents - grad_s).detach().float(),
                reduction="mean",
            )
            losses.append(self.dm_weight_source * loss_s)

        if grad_t is not None:
            loss_t = 0.5 * F.mse_loss(
                original_latents.float(),
                (original_latents - grad_t).detach().float(),
                reduction="mean",
            )
            losses.append(self.dm_weight_target * loss_t)

        total_dmd = torch.stack(losses).sum() if len(losses) > 0 else original_latents.new_zeros(())
        loss_dict = {"loss_dm": total_dmd}

        dm_log_dict = {
            "dmtrain_noisy_latents": noisy_latents.detach().float(),
            "dm_weight_source": float(self.dm_weight_source),
            "dm_weight_target": float(self.dm_weight_target),
        }
        dm_log_dict.update(log_extras)
        return loss_dict, dm_log_dict

    # This is the guidance-side denoising objective for fake_unet itself: predict the
    # epsilon used to corrupt generated latents at random timesteps.
    def compute_loss_fake(
        self,
        latents,
        text_embedding,
        uncond_embedding,
        unet_added_conditions=None,
        uncond_unet_added_conditions=None
    ):
    
        latents = latents.detach()
        batch_size = latents.shape[0]
        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            [batch_size], 
            device=latents.device,
            dtype=torch.long
        )
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        with self.network_context_manager:
            fake_noise_pred = predict_noise(
                unet=self.fake_unet,
                noisy_latents=noisy_latents,
                text_embeddings=text_embedding,
                uncond_embedding=uncond_embedding,
                timesteps=timesteps,
                guidance_scale=1.0,  # no guidance when training fake
                unet_added_conditions=unet_added_conditions,
                uncond_unet_added_conditions=uncond_unet_added_conditions,
            )


        fake_noise_pred = fake_noise_pred.float()

        fake_x0_pred = get_x0_from_noise(
            noisy_latents.double(), fake_noise_pred.double(), self.alphas_cumprod.double(), timesteps
        )

        # epsilon prediction loss 
        loss_fake = torch.mean(
            (fake_noise_pred.float() - noise.float())**2
        )

        loss_dict = {
            "loss_fake_mean": loss_fake,
        }

        fake_log_dict = {
            "faketrain_latents": latents.detach(),
            "faketrain_noisy_latents": noisy_latents.detach(),
            "faketrain_x0_pred": fake_x0_pred.detach()
        }
        
        return loss_dict, fake_log_dict
    
    # The target teacher is trained on real images, not generator outputs, so it can
    # act as an online teacher anchored to the real instance distribution.
    def compute_loss_target_teacher(
        self,
        real_image,
        text_embedding,
        uncond_embedding,
        unet_added_conditions=None,
        uncond_unet_added_conditions=None,
        ):
        """
        Standard EDM-style denoising loss for the Target Teacher (TT).
        Mirrors compute_loss_fake, but uses self.target_unet and real images by default.
        """
        assert self.target_teacher_unet is not None, "Target teacher is disabled."
        latents = self._as_latents_for_unet(real_image.detach())
        noise   = torch.randn_like(latents)
        B       = latents.size(0)

        timesteps = torch.randint(
            0, self.num_train_timesteps, [B],
            device=latents.device, dtype=torch.long
        )
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        with self.network_context_manager:
            pred = predict_noise(
                unet=self.target_teacher_unet,
                noisy_latents=noisy_latents,
                text_embeddings=text_embedding,
                uncond_embedding=uncond_embedding,         
                timesteps=timesteps,
                guidance_scale=1.0,     # CFG on TT if you want it
                unet_added_conditions=unet_added_conditions, # SDXL dict or None (SD1.5)
                uncond_unet_added_conditions=uncond_unet_added_conditions,
            )

        loss = torch.mean((pred.float() - noise.float())**2)


        # reconstruct x0 for logging
        tt_x0 = get_x0_from_noise(noisy_latents.double(), pred.double(),
                              self.alphas_cumprod.double(), timesteps).float()

        return {"loss_target_teacher_mean": loss}, {
            "target_latents": latents.detach(),
            "target_noisy_latents": noisy_latents.detach(),
            "tt_pred_x0": tt_x0.detach(),
            # return the per-sample timesteps used to create noisy_latents so
            # we can correctly run a backward denoising chain from each x_t
            "timesteps": timesteps.detach(),
        }

    def compute_generator_clean_cls_loss(self, 
        fake_image, text_embedding, 
        unet_added_conditions=None
    ):
        loss_dict = {} 

        pred_realism_on_fake_with_grad = self.compute_cls_logits(
            fake_image, 
            text_embedding=text_embedding, 
            unet_added_conditions=unet_added_conditions
        )
        loss_dict["gen_cls_loss"] = F.softplus(-pred_realism_on_fake_with_grad).mean()
        return loss_dict 

    # Called from SDUniModel during the generator turn. Only losses that are meant
    # to backprop into the generator should be assembled here.
    def generator_forward(
        self,
        image,
        text_embedding,
        uncond_embedding,
        unet_added_conditions=None,
        uncond_unet_added_conditions=None,
        teacher_text_embedding=None,
        generator_data_dict=None,
    ):
        loss_dict = {}
        log_dict = {}

        #image.requires_grad_(True)
        if not self.gan_alone:
            dm_dict, dm_log_dict = self.compute_distribution_matching_loss(
                latents=image,
                text_embedding=text_embedding,
                unet_added_conditions=unet_added_conditions,
                uncond_embedding=uncond_embedding,
                teacher_unet_added_conditions=generator_data_dict.get("teacher_unet_added_conditions", None),
                teacher_uncond_embedding=generator_data_dict.get("teacher_uncond_embedding", None),
                uncond_unet_added_conditions=uncond_unet_added_conditions,
                teacher_text_embedding=teacher_text_embedding,
            )

            loss_dict.update(dm_dict)
            log_dict.update(dm_log_dict)

        if self.cls_on_clean_image:
            clean_cls_loss_dict = self.compute_generator_clean_cls_loss(
                image, text_embedding, unet_added_conditions
            )
            loss_dict.update(clean_cls_loss_dict)

        if self.use_multihead_gan:
            # Critic should not update during G step
            self.multihead_gan.requires_grad_(False)

            lat = self._as_latents_for_unet(image)
            t   = self._gan_timesteps(lat.size(0), lat.device)

            if (self.gan_bce_random_t or self.gan_bce_timestep > 0):
                noise = torch.randn_like(lat)
                lat = self.scheduler.add_noise(lat, noise, t) # score x_t, not x_0

            # Text/conds for current batch
            cond = text_embedding
            added = None
            if isinstance(unet_added_conditions, dict):
                added = unet_added_conditions

            with self.network_context_manager:
                fake_scores = self.multihead_gan.score(lat, t, cond, added)

            # add G loss (try to make D believe fake is real)
            loss_g_gan = self.multihead_gan.g_loss(fake_scores) * self.args.gan_bce_weight_g
            loss_dict["gan_bce_g"] = loss_g_gan

        return loss_dict, log_dict 

    def compute_guidance_clean_cls_loss(
            self, real_image, fake_image, 
            real_text_embedding, fake_text_embedding,
            real_unet_added_conditions=None, 
            fake_unet_added_conditions=None
        ):
        pred_realism_on_real = self.compute_cls_logits(
            real_image.detach(), 
            text_embedding=real_text_embedding,
            unet_added_conditions=real_unet_added_conditions
        )
        pred_realism_on_fake = self.compute_cls_logits(
            fake_image.detach(), 
            text_embedding=fake_text_embedding,
            unet_added_conditions=fake_unet_added_conditions
        )

        log_dict = {
            "pred_realism_on_real": torch.sigmoid(pred_realism_on_real).squeeze(dim=1).detach(),
            "pred_realism_on_fake": torch.sigmoid(pred_realism_on_fake).squeeze(dim=1).detach()
        }

        classification_loss = F.softplus(pred_realism_on_fake).mean() + F.softplus(-pred_realism_on_real).mean()
        loss_dict = {
            "guidance_cls_loss": classification_loss
        }
        return loss_dict, log_dict 

    # Called during the guidance turn. This branch updates fake_unet and any attached
    # classifier/GAN heads using the real instance batch alongside generated samples.
    def guidance_forward(
        self,
        image,                       # fake RGB [B,3,H,W] in [0,1]
        text_embedding,              # cond for fake
        uncond_embedding,
        real_train_dict=None,        # must be provided when GAN is on
        unet_added_conditions=None,  # None (SD-1.5)
        uncond_unet_added_conditions=None
    ):
        
        # 0) We require real_train_dict to contain the training instance images.
        assert real_train_dict is not None and 'images' in real_train_dict, \
            "real_train_dict['images'] (target instance batch) is required for fake UNet loss."

        # 0a) if we’ll use clean-cls later, ensure real text conds exist
        if self.cls_on_clean_image:
            assert 'text_embedding' in real_train_dict, \
                "real_train_dict['text_embedding'] required when cls_on_clean_image=True"
            
        # Convert pixels [B,3,H,W] in [0,1] -> latents [B,4,h,w] using the shared VAE
        with torch.no_grad():
            train_latents = self._as_latents_for_unet(real_train_dict['images'])
        
        # Encode RGB → latent space that UNet expects (handles SD1.5 )
        latents = self._as_latents_for_unet(image)

        # 1) your existing diffusion 'fake' loss
        fake_dict, fake_log_dict = self.compute_loss_fake(
            latents=latents, 
            text_embedding=text_embedding, 
            uncond_embedding=uncond_embedding,
            unet_added_conditions=unet_added_conditions,
            uncond_unet_added_conditions=uncond_unet_added_conditions
        )
        
        loss_dict = fake_dict
        log_dict  = fake_log_dict

        # 2) optional classifier branch (unchanged)
        if self.cls_on_clean_image:
            clean_cls_loss_dict, clean_cls_log_dict = self.compute_guidance_clean_cls_loss(
                real_image=real_train_dict['images'], 
                fake_image=image,
                real_text_embedding=real_train_dict['text_embedding'],
                fake_text_embedding=text_embedding, 
                real_unet_added_conditions=real_train_dict.get('unet_added_conditions', None),
                fake_unet_added_conditions=unet_added_conditions
            )
            loss_dict.update(clean_cls_loss_dict)
            log_dict.update(clean_cls_log_dict)

        # 3) multi-head GAN D-step (BCE)
        if self.use_multihead_gan:
            assert real_train_dict is not None and 'images' in real_train_dict, \
                "real_train_dict with real images is required when multihead_gan is enabled."

            # Critic learns on this step
            self.multihead_gan.requires_grad_(True)

            # Convert pixels->latents if needed
            fake_lat = self._as_latents_for_unet(image.detach())
            real_img = real_train_dict['images']
            real_lat = self._as_latents_for_unet(real_img.detach())

            B = fake_lat.size(0)
            t = self._gan_timesteps(B, fake_lat.device)

            if (self.gan_bce_random_t or self.gan_bce_timestep > 0):
                nf = torch.randn_like(fake_lat)
                fake_lat = self.scheduler.add_noise(fake_lat, nf, t)
                nr = torch.randn_like(real_lat)
                real_lat = self.scheduler.add_noise(real_lat, nr, t)
                
            fake_cond  = text_embedding
            fake_added = unet_added_conditions
            real_cond  = real_train_dict['text_embedding']
            real_added = real_train_dict.get('unet_added_conditions', None)

        
            # No freezing: allow D-loss to backprop into UNet encoder (if your optimizer steps it)
            with self.network_context_manager:
                fake_scores = self.multihead_gan.score(fake_lat, t, fake_cond, fake_added)
                real_scores = self.multihead_gan.score(real_lat, t, real_cond, real_added)

            print("------------------------")
            print("[INSIDE GUIDANCE STEP]")
            fs = fake_scores.detach()
            rs = real_scores.detach()

            print(f"fake_scores  shape={tuple(fs.shape)}  "
                f"min={fs.min().item():.3e} max={fs.max().item():.3e} mean={fs.mean().item():.3e}")

            print(f"real_scores  shape={tuple(rs.shape)}  "
                f"min={rs.min().item():.3e} max={rs.max().item():.3e} mean={rs.mean().item():.3e}")

            print("GAN BCE WEIGHT (D):", float(self.gan_bce_weight_d))
            #----------
            print("[CHK] real/fake dtypes:", real_scores.dtype, fake_scores.dtype)
            with torch.cuda.amp.autocast(enabled=False):
                loss_d_unweighted = self.multihead_gan.d_loss(real_scores, fake_scores)  # keep tensors for autograd
            loss_d_gan = loss_d_unweighted * float(self.gan_bce_weight_d)   # use the same name as above
            loss_d_gan = loss_d_gan.to(torch.float32)   # belt-and-suspenders
            loss_dict["gan_bce_d"] = loss_d_gan  
            print("[CHK] d_loss dtype:", loss_d_unweighted.dtype)

            print("loss_d_gan (unweighted):", loss_d_unweighted.detach().item())
            print("loss_d_gan (weighted):", loss_d_gan.detach().item())
            v = loss_dict.get("gan_bce_d")
            print("[CHK/Trainer] has gan_bce_d?", isinstance(v, torch.Tensor), "finite?", (v.numel()==1 and torch.isfinite(v)))  
            print("grad_enabled:", torch.is_grad_enabled())
            print("logits dtypes:", real_scores.dtype, fake_scores.dtype,
                "requires_grad:", real_scores.requires_grad, fake_scores.requires_grad)
            print("targets present?", 'target_real' in locals(), 'target_fake' in locals())
            if 'target_real' in locals():
                print("target dtypes:", real_scores.dtype, fake_scores.dtype)
            print("loss parts:", loss_d_unweighted.dtype, "unw:", loss_d_unweighted.dtype)
            print("weighted:", (loss_d_unweighted * torch.as_tensor(self.gan_bce_weight_d,
                    device=loss_d_unweighted.device, dtype=loss_d_unweighted.dtype)).dtype)
            #try:
            log_dict["gan_bce_d"] = float(loss_d_gan.detach())
            log_dict["pred_realism_on_fake"] = fake_scores.detach()
            log_dict["pred_realism_on_real"] = real_scores.detach()
            log_dict["pred_prob_on_fake"] = torch.sigmoid(fake_scores).detach()
            log_dict["pred_prob_on_real"] = torch.sigmoid(real_scores).detach()
           # except Exception:
            #    pass

            self.multihead_gan.freeze_encoder(False)

        return loss_dict, log_dict


    # train_sd.py drives this dispatcher with exactly one active turn at a time so the
    # trainer can alternate generator, guidance, and target-teacher optimization steps.
    def forward(
        self,
        generator_turn=False,
        guidance_turn=False,
        teacher_turn=False,
        generator_data_dict=None, 
        guidance_data_dict=None,
        teacher_data_dict=None
    ):  
        if generator_turn:
            loss_dict, log_dict = self.generator_forward(
                image=generator_data_dict["image"],
                text_embedding=generator_data_dict["text_embedding"],
                uncond_embedding=generator_data_dict["uncond_embedding"],
                unet_added_conditions=generator_data_dict["unet_added_conditions"],
                uncond_unet_added_conditions=generator_data_dict["uncond_unet_added_conditions"],
                teacher_text_embedding=generator_data_dict.get("teacher_text_embedding", None),
                generator_data_dict=generator_data_dict,
            )

            # ---- NEW: hand off everything guidance_forward needs ----
            def _maybe_detach(x):
                return x.detach() if torch.is_tensor(x) else x

            gd = {
                "image": _maybe_detach(generator_data_dict.get("image")),
                "text_embedding": generator_data_dict.get("text_embedding"),
                "uncond_embedding": generator_data_dict.get("uncond_embedding"),
                "real_train_dict": generator_data_dict.get("real_train_dict", {}),

                # SDXL extras (keep None for SD-1.5)
                "unet_added_conditions": generator_data_dict.get("unet_added_conditions"),
                "uncond_unet_added_conditions": generator_data_dict.get("uncond_unet_added_conditions"),
            }
            log_dict["guidance_data_dict"] = gd
            return loss_dict, log_dict
            # ---------------------------------------------------------

        elif guidance_turn:
            return self.guidance_forward(
                image=guidance_data_dict["image"],
                text_embedding=guidance_data_dict["text_embedding"],
                uncond_embedding=guidance_data_dict["uncond_embedding"],
                real_train_dict=guidance_data_dict["real_train_dict"],
                unet_added_conditions=guidance_data_dict["unet_added_conditions"],
                uncond_unet_added_conditions=guidance_data_dict["uncond_unet_added_conditions"]
            )
        
        elif teacher_turn:
            return self.compute_loss_target_teacher(
                real_image=teacher_data_dict["real_image"],
                text_embedding=teacher_data_dict["text_embedding"],
                uncond_embedding=teacher_data_dict["uncond_embedding"],
                unet_added_conditions=teacher_data_dict.get("unet_added_conditions", None),
                uncond_unet_added_conditions=teacher_data_dict.get("uncond_unet_added_conditions", None),
            )
        
        else:
            raise NotImplementedError
# A single unified model that wraps both the generator and discriminator
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from main.utils import get_x0_from_noise, NoOpContext
from main.models.sd_guidance import SDGuidance
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate.utils import broadcast
from torch import nn
import torch 

# SDUniModel is the bridge between the trainer and the lower-level guidance
# modules. It owns the trainable generator UNet plus shared SD components, while
# SDGuidance owns the teacher/critic side of the pipeline.
class SDUniModel(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()

        self.args = args
        self.accelerator = accelerator
        # The trainer talks to one object, but internally responsibilities are split:
        # this class prepares text/latents/images, and guidance_model computes the
        # teacher, critic, and auxiliary losses for the current turn.
        self.guidance_model = SDGuidance(args, accelerator) 
        self.num_train_timesteps = self.guidance_model.num_train_timesteps
        self.num_visuals = args.grid_size * args.grid_size
        self.conditioning_timestep = args.conditioning_timestep 
        self.use_fp16 = args.use_fp16 
        self.gradient_checkpointing = args.gradient_checkpointing 
        self.backward_simulation = args.backward_simulation 

        self.cls_on_clean_image = args.cls_on_clean_image 
        self.denoising = args.denoising
        self.denoising_timestep = args.denoising_timestep 
        self.noise_scheduler = self.guidance_model.scheduler
        self.num_denoising_step = args.num_denoising_step 
        self.denoising_step_list = torch.tensor(
            list(range(self.denoising_timestep-1, 0, -(self.denoising_timestep//self.num_denoising_step))),
            dtype=torch.long,
            device=accelerator.device 
        )
        self.timestep_interval = self.denoising_timestep//self.num_denoising_step

        # For few-step denoising
        self.denoising_sigma_end = args.denoising_sigma_end

        # feedforward_model is the one-step generator updated during generator turns.
        # It shares the same SD base architecture as the teacher-side UNets.
        if args.initialie_generator:
            self.feedforward_model = UNet2DConditionModel.from_pretrained(
                args.model_id,
                subfolder="unet"
            ).float()

            self.feedforward_model.requires_grad_(True)

            if args.generator_lora:
                raise ValueError("generator_lora is not kept in the SD1.5-only cleanup yet.")
            
            if self.gradient_checkpointing:
                self.feedforward_model.enable_gradient_checkpointing()
        
            
        # Tokenizer, text encoder, VAE, and scheduler are shared utilities used for
        # generation, denoising prep, logging, and pause-time sampling.
        # Shared: scheduler from the model_id
        self.scheduler = DDIMScheduler.from_pretrained(
            args.model_id, subfolder="scheduler", revision=getattr(args, "revision", None)
        )

        # VAE (dtype only matters for SD-1.5)
        vae_dtype = torch.float16 if getattr(args, "use_fp16", False) else torch.float32

        # SD-1.5 path
        self.tokenizer = CLIPTokenizer.from_pretrained(
            args.model_id, subfolder="tokenizer", revision=getattr(args, "revision", None)
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.model_id, subfolder="text_encoder", revision=getattr(args, "revision", None)
        ).to(accelerator.device)
        self.text_encoder.requires_grad_(False)
        self.alphas_cumprod = self.guidance_model.alphas_cumprod.to(accelerator.device)
        self.vae = AutoencoderKL.from_pretrained(
            args.model_id, subfolder="vae", revision=getattr(args, "revision", None), torch_dtype=vae_dtype 
        )
        self.guidance_model.vae = self.vae 

        # We intentionally do NOT set self.unet here; the trainable UNet lives in self.feedforward_model.


        # put VAE on the right device first
        self.vae.to(self.accelerator.device)

        # only use fp16 for VAE on CUDA; keep fp32 on CPU
        if self.use_fp16 and self.accelerator.device.type == "cuda":
            self.vae.to(dtype=torch.float16)
        else:
            self.vae.to(dtype=torch.float32)

        self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_fp16 else NoOpContext()

        # Guidance/discriminator classifier head
        if getattr(args, "cls_on_clean_image", False):
            feat_dim = 1024
            self.guidance_classifier = nn.Linear(feat_dim, args.num_classes)

        # Generator-side classifier head (if you use --gen_cls_loss)
        if getattr(args, "gen_cls_loss", False):
            gen_feat_dim = 256
            self.gen_classifier = nn.Linear(gen_feat_dim, args.num_classes)

    # decode_image is used for trainer logging, pause-time generations, and any
    # path that needs to convert latent-space predictions back to RGB for humans.
    def decode_image(self, latents):
        vae = self.vae
        vae_device = next(vae.parameters()).device
        vae_dtype  = next(vae.parameters()).dtype

        # If VAE is on CPU, force fp32 (CPU conv2d doesn't support float16)
        if vae_device.type == "cpu" and vae_dtype == torch.float16:
            vae_dtype = torch.float32
            vae.to(dtype=vae_dtype)

        # VAE expects unscaled latents, in its own dtype/device
        latents = (latents / vae.config.scaling_factor).to(device=vae_device, dtype=vae_dtype)

        # Avoid autocast surprises during decode
        use_cuda = (vae_device.type == "cuda")
        with torch.autocast("cuda", enabled=False) if use_cuda else torch.no_grad():
            image = vae.decode(latents).sample  # [B,3,H,W], same dtype as VAE

        return image.float()  # hand back fp32 for downstream

    # Helpers for few step denoising
    @torch.no_grad()
    def _make_sigma_schedule(self, sigma_start: torch.Tensor, K: int, sigma_end: float) -> torch.Tensor:
        if sigma_start.ndim > 1:
            sigma_start = sigma_start.view(sigma_start.shape[0], -1)[:, 0]
        B = sigma_start.shape[0]
        s0 = sigma_start.clamp_min(sigma_end + 1e-8)
        sK = torch.full_like(s0, float(sigma_end))
        t0, tK = torch.log(s0), torch.log(sK)

        grid = torch.linspace(0, 1, steps=K, device=s0.device, dtype=s0.dtype).unsqueeze(1)
        logs = t0.unsqueeze(0) * (1 - grid) + tK.unsqueeze(0) * grid
        sigmas = torch.exp(logs)
        return sigmas

    def _re_noise_sigma(self, x0: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
        # using sigma EDM corruption x = x0 + sigma * eps.
        if sigma_next.ndim == 1:
            sigma_next = sigma_next.view(-1, 1, 1, 1)
        return x0 + sigma_next * torch.randn_like(x0)

    @torch.no_grad()
    # This helper is the multi-step SD1.5 sampler used for optional pause-time or
    # debug generation. It is separate from the training path, which is mostly 1-step.
    def sample_ddim_sd15(
        self,
        prompt_texts,                 # List[str], length B
        num_inference_steps=None,       # link it to the arg gen_num_step
        guidance_scale=7.5,
        height=None,
        width=None,
        seeds=None,                   # Optional[List[int]] or None
    ):
        device = self.accelerator.device

        # 1) Text encode (student path, same tokenizer/encoder you already load)
        tok = self.tokenizer(
            list(prompt_texts),
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        ).input_ids.to(device)
        text_emb = self.text_encoder(tok)[0].float()             # [B, L, 768]

        # Uncond (already built in Trainer, but re-create here for self-containment)
        uncond_ids = self.tokenizer(
            [""] * len(prompt_texts),
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        ).input_ids.to(device)
        uncond_emb = self.text_encoder(uncond_ids)[0].float()    # [B, L, 768]

        # 2) Init latents
        B = len(prompt_texts)
        h = (height or self.accelerator.state.deepspeed_plugin.mpu if False else self.args.resolution) // 8
        w = (width  or self.args.resolution) // 8
        if seeds is None:
            noise = torch.randn(B, 4, h, w, device=device, dtype=torch.float32)
        else:
            noises = []
            for s in seeds:
                g = torch.Generator(device=device).manual_seed(int(s))
                noises.append(torch.randn(1, 4, h, w, generator=g, device=device, dtype=torch.float32))
            noise = torch.cat(noises, dim=0)

        latents = noise

        # 3) Prepare scheduler
        num_inference_steps = num_inference_steps or 1 
        self.scheduler.set_timesteps(num_inference_steps, device=device)  # e.g., 20/30/50…
        latents = latents * self.scheduler.init_noise_sigma               # key detail

        # 4) Denoising loop (DDIM)
        for t in self.scheduler.timesteps:
            # CFG: concat uncond/cond latents & embeddings
            model_in = torch.cat([latents, latents], dim=0)
            embeds   = torch.cat([uncond_emb, text_emb], dim=0)

            # UNet ε prediction
            eps = self.feedforward_model(model_in, t.expand(2*B), embeds, added_cond_kwargs=None).sample
            eps_uncond, eps_text = eps.chunk(2, dim=0)
            eps = eps_uncond + guidance_scale * (eps_text - eps_uncond)

            # Scheduler step
            out = self.scheduler.step(eps, t, latents)
            latents = out.prev_sample

        # 5) Decode to image in [-1,1]
        imgs = self.decode_image(latents)
        return imgs.clamp(-1, 1)

    @torch.no_grad()
    def sample_backward(self, noisy_image, real_text_embedding):
        batch_size =  noisy_image.shape[0]
        device = noisy_image.device
        unet_added_conditions = None

        # we choose a random step and share it across all gpu
        selected_step = torch.randint(low=0, high=self.num_denoising_step, size=(1,), device=device, dtype=torch.long)
        selected_step = broadcast(selected_step, from_process=0)

        # set a default value in case we don't enter the loop 
        # it will be overwriten in the pure_noise_mask check later 
        generated_image = noisy_image  

        for constant in self.denoising_step_list[:selected_step]:
            current_timesteps = torch.ones(batch_size, device=device, dtype=torch.long)  *constant

            generated_noise = self.feedforward_model(
                noisy_image, current_timesteps, real_text_embedding, added_cond_kwargs=None
            ).sample

            generated_image = get_x0_from_noise(
                noisy_image, generated_noise.double(), self.alphas_cumprod.double(), current_timesteps
            ).float()

            next_timestep = current_timesteps - self.timestep_interval 
            noisy_image = self.noise_scheduler.add_noise(
                generated_image, torch.randn_like(generated_image), next_timestep
            ).to(noisy_image.dtype)  

        return_timesteps = self.denoising_step_list[selected_step] * torch.ones(batch_size, device=device, dtype=torch.long)
        return generated_image, return_timesteps

    @torch.no_grad()
    # In denoising mode we build a training batch around real images: choose the
    # denoising timestep(s), encode the corresponding text, and construct x_t.
    def prepare_denoising_data(self, denoising_dict, real_train_dict, noise):

        indices = torch.randint(
            0, self.num_denoising_step, (noise.shape[0],), device=noise.device, dtype=torch.long
        )
        timesteps = self.denoising_step_list.to(noise.device)[indices]

        text_embedding, pooled_text_embedding = self.text_encoder(denoising_dict)

        if real_train_dict is not None:
            real_text_embedding, real_pooled_text_embedding = self.text_encoder(real_train_dict)

            real_train_dict['text_embedding'] = real_text_embedding

            real_unet_added_conditions = {
                "time_ids": self.add_time_ids.repeat(len(real_text_embedding), 1),
                "text_embeds": real_pooled_text_embedding
            }
            real_train_dict['unet_added_conditions'] = real_unet_added_conditions

        if self.backward_simulation:
            # we overwrite the denoising timesteps 
            # note: we also use uncorrelated noise 
            clean_images, timesteps = self.sample_backward(torch.randn_like(noise), text_embedding, pooled_text_embedding) 
        else:
            clean_images = denoising_dict['images'].to(noise.device)

        noisy_image = self.noise_scheduler.add_noise(
            clean_images, noise, timesteps
        )

        # set last timestep to pure noise
        pure_noise_mask = (timesteps == (self.num_train_timesteps-1))
        noisy_image[pure_noise_mask] = noise[pure_noise_mask]

        return timesteps, text_embedding, pooled_text_embedding, real_train_dict, noisy_image

    @torch.no_grad()
    # In the default 1-step generation path, the generator starts directly from noise
    # at the fixed conditioning timestep and uses prompt ids from the student stream.
    def prepare_pure_generation_data(self, ids_for_gen, real_train_dict, noise):
        """
        ids_for_gen:
        - SD-1.5: LongTensor [B,L] or [B,1,L]
        Returns:
        text_embedding:        [B, Lc, C] (SD1.5) 
        real_train_dict:       updated with 'text_embedding' and 'unet_added_conditions' when provided
        noisy_image:           == noise
        """
        te_device = next(self.text_encoder.parameters()).device

        # ---- Encode student (generator) text ----
        # SD-1.5: ids_for_gen is a LongTensor; squeeze possible [B,1,L] -> [B,L]
        ids = ids_for_gen
        if ids.dim() == 3 and ids.size(1) == 1:
            ids = ids.squeeze(1)
        ids = ids.to(te_device)
        out = self.text_encoder(ids)
        # HuggingFace CLIPTextModel returns (last_hidden_state, pooled_output)
        text_embedding = out[0].float()
        # Not all CLIP configs return pooled; be defensive
        pooled_text_embedding = (out[1].float() if len(out) > 1 and out[1] is not None else None)

        # ---- Optionally encode "real" text for classifier / auxiliary losses ----
        if real_train_dict is not None:
            real_ids = real_train_dict.get("teacher_input_ids_one", real_train_dict["text_input_ids_one"])
            if real_ids.dim() == 3 and real_ids.size(1) == 1:
                real_ids = real_ids.squeeze(1)
            real_ids = real_ids.to(te_device)
            real_out = self.text_encoder(real_ids)
            real_train_dict["text_embedding"] = real_out[0].float()
            real_train_dict["unet_added_conditions"] = None

        noisy_image = noise
        return text_embedding, pooled_text_embedding, real_train_dict, noisy_image


    # This dispatcher mirrors the trainer's alternating loop: exactly one of
    # generator_turn, guidance_turn, or teacher_turn should be active per call.
    def forward(self, noise, text_embedding, uncond_embedding, 
        visual=False, denoising_dict=None,
        real_train_dict=None,
        compute_generator_gradient=True,
        generator_turn=False,
        guidance_turn=False,
        teacher_turn=False, 
        guidance_data_dict=None, 
        teacher_data_dict=None, 
        generator_data_dict=None   
    ):
        assert int(generator_turn) + int(guidance_turn) + int(teacher_turn) == 1
        if teacher_turn:
            return self.guidance_model(
                generator_turn=False, guidance_turn=False, teacher_turn=True,
                teacher_data_dict=teacher_data_dict
            )

        is_text_dict = isinstance(text_embedding, dict)
        is_real_dict = isinstance(real_train_dict, dict)
        def _safe_get(d, k, default=None):
            return d.get(k, default) if isinstance(d, dict) else default

        # We do NOT encode text here.
        # SD-1.5: we will encode just-in-time (right before use) from *_input_ids.
        # Normalize keys if caller passed a single-stream dict.
        if is_text_dict and ("student_input_ids_one" not in text_embedding):
            text_embedding["student_input_ids_one"] = text_embedding["text_input_ids_one"]
            text_embedding["teacher_input_ids_one"] = text_embedding["text_input_ids_one"]

        def _fix_ids(t):
            if torch.is_tensor(t) and t.dim() == 3 and t.size(1) == 1:
                return t.squeeze(1)
            return t

        if is_text_dict:
            for k in ("student_input_ids_one", "teacher_input_ids_one"):
                if k in text_embedding:
                    text_embedding[k] = _fix_ids(text_embedding[k])
        
        # Generator turn: build the current fake sample, then hand the resulting
        # tensors to SDGuidance so it can compute generator-side losses.
        if generator_turn:
            # --- cache prompts before we overwrite text_embedding with tensors ---
            cached_raw_prompts = (_safe_get(text_embedding, "raw_prompt") if isinstance(text_embedding, dict) else None)
            cached_real_raw_prompts = ( _safe_get(real_train_dict, "raw_prompt") if isinstance(real_train_dict, dict) else None)
            # train generator only
            self.feedforward_model.requires_grad_(compute_generator_gradient)
            self.guidance_model.requires_grad_(False)

            _student_ids_pack = text_embedding["student_input_ids_one"]                 # LongTensor [B,L] or [B,1,L]
            _teacher_ids_pack = text_embedding.get("teacher_input_ids_one", None)       # LongTensor or None
            
            teacher_text_embedding = None
            if is_text_dict:
                t_ids = _safe_get(text_embedding, "teacher_input_ids_one", None)
                if t_ids is not None:
                    if t_ids.dim() == 3 and t_ids.size(1) == 1:
                        t_ids = t_ids.squeeze(1)
                    t_ids = t_ids.to(next(self.text_encoder.parameters()).device)
                    t_out = self.text_encoder(t_ids)          # CLIPTextModel
                    teacher_text_embedding = t_out[0].float() # last_hidden_state
           
            teacher_text_emb = None
            teacher_unet_added_conds = None
            teacher_uncond_emb = None
            if self.denoising:
                # we ignore the text_embedding, uncond_embedding passed to the model 
                timesteps, text_embedding, pooled_text_embedding, real_train_dict, noisy_image = self.prepare_denoising_data(
                    denoising_dict, real_train_dict, noise
                )
            else:
                timesteps = torch.ones(noise.shape[0], device=noise.device, dtype=torch.long) * self.conditioning_timestep
                # Use the student's token IDs as the input to the text encoder
                ids_for_gen = text_embedding["student_input_ids_one"]  # [B, L]
                text_embedding, pooled_text_embedding, real_train_dict, noisy_image = self.prepare_pure_generation_data(
                    ids_for_gen, real_train_dict, noise
                )
                # Build teacher embeddings (ONLY to feed real_unet in Source-DM)
                # NOTE: we may already have teacher_text_embedding for SD1.5 from earlier — reuse it if present.
                if (teacher_text_emb is None) and (teacher_text_embedding is not None):
                    teacher_text_emb = teacher_text_embedding

                # Initialize SDXL extras by keeping None (from dmd2 initialization but we are not using sdxl here ).
                teacher_unet_added_conds =  None
                teacher_uncond_emb = None

                # SD-1.5: only compute if we didn’t already build it above
                if (_teacher_ids_pack is not None) and (teacher_text_emb is None):
                    t_ids = _teacher_ids_pack
                    if t_ids.dim() == 3 and t_ids.size(1) == 1:
                        t_ids = t_ids.squeeze(1)
                    t_out = self.text_encoder(t_ids.to(self.accelerator.device))
                    teacher_text_emb = t_out[0].float()
                    # SD-1.5 real_unet path needs no added conds; reuse uncond embedding
                    teacher_unet_added_conds = None
                    teacher_uncond_emb = uncond_embedding

    
            unet_added_conditions = None
            uncond_unet_added_conditions = None

            # ===  1-step generation (SD-1.5 path) ===
            if compute_generator_gradient:
                with self.network_context_manager:
                    generated_noise = self.feedforward_model(
                        noisy_image, timesteps.long(), 
                        text_embedding, added_cond_kwargs=unet_added_conditions
                    ).sample
            else:
                if self.gradient_checkpointing:
                    self.accelerator.unwrap_model(self.feedforward_model).disable_gradient_checkpointing()

                with torch.no_grad():
                    generated_noise = self.feedforward_model(
                        noisy_image, timesteps.long(), 
                        text_embedding, added_cond_kwargs=unet_added_conditions
                    ).sample

                if self.gradient_checkpointing:
                    self.accelerator.unwrap_model(self.feedforward_model).enable_gradient_checkpointing()
            # this assume that all teacher models use epsilon prediction (which is true for SDv1.5)
            generated_image = get_x0_from_noise(
                noisy_image.double(), 
                generated_noise.double(), self.alphas_cumprod.double(), timesteps
            ).float()

            if compute_generator_gradient:
                # This bundle is the contract between SDUniModel and SDGuidance for
                # generator-side losses: fake image, text conds, real batch context,
                # and any teacher-specific embeddings prepared upstream.
                generator_data_dict = {
                    "image": generated_image,
                    "text_embedding": text_embedding,
                    "pooled_text_embedding": pooled_text_embedding,
                    "uncond_embedding": uncond_embedding,
                    "real_train_dict": real_train_dict,
                    "unet_added_conditions": unet_added_conditions,
                    "uncond_unet_added_conditions": uncond_unet_added_conditions,
                    "teacher_text_embedding": teacher_text_emb,
                    "teacher_unet_added_conditions": teacher_unet_added_conds,
                    "teacher_uncond_embedding": teacher_uncond_emb,
                } 
   
                # avoid any side effects of gradient accumulation
                self.guidance_model.requires_grad_(False)
                loss_dict, log_dict = self.guidance_model(
                    generator_turn=True,
                    guidance_turn=False,
                    generator_data_dict=generator_data_dict
                )

                self.guidance_model.requires_grad_(True)
            else:
                loss_dict = {}
                log_dict = {} 

            if visual:
                decode_key = [
                    "dmtrain_pred_real_image", "dmtrain_pred_fake_image"
                ]

                with torch.no_grad():
                    if compute_generator_gradient and not self.args.gan_alone:
                        for key in ("dmtrain_pred_real_image", "dmtrain_pred_fake_image"):
                            if key in log_dict:
                                log_dict[key + "_decoded"] = self.decode_image(
                                    log_dict[key].detach()[:self.num_visuals]
                                )
                    
                    if self.use_fp16 :
                        log_dict["generated_image"] = self.decode_image(generated_image[:self.num_visuals].detach())
                    else:
                        log_dict["generated_image"] = self.decode_image(generated_image[:self.num_visuals].detach())

                    if self.denoising:
                        if self.use_fp16 :
                            log_dict["original_clean_image"] = self.decode_image(denoising_dict['images'].detach()[:self.num_visuals])
                        else:
                            log_dict["original_clean_image"] = self.decode_image(denoising_dict['images'].detach()[:self.num_visuals])

                    if cached_raw_prompts is not None:
                        log_dict["raw_prompt"] = cached_raw_prompts
                    if cached_real_raw_prompts is not None:
                        log_dict["real_raw_prompt"] = cached_real_raw_prompts


            # guidance_data_dict lets the next trainer phase reuse the exact same
            # generated batch when it updates fake_unet/classifier/GAN components.
            log_dict["guidance_data_dict"] = {
                "image": generated_image.detach(),
                "text_embedding": text_embedding.detach(),
                "pooled_text_embedding": pooled_text_embedding.detach(),
                "uncond_embedding": uncond_embedding.detach(),
                "real_train_dict": real_train_dict,
                "unet_added_conditions": unet_added_conditions,
                "uncond_unet_added_conditions": uncond_unet_added_conditions
            }

            log_dict['denoising_timestep'] = timesteps

        # Guidance turn: consume the detached fake batch produced above and update
        # fake_unet plus any attached classifier or GAN heads.
        elif guidance_turn:
            # train guidance only (fake_unet + classifier)
            self.feedforward_model.requires_grad_(False)
            self.guidance_model.requires_grad_(True)

            assert guidance_data_dict is not None
            loss_dict, log_dict = self.guidance_model(
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=guidance_data_dict,
            )   
            if self.gradient_checkpointing:
                self.accelerator.unwrap_model(self.feedforward_model).disable_gradient_checkpointing()

        return loss_dict, log_dict
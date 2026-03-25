"""
Unified Dhariwal diffusion model wrapper.

This module defines `dhariwalUniModel`, a single nn.Module that ties together:
- a Dhariwal-style guidance model (`dhariwalGuidance`) providing the
  discriminator / critic and noise schedule (min/max step, Karras sigmas),
- a feedforward generator initialized from the guidance model's UNet
  (`fake_unet`), used as the distilled student.

Key features:
- Supports both one-step and few-step (K-step) EDM-style denoising via
  `_make_sigma_schedule` and `_re_noise`, with a configurable terminal
  sigma (`denoising_sigma_end`).
- Exposes a unified `forward` that alternates between:
    * generator updates (optionally computing gradients through the
      generator while freezing the guidance model), and
    * guidance / critic updates driven by `guidance_data_dict`.
- Intended to be used in the DMD2-style distillation loop, where the
  generator learns from Dhariwal guidance and can be run in a few-step
  unrolled sampling mode at test time.
"""

# A single unified model that wraps both the generator and discriminator
from main.dhariwal.dhariwal_guidance import dhariwalGuidance
from torch import nn
import torch 
import copy

class dhariwalUniModel(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()

        self.guidance_model = dhariwalGuidance(args, accelerator) 

        self.guidance_min_step = self.guidance_model.min_step
        self.guidance_max_step = self.guidance_model.max_step

        if args.initialie_generator:
            self.feedforward_model = copy.deepcopy(self.guidance_model.fake_unet)
        else:
            raise NotImplementedError("Only support initializing generator from guidance model.")

        self.feedforward_model.requires_grad_(True)
        self.accelerator = accelerator 
        self.num_train_timesteps = args.num_train_timesteps

        # ---- Few-step (K-step) options ----
        self.denoising = getattr(args, "denoising", False)                 # bool
        self.num_denoising_step = int(getattr(args, "num_denoising_step", 1))  # K
        self.denoising_sigma_end = float(getattr(args, "denoising_sigma_end", 0.5))  # last sigma

        # we will re-use the guidance model's Karras schedule if you want later;
        # not strictly needed for K-step but handy for sanity checks
        self.karras_sigmas = getattr(self.guidance_model, "karras_sigmas", None)

    
    # helpers for few-steps denoising
    def _make_sigma_schedule(self, sigma_start: torch.Tensor, K: int, sigma_end: float) -> torch.Tensor:
        """
        Build a monotonically decreasing list of sigmas of length K, starting at sigma_start and
        ending near sigma_end (geometric spacing is nicer than linear for noise scales).
        sigma_start: [B] or [B,1,1,1] or scalar tensor.
        Returns: [K, B] tensor of per-step sigmas.
        """
        if sigma_start.ndim > 1:
            sigma_start = sigma_start.view(sigma_start.shape[0], -1)[:, 0]  # [B]
        B = sigma_start.shape[0]
        s0 = sigma_start.clamp_min(sigma_end + 1e-8)                        # ensure > end
        sK = torch.full_like(s0, float(sigma_end))
        # geometric spacing in log space
        t0 = torch.log(s0)
        tK = torch.log(sK)
        # steps: K values from t0 -> tK inclusive
        grid = torch.linspace(0, 1, steps=K, device=s0.device, dtype=s0.dtype).unsqueeze(1)  # [K,1]
        logs = t0.unsqueeze(0) * (1 - grid) + tK.unsqueeze(0) * grid                         # [K,B]
        sigmas = torch.exp(logs)                                                             # [K,B]
        return sigmas

    def _re_noise(self, x0: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
        """EDM corruption x = x0 + sigma * eps."""
        if sigma_next.ndim == 1:
            sigma_next = sigma_next.view(-1, 1, 1, 1)
        return x0 + sigma_next * torch.randn_like(x0)

    def _generator_unroll(self, z: torch.Tensor, sigma0: torch.Tensor, labels: torch.Tensor):
        """
        Unroll the generator K times:
        x^(0) = z (at sigma0)  -> x0_hat^(0)
        x^(1) = re-noise(x0_hat^(0), sigma1)
        ...
        return x0_hat^(K-1)
        """
        K = max(1, int(self.num_denoising_step))
        if K == 1:
            return self.feedforward_model(z, sigma0, labels)

        # Build decreasing sigma schedule [K, B]
        sigmas = self._make_sigma_schedule(sigma0, K, self.denoising_sigma_end)  # [K,B]
        x = z
        last_x0 = None
        for i in range(K):
            si = sigmas[i]                         # [B]
            x0_hat = self.feedforward_model(x, si, labels)   # returns x0_hat
            last_x0 = x0_hat
            if i + 1 < K:
                s_next = sigmas[i + 1]            # [B]
                x = self._re_noise(x0_hat, s_next)
        return last_x0
    # ----------------------------------------


    def forward(self, scaled_noisy_image,
        timestep_sigma, labels,  
        real_train_dict=None,
        compute_generator_gradient=False,
        generator_turn=False,
        guidance_turn=False,
        guidance_data_dict=None
    ):        
        assert (generator_turn and not guidance_turn) or (guidance_turn and not generator_turn) 

        if generator_turn:
            if not compute_generator_gradient:
                with torch.no_grad():
                    if self.denoising and self.num_denoising_step > 1:
                        generated_image = self._generator_unroll(scaled_noisy_image, timestep_sigma, labels)
                    else:
                        generated_image = self.feedforward_model(scaled_noisy_image, timestep_sigma, labels)
            else:
                if self.denoising and self.num_denoising_step > 1:
                    generated_image = self._generator_unroll(scaled_noisy_image, timestep_sigma, labels)
                else:
                    generated_image = self.feedforward_model(scaled_noisy_image, timestep_sigma, labels)

            if compute_generator_gradient:
                generator_data_dict = {
                    "image": generated_image,
                    "label": labels,
                    "real_train_dict": real_train_dict
                }

                # as we don't need to compute gradient for guidance model
                # we disable gradient to avoid side effects (in GAN Loss computation)
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

            log_dict['generated_image'] = generated_image.detach() 
            log_dict['generated_image_undetached'] = generated_image

            log_dict['guidance_data_dict'] = {
                "image": generated_image.detach(),
                "label": labels.detach() if labels is not None else None,
                "real_train_dict": real_train_dict
            }

        elif guidance_turn:
            assert guidance_data_dict is not None 
            loss_dict, log_dict = self.guidance_model(
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=guidance_data_dict
            )

        return loss_dict, log_dict
"""
Dhariwal UNet → EDM-style wrapper and network factory.

This module exposes a Dhariwal/ADM UNet (from guided-diffusion) under an
EDMPrecond-like API, so the rest of the code can treat it as an EDM-style
generator:

- `DhariwalUNetAdapter`:
  * Maps EDM noise scales σ to DDPM timesteps t via ᾱ = 1 / (1 + σ²),
    and rescales inputs accordingly (x_t = x / sqrt(1 + σ²)).
  * Implements `forward(x, sigma, class_labels=None, return_bottleneck=False)`
    which returns either:
      - x₀̂ (EDM reconstruction) for sampling / distillation, or
      - middle-block bottleneck features when `return_bottleneck=True`
        (used for multi-head GAN losses).
  * Provides multi-scale feature extraction hooks
    (`extract_multi_scale_features`) for discriminator / guidance heads.
  * Adds a dummy `.model.map_augment` attribute so higher-level code can
    safely delete/override augmentation without touching guided-diffusion internals.
  * Includes a forgiving weight loader to handle checkpoints with nested
    keys (`ema`, `model`, `state_dict`, …).

- `load_pt_with_logs(...)`: utility to load a checkpoint into the adapter
  with strict or non-strict reporting.

- `get_edm_network(args)`: small factory that builds a `DhariwalUNetAdapter`
  with the resolution, label_dim, and precision specified in `args`.

This file is the bridge between guided-diffusion’s ADM UNet and the DMD2 /
EDM-style distillation and evaluation code.
"""


import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import Optional

# guided_diffusion
from guided_diffusion import script_util

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _onehot_to_class_index(labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if labels is None:
        return None
    if labels.ndim == 2:
        return labels.argmax(dim=1).long()
    return labels.long()

def _map_sigma_to_t(sigmas: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    Map EDM noise scale sigma -> nearest DDPM timestep index t via:
      alpha_bar_target = 1 / (1 + sigma^2)
    """
    if sigmas.ndim > 1:
        sigmas = sigmas.reshape(sigmas.shape[0], -1)[:, 0]
    alpha_bar_target = 1.0 / (1.0 + sigmas**2)  # [B]
    # find nearest t (alphas_cumprod is [T] descending)
    d = (alpha_bar_target[:, None] - alphas_cumprod[None, :]).abs()
    t = torch.argmin(d, dim=1).long()
    return t

# --------------------------------------------------------------------
# Adapter that mimics EDMPrecond but runs an ADM (Dhariwal) UNet
# --------------------------------------------------------------------

class DhariwalUNetAdapter(nn.Module):
    """
    Forward API matches EDMPrecond:

        forward(x, sigma, class_labels=None, return_bottleneck=False) -> x0_hat or features

    where:
      - x is EDM-corrupted: x = x0 + sigma * eps
      - sigma can be [B], [B,1], or [B,1,1,1]
    """
    def __init__(self, image_size: int, class_cond: bool, use_fp16: bool):
        super().__init__()

        # Build ADM UNet + diffusion for schedule
        defaults = script_util.model_and_diffusion_defaults()
        defaults.update(
            dict(
                image_size=image_size,
                class_cond=class_cond,
                attention_resolutions="32,16,8",  # good for 256x256
                num_channels=256,
                noise_schedule="linear",
                diffusion_steps=1000,
                learn_sigma=True, # not used, but keep consistent with DhariwalGuidance
                use_fp16=use_fp16,
                num_head_channels=64,
                num_res_blocks=2,
                resblock_updown=True,
                use_scale_shift_norm=True,
            )
        )
        self.unet, self.diffusion = script_util.create_model_and_diffusion(**defaults)

        self._feat_buf = {}
        self._feat_hooks = {}

        acp = self.diffusion.alphas_cumprod
        if not torch.is_tensor(acp):
            acp = torch.from_numpy(acp)
        self.register_buffer("alphas_cumprod", acp.float(), persistent=False)

        # Middle-block feature hook (for return_bottleneck=True)
        self._mid_feat = None
        self._hook = None

        # ---- EDMPrecond-compat: provide a `.model` with `.map_augment` attribute ----
        # Your code does: del net.model.map_augment; net.model.map_augment = None
        # We'll expose the inner UNet as `.model` and add a dummy attr so those lines are no-ops.
        self.model = self.unet
        # Ensure attribute exists; deleting & re-adding will be harmless.
        setattr(self.model, "map_augment", None)

    # ---------- feature extraction hooks (for multi-head GAN) ---------- 
    def _resolve_feature_layers(self, requested):
        # Return an ordered dict of {name: module} we can hook.
        layers = {}
        # Prefer guided_diffusion-style names if present:
        if hasattr(self.unet, 'input_blocks'):
            for i, m in enumerate(self.unet.input_blocks):
                layers[f'in{i}'] = m
        if hasattr(self.unet, 'middle_block'):
            layers['mid'] = self.unet.middle_block

        if requested == 'all':
            return layers
        # requested like "in2,in4,mid"
        selected = {}
        for tok in [t.strip() for t in requested.split(',') if t.strip()]:
            if tok in layers:
                selected[tok] = layers[tok]
            elif tok.isdigit() and f'in{tok}' in layers:
                selected[f'in{tok}'] = layers[f'in{tok}']
        return selected if selected else layers  # fallback to all if nothing matched

    def extract_multi_scale_features(self, x_t, t, y, requested='all'):
        # Install hooks
        modules = self._resolve_feature_layers(requested)
        self._feat_buf = {}
        def _mk_hook(name):
            def hook(_m, _inp, out):
                self._feat_buf[name] = out
            return hook
        for name, mod in modules.items():
            self._feat_hooks[name] = mod.register_forward_hook(_mk_hook(name))
        # one forward to populate buffers
        _ = self._eps_pred(x_t, t, y)
        # remove hooks
        for h in self._feat_hooks.values():
            h.remove()
        self._feat_hooks = {}
        return self._feat_buf  # dict name->Tensor

    # ---------- weight loading ----------
    def load_state_dict_forgiving(self, state):
        """
        Supports plain state_dict or dicts with keys like 'model', 'ema', 'state_dict'.
        Tries strict, then falls back to non-strict with a printout.
        """
        if isinstance(state, dict) and any(k in state for k in ("ema", "model", "state_dict")):
            for k in ("ema", "model", "state_dict"):
                if k in state and isinstance(state[k], dict):
                    state = state[k]
                    break
        try:
            self.unet.load_state_dict(state, strict=True)
            print("[DhariwalUNetAdapter] Loaded weights (strict).")
        except Exception as e:
            msg = self.unet.load_state_dict(state, strict=False)
            print("[DhariwalUNetAdapter] Loaded weights (non-strict).")
            print("  Missing keys:", msg.missing_keys)
            print("  Unexpected keys:", msg.unexpected_keys)

    # ---------- internal helpers ----------
    def _install_middle_hook(self):
        if self._hook is not None:
            return
        def _capture(module, inp, out):
            self._mid_feat = out
        # guided_diffusion UNet has .middle_block
        self._hook = self.unet.middle_block.register_forward_hook(_capture)

    def _remove_middle_hook(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _eps_pred(self, x_t, t, y):
        if y is None:
            out = self.unet(x_t, t)
        else:
            out = self.unet(x_t, t, y)
        # If learn_sigma=True, output would be [eps, var]; we set learn_sigma=False above,
        # but keep this guard so it works even if configs change.
        if out.shape[1] == x_t.shape[1] * 2:
            out, _ = torch.chunk(out, 2, dim=1)
        return out

    # ---------- main forward ----------
    def forward(self, x: torch.Tensor, sigma: torch.Tensor, class_labels=None, return_bottleneck: bool=False):
        """
        Mimics EDMPrecond.forward. Returns:
          - x0_hat when return_bottleneck=False
          - middle-block features when return_bottleneck=True
        """
        B = x.size(0)

        # Normalize sigma shapes
        if sigma.ndim == 1:
            sigma_flat = sigma
            sigma_reshaped = sigma.view(B, 1, 1, 1)
        else:
            sigma_reshaped = sigma
            sigma_flat = sigma.view(B, -1)[:, 0]

        # Map EDM (x = x0 + sigma*eps) -> DDPM timestep & state:
        #   alpha_bar = 1/(1+sigma^2)
        #   x_t = x / sqrt(1+sigma^2)
        c = 1.0 / torch.sqrt(1.0 + sigma_reshaped**2)   # [B,1,1,1]
        x_t = c * x
        t = _map_sigma_to_t(sigma_flat, self.alphas_cumprod)  # [B]
        y = _onehot_to_class_index(class_labels)              # None or [B]

        if return_bottleneck:
            self._install_middle_hook()
            _ = self._eps_pred(x_t, t, y)   # run once to trigger hook
            feat = self._mid_feat
            self._mid_feat = None
            self._remove_middle_hook()
            return feat  # [B, C_mid, H_mid, W_mid]

        # Predict epsilon and reconstruct x0_hat under EDM corruption:
        # With the mapping above, reconstruction simplifies to x0_hat = x - sigma * eps_hat
        eps_hat = self._eps_pred(x_t, t, y)
        x0_hat = x - sigma_reshaped * eps_hat
        return x0_hat


# --------------------------------------------------------------------
# Public functions referenced by your existing code
# --------------------------------------------------------------------

def load_pt_with_logs(precond: nn.Module, ckpt_path: str, try_inner: bool=True):
    print(f"[load_pt_with_logs] Loading weights from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if hasattr(precond, "load_state_dict_forgiving"):
        precond.load_state_dict_forgiving(state)
    else:
        try:
            precond.load_state_dict(state, strict=True)
            print("[load_pt_with_logs] strict load OK.")
        except Exception:
            msg = precond.load_state_dict(state, strict=False)
            print("[load_pt_with_logs] non-strict load. Missing:", msg.missing_keys, "Unexpected:", msg.unexpected_keys)
    return precond


def get_edm_network(args):
    """
    Build the Dhariwal (ADM) UNet wrapped to look like EDMPrecond.
    """
    class_cond = (args.label_dim or 0) > 0
    net = DhariwalUNetAdapter(
        image_size=args.resolution,      # 256 for FFHQ
        class_cond=class_cond,           # False for unconditional FFHQ
        use_fp16=args.use_fp16,         # True/False
    )
    return net

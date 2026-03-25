# main/multihead_gan_bce.py
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import math

# This critic reuses intermediate UNet encoder features instead of owning a
# separate discriminator backbone. Each tapped block gets a tiny per-feature
# head, and the final critic score is the mean across those heads.
class MultiHeadGlobalBCEGan(nn.Module):
    """
    Multi-head BCE critic for diffusers' UNet2DConditionModel.
    - Registers forward hooks on all .down_blocks (encoder) + .mid_block.
    - For each captured feature [B,C,H,W], apply Conv1x1(C->1) + AdaptiveAvgPool2d(1) -> [B,1,1,1] -> [B].
    - Average logits across heads to get one scalar logit per image.
    """
    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet
        self.heads = nn.ModuleDict()     # built lazily
        self._buf: Dict[str, torch.Tensor] = {}
        self._hooks = {}
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self._freeze_backbone = False

    # During discriminator-style updates we sometimes want gradients to stop at
    # the tapped UNet features so the critic trains without pushing on the UNet.
    def freeze_encoder(self, flag: bool = True):
        self._freeze_backbone = flag

    from contextlib import contextmanager
    @contextmanager
    def freeze_ctx(self):
        old = self._freeze_backbone
        self._freeze_backbone = True
        try:
            yield
        finally:
            self._freeze_backbone = old
    
    # ---------- hook management ----------
    def _modules_to_hook(self) -> Dict[str, nn.Module]:
        layers: Dict[str, nn.Module] = {}
        # --- Diffusers UNet2DConditionModel style ---
        if hasattr(self.unet, "down_blocks") and hasattr(self.unet, "mid_block"):
            for i, m in enumerate(self.unet.down_blocks):
                layers[f"down{i}"] = m
            layers["mid"] = self.unet.mid_block
            return layers

        # --- Guided-diffusion style (older SD repos) ---
        if hasattr(self.unet, "input_blocks") and hasattr(self.unet, "middle_block"):
            # NOTE: input_blocks are finer-grained; this yields one head per small block.
            # If you want strictly one per *encoder stage* here too,
            # wrap/aggregate on your side or keep this legacy behavior.
            for i, m in enumerate(self.unet.input_blocks):
                layers[f"in{i}"] = m
            layers["mid"] = self.unet.middle_block
            return layers

        raise AttributeError("UNet must expose either (.down_blocks,.mid_block) or (.input_blocks,.middle_block)")

    # Hooked activations are stored by logical stage name in self._buf so later
    # scoring code can stay agnostic to the underlying UNet implementation.
    def _register(self, modules: Dict[str, nn.Module]):
        self._buf.clear()
        self._hooks = {}

        def make_hook(name: str):
            def _hook(_m, _in, out):
                feat = out
                # Diffusers down_block: (hidden_states, res_samples)
                if isinstance(out, (list, tuple)):
                    # take the block's final hidden_states (end-of-block output)
                    feat = out[0]
                elif isinstance(out, dict):
                    # be robust to dict outputs
                    feat = out.get("hidden_states", out.get("sample", out))
                # ensure we store a BCHW tensor
                if isinstance(feat, (list, tuple)):  # rare nested case
                    feat = feat[0]
                # freeze encoder if feature encoder in multihead discriminator loss computation
                if self._freeze_backbone:
                    feat = feat.detach()

                self._buf[name] = feat
            return _hook

        for name, mod in modules.items():
            self._hooks[name] = mod.register_forward_hook(make_hook(name))

    def _remove(self):
        for h in self._hooks.values():
            h.remove()
        self._hooks.clear()

    # Heads are created lazily because the channel dimensions come from the first
    # real UNet forward; this keeps the critic compatible with different UNet widths.
    @torch.no_grad()
    def _ensure_heads(self):
        if len(self.heads) > 0:
            return
        assert len(self._buf) > 0, "Need tapped features in _buf before building heads."
        # Infer device/dtype from any tapped feature (all should match under autocast)
        sample = next(iter(self._buf.values()))
        device, dtype = sample.device, sample.dtype

        new_heads = {}
        """
        for name, feat in self._buf.items():
            c = int(feat.shape[1])
            g = math.gcd(32, c) or 1
            new_heads[name] = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=g, num_channels=c),
                nn.SiLU(),
                nn.Conv2d(c, c, kernel_size=4, stride=4, padding=0, bias=False),
                nn.GroupNorm(num_groups=g, num_channels=c),
                nn.SiLU(),
                nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.AdaptiveAvgPool2d(1),
            )
          """
        # The currently active design is intentionally minimal: a 1x1 projection
        # plus global pooling turns each stage feature into one scalar logit.
        #If linear head switch needed
        for name, feat in self._buf.items():
            c = int(feat.shape[1])
            new_heads[name] = nn.Sequential(
                nn.Conv2d(c, 1, kernel_size=1, bias=True),
                nn.AdaptiveAvgPool2d(1),
        )
    
        self.heads = nn.ModuleDict(new_heads).to(device=device, dtype=dtype)

    # This is the two-pass path used when the caller wants to reuse one UNet pass:
    # first cache tapped features, then score them later with score_from_cached().
    def populate_taps(self, latents, timesteps, cond, added_cond=None):
        """
        Run UNet once to populate hook buffers (self._buf) — no logits returned.
        Must be called before score_from_cached().
        """
        # Register hooks on encoder + mid
        modules = self._modules_to_hook()
        self._register(modules)
        try:
            out = self.unet(
                latents,
                timesteps,
                encoder_hidden_states=cond,
                added_cond_kwargs=added_cond
            )
            # touch .sample if diffusers output
            _ = out.sample if hasattr(out, "sample") else out
        finally:
            # hooks only needed for this pass
            self._remove()

        # Build heads lazily the first time, now that _buf has shapes
        p = next(self.unet.parameters())
        self._ensure_heads()
        # no return value needed; taps live in self._buf


    # score_from_cached() only touches critic heads. It assumes the UNet forward
    # already happened and self._buf still contains stage features from that pass.
    def score_from_cached(self):
        """
        Compute critic logits using features already in self._buf (filled by populate_taps()).
        Keeps gradients for critic params; UNet was detached by hooks when frozen.
        """
        assert len(self._buf) > 0, "populate_taps() must be called before score_from_cached()."

        # Ensure heads exist (no-op if already built)
        p = next(self.unet.parameters())
        self._ensure_heads()

        logits = []
        for name, feat in self._buf.items():
            head = self.heads[name]          # head exists for each tapped feature
            p0 = next(head.parameters())
            self.heads[name] = head
            z = head(feat)  # [B,1,1,1]
            z = z.view(z.size(0))            # [B]
            logits.append(z)

        # average over heads -> [B]
        out = torch.stack(logits, dim=0).mean(dim=0)

        # clear cache after use (keeps memory in check)
        self._buf.clear()
        return out

    # ---------- scoring ----------
    # score() is the one-shot convenience path: run the UNet, capture taps, apply
    # the critic heads immediately, and return one logit per sample.
    def score(
        self,
        x: torch.Tensor,                # latents [B,4,H,W] or pixels already encoded upstream
        t: torch.Tensor,                # [B] long
        cond: torch.Tensor,             # text embeddings
        added_cond: Optional[dict] = None,   # SDXL dict, else None
    ) -> torch.Tensor:
        """
        Returns [B] raw logits: mean over head logits.
        """
        # Capture feats
        modules = self._modules_to_hook()          # <— get the blocks to hook
        self._register(modules)                    # <— pass them in
        try:
            out = self.unet(x, t, cond, added_cond_kwargs=added_cond)
            _ = out.sample if hasattr(out, "sample") else out
        finally:
            self._remove()

        # Build heads once we know channels
        p = next(self.unet.parameters())
        self._ensure_heads()

        if self.training and len(self.heads) and torch.cuda.current_device() == 0:
            print("[MHGAN] heads:", list(self.heads.keys()))

        # Apply heads and average
        logits_per_head: List[torch.Tensor] = []
        for name, feat in self._buf.items():
            h = self.heads[name]

            p0 = next(h.parameters())
            self.heads[name] = h
        
            z = h(feat)          # [B,1,1,1]
            z = z.view(z.size(0))
            logits_per_head.append(z)
        return torch.stack(logits_per_head, dim=0).mean(dim=0)  # [B]

    # The critic uses standard non-saturating BCE targets: real->1, fake->0 for D,
    # and fake->1 for G.
    # ----------- BCE losses ----------
    def d_loss(self, real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
        return self.bce(real_scores, torch.ones_like(real_scores)) + \
               self.bce(fake_scores, torch.zeros_like(fake_scores))

    def g_loss(self, fake_scores: torch.Tensor) -> torch.Tensor:
        return self.bce(fake_scores, torch.ones_like(fake_scores))

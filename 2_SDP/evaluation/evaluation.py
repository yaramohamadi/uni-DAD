import clip
import torch
from torchvision import transforms

# ---------------- DINO ----------------
class DINOEvaluator(object):
    """
    DINO-ViT-S/16 wrapper. Expects BCHW in [-1,1] at 224×224.
    """
    def __init__(self, device: str = "cuda") -> None:
        # pick an actual torch.device
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

        # 1) Load model ON the chosen device
        try:
            self.model = torch.hub.load(
                'facebookresearch/dino:main', 'dino_vits16', map_location=self.device
            )
        except Exception:
            from timm import create_model
            model = None
            for name in ("vit_small_patch16_224.dino", "vit_small_patch16_224_dino", "vit_small_patch16_224"):
                try:
                    model = create_model(name, pretrained=True)
                    break
                except Exception:
                    continue
            if model is None:
                raise RuntimeError("Could not load a DINO ViT-S/16 model.")
            self.model = model

        # 2) Move to device, eval, no grads
        self.model = self.model.to(self.device)
        self.model.eval().requires_grad_(False)

        # Inputs expected: [-1,1] BCHW. Convert to ImageNet norm
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),  # [-1,1] -> [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor, norm: bool = True) -> torch.Tensor:
        """
        images: BCHW (or CHW) in [-1,1], already 224×224
        returns: B×D (float32), optionally L2-normalized
        """
        if images is None or images.numel() == 0:
            return torch.empty(0, 0, device=self.device)

        # Use the model's actual device (in case someone re-moved it)
        model_device = next(self.model.parameters()).device

        x = images
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(model_device, non_blocking=True).float()
        x = self.preprocess(x)

        feats = self.model(x)

        # Normalize return types/shapes
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        elif isinstance(feats, dict):
            for k in ("feat", "features", "last_hidden_state", "x", "logits"):
                if k in feats:
                    feats = feats[k]
                    break

        if feats.dim() == 4:
            feats = feats.mean(dim=(2, 3))     # B×C×H×W -> B×C
        elif feats.dim() == 3:
            feats = feats.mean(dim=1)          # B×T×D -> B×D

        feats = feats.float()
        if norm and feats.numel() > 0:
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats

    @torch.no_grad()
    def img_to_img_similarity(self, src_images: torch.Tensor, generated_images: torch.Tensor) -> torch.Tensor:
        if src_images is None or src_images.numel() == 0:
            return torch.tensor(float("nan"), device=next(self.model.parameters()).device)
        if generated_images is None or generated_images.numel() == 0:
            return torch.tensor(float("nan"), device=next(self.model.parameters()).device)

        src = self.encode_images(src_images, norm=True)
        gen = self.encode_images(generated_images, norm=True)
        if src.numel() == 0 or gen.numel() == 0:
            return torch.tensor(float("nan"), device=next(self.model.parameters()).device)
        return (src @ gen.T).mean()


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073] # check if could be problematic with imagenet mean/std
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

class CLIPEvaluator(object):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        self.device = device
        self.model, _ = clip.load(clip_model, device=self.device)
        # Loader already gives BCHW in [-1,1] and size 224×224
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),  # [-1,1] -> [0,1]
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])                                    # + skip convert PIL to tensor

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        assert images.shape[-1] == 224 and images.shape[-2] == 224, "Expected 224×224 inputs; got {}".format(images.shape)

        x = self.preprocess(images).to(self.device, non_blocking=True)
        return self.model.encode_image(x)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()

    def txt_to_img_similarity(self, text, generated_images):
        text_features    = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()


class ImageDirEvaluator(CLIPEvaluator):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        super().__init__(device, clip_model)
        # Add a DINO helper alongside CLIP
        self.dino = DINOEvaluator(device=device)

    def evaluate(self, gen_samples, src_images, target_text):
        clip_i = self.img_to_img_similarity(src_images, gen_samples)
        clip_t = self.txt_to_img_similarity(target_text.replace("*", ""), gen_samples)
        dino_i = self.dino.img_to_img_similarity(src_images, gen_samples)
        return clip_i, clip_t, dino_i

    # Optional direct accessors if you want the same call pattern as CLIP
    def dino_img_to_img_similarity(self, src_images, generated_images):
        # optional guard for empty source
        if src_images is None or (hasattr(src_images, "numel") and src_images.numel() == 0):
            return torch.tensor(float("nan"), device=self.dino.model.device if hasattr(self.dino, "model") else "cpu")
        return self.dino.img_to_img_similarity(src_images, generated_images)

"""
Evaluation utilities for Dhariwal/EDM generative models.

This module bundles all the heavy lifting needed to evaluate generated images:
- FewShotDataset / ImageFolderDataset to load real images from disk for
  few-shot or FID-style evaluation.
- InceptionV3 wrapper plus patched FID Inception blocks (A/C/E) to reproduce
  the TensorFlow FID implementation (adapted from mseitzer/pytorch-fid).
- Functions to compute Inception activations, mean/covariance statistics,
  and Frechet Inception Distance (FID), both from folders and from tensors.
- FIDLoss nn.Module for training-time FID regularization against a reference
  stats npz.
- Evaluator class that, given a tensor of fake images and a real npz:
    * computes FID,
    * computes intra-LPIPS diversity using a few-shot real dataset,
    * computes precision/recall/density/coverage via PRDC metrics, with
      optional realism score (adapted from clovaai/generative-evaluation-prdc).

All functions assume images are RGB, and most metrics expect inputs in
[0,1] float32 or [-1,1] (with optional normalization handled internally).
"""


from typing import Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import sklearn.metrics
import sys
import numpy as np


class FewShotDataset(Dataset):
    """
    Loads all images from a single folder (no subfolders).
    Labels are just the index (0..N-1).
    """
    def __init__(self, root_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 extensions=("jpg", "jpeg", "png", "bmp", "webp")) -> None:
        self.root_dir = root_dir
        self.transform = transform
        exts = tuple(e.lower() for e in extensions)

        self.paths = []
        for fn in sorted(os.listdir(root_dir)):
            p = os.path.join(root_dir, fn)
            if os.path.isfile(p) and fn.lower().endswith(exts):
                self.paths.append(p)

        if not self.paths:
            raise RuntimeError(f"No images found in {root_dir} with extensions {exts}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple:
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = idx
        return img, label

#----------------------------------------------------------------------------

# FID Score

import os

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

FID_WEIGHTS_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"  # noqa: E501


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3,  # Final average pooling features
    }

    def __init__(
        self,
        output_blocks=(DEFAULT_BLOCK_INDEX,),
        resize_input=True,
        normalize_input=True,
        requires_grad=False,
        use_fid_inception=True,
    ):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert (
            self.last_needed_block <= 3
        ), "Last possible output block index is 3"

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(weights="DEFAULT")

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(
                x, size=(299, 299), mode="bilinear", align_corners=False
            )

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def _inception_v3(*args, **kwargs):
    """Wraps `torchvision.models.inception_v3`"""
    try:
        version = tuple(map(int, torchvision.__version__.split(".")[:2]))
    except ValueError:
        # Just a caution against weird version strings
        version = (0,)

    # Skips default weight inititialization if supported by torchvision
    # version. See https://github.com/mseitzer/pytorch-fid/issues/28.
    if version >= (0, 6):
        kwargs["init_weights"] = False

    # Backwards compatibility: `weights` argument was handled by `pretrained`
    # argument prior to version 0.13.
    if version < (0, 13) and "weights" in kwargs:
        if kwargs["weights"] == "DEFAULT":
            kwargs["pretrained"] = True
        elif kwargs["weights"] is None:
            kwargs["pretrained"] = False
        else:
            raise ValueError(
                "weights=={} not supported in torchvision {}".format(
                    kwargs["weights"], torchvision.__version__
                )
            )
        del kwargs["weights"]

    return torchvision.models.inception_v3(*args, **kwargs)


def fid_inception_v3():
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008, aux_logits=False, weights=None)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class FIDInceptionA(torchvision.models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


IMAGE_EXTENSIONS = {
    "bmp",
    "jpg",
    "jpeg",
    "pgm",
    "png",
    "ppm",
    "tif",
    "tiff",
    "webp",
}


class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform, max_images=None):
        paths = sorted([
            os.path.join(folder, fname)
            for fname in os.listdir(folder)
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
        if len(paths) == 0:
            raise ValueError(f"No images found under: {folder}")
        if (max_images is not None) and (len(paths) > max_images):
            paths = paths[:max_images]
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]).convert("RGB") as img:
            return self.transform(img)


class FakeImageDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def get_activations(
    samples, model, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    model.eval()

    if batch_size > len(samples):
        print(
            (
                "Warning: batch size is bigger than the data size. "
                "Setting batch size to data size"
            )
        )
        batch_size = len(samples)

    dataset = FakeImageDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    pred_arr = np.empty((len(samples), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(
    files, model, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act


def compute_statistics_of_path(path, batch_size=32, device="cuda", dims=2048,
                               num_workers=0, max_images=5000):
    """
    Compute FID stats for up to `max_images` images from `path`.
    If there are fewer images than `max_images`, use them all.
    """

    if path.endswith(".npz"):
        with np.load(path) as f:
            print("keys are ----------------------------")
            print(f.keys())
            return f["mu"][:], f["sigma"][:], f["act"][:]

    transform = transforms.Compose([transforms.ToTensor()])  # [0,1] float32

    dataset = ImageFolderDataset(path, transform, max_images=max_images)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=False if num_workers == 0 else True,
        pin_memory=False
    )

    # Stack (note: up to 5000 images can still be large; keep batch_size moderate)
    all_samples = []
    for batch in tqdm(dataloader, desc=f"Loading images from {path} (<= {len(dataset)} imgs)"):
        all_samples.append(batch)
    samples_tensor = torch.cat(all_samples, dim=0)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], resize_input=True, normalize_input=True).to(device).eval()

    mu, sigma, act = calculate_activation_statistics(
        samples_tensor, model, batch_size, dims, device, num_workers
    )
    return mu, sigma, act



def compute_statistics_of_tensor(
    samples, model, batch_size, dims, device, num_workers=1
):
    m, s, _ = calculate_activation_statistics(
        samples, model, batch_size, dims, device, num_workers
    )
    return m, s


def calculate_fid_given_paths(
    samples, path, batch_size, device, dims, num_workers=1
):
    """Calculates the FID of two paths"""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1, _ = compute_statistics_of_path(path)
    m2, s2 = compute_statistics_of_tensor(
        samples, model, batch_size, dims, device, num_workers
    )
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def calc_fid_score(
    samples, path, batch_size=50, dims=2048, device=None, num_workers=None
):
    if device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(device)

    if num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = num_workers

    fid_value = calculate_fid_given_paths(
        samples, path, batch_size, device, dims, num_workers
    )
    return fid_value


class FIDLoss(nn.Module):
    def __init__(
        self, npz_path, batch_size=10, dims=2048, num_workers=None
    ):
        super().__init__()
        self.npz_path = npz_path
        self.batch_size = batch_size
        self.dims = dims

        if num_workers is None:
            try:
                num_cpus = len(os.sched_getaffinity(0))
            except AttributeError:
                num_cpus = os.cpu_count()
            self.num_workers = min(num_cpus, 8) if num_cpus is not None else 0
        else:
            self.num_workers = num_workers

    def forward(self, samples):
        fid_value = calculate_fid_given_paths(
            samples,
            self.npz_path,
            batch_size=self.batch_size,
            device=samples.device,
            dims=self.dims,
            num_workers=self.num_workers,
        )
        return fid_value


#----------------------------------------------------------------------------

# Evaluation util

from tqdm import tqdm

import numpy as np

import torch
import lpips
from torch.utils.data import DataLoader
from torchvision import transforms


def load_real_features_from_npz(npz_path: str) -> np.ndarray:
    with np.load(npz_path) as f:
        for k in ('act', ):
            if k in f:
                return f[k].astype(np.float32)
    raise ValueError(
        f"{npz_path} has no per-image features. "
        f"Store them under key 'act' (see build_real_features_npz)."
    )

class Evaluator:
    def __init__(
        self,
        args,
        fake_images,
        fid_npz_path,
        cluster_size,
        device="cuda",
    ):
        assert len(fake_images.shape) == 4

        self.lpips_fn = lpips.LPIPS(net="vgg").to(device)
        self.lpips_fn.eval()
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
        dataset = FewShotDataset(root_dir=args.fewshotdataset, transform=transform)
        self.real_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.fake_images = fake_images
        self.cluster_size = cluster_size
        self.fid_npz_path = fid_npz_path
        self.args = args

    def calc_intra_lpips(self, device: str = "cuda") -> float:
        cluster = {i: [] for i in range(10)}
        b, _, _, _ = self.fake_images.shape
        for i in tqdm(range(b)):
            dists = []
            for batch in self.real_loader:
                real_image, _ = batch
                if self.args.normalization:
                    real_image = real_image * 2 - 1
                real_image = real_image.to(device)
                with torch.no_grad():
                    dist = self.lpips_fn(
                        self.fake_images[i, :, :, :].unsqueeze(0).cuda(),
                        real_image,
                    )
                    dists.append(dist.item())
            cluster[int(np.argmin(dists))].append(i)

        dists = []
        cluster = {c: cluster[c][: self.cluster_size] for c in cluster}

        for c in tqdm(cluster):
            temp = []
            cluster_length = len(cluster[c])
            for i in tqdm(range(cluster_length)):
                img1 = (
                    self.fake_images[cluster[c][i], :, :, :].unsqueeze(0).cuda()
                )
                for j in range(i + 1, cluster_length):
                    img2 = (
                        self.fake_images[cluster[c][j], :, :, :]
                        .unsqueeze(0)
                        .cuda()
                    )
                    with torch.no_grad():
                        pairwise_dist = self.lpips_fn(img1, img2)
                        temp.append(pairwise_dist.item())
            if temp:
                dists.append(np.mean(temp))
            else:
                print("**************EMPTY****************")
                dists.append(0)
        dists = np.array(dists)
        intra_lpips = dists[~np.isnan(dists)].mean()
        return intra_lpips

    def calc_fid(self):
        return calc_fid_score(
            self.fake_images, self.fid_npz_path, num_workers=0
        )

    def calc_precision_recall(
        self,
        device: str = "cuda",
        nearest_k: int = 5,
        dims: int = 2048,
        batch_size: int = 64,
        num_workers: int = 0,
        max_fake: int = 5000,
        realism: bool = False,
        return_all: bool = False,
    ):
        # Inception head (same as FID)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        inc = InceptionV3([block_idx], resize_input=True, normalize_input=True).to(device).eval()

        # ---- REAL: load per-image features from npz ----
        # Prefer the FID npz if it already includes feats; otherwise point to a sibling e.g. ffhq_feats.npz
        try:
            real_feats = load_real_features_from_npz(self.fid_npz_path)  # [Nr, D] float32
        except Exception:
            # try "<base>_feats.npz"
            base, ext = os.path.splitext(self.fid_npz_path)
            feats_npz = base + "_feats.npz"
            real_feats = load_real_features_from_npz(feats_npz)

        # ---- FAKE: compute features from the generated tensor ----
        fake = self.fake_images
        if max_fake is not None:
            fake = fake[:max_fake]

        if fake.min() < 0:   # ensure [0,1]
            fake = (fake + 1) / 2

        fake_feats = get_activations(
            fake.to(device), inc, batch_size=batch_size, dims=dims,
            device=device, num_workers=num_workers
        ).astype(np.float32)

        prdc_out = compute_prdc(
            real_features=real_feats,
            fake_features=fake_feats,
            nearest_k=nearest_k,
            realism=realism
        )
        return prdc_out if return_all else (float(prdc_out['precision']), float(prdc_out['recall']))


# -------------------- Precision and Recall --------------------
"""
prdc from https://github.com/clovaai/generative-evaluation-prdc 
Copyright (c) 2020-present NAVER Corp.
MIT license
Modified to also report realism score from https://arxiv.org/abs/1904.06991
"""

import numpy as np
import sklearn.metrics
import sys

def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k, realism=False):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    print('Num real: {} Num fake: {}'
          .format(real_features.shape[0], fake_features.shape[0]), file=sys.stderr)

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    d = dict(precision=precision, recall=recall,
                density=density, coverage=coverage)

    if realism:
        """
        Large errors, even if they are rare, would undermine the usefulness of the metric.
        We tackle this problem by discarding half of the hyperspheres with the largest radii.
        In other words, the maximum in Equation 3 is not taken over all φr ∈ Φr but only over 
        those φr whose associated hypersphere is smaller than the median.
        """
        mask = real_nearest_neighbour_distances < np.median(real_nearest_neighbour_distances)

        d['realism'] = (
                np.expand_dims(real_nearest_neighbour_distances[mask], axis=1)/distance_real_fake[mask]
        ).max(axis=0)

    return d
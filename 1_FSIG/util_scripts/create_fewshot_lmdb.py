"""
few_shot_lmdb.py

Build an LMDB dataset from a (few-shot) image folder.

Features:
- Recursively scans `--data_path` for images with extensions: .png, .jpg, .jpeg, .webp.
- Automatically builds or reuses `dataset.json` containing (relative_path -> class_id) pairs.
  * If `dataset.json` exists, it is loaded.
  * Otherwise, labels are inferred from a class-per-subdir structure:
        data_path/
          class_0/ *.png, *.jpg, ...
          class_1/ ...
          ...
    and a new `dataset.json` is written.
- Stores images as CHW uint8 in LMDB with keys:
      images_{idx}_data
      labels_{idx}_data
- Also stores metadata:
      images_shape, labels_shape, images_dtype, labels_dtype

This script is useful to convert few-shot classification datasets into a compact LMDB
format for faster IO during training.

Example:
    python few_shot_lmdb.py \\
        --data_path 1_FSIG/datasets/targets/10_babies/0 \\
        --lmdb_path 1_FSIG/datasets/targets/10_babies_lmdb \\
        --force_rgb

Arguments:
    --data_path   Root folder of images.
    --lmdb_path   Output LMDB directory.
    --chunk_size  Number of images per LMDB commit (default: 2048).
    --map_size_gb LMDB map size in GB (default: 64).
    --force_rgb   Convert all images to RGB (3 channels).
"""


from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import torch
import lmdb
import glob
import json
import os
from pathlib import Path

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")

def list_images_rec(root):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    files = sorted(files)
    return files

def build_or_load_labels(data_path):
    label_path = os.path.join(data_path, "dataset.json")
    if os.path.exists(label_path):
        labels = dict(json.load(open(label_path))["labels"])
        return labels

    # Build labels from folder structure (class-per-subdir)
    print(f"[INFO] {label_path} not found. Creating it...")
    labels = {}
    class_dirs = sorted([p for p in glob.glob(os.path.join(data_path, "*")) if os.path.isdir(p)])
    class_to_idx = {os.path.basename(d): i for i, d in enumerate(class_dirs)}

    total = 0
    for cls_name, idx in class_to_idx.items():
        for ext in IMG_EXTS:
            for fp in glob.glob(os.path.join(data_path, cls_name, f"*{ext}")):
                key = os.path.join(cls_name, os.path.basename(fp))
                labels[key] = idx
                total += 1

    with open(label_path, "w") as f:
        json.dump({"labels": list(labels.items())}, f, indent=4)
    print(f"[INFO] Created dataset.json with {total} entries.")
    return labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="root folder of images")
    parser.add_argument("--lmdb_path", type=str, required=True, help="path to LMDB dir")
    parser.add_argument("--chunk_size", type=int, default=2048, help="images per LMDB commit")
    parser.add_argument("--map_size_gb", type=float, default=64.0, help="LMDB map size (GB)")
    parser.add_argument("--force_rgb", action="store_true", help="convert to RGB")
    args = parser.parse_args()

    # Collect files (flat or nested)
    files = list_images_rec(args.data_path)
    if len(files) == 0:
        raise ValueError(f"No images found under: {args.data_path}")

    # Load or build labels
    labels_dict = build_or_load_labels(args.data_path)

    # Compute shape using the first image
    with Image.open(files[0]) as im0:
        if args.force_rgb:
            im0 = im0.convert("RGB")
        arr0 = np.array(im0)
    if arr0.ndim != 3 or arr0.shape[2] not in (1, 3, 4):
        raise ValueError(f"Unexpected image shape for {files[0]}: {arr0.shape}")
    H, W = arr0.shape[:2]
    C = 3 if args.force_rgb else (arr0.shape[2] if arr0.ndim == 3 else 1)

    # LMDB open with sane flags for NFS
    map_size = int(args.map_size_gb * (1024**3))
    env = lmdb.open(
        args.lmdb_path,
        map_size=map_size,
        subdir=True,
        lock=True,
        readahead=False,   # better when dataset is bigger than RAM / NFS
        writemap=False,    # safer default on network FS
        map_async=True,    # queue disk writes; call env.sync() at the end
    )

    # Write loop (streaming, chunked)
    txn = env.begin(write=True)
    n_written = 0

    pbar = tqdm(files, desc="Writing LMDB")
    for fp in pbar:
        # derive label key as "<class>/<filename>"
        parts = Path(fp).parts
        # try to find the class folder relative to data_path
        rel = str(Path(fp).relative_to(args.data_path))
        cls_and_file = rel.replace("\\", "/")  # normalize
        # If dataset.json was created from class folders, labels_dict expects "<class>/<file>"
        # If there are no class folders, you can use a single-class setting or adapt here:
        if cls_and_file not in labels_dict:
            # fallback: no-class structure -> single class 0
            # or raise error; here we choose single class
            label = 0
        else:
            label = labels_dict[cls_and_file]

        with Image.open(fp) as im:
            if args.force_rgb:
                im = im.convert("RGB")
            arr = np.array(im, dtype=np.uint8)
        if arr.ndim == 2:  # grayscale
            arr = np.expand_dims(arr, -1)
        if arr.shape[2] != C:
            # normalize channels according to first image
            if C == 3:
                arr = np.array(Image.fromarray(arr.squeeze() if arr.shape[2] == 1 else arr).convert("RGB"), dtype=np.uint8)
            elif C == 1:
                arr = np.array(Image.fromarray(arr).convert("L"), dtype=np.uint8)[..., None]
        arr = arr.transpose(2, 0, 1).copy(order="C")  # CHW

        # Put bytes
        txn.put(f"images_{n_written}_data".encode(), arr.tobytes())
        txn.put(f"labels_{n_written}_data".encode(), np.int64(label).tobytes())
        n_written += 1

        # Commit periodically
        if (n_written % args.chunk_size) == 0:
            txn.commit()
            txn = env.begin(write=True)
            pbar.set_postfix(committed=n_written)

    # Final commit
    txn.commit()

    # Store metadata (no huge prints)
    with env.begin(write=True) as meta_txn:
        meta_txn.put(b"images_shape", f"{n_written} {C} {H} {W}".encode())
        meta_txn.put(b"labels_shape", f"{n_written}".encode())
        meta_txn.put(b"images_dtype", b"uint8")
        meta_txn.put(b"labels_dtype", b"int64")

    env.sync()   # flush async writes
    env.close()
    print(f"[INFO] Done. Wrote {n_written} items to {args.lmdb_path} with shape [{n_written}, {C}, {H}, {W}].")

if __name__ == "__main__":
    main()

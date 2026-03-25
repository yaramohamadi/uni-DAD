# create_instance_lmdb.py is the second stage of the data pipeline: it reads the
# metadata manifest and packs images, prompts, and labels into one LMDB that the
# training-time dataset wrappers can read efficiently.
# create_instance_lmdb.py
from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np
import argparse
import lmdb, json, os

def put_kv(txn, key_str, value_bytes):
    txn.put(key_str.encode("utf-8"), value_bytes)

# The writer keeps image/prompt/label rows on the same numeric index so later
# dataset classes can reconstruct aligned samples with simple key lookups.
def store_images_labels_prompts(env, triplets, start_index=0):
    """
    triplets: list of (np.uint8 CHW image, int label, utf8 prompt)
    """
    with env.begin(write=True) as txn:
        for i, (img_chw, label, prompt) in enumerate(triplets):
            idx = start_index + i
            put_kv(txn, f"images_{idx:06d}_data", img_chw.tobytes())
            put_kv(txn, f"labels_{idx:06d}_data", np.int64(label).tobytes())
            put_kv(txn, f"prompts_{idx:06d}_data", prompt.encode("utf-8"))

# This script normalizes raw files into the LMDB layout expected by
# lmdb_dataset.py, sd_image_dataset.py, and sd_text_dataset.py.
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="root folder containing images")
    p.add_argument("--metadata_json", required=True, help="JSON with items[{file,prompt,label?}]")
    p.add_argument("--lmdb_path", required=True, help="output .lmdb directory")
    p.add_argument("--image_size", type=int, default=None, help="optional, resize shorter side")
    p.add_argument("--map_gb", type=float, default=100.0, help="LMDB map size in GB")
    p.add_argument("--target_size", type=int, default=512, help="Final square size (W=H=target_size)")
    args = p.parse_args()

    # IMPORTANT: map_size must be <= free disk space
    env = lmdb.open(args.lmdb_path, map_size=int(args.map_gb * (1024**3)))

    meta = json.load(open(args.metadata_json))
    items = meta["items"]
    # Allow fallback to “discover all images under data_root” if no items, but prompts need JSON.
    if not items:
        raise ValueError("metadata_json has no items. Provide file/prompt pairs.")

    triplets = []
    H = W = C = None

    # Iterate over manifest rows, load/crop/resize each image, and convert it to
    # the CHW uint8 format that all LMDB readers in this project expect.
    for it in tqdm(items, desc="Loading"):
        rel = it["file"]
        prompt = it["prompt"]
        label = int(it.get("label", 0))

        path = os.path.join(args.data_root, rel)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        # Load image -> CHW uint8
        img = Image.open(path).convert("RGB")
        # Make every image exactly target_size x target_size (center-crop + resize)
        target = args.target_size  # e.g., 512
        img = ImageOps.fit(img, (target, target), method=Image.BICUBIC)
        if args.image_size is not None:
            # simple longest-side preserve? here: resize shorter side to image_size
            w, h = img.size
            if min(w, h) != args.image_size:
                if w < h:
                    new_w = args.image_size
                    new_h = int(h * (args.image_size / w))
                else:
                    new_h = args.image_size
                    new_w = int(w * (args.image_size / h))
                img = img.resize((new_w, new_h), Image.BICUBIC)
        
        img_np = np.array(img, dtype=np.uint8)
        # Transpose to CHW
        img_chw = img_np.transpose(2, 0, 1)

        if H is None:
            C, H, W = 3, args.target_size, args.target_size

        else:
            # (Optional) center-crop to match first image size if sizes vary
            c, h, w = img_chw.shape
            assert c == C, "All images must have 3 channels (RGB)"
            if h != H or w != W:
                # center-crop to (H, W)
                top = max(0, (h - H) // 2); left = max(0, (w - W) // 2)
                img_chw = img_chw[:, top:top+H, left:left+W]
                if img_chw.shape[1:] != (H, W):
                    raise ValueError(f"Image {path} has size {h}x{w}, unable to match first image size {H}x{W}")

        triplets.append((img_chw, label, prompt))

    # Write rows
    store_images_labels_prompts(env, triplets, start_index=0)

    # Shape keys are part of the reader contract: downstream datasets use them to
    # infer dataset length and how to decode each stored array.
    # Save shapes
    N = len(triplets)
    with env.begin(write=True) as txn:
        # images: N C H W
        img_shape_str = f"{N} {C} {H} {W}"
        put_kv(txn, "images_shape", img_shape_str.encode("utf-8"))

        # labels: N 1
        lbl_shape_str = f"{N} 1"
        put_kv(txn, "labels_shape", lbl_shape_str.encode("utf-8"))

        # prompts: just N (variable-length strings)
        prm_shape_str = f"{N}"
        put_kv(txn, "prompts_shape", prm_shape_str.encode("utf-8"))

    print(f"Done. Wrote {N} samples to {args.lmdb_path}")

if __name__ == "__main__":
    main()

"""
DATA_ROOT="${DATA_ROOT:-./data/instance_images}"
META_JSON="${META_JSON:-./data/metadata.json}"
LMDB_PATH="${LMDB_PATH:-./data/instances.lmdb}"
TARGET_SIZE="${TARGET_SIZE:-512}"
MAP_GB="${MAP_GB:-50}"

python data/create_instance_lmdb.py \
  --data_root "$DATA_ROOT" \
  --metadata_json "$META_JSON" \
  --lmdb_path "$LMDB_PATH" \
  --map_gb "$MAP_GB" \
  --target_size "$TARGET_SIZE"
"""
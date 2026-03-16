"""
resize_in_place.py

Utility script to resize all images in a given folder to a fixed resolution
and convert them to PNG format in-place.

⚠️ This script is destructive:
    - Original files are removed after conversion.
    - Run on a backup or a copy of your data if you want to keep the originals.

Example:
    folder = "0_myfiles_face/datasets/targets/5_babies/0"
    resize_in_place(folder, size=(256, 256))
"""

import os
from PIL import Image

def resize_in_place(folder, size=(256, 256)):
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)

        if not os.path.isfile(fpath):
            continue

        try:
            with Image.open(fpath) as img:
                img = img.convert("RGB")
                img_resized = img.resize(size, Image.BILINEAR)

                # always save as PNG
                base, _ = os.path.splitext(fname)
                new_path = os.path.join(folder, f"{base}.png")
                img_resized.save(new_path, format="PNG")

            # delete old file if it wasn't the new PNG
            if not fname.lower().endswith(".png") or fpath != new_path:
                os.remove(fpath)

            print(f"Resized and saved: {new_path}")

        except Exception as e:
            print(f"Skipping {fname}: {e}")

if __name__ == "__main__":
    folder = "0_myfiles_face/datasets/targets/5_babies/0"
    resize_in_place(folder, size=(256, 256))

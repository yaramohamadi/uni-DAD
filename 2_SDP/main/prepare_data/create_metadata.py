# create_metadata.py is the first stage of the instance-data pipeline.
# It turns a folder tree of subject images into a stable JSON manifest that
# later scripts use to build LMDBs and keep label/prompt assignments reproducible.
import os, json, argparse, glob
from collections import OrderedDict

# SPECIFY THE CLASS FOR EACH SUBJECT (stable mapping for reproducibility) - ex. Dreambooth dataset 
# (subject, class) pairs - the class is used for prompting and evaluation, and can be shared across subjects (e.g. multiple dog subjects can all be "dog" class)
# Note: the subject names must match the folder names in data_root, and the class names are used for prompting and evaluation (e.g. "a prt {class}")
# This mapping is intentionally fixed in source control so experiments, prompt
# generation, and evaluation all agree on which semantic class each subject belongs to.
SUBJECT_TO_CLASS = OrderedDict([
    ("backpack", "backpack"),
    ("backpack_dog", "backpack"),
    ("bear_plushie", "stuffed animal"),
    ("berry_bowl", "bowl"),
    ("can", "can"),
    ("candle", "candle"),
    ("cat", "cat"),
    ("cat2", "cat"),
    ("clock", "clock"),
    ("colorful_sneaker", "sneaker"),
    ("dog", "dog"),
    ("dog2", "dog"),
    ("dog3", "dog"),
    ("dog5", "dog"),
    ("dog6", "dog"),
    ("dog7", "dog"),
    ("dog8", "dog"),
    ("duck_toy", "toy"),
    ("fancy_boot", "boot"),
    ("grey_sloth_plushie", "stuffed animal"),
    ("monster_toy", "toy"),
    ("pink_sunglasses", "glasses"),
    ("poop_emoji", "toy"),
    ("rc_car", "toy"),
    ("red_cartoon", "cartoon"),
    ("robot_toy", "toy"),
    ("shiny_sneaker", "sneaker"),
    ("teapot", "teapot"),
    ("vase", "vase"),
    ("wolf_plushie", "stuffed animal"),
])

# The output JSON is the contract consumed by create_instance_lmdb.py: each item
# carries a relative file path plus the prompt and label that should travel with it.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="Root directory containing one subfolder per subject, each with images like XX.jpg")
    ap.add_argument("--out_json", required=True, help="Output metadata.json path")
    ap.add_argument("--extensions", nargs="+", default=[".jpg", ".jpeg", ".png"],
                    help="Image extensions to include (case-insensitive)")
    args = ap.parse_args()

    # Normalize extensions to lower-case
    exts = set(e.lower() for e in args.extensions)

    # Build a stable class_to_id mapping
    classes = sorted(set(SUBJECT_TO_CLASS.values()))
    class_to_id = {c: i for i, c in enumerate(classes)}

    items = []
    missing_subjects = []
    empty_subjects = []

    # Walk the canonical subject list in a deterministic order so the resulting
    # manifest stays stable across machines and repeated preprocessing runs.
    for subject, cls in SUBJECT_TO_CLASS.items():
        subj_dir = os.path.join(args.data_root, subject)
        if not os.path.isdir(subj_dir):
            missing_subjects.append(subject)
            continue

        # Grab images like XX.jpg (but we’ll accept any basename with allowed extensions)
        paths = []
        for pattern in ["*.jpg", "*.jpeg", "*.png", "*.*"]:
            paths.extend(glob.glob(os.path.join(subj_dir, pattern)))
        # Filter by extension set
        paths = [p for p in sorted(set(paths))
                 if os.path.splitext(p)[1].lower() in exts]

        if not paths:
            empty_subjects.append(subject)
            continue

        # Each manifest row becomes one LMDB row later on, so file/prompt/label
        # alignment is established here and preserved through the rest of the pipeline.
        for p in paths:
            rel = os.path.relpath(p, args.data_root)
            prompt = f"a prt {cls}"
            label = class_to_id[cls]

            items.append({
                "file": rel.replace("\\", "/"),
                "prompt": prompt,
                "label": label,
                "subject": subject,
                "class": cls
            })

    meta = {
        "class_to_id": class_to_id,
        "items": items
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(items)} items to {args.out_json}")
    if missing_subjects:
        print(f"[WARN] Missing subject folders: {missing_subjects}")
    if empty_subjects:
        print(f"[WARN] No images found in: {empty_subjects}")

if __name__ == "__main__":
    main()


"""
DATA_ROOT=path/to/data
META_JSON=path/to/metadata.json
EXTENSIONS=".jpg .jpeg .png"

python data/create_metadata.py \
  --data_root "$DATA_ROOT" \
  --out_json "$META_JSON" \
  --extensions "${EXTENSIONS[@]}"
"""
# Prepare Data

This folder contains the preprocessing steps used to build the instance LMDB consumed by the SDP training pipeline.

## What each script does

- `download_instances.sh`: Stage 0. Either downloads the archived DreamBooth subject folders or copies your own local subject-image folders into `data/instance_images`.
- `create_metadata.py`: Stage 1. Scans a root folder of subject images and writes a `metadata.json` manifest.
- `create_instance_lmdb.py`: Stage 2. Reads `metadata.json`, center-crops/resizes each image, and writes images, labels, and prompts into one LMDB.
- `lmdb_dataset.py`: Generic LMDB reader that returns image + label + prompt triples.
- `sd_image_dataset.py`: Stable Diffusion image-side dataset view used when training code needs RGB images and prompts.
- `sd_text_dataset.py`: Stable Diffusion text-side dataset view used when training code only needs prompt token ids.

## Expected input layout

`create_metadata.py` expects one subfolder per subject under `--data_root`.
The folder names must match the keys in `SUBJECT_TO_CLASS` inside `create_metadata.py`, because that mapping decides:

- which class name each subject belongs to
- which prompt is written for each image
- which integer label is assigned in the manifest

Each image row is stored with the prompt format:

```text
a prt <class>
```

## Optional: Populate `data/instance_images`

`download_dreambooth_instances.sh` can fill `data/instance_images` in two ways.

### Option A: Download the DreamBooth instance set

```bash
bash main/prepare_data/download_dreambooth_instances.sh
```

In this mode the script:

- downloads the archived `google/dreambooth` `dataset/` folder
- copies all subject folders into `data/instance_images`
- copies `prompts_and_classes.txt` into `data/prompts_and_classes.txt` when available
- skips any subject folder that already exists locally unless overwrite is enabled

### Option B: Copy your own local subject images

If you already have images on your machine, you can import them instead of downloading DreamBooth.

For a folder that already contains one subfolder per subject:

```bash
bash main/prepare_data/download_dreambooth_instances.sh \
  --local-source /path/to/my_subjects_root
```

For one folder that contains images for a single subject directly:

```bash
bash main/prepare_data/download_dreambooth_instances.sh \
  --local-source /path/to/my_subject \
  --subject-name my_subject
```

### Overwrite existing folders

```bash
bash main/prepare_data/download_dreambooth_instances.sh \
  --force-overwrite
```

## Step 1: Build the metadata manifest

Run `create_metadata.py` on your raw instance-image folders.

```bash
python main/prepare_data/create_metadata.py \
  --data_root ./data/instance_images \
  --out_json ./data/metadata.json \
  --extensions .jpg .jpeg .png
```

This writes a JSON file with:

- `class_to_id`: stable class-to-label mapping
- `items`: one row per image with `file`, `prompt`, `label`, `subject`, and `class`

If a subject folder is missing or empty, the script prints a warning.

## Step 2: Convert the manifest into LMDB

Run `create_instance_lmdb.py` after `metadata.json` exists.

```bash
python main/prepare_data/create_instance_lmdb.py \
  --data_root ./data/instance_images \
  --metadata_json ./data/metadata.json \
  --lmdb_path ./data/instances.lmdb \
  --map_gb 50 \
  --target_size 512
```

This script:

- reads the manifest rows in order
- loads each image from `data_root/file`
- converts it to RGB
- center-crops and resizes it to `target_size x target_size`
- stores aligned `images_*`, `labels_*`, and `prompts_*` entries in LMDB
- writes shape keys such as `images_shape`, `labels_shape`, and `prompts_shape`

Important options:

- `--target_size`: final square image size written to LMDB
- `--map_gb`: LMDB map size; increase it if LMDB runs out of space
- `--image_size`: optional extra resize path kept in the script, but the main square output is controlled by `--target_size`

## How the training pipeline uses the result

`train_sd.py` does not read raw image folders directly. It reads the generated LMDB through the dataset wrappers in this folder:

- `SDImageDatasetLMDB` for image-space samples and prompts
- `SDTextDatasetLMDB` for prompt-only/tokenized batches

So the normal preparation flow is:

1. organize instance images into subject folders
2. run `create_metadata.py`
3. run `create_instance_lmdb.py`
4. point training to the generated `instances.lmdb`

## Output files to expect

After preprocessing, the usual files are:

- `data/metadata.json`
- `data/instances.lmdb/`

These are the two artifacts that connect raw subject images to the SDP training pipeline.

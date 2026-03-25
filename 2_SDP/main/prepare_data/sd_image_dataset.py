# sd_image_dataset.py is the image-side training view of the LMDB. The trainer uses
# it when guidance or denoising code needs RGB samples, prompts, and optional labels.
from main.utils import retrieve_row_from_lmdb, get_array_shape_from_lmdb
from torch.utils.data import Dataset
import numpy as np 
import torch
import lmdb 
from typing import Optional, Any, Dict 

# Older/preprocessed LMDBs in this project do not always agree on prompt key naming,
# so prompt lookup is intentionally permissive to keep training compatible.
def _get_prompt_from_lmdb(env, idx: int) -> str:
    """
    Fetch a prompt string for a given index from LMDB, robust to key naming.
    Tries multiple key variants and finally falls back to a cursor prefix scan.
    """
    key_variants = [
        f"prompts_{idx:06d}_data",
        f"prompts_{idx}_data",
        f"prompts_{idx:06d}",
        f"prompts_{idx}",
    ]
    with env.begin(write=False) as txn:
        # Try the common variants first
        for k in key_variants:
            b = txn.get(k.encode("utf-8"))
            if b is not None:
                try:
                    return b.decode("utf-8")
                except UnicodeDecodeError:
                    # Try latin-1 as a fallback in case of odd writer encodings
                    try:
                        return b.decode("latin-1")
                    except Exception:
                        pass
        # Fallback: scan for any key starting with "prompts_{idx}"
        prefix = f"prompts_{idx}".encode("utf-8")
        cur = txn.cursor()
        try:
            # LMDB keys are sorted; seek to the first possible position
            if cur.set_range(prefix):
                for k, v in cur:
                    if not k.startswith(prefix):
                        break
                    # Accept keys like: prompts_<idx>, prompts_<idx>_data, prompts_<idx>_something
                    try:
                        return v.decode("utf-8")
                    except UnicodeDecodeError:
                        try:
                            return v.decode("latin-1")
                        except Exception:
                            continue
        except Exception:
            # Cursor errors shouldn't crash training; we fall through to the final error
            pass

    raise KeyError(
        f"Unable to find a prompt for index {idx}. Tried variants: "
        f"{', '.join(key_variants)} and a prefix scan."
    )


# This wrapper prepares samples for Stable Diffusion-style training: images are
# returned in [-1,1], prompts stay available as raw strings, and tokenization is optional.
class SDImageDatasetLMDB(Dataset):
    def __init__(self, dataset_path, tokenizer_one: Optional[Any] = None): #set tokenizer as optional for when creating evaluation ds
        self.KEY_TO_TYPE = {
            'images': np.uint8
            #'latents': np.float16
        }
        self.dataset_path = dataset_path
        self.tokenizer_one = tokenizer_one

        #self.env = lmdb.open(dataset_path, readonly=True, lock=False, readahead=False, meminit=False)
        #self.latent_shape = get_array_shape_from_lmdb(self.env, "latents")

        #self.length = self.latent_shape[0]
        #self.env = lmdb.open(dataset_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.env = lmdb.open(dataset_path, readonly=True, lock=False, readahead=False, meminit=False)
        # LMDB stores 'images_shape' = "N C H W"
        self.image_shape = get_array_shape_from_lmdb(self.env, "images")
        self.length = self.image_shape[0]
        print(f"Dataset length: {self.length}")
    
        
    def __len__(self):
        return self.length

        # The training pipeline reads the same LMDB row through different dataset
        # adapters; this view focuses on image-space tensors plus prompt metadata.
    def __getitem__(self, idx):
        #image = retrieve_row_from_lmdb( 
        # # self.env, 
        # # "latents", self.KEY_TO_TYPE['latents'], self.latent_shape[1:], idx #) 
        # #image = torch.tensor(image, dtype=torch.float32) 
        # # Read uint8 CHW image and scale to [0,1] float32

        # ---- Image: CHW uint8 -> float32 [0,1] -> [-1,1] ----
        img_np = retrieve_row_from_lmdb(
            self.env,
            "images", self.KEY_TO_TYPE["images"], self.image_shape[1:], idx
        ).astype(np.float32) / 255.0   # CHW in [0,1]
        image = torch.from_numpy(img_np) * 2.0 - 1.0   # CHW in [-1,1]

        # ---- Prompt (robust retrieval) ----
        prompt: str
        try:
            prompt = _get_prompt_from_lmdb(self.env, idx)
        except Exception:
            prompt = ""

        # ---- Optional: class label (if present in LMDB) ----
        for key, dtype in (("class_labels", np.int64), ("labels", np.int64), ("label", np.int64)):
            try:
                lab_np = retrieve_row_from_lmdb(self.env, key, dtype, (), idx)  # scalar
                class_label = int(lab_np)
                break
            except Exception:
                continue

        # ---- Base output dict (always returned) ----
        output_dict: Dict[str, Any] = {
            "images": image,                 # CHW in [-1,1]
            "prompt": prompt,                # keep raw prompt string (handy in debug)
        }

        if class_label is not None:
            output_dict["class_labels"] = class_label  # <-- standardize the name

        # ---- Tokenize only if tokenizer(s) provided ----
        # Tokenization happens here only when the caller wants image samples and text
        # ids together; otherwise the raw prompt stays available for later handling.
        if self.tokenizer_one is not None:
            text_input_ids_one = self.tokenizer_one(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer_one.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            output_dict["text_input_ids_one"] = text_input_ids_one

        return output_dict


# sd_text_dataset.py is the text-only training view of the LMDB. The generator and
# guidance prompt streams use it when they only need token ids and raw prompt text.
import matplotlib
matplotlib.use('Agg')
import torch
import lmdb


# load text data from lmdb instance and prompts 
# Like the image-side loader, this helper accepts multiple key formats so prompt-only
# training remains robust to older LMDB variants.
def _lmdb_get_text(env, array_name, idx: int) -> str:
    with env.begin(write=False) as txn:
        tried = []
        for pad in (6, 5, 4, 3, 2, 1, 0):
            key = (f"{array_name}_{idx:0{pad}d}_data" if pad else f"{array_name}_{idx}_data").encode("utf-8")
            tried.append(key.decode())
            b = txn.get(key)
            if b is not None:
                try:
                    return b.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        return b.decode("latin-1")
                    except Exception:
                        pass
        # Fallback: scan for any key starting with "prompts_{idx}"
        prefix=f"{array_name}_{idx}".encode("utf-8")
        cur = txn.cursor()
        try:
            if cur.set_range(prefix):
                for k, v in cur:
                    if not k.startswith(prefix):
                        break
                    try:
                        return v.decode("utf-8")
                    except UnicodeDecodeError:
                        try:
                            return v.decode("latin-1")
                        except Exception:
                            continue
        except Exception:
            pass
    raise KeyError(f"Missing text for idx={idx}. Tried: {tried}")
    
# load image data from lmdb instance
def _lmdb_get_shape(env, array_name):
    with env.begin(write=False) as txn:
        b = txn.get(f"{array_name}_shape".encode("utf-8"))
        if b is None:
            raise KeyError(f"Missing shape key: {array_name}_shape")
        parts = b.decode("utf-8").strip().split()
        ints = tuple(int(p) for p in parts)
        return ints if len(ints) > 1 else ints[0]

from torch.utils.data import Dataset

# This dataset exposes the prompt stream expected by train_sd.py: token ids for the
# current sample plus the raw prompt for logging/debugging and teacher-prompt handling.
class SDTextDatasetLMDB(Dataset):
    """
    Reads 'prompts' from an LMDB created with keys: prompts_{idx:06d}_data (utf-8 string),
    and a shape key 'prompts_shape' = "N".
    """
    def __init__(self, lmdb_path, tokenizer_one):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        shape = _lmdb_get_shape(self.env, "prompts")   # -> (N,) or N
        self.N = shape[0] if isinstance(shape, tuple) else int(shape)
        self.tokenizer_one = tokenizer_one
       

    def __len__(self):
        return self.N

        # Tokenize one prompt row from the shared LMDB and keep the original string so
        # later pipeline stages can derive rare-token and teacher variants from it.
    def __getitem__(self, idx):
        prompt = _lmdb_get_text(self.env, "prompts", idx)
        ids = self.tokenizer_one(
            [prompt], max_length=self.tokenizer_one.model_max_length,
            return_tensors="pt", padding="max_length", truncation=True
        ).input_ids[0].to(dtype=torch.long)          # [1, L]
        # match existing code path that expects squeeze(1) later:
        return {
            "text_input_ids_one": ids,
            "raw_prompt": prompt
            }  # [1,1,L] -> collate -> [B,1,L]

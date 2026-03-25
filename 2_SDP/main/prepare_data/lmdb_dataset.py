# This is the generic LMDB reader for image + label + prompt triples written by
# create_instance_lmdb.py. It is useful for evaluation/debug paths that want the
# full sample instead of the SD-specific text-only or image-only views.
from main.utils import retrieve_row_from_lmdb, get_array_shape_from_lmdb
from torch.utils.data import Dataset
import numpy as np, torch, lmdb 
import torch
import lmdb 



# For retrieving text prompts, we stored them as bytes in LMDB. This helper decodes them back to strings.
# Note: this assumes UTF-8 encoding, which is standard for text. If you used a different encoding when writing, adjust accordingly.
def retrieve_text_from_lmdb(env, array_name, idx):
    key = f"{array_name}_{idx:06d}_data".encode("utf-8")
    with env.begin(write=False) as txn:
        b = txn.get(key)
        if b is None:
            raise KeyError(f"Missing key {key!r}")
        return b.decode("utf-8")

# Unlike the SD-specific dataset wrappers, LMDBDataset returns a simple combined
# sample dict and keeps the image in [0,1] for general-purpose consumers.
class LMDBDataset(Dataset):
    # LMDB version of an ImageDataset. It is suitable for large datasets.
    def __init__(self, dataset_path):
        # for supporting new datasets, please adapt the data type according to the one used in "main/data/create_imagenet_lmdb.py"
        self.KEY_TO_TYPE = {
            'labels': np.int64,
            'images': np.uint8,
        }

        self.dataset_path = dataset_path
        
        self.env = lmdb.open(dataset_path, readonly=True, lock=False, readahead=False, meminit=False)

        self.image_shape = get_array_shape_from_lmdb(self.env, 'images')
        self.label_shape = get_array_shape_from_lmdb(self.env, 'labels')
        # For prompts, we only stored N; return an int or a tuple (N,)
        self.prompt_len = get_array_shape_from_lmdb(self.env, 'prompts')
        # Some DMD2 utils may return a tuple-of-ints; normalize:
        self.N = int(self.image_shape[0])


    def __len__(self):
        return self.N

        # Rebuild the aligned row that was written under the same index in the LMDB:
        # RGB image, class label, and raw prompt string.
    def __getitem__(self, idx):
        # final ground truth rgb image 
        image = retrieve_row_from_lmdb(
            self.env, 
            "images", self.KEY_TO_TYPE['images'], self.image_shape[1:], idx
        )
        image = torch.tensor(image, dtype=torch.float32) 

        label =  retrieve_row_from_lmdb(
            self.env, 
            "labels", self.KEY_TO_TYPE['labels'], self.label_shape[1:], idx
        )

        label = torch.tensor(label, dtype=torch.long)
        image = (image / 255.0)
        prompt = retrieve_text_from_lmdb(self.env, "prompts", idx)


        output_dict = { 
            'images': image,
            'class_labels': label,
            'prompts': prompt
        }
        
        return output_dict

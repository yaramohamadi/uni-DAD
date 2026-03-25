from main.utils import retrieve_row_from_lmdb, get_array_shape_from_lmdb
from torch.utils.data import Dataset
from tqdm import tqdm 
import numpy as np 
import torch
import lmdb 
import glob 
import os
# add at top
from typing import Optional, Callable

class LMDBDataset(Dataset):
    def __init__(self, dataset_path, transform: Optional[Callable] = None):
        self.KEY_TO_TYPE = {'labels': np.int64, 'images': np.uint8}
        self.dataset_path = dataset_path
        self.transform = transform  # <— NEW

        self.env = lmdb.open(dataset_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.image_shape = get_array_shape_from_lmdb(self.env, 'images')
        self.label_shape = get_array_shape_from_lmdb(self.env, 'labels')

    def __len__(self):
        return self.image_shape[0]

    def __getitem__(self, idx):
        image = retrieve_row_from_lmdb(self.env, "images", self.KEY_TO_TYPE['images'], self.image_shape[1:], idx)
        image = torch.tensor(image, dtype=torch.float32) / 255.0            # [C,H,W] in [0,1]

        label = retrieve_row_from_lmdb(self.env, "labels", self.KEY_TO_TYPE['labels'], self.label_shape[1:], idx)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            # transform should accept a tensor [C,H,W] in [0,1] and return same
            image = self.transform(image)

        return {'images': image, 'class_labels': label}
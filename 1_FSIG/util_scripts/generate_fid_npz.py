"""
precompute_fid_stats.py

Utility script to precompute and save Inception statistics (FID features)
for a given real image dataset.

- Adds the project root to PYTHONPATH so that `main.dhariwal.evaluation_util`
  can be imported.
- Uses `compute_statistics_of_path(image_path)` to compute:
    * mu   : mean of Inception activations
    * sigma: covariance matrix of Inception activations
    * act  : raw activation vectors (optional, kept here for reuse)
- Saves these statistics into a .npz file (mu, sigma, act) that can be
  reused later for FID evaluation without recomputing features.
"""

import os
import numpy as np
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from main.dhariwal.evaluation_util import compute_statistics_of_path

# Set path to your real dataset
image_path = "FFHQ_src/datasets/fid_folders/cat_resized" 

# Compute Inception features (mean, covariance)
mu, sigma, act = compute_statistics_of_path(image_path)

# Save them into .npz file
np.savez("FFHQ_src/datasets/fid_npz/cat.npz", mu=mu, sigma=sigma, act=act)

print("Saved metfaces.npz successfully.")
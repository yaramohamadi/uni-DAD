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

import sys
from pathlib import Path

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from main.dhariwal.evaluation_util import compute_statistics_of_path

# Example paths; adapt these to your dataset before running the script.
image_path = PROJECT_ROOT / "datasets" / "fid_folders" / "cat_resized"
output_path = PROJECT_ROOT / "datasets" / "fid_npz" / "cat.npz"

# Compute Inception features (mean, covariance)
mu, sigma, act = compute_statistics_of_path(str(image_path))

# Save them into .npz file
np.savez(output_path, mu=mu, sigma=sigma, act=act)

print(f"Saved FID stats to {output_path}.")

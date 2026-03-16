import wandb
import os

import wandb
api = wandb.Api()
run = api.run("/yara-mohammadi-bahram-1-ecole-superieure-de-technologie/SWEEP_BABIES_SRC_TGT/runs/sfvvxshz")

for f in run.files():
    f.download(root="wandb_train_babies_s0.75_t0.25")
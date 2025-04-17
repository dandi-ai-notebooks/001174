#!/usr/bin/env python3
"""
Script: plot_max_projection.py
Description: Load NWB file from Dandi and compute a max projection of the first 100 OnePhotonSeries frames, then save as PNG.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import remfile
import h5py
import pynwb

# Hard-coded NWB asset URL for sub-Q/sub-Q_ophys.nwb
URL = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"

# Load remote NWB file
remote_file = remfile.File(URL)
h5f = h5py.File(remote_file, mode="r")
io = pynwb.NWBHDF5IO(file=h5f, mode="r")
nwb = io.read()

# Access the raw OnePhoton data
ops = nwb.acquisition["OnePhotonSeries"]
# Load first 100 frames to limit memory/network load
subset = ops.data[0:100, :, :]

# Compute max projection across time axis
max_proj = np.max(subset, axis=0)

# Plot and save the max projection
plt.figure(figsize=(6, 6))
plt.imshow(max_proj, cmap="gray")
plt.title("Max Projection of First 100 Frames")
plt.axis("off")

# Ensure explore directory exists and save figure
os.makedirs("explore", exist_ok=True)
plt.savefig("explore/max_projection.png", bbox_inches="tight")
plt.close()
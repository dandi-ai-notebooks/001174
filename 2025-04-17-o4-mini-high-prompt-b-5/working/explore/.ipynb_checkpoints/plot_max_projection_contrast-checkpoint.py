#!/usr/bin/env python3
"""
Script: plot_max_projection_contrast.py
Description: Load NWB file, compute max projection over first 100 frames, apply contrast stretching, and save as PNG.
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
subset = ops.data[0:100, :, :]

# Compute max projection across time axis
max_proj = np.max(subset, axis=0)

# Contrast stretching using 2nd and 98th percentiles
p_low, p_high = np.percentile(max_proj, (2, 98))
stretched = np.clip((max_proj - p_low) / (p_high - p_low), 0, 1)

# Plot and save the contrast-stretched max projection
plt.figure(figsize=(6, 6))
plt.imshow(stretched, cmap="gray")
plt.title("Contrast-Stretched Max Projection")
plt.axis("off")

os.makedirs("explore", exist_ok=True)
plt.savefig("explore/max_projection_contrast.png", bbox_inches="tight")
plt.close()
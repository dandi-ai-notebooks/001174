#!/usr/bin/env python3
"""
Script: plot_first_frame.py
Description: Load NWB file from Dandi to extract the first OnePhotonSeries frame and save it as a PNG.
"""

import os
import matplotlib.pyplot as plt
import remfile
import h5py
import pynwb

# Hard-coded NWB asset URL for sub-Q/sub-Q_ophys.nwb
URL = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"

# Load remote NWB file
remote_file = remfile.File(URL)
h5f = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5f, mode="r")
nwb = io.read()

# Extract the first imaging frame
frame = nwb.acquisition["OnePhotonSeries"].data[0, :, :]

# Plot and save the frame
plt.figure(figsize=(6, 6))
plt.imshow(frame, cmap="gray")
plt.axis("off")

# Ensure explore directory exists and save figure
os.makedirs("explore", exist_ok=True)
plt.savefig("explore/first_frame.png", bbox_inches="tight")
plt.close()
#!/usr/bin/env python3
"""
Script: plot_first_frame.py
Purpose: Load remote NWB file and plot the first imaging frame from OnePhotonSeries.
"""
import os
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt

# URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"

# Load remote NWB file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract the first imaging frame
one_photon = nwb.acquisition["OnePhotonSeries"]
frame = one_photon.data[0, :, :]

# Plot the frame
plt.figure(figsize=(6, 4))
plt.imshow(frame, cmap='gray')
plt.title("First Imaging Frame")
plt.axis('off')

# Save the figure
output_dir = os.path.dirname(__file__)
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "first_frame.png")
plt.savefig(output_path, bbox_inches='tight')
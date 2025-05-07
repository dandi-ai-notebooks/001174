#!/usr/bin/env python3
"""
Script: plot_roi_traces.py
Purpose: Load remote NWB file and plot ROI fluorescence traces for the first 3 ROIs over the first 100 time points.
"""
import os
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"

# Load remote NWB file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract the first 3 ROI fluorescence traces for the first 100 time points
roi_response = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["RoiResponseSeries"]
data = roi_response.data[:100, :3]  # first 100 timepoints, first 3 ROIs
times = np.arange(data.shape[0]) * (1.0 / roi_response.rate)

# Plot ROI traces
plt.figure(figsize=(8, 4))
for i in range(data.shape[1]):
    plt.plot(times, data[:, i], label=f"ROI {i}")
plt.xlabel("Time (s)")
plt.ylabel("Fluorescence (a.u.)")
plt.title("ROI Fluorescence Traces (First 3 ROIs, First 100 Frames)")
plt.legend()
plt.tight_layout()

# Save the figure
output_dir = os.path.dirname(__file__)
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "roi_traces.png")
plt.savefig(output_path)
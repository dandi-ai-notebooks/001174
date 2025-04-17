#!/usr/bin/env python3
"""
Script: plot_event_amplitude_series.py
Description: Load NWB file, extract event amplitude time series for ROI 0 (first ROI),
and plot the first 1000 timepoints, saving as PNG.
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

# Extract event amplitude data
event_amp = nwb.processing["ophys"]["EventAmplitude"].data  # shape (time, n_rois)
rate = nwb.processing["ophys"]["EventAmplitude"].rate
start = nwb.processing["ophys"]["EventAmplitude"].starting_time or 0.0

# Choose first ROI and first 1000 samples
roi_index = 0
n_points = min(1000, event_amp.shape[0])
amp_series = event_amp[0:n_points, roi_index]

# Build time vector
time = start + np.arange(n_points) * rate

# Plot time series
plt.figure(figsize=(8, 3))
plt.plot(time, amp_series, lw=1)
plt.xlabel("Time (s)")
plt.ylabel("Event Amplitude (fluorescence)")
plt.title(f"Event Amplitude Time Series (ROI {roi_index})")
plt.tight_layout()

# Save figure
os.makedirs("explore", exist_ok=True)
plt.savefig("explore/event_amplitude_series.png", dpi=150)
plt.close()
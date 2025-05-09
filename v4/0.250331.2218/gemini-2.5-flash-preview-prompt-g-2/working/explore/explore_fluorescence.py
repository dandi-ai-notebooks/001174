# This script loads and inspects the Fluorescence data from an NWB file.
# It plots the fluorescence traces for a few ROIs.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os

# Load
url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get Fluorescence data
fluorescence_data = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["RoiResponseSeries"].data

print(f"Fluorescence data shape: {fluorescence_data.shape}")

# Plot fluorescence traces for first few ROIs
num_rois_to_plot = 5
time_points_to_plot = fluorescence_data.shape[0] # Plot all time points

plt.figure(figsize=(12, 6))
for i in range(min(num_rois_to_plot, fluorescence_data.shape[1])):
    plt.plot(np.arange(time_points_to_plot) / nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["RoiResponseSeries"].rate, fluorescence_data[:time_points_to_plot, i] + i * 100, label=f'ROI {i}') # Adding offset for clarity

plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (offset for clarity)')
plt.title('Fluorescence Traces for Selected ROIs')
# plt.legend() # Don't show legend due to offsets
plt.tight_layout()

# Save the plot to a file
if not os.path.exists('explore'):
    os.makedirs('explore')
plt.savefig('explore/fluorescence_traces.png')
plt.close()

print("Fluorescence traces plot saved to explore/fluorescence_traces.png")
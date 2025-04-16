"""
This script provides a fixed version of the fluorescence trace plotting
to ensure we can see the actual neural activity patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import h5py
import remfile
import seaborn as sns
from scipy.stats import pearsonr

# Load
url = "https://api.dandiarchive.org/api/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"Session description: {nwb.session_description}")

# Get the fluorescence data
fluorescence = nwb.processing["ophys"].data_interfaces["Fluorescence"]
roi_response_series = fluorescence.roi_response_series["RoiResponseSeries"]
fluorescence_data = roi_response_series.data[:]
sampling_rate = roi_response_series.rate
num_frames = fluorescence_data.shape[0]
num_rois = fluorescence_data.shape[1]

print(f"Fluorescence data shape: {fluorescence_data.shape}")
print(f"Data type: {fluorescence_data.dtype}")
print(f"Min value: {np.min(fluorescence_data)}, Max value: {np.max(fluorescence_data)}")

# Create time vector
time = np.arange(num_frames) / sampling_rate

# Plot fluorescence traces in a different way, with vertical offsets
selected_rois = [0, 5, 10, 15, 20, 25, 30, 35]  # Select 8 ROIs
plt.figure(figsize=(14, 10))

# Plot a subsection of the data to see more detail
time_window = 200  # seconds - shorter window to see more detail
frames_to_plot = int(time_window * sampling_rate)
if frames_to_plot > len(time):
    frames_to_plot = len(time)

# Using a different approach to plot with offsets
offset = 0
for roi_idx in selected_rois:
    trace = fluorescence_data[:frames_to_plot, roi_idx]
    # Instead of normalizing, we'll use the raw trace with offsets
    plt.plot(time[:frames_to_plot], trace + offset, label=f'ROI {roi_idx}')
    offset += np.max(trace) - np.min(trace) + 0.5  # Add offset for next trace

plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (a.u.) + offset')
plt.title(f'Fluorescence Traces for Selected ROIs (First {time_window} seconds)')
plt.xlim(0, time_window)
plt.legend()
plt.tight_layout()
plt.savefig('explore/fluorescence_traces_fixed.png')
plt.close()

print("Fixed plot saved.")
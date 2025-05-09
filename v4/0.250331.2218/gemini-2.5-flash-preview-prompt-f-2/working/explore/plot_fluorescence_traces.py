# This script loads and plots fluorescence traces for a subset of ROIs.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Define the NWB file URL
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"

# Load the NWB file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the Fluorescence data and RoiResponseSeries
fluorescence_module = nwb.processing["ophys"]["Fluorescence"]
roi_response_series = fluorescence_module.roi_response_series["RoiResponseSeries"]

# Load a subset of the data: first 5 ROIs and first 1000 time points
num_rois_to_plot = 5
num_time_points_to_plot = 1000
traces = roi_response_series.data[:num_time_points_to_plot, :num_rois_to_plot]

# Get the timestamps for the subset
# The rate is constant, so we can generate times based on starting_time and rate
starting_time = roi_response_series.starting_time
rate = roi_response_series.rate
timestamps = starting_time + np.arange(num_time_points_to_plot) / rate

# Plot the traces
plt.figure(figsize=(12, 6))
for i in range(num_rois_to_plot):
    plt.plot(timestamps, traces[:, i], label=f'ROI {i+1}')

plt.title('Fluorescence Traces for a Subset of ROIs')
plt.xlabel('Time (s)')
plt.ylabel(f'Fluorescence ({roi_response_series.unit})')
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig('explore/fluorescence_traces.png')
plt.close()

# Close the NWB file (optional)
io.close()
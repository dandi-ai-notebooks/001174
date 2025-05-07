# This script loads an NWB file and plots the fluorescence traces for all ROIs.
# It also prints some basic information about the NWB file and the fluorescence data.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"NWB File Identifier: {nwb.identifier}")
print(f"Session Description: {nwb.session_description}")
print(f"Session Start Time: {nwb.session_start_time}")

# Get fluorescence data
fluorescence_series = nwb.processing['ophys']['Fluorescence']['RoiResponseSeries']
print(f"Fluorescence data shape: {fluorescence_series.data.shape}")
print(f"Fluorescence data unit: {fluorescence_series.unit}")
print(f"Fluorescence data rate: {fluorescence_series.rate} Hz")

# It's important to load only a subset of data for plotting if the dataset is large.
# For this exploration, let's take the first 1000 timepoints if available, or all if less.
num_timepoints_to_plot = min(1000, fluorescence_series.data.shape[0])
data_to_plot = fluorescence_series.data[:num_timepoints_to_plot, :]

# Get timestamps for the selected data
# We need to calculate the timestamps based on the rate and starting_time
# or load them if they are explicitly stored.
# The nwb-file-info doesn't show explicit timestamps for RoiResponseSeries, so we calculate.
timestamps = np.arange(num_timepoints_to_plot) / fluorescence_series.rate + fluorescence_series.starting_time

# Plot fluorescence traces
sns.set_theme()
plt.figure(figsize=(15, 5))
for i in range(data_to_plot.shape[1]):
    plt.plot(timestamps, data_to_plot[:, i], label=f'ROI {i+1}') # Assuming ROI IDs are 1-indexed for plotting

plt.xlabel(f"Time ({fluorescence_series.starting_time_unit})")
plt.ylabel(f"Fluorescence ({fluorescence_series.unit})")
plt.title(f"Fluorescence Traces (First {num_timepoints_to_plot} timepoints)")
# plt.legend() # Avoid legend if there are too many ROIs, it can clutter the plot.
plt.savefig("explore/fluorescence_traces.png")
plt.close()

print("Fluorescence traces plot saved to explore/fluorescence_traces.png")

io.close()
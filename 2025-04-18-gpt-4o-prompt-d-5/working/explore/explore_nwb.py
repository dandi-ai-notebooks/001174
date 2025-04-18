"""
This script explores the NWB file from the Dandiset 001174. It aims to visualize imaging data from
the 'OnePhotonSeries' and fluorescence data from 'RoiResponseSeries'. Additionally, it will
produce images of the data's amplitude over time to understand the patterns better.

Plots will be saved as PNG files in the explore/ directory.

The script does not perform plt.show() to avoid hang-ups.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Set up plotting
plt.style.use('default')

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Plot settings
def save_plot(filename):
    plt.savefig(f"explore/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Visualize one plane from the OnePhotonSeries
fig, ax = plt.subplots()
plane_data = nwb.acquisition["OnePhotonSeries"].data[0, :, :] # First Plane
ax.imshow(plane_data, cmap='viridis')
ax.set_title('OnePhotonSeries: First Plane')
save_plot("onephotonseries_first_plane")

# Visualize Fluorescence data's time series from RoiResponseSeries
fluorescence_data = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["RoiResponseSeries"].data[:100, 0]  # First 100 time points for ROI 0
fig, ax = plt.subplots()
ax.plot(fluorescence_data, label='Fluorescence')
ax.set_title('Fluorescence over time')
ax.set_xlabel('Time')
ax.set_ylabel('Fluorescence Intensity')
save_plot("fluorescence_time_series")

# Visualize Event Amplitude for first 10 rows
event_amplitude_data = nwb.processing["ophys"].data_interfaces["EventAmplitude"].data[0:10, :]
fig, ax = plt.subplots()
ax.imshow(event_amplitude_data, aspect='auto', cmap='hot')
ax.set_title('Event Amplitudes (first 10 rows)')
ax.set_xlabel('Cells')
ax.set_ylabel('Time')
save_plot("event_amplitudes_first_10_rows")

# Clean up
io.close()
h5_file.close()
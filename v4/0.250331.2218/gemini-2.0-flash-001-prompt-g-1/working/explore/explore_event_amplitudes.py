import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Script to explore the EventAmplitude data

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the EventAmplitude data
processing = nwb.processing
ophys = processing["ophys"]
data_interfaces = ophys.data_interfaces
EventAmplitude = data_interfaces["EventAmplitude"]
amplitudes = EventAmplitude.data

# Plot the first 10 event amplitudes for the first 10 ROIs
num_rois = 10
num_events = 10
fig, axes = plt.subplots(num_rois, 1, figsize=(8, 10))
for i in range(num_rois):
    axes[i].plot(amplitudes[:num_events, i])
    axes[i].set_ylabel(f'ROI {i}')
plt.suptitle(f'First {num_events} Event Amplitudes for First {num_rois} ROIs')

# Save plot to a file
plt.savefig("explore/event_amplitudes.png")
plt.close()
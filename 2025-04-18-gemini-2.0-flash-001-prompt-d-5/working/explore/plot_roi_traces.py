"""
This script loads the fluorescence traces for a few ROIs and plots them.
"""
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get the fluorescence traces
roi_response_series = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["RoiResponseSeries"]
data = roi_response_series.data
num_frames = data.shape[0]
num_rois = data.shape[1]

# Select a few ROIs to plot
roi_ids = [0, 1, 2]

# Create a time vector
time = np.arange(0, num_frames) / roi_response_series.rate

# Plot the fluorescence traces for the selected ROIs
plt.figure(figsize=(10, 5))
for roi_id in roi_ids:
    plt.plot(time, data[:, roi_id], label=f"ROI {roi_id}")

plt.xlabel("Time (s)")
plt.ylabel("Fluorescence")
plt.title("Fluorescence Traces for Selected ROIs")
plt.legend()
plt.savefig("explore/roi_traces.png")
plt.close()
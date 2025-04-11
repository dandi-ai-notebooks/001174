import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f, mode='r')
nwb = io.read()

# Extract fluorescence data
roi_response_series = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["RoiResponseSeries"]
data = roi_response_series.data
timestamps = roi_response_series.timestamps

# Limit data to the first 1000 timepoints
num_timepoints = min(1000, len(data))
data = data[:num_timepoints, :]

# Handle missing timestamps
if timestamps is None:
    rate = roi_response_series.rate
    timestamps = np.arange(num_timepoints) / rate
else:
    timestamps = timestamps[:num_timepoints]

# Compute mean fluorescence across ROIs
mean_fluorescence = np.mean(data, axis=1)

# Generate plot
plt.figure(figsize=(10, 6))
plt.plot(timestamps, mean_fluorescence)
plt.xlabel("Time (s)")
plt.ylabel("Mean Fluorescence")
plt.title("Mean Fluorescence Across ROIs Over Time")
plt.savefig("tmp_scripts/mean_fluorescence.png")
plt.close()
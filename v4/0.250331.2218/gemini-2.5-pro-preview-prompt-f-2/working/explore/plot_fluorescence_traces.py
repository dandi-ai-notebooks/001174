# Objective: Plot fluorescence traces for all ROIs to get an overview of neural activity.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Apply seaborn theme
sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Specify mode='r'
nwbfile = io.read()

# Access RoiResponseSeries data
roi_response_series = nwbfile.processing['ophys']['Fluorescence']['RoiResponseSeries']
data = roi_response_series.data[:]  # Load data into memory
timestamps = np.linspace(0, data.shape[0] / roi_response_series.rate, data.shape[0])

# Plot fluorescence traces
plt.figure(figsize=(15, 10))
for i in range(data.shape[1]):
    plt.plot(timestamps, data[:, i] + i * np.max(data[:, i]) * 1.5, label=f'ROI {i+1}') # Offset traces for visibility
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (arbitrary units, offset)')
plt.title('Fluorescence Traces for All ROIs')
plt.savefig('explore/fluorescence_traces.png')
plt.close()

print("Saved plot to explore/fluorescence_traces.png")

io.close() # Close the NWBHDF5IO object
# It's good practice to close the h5py.File, though remfile handles its closure.
# remote_file.close() # remfile will close it on garbage collection or explicit close
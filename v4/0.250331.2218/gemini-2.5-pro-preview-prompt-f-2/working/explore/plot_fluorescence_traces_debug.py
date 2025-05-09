# Objective: Debug fluorescence traces plot. Plot only the first ROI.

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
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwbfile = io.read()

# Access RoiResponseSeries data
roi_response_series = nwbfile.processing['ophys']['Fluorescence']['RoiResponseSeries']
data = roi_response_series.data[:]  # Load data into memory

print(f"Data shape: {data.shape}")
print(f"ROI response series rate: {roi_response_series.rate}")

if data.shape[0] > 0 and data.shape[1] > 0:
    timestamps = np.linspace(0, data.shape[0] / roi_response_series.rate, data.shape[0])

    # Plot fluorescence trace for the first ROI
    plt.figure(figsize=(15, 5))
    plt.plot(timestamps, data[:, 0])
    plt.xlabel('Time (s)')
    plt.ylabel('Fluorescence (ROI 1)')
    plt.title('Fluorescence Trace for First ROI')
    plt.savefig('explore/fluorescence_trace_roi1.png')
    plt.close()
    print("Saved plot to explore/fluorescence_trace_roi1.png")
else:
    print("No data to plot or no ROIs found.")

io.close()
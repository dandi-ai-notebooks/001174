# This script plots the calcium activity traces (fluorescence) over time for the first 5 ROIs (cells)
# in the NWB file, using the Fluorescence RoiResponseSeries data. Useful for visualizing real neural dynamics.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

roi_resp = nwb.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['RoiResponseSeries']
fluor_data = roi_resp.data[:, :5]  # first 5 ROIs, all timepoints
n_time = fluor_data.shape[0]
rate = roi_resp.rate
time = np.arange(n_time) / rate

plt.figure(figsize=(10, 6))
for i in range(fluor_data.shape[1]):
    plt.plot(time, fluor_data[:, i], label=f'ROI {i}')
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (a.u.)')
plt.title('Calcium Fluorescence Traces for First 5 ROIs')
plt.legend()
plt.tight_layout()
plt.savefig('explore/fluor_traces.png')
plt.close()
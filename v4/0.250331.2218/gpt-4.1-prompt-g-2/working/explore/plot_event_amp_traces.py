# This script plots the event amplitude traces for the first 5 ROIs (cells)
# in the NWB file, using the ophys EventAmplitude data. This measurement may reflect
# event-detection processed traces, which can sometimes be sparser or highlight specific transients.

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

event_amp = nwb.processing['ophys'].data_interfaces['EventAmplitude']
amp_data = event_amp.data[:, :5]  # first 5 ROIs, all timepoints
n_time = amp_data.shape[0]
rate = event_amp.rate
time = np.arange(n_time) / rate

plt.figure(figsize=(10, 6))
for i in range(amp_data.shape[1]):
    plt.plot(time, amp_data[:, i], label=f'ROI {i}')
plt.xlabel('Time (s)')
plt.ylabel('Event Amplitude (a.u.)')
plt.title('Event Amplitude Traces for First 5 ROIs')
plt.legend()
plt.tight_layout()
plt.savefig('explore/event_amp_traces.png')
plt.close()
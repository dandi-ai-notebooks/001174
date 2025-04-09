# This script loads calcium fluorescence traces and associated event amplitudes
# for a few randomly selected ROIs, and plots them to visualize calcium activity dynamics.
# The resulting figure will be saved to fluor_events_examples.png.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwb = io.read()

flu_data = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["RoiResponseSeries"].data
event_data = nwb.processing["ophys"].data_interfaces["EventAmplitude"].data
rate = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["RoiResponseSeries"].rate
t = np.arange(flu_data.shape[0]) / rate

num_rois = flu_data.shape[1]
np.random.seed(42)
roi_inds = np.random.choice(num_rois, size=min(5, num_rois), replace=False)

plt.figure(figsize=(12, 8))
for idx, roi in enumerate(roi_inds, start=1):
    plt.subplot(2, len(roi_inds), idx)
    plt.plot(t, flu_data[:, roi], color='blue')
    plt.title(f'Fluor ROI {roi}')
    plt.xlabel('Time (s)')
    plt.ylabel('Fluorescence')
    plt.subplot(2, len(roi_inds), idx + len(roi_inds))
    plt.plot(t, event_data[:, roi], color='orange')
    plt.title(f'EventAmp ROI {roi}')
    plt.xlabel('Time (s)')
    plt.ylabel('Event amplitude')

plt.tight_layout()
plt.savefig('tmp_scripts/fluor_events_examples.png')
# no plt.show()

io.close()
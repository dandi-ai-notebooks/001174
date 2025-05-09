# This script loads the Fluorescence and EventAmplitude traces for 3 example ROIs from the NWB file.
# It plots the first 1000 samples of each trace to illustrate their timeseries, with time on the x-axis.
# The ROI IDs used for plotting are shown in the legend.
# Output: explore/fluorescence_traces.png and explore/eventamplitude_traces.png

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
N_TRACES = 3
N_SAMPLES = 1000

# Load NWB file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

processing = nwb.processing["ophys"]
# Fluorescence
fluor = processing.data_interfaces["Fluorescence"].roi_response_series["RoiResponseSeries"]
fluor_data = fluor.data  # h5py.Dataset, shape (6041, 40)
roi_ids = fluor.rois.table.id[:]
t = np.arange(N_SAMPLES) / fluor.rate  # seconds

plt.figure(figsize=(8, 5))
for i in range(N_TRACES):
    trace = fluor_data[:N_SAMPLES, i]
    plt.plot(t, trace, label=f"ROI {roi_ids[i]}")
plt.xlabel("Time (s)")
plt.ylabel("Fluorescence")
plt.title("Fluorescence traces (first 3 ROIs)")
plt.legend()
plt.tight_layout()
plt.savefig("explore/fluorescence_traces.png", dpi=150)
plt.close()

# EventAmplitude
eventa = processing.data_interfaces["EventAmplitude"]
eventa_data = eventa.data  # h5py.Dataset, shape (6041, 40)
roi_ids_evt = eventa.rois.table.id[:]

plt.figure(figsize=(8, 5))
for i in range(N_TRACES):
    trace = eventa_data[:N_SAMPLES, i]
    plt.plot(t, trace, label=f"ROI {roi_ids_evt[i]}")
plt.xlabel("Time (s)")
plt.ylabel("Event amplitude")
plt.title("Event Amplitude traces (first 3 ROIs)")
plt.legend()
plt.tight_layout()
plt.savefig("explore/eventamplitude_traces.png", dpi=150)
plt.close()

print("Done. Traces saved to explore/.")
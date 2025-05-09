# Objective: Plot fluorescence traces for all ROIs with improved offsetting.

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
data = roi_response_series.data[:]  # (time, rois)
timestamps = np.arange(data.shape[0]) / roi_response_series.rate

# Get ROI IDs
try:
    roi_ids = nwbfile.processing['ophys']['ImageSegmentation']['PlaneSegmentation'].id[:]
except Exception:
    roi_ids = [f"ROI {i+1}" for i in range(data.shape[1])]


# Normalize each trace (e.g., to its own max or z-score) for better visualization if ranges widely vary
# For simplicity here, we'll just use an offset.
# A more robust offset might be based on standard deviations or percentiles.

# Calculate an offset for plotting
# We'll scale traces and then offset them.
# Let's try to make the offset adaptive.
# We can normalize each trace to its own peak or a robust range.
# For now, let's use a simpler offset based on a fraction of the overall data range.

offset_scale = 0
if data.shape[1] > 1: # only add offset if more than one ROI
    valid_trace_peaks = [np.max(data[:, i]) - np.min(data[:,i]) for i in range(data.shape[1]) if (np.max(data[:, i]) - np.min(data[:,i])) > 1e-6]
    if valid_trace_peaks:
        offset_scale = np.percentile(valid_trace_peaks, 75) # Use 75th percentile of peak-to-peak values as offset
    if offset_scale == 0: # fallback if all traces are flat or nearly flat
        offset_scale = 1.0


plt.figure(figsize=(18, 10))
for i in range(data.shape[1]):
    trace_to_plot = data[:, i]
    # Apply offset for visibility, ensure it's significant enough
    offset = i * offset_scale * 1.1 
    plt.plot(timestamps, trace_to_plot + offset, label=str(roi_ids[i]))

plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (offset for clarity)')
plt.title(f'Fluorescence Traces for {data.shape[1]} ROIs')
if data.shape[1] <= 10: # Only show legend if not too many ROIs
    plt.legend(title="ROI ID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
plt.savefig('explore/fluorescence_traces_revised.png')
plt.close()

print("Saved plot to explore/fluorescence_traces_revised.png")

io.close()
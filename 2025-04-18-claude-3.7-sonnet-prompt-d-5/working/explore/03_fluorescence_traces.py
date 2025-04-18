"""
This script extracts and visualizes the fluorescence traces from the NWB file.
These traces represent the calcium activity of each detected ROI over time.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the fluorescence data
ophys = nwb.processing["ophys"]
fluor = ophys.data_interfaces["Fluorescence"]
roi_response = fluor.roi_response_series["RoiResponseSeries"]

# Get the event amplitude data
event_amp = ophys.data_interfaces["EventAmplitude"]

# Extract some basic information
num_timepoints = roi_response.data.shape[0]
num_rois = roi_response.data.shape[1]
sampling_rate = roi_response.rate
duration_seconds = num_timepoints / sampling_rate
duration_minutes = duration_seconds / 60

print(f"Number of time points: {num_timepoints}")
print(f"Number of ROIs: {num_rois}")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Duration: {duration_seconds:.2f} seconds ({duration_minutes:.2f} minutes)")

# Create a time vector (in seconds)
time_vector = np.arange(num_timepoints) / sampling_rate

# Only load a subset of data to prevent memory issues
# We'll analyze the first 2 minutes of data (about 1200 time points)
subset_size = min(int(120 * sampling_rate), num_timepoints)
print(f"Analyzing first {subset_size} time points ({subset_size/sampling_rate:.2f} seconds)")

# Load the fluorescence data for the subset
fluorescence_data = roi_response.data[:subset_size, :]
event_amplitude_data = event_amp.data[:subset_size, :]

# Plot the fluorescence traces for each ROI
plt.figure(figsize=(15, 10))
for i in range(num_rois):
    # Normalize the trace to make it easier to visualize
    trace = fluorescence_data[:, i]
    normalized_trace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace))
    
    # Plot with offset for better visibility
    plt.plot(time_vector[:subset_size], normalized_trace + i*1.5, label=f'ROI {i}')

plt.xlabel('Time (seconds)')
plt.ylabel('Normalized Fluorescence (offset for visibility)')
plt.title('Fluorescence Traces for Each ROI (First 2 minutes)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("explore/fluorescence_traces.png")

# Plot the event amplitude traces for each ROI
plt.figure(figsize=(15, 10))
for i in range(num_rois):
    # Normalize the trace to make it easier to visualize
    trace = event_amplitude_data[:, i]
    normalized_trace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace))
    
    # Plot with offset for better visibility
    plt.plot(time_vector[:subset_size], normalized_trace + i*1.5, label=f'ROI {i}')

plt.xlabel('Time (seconds)')
plt.ylabel('Normalized Event Amplitude (offset for visibility)')
plt.title('Event Amplitude Traces for Each ROI (First 2 minutes)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("explore/event_amplitude_traces.png")

# Compute correlation matrix between ROI fluorescence traces
correlation_matrix = np.corrcoef(fluorescence_data.T)

# Plot correlation matrix heatmap
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.title('Correlation Between ROI Fluorescence Traces')
plt.xlabel('ROI')
plt.ylabel('ROI')
plt.xticks(np.arange(num_rois))
plt.yticks(np.arange(num_rois))
plt.tight_layout()
plt.savefig("explore/fluorescence_correlation_matrix.png")

# Extract a shorter time window to see finer details (30 seconds)
short_window = min(int(30 * sampling_rate), num_timepoints)

# Plot this shorter window for better visualization of details
plt.figure(figsize=(15, 10))
for i in range(num_rois):
    # Normalize the trace
    trace = fluorescence_data[:short_window, i]
    normalized_trace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace))
    
    # Plot with offset
    plt.plot(time_vector[:short_window], normalized_trace + i*1.5, label=f'ROI {i}')

plt.xlabel('Time (seconds)')
plt.ylabel('Normalized Fluorescence (offset for visibility)')
plt.title('Fluorescence Traces for Each ROI (30-second window)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("explore/fluorescence_traces_short.png")

print("Fluorescence trace analysis completed.")
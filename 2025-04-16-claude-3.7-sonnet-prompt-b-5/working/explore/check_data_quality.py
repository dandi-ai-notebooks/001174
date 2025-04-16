"""
This script checks the quality of the fluorescence and event amplitude data
to identify which ROIs have valid data vs. NaN values.
"""

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import h5py
import remfile

# Load
url = "https://api.dandiarchive.org/api/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"Session description: {nwb.session_description}")

# Get the fluorescence data
fluorescence = nwb.processing["ophys"].data_interfaces["Fluorescence"]
roi_response_series = fluorescence.roi_response_series["RoiResponseSeries"]
fluorescence_data = roi_response_series.data[:]
sampling_rate = roi_response_series.rate
num_rois = fluorescence_data.shape[1]

print(f"Fluorescence data shape: {fluorescence_data.shape}")

# Get the event amplitude data
event_amplitude = nwb.processing["ophys"].data_interfaces["EventAmplitude"]
event_data = event_amplitude.data[:]
print(f"Event amplitude data shape: {event_data.shape}")

# Check for NaN values in fluorescence data
fluo_nan_count = np.isnan(fluorescence_data).sum(axis=0)
print("\nNaN count per ROI (fluorescence data):")
for i in range(num_rois):
    print(f"ROI {i}: {fluo_nan_count[i]} NaN values")

# Check for NaN values in event amplitude data
event_nan_count = np.isnan(event_data).sum(axis=0)
print("\nNaN count per ROI (event amplitude data):")
for i in range(num_rois):
    print(f"ROI {i}: {event_nan_count[i]} NaN values")

# Find ROIs with valid data
valid_fluo_rois = np.where(fluo_nan_count == 0)[0]
print(f"\nROIs with valid fluorescence data: {valid_fluo_rois}")

valid_event_rois = np.where(event_nan_count == 0)[0]
print(f"\nROIs with valid event amplitude data: {valid_event_rois}")

# Plot a few traces from valid ROIs for event amplitude data
if len(valid_event_rois) > 0:
    plt.figure(figsize=(14, 10))
    time = np.arange(event_data.shape[0]) / sampling_rate
    time_window = 200  # seconds
    frames_to_plot = int(time_window * sampling_rate)
    if frames_to_plot > len(time):
        frames_to_plot = len(time)
        
    for i, roi_idx in enumerate(valid_event_rois[:5]):  # Plot up to 5 valid ROIs
        plt.plot(time[:frames_to_plot], event_data[:frames_to_plot, roi_idx] + i*2, 
                 label=f'ROI {roi_idx}')
        
    plt.xlabel('Time (s)')
    plt.ylabel('Event Amplitude + offset')
    plt.title(f'Event Amplitude for Valid ROIs (First {time_window} seconds)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('explore/valid_event_traces.png')
    plt.close()
    print("\nPlotted valid event amplitude traces.")
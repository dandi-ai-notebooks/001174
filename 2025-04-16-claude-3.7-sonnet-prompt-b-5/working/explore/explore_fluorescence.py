"""
This script explores the fluorescence traces in the NWB file, including:
1. Plotting the fluorescence traces for multiple ROIs
2. Examining correlations between ROIs
3. Visualizing periods of high activity
"""

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import h5py
import remfile
import seaborn as sns
from scipy.stats import pearsonr

# Load
url = "https://api.dandiarchive.org/api/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"Session description: {nwb.session_description}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Date: {nwb.session_start_time.date()}")

# Get the fluorescence data
fluorescence = nwb.processing["ophys"].data_interfaces["Fluorescence"]
roi_response_series = fluorescence.roi_response_series["RoiResponseSeries"]
fluorescence_data = roi_response_series.data[:]
sampling_rate = roi_response_series.rate
num_frames = fluorescence_data.shape[0]
num_rois = fluorescence_data.shape[1]

print(f"Fluorescence data shape: {fluorescence_data.shape}")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Duration: {num_frames/sampling_rate:.2f} seconds")
print(f"Number of ROIs: {num_rois}")

# Get the event amplitude data
event_amplitude = nwb.processing["ophys"].data_interfaces["EventAmplitude"]
event_data = event_amplitude.data[:]
print(f"Event amplitude data shape: {event_data.shape}")

# Create time vector
time = np.arange(num_frames) / sampling_rate

# Plot fluorescence traces for a few ROIs
selected_rois = [0, 5, 10, 15, 20, 25, 30, 35]  # Select 8 ROIs spread across the dataset
plt.figure(figsize=(14, 10))

# Plot a subsection of the data (first 500 seconds) to see more detail
time_window = 500  # seconds
frames_to_plot = int(time_window * sampling_rate)
if frames_to_plot > len(time):
    frames_to_plot = len(time)
    
for i, roi_idx in enumerate(selected_rois):
    # Normalize the trace for better visualization
    trace = fluorescence_data[:frames_to_plot, roi_idx]
    normalized_trace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace)) + i
    
    plt.plot(time[:frames_to_plot], normalized_trace, label=f'ROI {roi_idx}')

plt.xlabel('Time (s)')
plt.ylabel('Normalized Fluorescence (a.u.)')
plt.title(f'Fluorescence Traces for Selected ROIs (First {time_window} seconds)')
plt.legend()
plt.xlim(0, time_window)  # Set x-axis limits explicitly
plt.tight_layout()
plt.savefig('explore/fluorescence_traces.png')
plt.close()

# Calculate correlation matrix between ROIs
corr_matrix = np.zeros((len(selected_rois), len(selected_rois)))
for i, roi_i in enumerate(selected_rois):
    for j, roi_j in enumerate(selected_rois):
        corr, _ = pearsonr(fluorescence_data[:, roi_i], fluorescence_data[:, roi_j])
        corr_matrix[i, j] = corr

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
            xticklabels=[f'ROI {idx}' for idx in selected_rois],
            yticklabels=[f'ROI {idx}' for idx in selected_rois])
plt.title('Correlation Matrix Between Selected ROIs')
plt.tight_layout()
plt.savefig('explore/roi_correlation_matrix.png')
plt.close()

# Plot event amplitude data for a few ROIs
plt.figure(figsize=(14, 10))
for i, roi_idx in enumerate(selected_rois):
    # Normalize the trace for better visualization
    event_trace = event_data[:frames_to_plot, roi_idx]
    normalized_event = (event_trace - np.min(event_trace)) / (np.max(event_trace) - np.min(event_trace)) + i
    
    plt.plot(time[:frames_to_plot], normalized_event, label=f'ROI {roi_idx}')

plt.xlabel('Time (s)')
plt.ylabel('Normalized Event Amplitude (a.u.)')
plt.title(f'Event Amplitude for Selected ROIs (First {time_window} seconds)')
plt.legend()
plt.tight_layout()
plt.savefig('explore/event_amplitude_traces.png')
plt.close()

# Identify periods of high synchronous activity
# Calculate the sum of event amplitudes across all ROIs at each timepoint
total_activity = np.sum(event_data, axis=1)
activity_threshold = np.percentile(total_activity, 95)  # Top 5% of activity
high_activity_periods = time[total_activity > activity_threshold]

# Plot overall activity and high-activity periods
plt.figure(figsize=(14, 6))
plt.plot(time, total_activity, 'k-', alpha=0.7, label='Total activity')
plt.axhline(activity_threshold, color='r', linestyle='--', label='High activity threshold')
plt.scatter(high_activity_periods, np.ones_like(high_activity_periods) * activity_threshold,
           color='r', alpha=0.5, label='High activity periods')
plt.xlabel('Time (s)')
plt.ylabel('Sum of event amplitudes')
plt.title('Total neural activity and identification of high activity periods')
plt.legend()
plt.tight_layout()
plt.savefig('explore/high_activity_periods.png')
plt.close()

print("Exploration complete. Check the saved PNG files.")
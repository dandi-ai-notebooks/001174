'''
This script explores the temporal patterns of calcium activity across all ROIs.
It will:
1. Create a heatmap of all ROIs' activity over time
2. Generate plots showing overall activity patterns
3. Look for temporal dynamics in the calcium imaging data
'''

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/b4e6bbf7-0564-4628-b8f0-680fd9b8d4ea/download/"
print(f"Loading NWB file from: {url}")

remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the fluorescence data
print("Accessing fluorescence data...")
fluorescence = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["RoiResponseSeries"]
fluorescence_data = fluorescence.data[:]
num_rois = fluorescence_data.shape[1]
num_timepoints = fluorescence_data.shape[0]
print(f"Fluorescence data shape: {fluorescence_data.shape}")

# Create a time vector
sampling_rate = fluorescence.rate
time = np.arange(num_timepoints) / sampling_rate
duration_minutes = time[-1] / 60
print(f"Recording duration: {duration_minutes:.2f} minutes")

# For better visualization, let's take a subset of the data
# We'll take either the first 5 minutes or the full recording if it's shorter than 5 minutes
five_min_samples = int(5 * 60 * sampling_rate)
subset_samples = min(five_min_samples, num_timepoints)
subset_data = fluorescence_data[:subset_samples, :]
subset_time = time[:subset_samples]

print(f"Using {subset_samples} samples for visualization (approximately {subset_samples/sampling_rate/60:.2f} minutes)")

# Create a heatmap of all ROIs' activity over time
plt.figure(figsize=(15, 8))
sns.heatmap(subset_data.T, cmap='viridis', robust=True)
plt.xlabel('Time Sample Index')
plt.ylabel('ROI')
plt.title('Fluorescence Activity Heatmap')
plt.tight_layout()
plt.savefig('tmp_scripts/fluorescence_heatmap.png')

# Plot the sum of activity across all ROIs over time
# This gives an idea of overall population activity
total_activity = np.sum(fluorescence_data, axis=1)
plt.figure(figsize=(15, 5))
plt.plot(time, total_activity)
plt.xlabel('Time (seconds)')
plt.ylabel('Sum of Fluorescence Across All ROIs')
plt.title('Total Population Activity Over Time')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('tmp_scripts/total_activity.png')

# Create a raster plot of event activity
# We'll use the event amplitude data for this
event_amplitude = nwb.processing["ophys"].data_interfaces["EventAmplitude"]
event_data = event_amplitude.data[:]

# Threshold the events to identify significant activity
# A simple approach is to consider any non-zero value as an event
binary_events = (event_data > 0).astype(int)

plt.figure(figsize=(15, 8))
plt.imshow(binary_events.T, aspect='auto', cmap='binary', 
           extent=[0, time[-1], 0, num_rois])
plt.xlabel('Time (seconds)')
plt.ylabel('ROI')
plt.title('Neural Event Raster Plot')
plt.tight_layout()
plt.savefig('tmp_scripts/event_raster.png')

# Plot the number of active neurons at each timepoint
active_count = np.sum(binary_events, axis=1)
plt.figure(figsize=(15, 5))
plt.plot(time, active_count)
plt.xlabel('Time (seconds)')
plt.ylabel('Number of Active ROIs')
plt.title('Number of Simultaneously Active Neurons Over Time')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('tmp_scripts/active_count.png')

# Create a raster-like plot showing top 5 most active ROIs
# First, calculate the total activity (sum of fluorescence) per ROI
roi_total_activity = np.sum(fluorescence_data, axis=0)
# Get the top 5 most active ROIs
top_roi_indices = np.argsort(roi_total_activity)[-5:][::-1]  # Sort descending

# Plot the fluorescence traces for the top 5 ROIs
plt.figure(figsize=(15, 10))
subset_time_min = subset_time / 60  # Convert to minutes for better readability

for i, roi_idx in enumerate(top_roi_indices):
    # Normalize the data for better visualization
    normalized_trace = subset_data[:, roi_idx] / np.max(subset_data[:, roi_idx])
    # Offset each trace to prevent overlap
    offset_trace = normalized_trace + i
    plt.plot(subset_time_min, offset_trace, label=f'ROI {roi_idx}')

plt.xlabel('Time (minutes)')
plt.ylabel('Normalized Fluorescence (offset for clarity)')
plt.title('Top 5 Most Active ROIs')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('tmp_scripts/top_active_rois.png')

print("Script completed successfully!")
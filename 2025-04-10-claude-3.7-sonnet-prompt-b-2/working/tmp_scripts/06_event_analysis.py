# This script analyzes neural events and checks for data quality issues

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

# Get the fluorescence traces and event amplitudes
fluorescence = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["RoiResponseSeries"]
fluor_data = fluorescence.data[:]
event_amplitudes = nwb.processing["ophys"].data_interfaces["EventAmplitude"]
event_data = event_amplitudes.data[:]

# Check for NaN values
fluor_nan_count = np.isnan(fluor_data).sum()
event_nan_count = np.isnan(event_data).sum()
print(f"NaN values in fluorescence data: {fluor_nan_count}")
print(f"NaN values in event data: {event_nan_count}")

# If there are NaN values, replace them with zeros for analysis
if fluor_nan_count > 0:
    fluor_data = np.nan_to_num(fluor_data)
if event_nan_count > 0:
    event_data = np.nan_to_num(event_data)

# Print shape and basic statistics
print(f"Fluorescence data shape: {fluor_data.shape}")
print(f"Event data shape: {event_data.shape}")

# Check for any all-NaN or all-zero rows/columns
nan_rows = np.isnan(fluor_data).all(axis=1).sum()
nan_cols = np.isnan(fluor_data).all(axis=0).sum()
zero_rows = (fluor_data == 0).all(axis=1).sum()
zero_cols = (fluor_data == 0).all(axis=0).sum()
print(f"All-NaN rows: {nan_rows}, All-NaN columns: {nan_cols}")
print(f"All-zero rows: {zero_rows}, All-zero columns: {zero_cols}")

# Calculate time axis
fluor_rate = fluorescence.rate
time_axis = np.arange(fluor_data.shape[0]) / fluor_rate

# First, let's look at event detection - detect calcium events based on threshold
# A common approach is to detect events that exceed a certain number of standard deviations above baseline
def detect_events(trace, threshold_sd=2.5):
    # Calculate baseline and standard deviation
    baseline = np.median(trace)
    mad = np.median(np.abs(trace - baseline))
    threshold = baseline + threshold_sd * mad / 0.6745  # Convert MAD to SD
    
    # Detect threshold crossings
    events = trace > threshold
    
    # Return binary events
    return events

# Calculate events for each neuron
print("Detecting events for each neuron...")
event_counts = []
for i in range(fluor_data.shape[1]):
    events = detect_events(fluor_data[:, i])
    event_counts.append(events.sum())

print(f"Total events detected: {sum(event_counts)}")
print(f"Average events per neuron: {np.mean(event_counts):.2f}")
print(f"Min events: {np.min(event_counts)}, Max events: {np.max(event_counts)}")

# Plot event counts by neuron
plt.figure(figsize=(10, 6))
plt.bar(np.arange(len(event_counts)), event_counts)
plt.xlabel('Neuron ID')
plt.ylabel('Number of Events')
plt.title('Calcium Events by Neuron')
plt.grid(True, alpha=0.3)
plt.savefig('tmp_scripts/event_counts.png')

# Identify most active neurons based on event count
top_neuron_indices = np.argsort(event_counts)[-5:][::-1]  # Get top 5 most active neurons
print(f"Top 5 most active neurons: {top_neuron_indices}")

# Plot event amplitudes for top neurons
plt.figure(figsize=(15, 8))
for i, neuron_idx in enumerate(top_neuron_indices):
    # Offset traces for better visualization
    offset = i * 3
    plt.plot(time_axis, event_data[:, neuron_idx] + offset, label=f"Neuron {neuron_idx}")

plt.xlabel("Time (s)")
plt.ylabel("Event Amplitude (A.U.) + offset")
plt.title("Event Amplitudes for Top 5 Most Active Neurons")
plt.legend()
plt.savefig('tmp_scripts/top_neurons_events.png')

# Create a raster plot to visualize events across all neurons
plt.figure(figsize=(15, 8))
all_events = np.zeros((fluor_data.shape[1], fluor_data.shape[0]), dtype=bool)

for i in range(fluor_data.shape[1]):
    all_events[i, :] = detect_events(fluor_data[:, i])

# Plot raster
plt.imshow(all_events, aspect='auto', cmap='binary', 
           extent=[0, time_axis[-1], 0, all_events.shape[0]])
plt.xlabel('Time (s)')
plt.ylabel('Neuron ID')
plt.title('Raster Plot of Calcium Events')
plt.colorbar(label='Event Detected')
plt.savefig('tmp_scripts/event_raster.png')

# Create a heatmap showing activity over time
# We'll take a window of the data to avoid too much compression
time_window = 300  # 5 minutes (300 seconds)
start_idx = 0
end_idx = int(time_window * fluor_rate)

# Extract window data
window_data = fluor_data[start_idx:end_idx, :]
window_time = time_axis[start_idx:end_idx]

# Z-score normalize each neuron's trace for better visualization
z_scored = np.zeros_like(window_data)
for i in range(window_data.shape[1]):
    trace = window_data[:, i]
    z_scored[:, i] = (trace - np.mean(trace)) / np.std(trace)

# Plot heatmap
plt.figure(figsize=(15, 8))
plt.imshow(z_scored.T, aspect='auto', cmap='viridis', 
           extent=[0, time_window, 0, window_data.shape[1]])
plt.xlabel('Time (s)')
plt.ylabel('Neuron ID')
plt.title('Neural Activity Heatmap (Z-scored)')
plt.colorbar(label='Z-score')
plt.savefig('tmp_scripts/activity_heatmap.png')
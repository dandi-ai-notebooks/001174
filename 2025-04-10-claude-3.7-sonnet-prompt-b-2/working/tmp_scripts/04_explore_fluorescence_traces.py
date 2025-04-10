# This script explores the fluorescence traces and activity patterns of the neurons

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

# Get the fluorescence traces
fluorescence = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["RoiResponseSeries"]
fluor_data = fluorescence.data[:]
fluor_rate = fluorescence.rate
print(f"Fluorescence data shape: {fluor_data.shape}")
print(f"Sampling rate: {fluor_rate} Hz")

# Get the event amplitudes
event_amplitudes = nwb.processing["ophys"].data_interfaces["EventAmplitude"]
event_data = event_amplitudes.data[:]
print(f"Event amplitudes shape: {event_data.shape}")

# Calculate time axis in seconds
num_samples = fluor_data.shape[0]
time_axis = np.arange(num_samples) / fluor_rate

# Plot fluorescence traces for a subset of neurons
num_neurons_to_plot = 5
selected_neurons = np.random.choice(fluor_data.shape[1], num_neurons_to_plot, replace=False)

plt.figure(figsize=(15, 8))
for i, neuron_idx in enumerate(selected_neurons):
    # Offset traces for better visualization
    offset = i * 1.5
    plt.plot(time_axis, fluor_data[:, neuron_idx] + offset, label=f"Neuron {neuron_idx}")

plt.xlabel("Time (s)")
plt.ylabel("Fluorescence (A.U.) + offset")
plt.title("Fluorescence Traces for Selected Neurons")
plt.legend()
plt.savefig('tmp_scripts/fluorescence_traces.png')

# Plot event amplitudes for the same neurons
plt.figure(figsize=(15, 8))
for i, neuron_idx in enumerate(selected_neurons):
    # Offset traces for better visualization
    offset = i * 1.5
    plt.plot(time_axis, event_data[:, neuron_idx] + offset, label=f"Neuron {neuron_idx}")

plt.xlabel("Time (s)")
plt.ylabel("Event Amplitude (A.U.) + offset")
plt.title("Event Amplitudes for Selected Neurons")
plt.legend()
plt.savefig('tmp_scripts/event_amplitudes.png')

# Plot a zoomed-in segment of traces to see details
segment_start = 100  # seconds
segment_duration = 50  # seconds
start_idx = int(segment_start * fluor_rate)
end_idx = int((segment_start + segment_duration) * fluor_rate)
segment_time = time_axis[start_idx:end_idx]

plt.figure(figsize=(15, 8))
for i, neuron_idx in enumerate(selected_neurons):
    # Offset traces for better visualization
    offset = i * 1.5
    plt.plot(segment_time, fluor_data[start_idx:end_idx, neuron_idx] + offset, label=f"Neuron {neuron_idx}")

plt.xlabel("Time (s)")
plt.ylabel("Fluorescence (A.U.) + offset")
plt.title(f"Zoomed Fluorescence Traces ({segment_start}-{segment_start+segment_duration}s)")
plt.legend()
plt.savefig('tmp_scripts/zoomed_fluorescence.png')

# Calculate correlation matrix between neurons
correlation_matrix = np.corrcoef(fluor_data.T)
print(f"Correlation matrix shape: {correlation_matrix.shape}")

# Plot correlation matrix
plt.figure(figsize=(10, 8))
im = plt.imshow(correlation_matrix, cmap='viridis')
plt.colorbar(im, label='Correlation Coefficient')
plt.title('Correlation Matrix Between Neurons')
plt.xlabel('Neuron ID')
plt.ylabel('Neuron ID')
plt.savefig('tmp_scripts/correlation_matrix.png')

# Analyze activity patterns
# Calculate mean activity of each neuron
mean_activity = np.mean(fluor_data, axis=0)
# Calculate peak activity of each neuron
peak_activity = np.max(fluor_data, axis=0)
# Calculate standard deviation of activity
std_activity = np.std(fluor_data, axis=0)

# Create a table of neuron statistics
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.bar(np.arange(len(mean_activity)), mean_activity)
plt.xlabel('Neuron ID')
plt.ylabel('Mean Fluorescence')
plt.title('Mean Activity by Neuron')

plt.subplot(1, 3, 2)
plt.bar(np.arange(len(peak_activity)), peak_activity)
plt.xlabel('Neuron ID')
plt.ylabel('Peak Fluorescence')
plt.title('Peak Activity by Neuron')

plt.subplot(1, 3, 3)
plt.bar(np.arange(len(std_activity)), std_activity)
plt.xlabel('Neuron ID')
plt.ylabel('Std Dev of Fluorescence')
plt.title('Activity Variability by Neuron')

plt.tight_layout()
plt.savefig('tmp_scripts/neuron_statistics.png')
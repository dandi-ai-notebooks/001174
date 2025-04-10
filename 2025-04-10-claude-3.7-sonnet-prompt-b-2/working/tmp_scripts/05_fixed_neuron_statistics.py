# This script creates proper neuron statistics plots

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

# Calculate time axis in seconds
fluor_rate = fluorescence.rate
num_samples = fluor_data.shape[0]
time_axis = np.arange(num_samples) / fluor_rate

# Analyze activity patterns
# Calculate mean activity of each neuron
mean_activity = np.mean(fluor_data, axis=0)
# Calculate peak activity of each neuron
peak_activity = np.max(fluor_data, axis=0)
# Calculate standard deviation of activity
std_activity = np.std(fluor_data, axis=0)

print("Mean activity range:", np.min(mean_activity), "to", np.max(mean_activity))
print("Peak activity range:", np.min(peak_activity), "to", np.max(peak_activity))
print("Std dev range:", np.min(std_activity), "to", np.max(std_activity))

# Sort neurons by mean activity
sorted_indices = np.argsort(mean_activity)
top_neurons = sorted_indices[-5:]  # Top 5 most active neurons
print("Top 5 most active neurons:", top_neurons)

# Create a table of neuron statistics with proper scaling
plt.figure(figsize=(15, 5))

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
plt.savefig('tmp_scripts/fixed_neuron_statistics.png')

# Create a scatter plot to compare mean vs. peak activity
plt.figure(figsize=(8, 6))
plt.scatter(mean_activity, peak_activity, alpha=0.7)

# Add labels for top neurons
for neuron_id in top_neurons:
    plt.annotate(f"{neuron_id}", 
                (mean_activity[neuron_id], peak_activity[neuron_id]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9)

plt.xlabel('Mean Fluorescence')
plt.ylabel('Peak Fluorescence')
plt.title('Relationship Between Mean and Peak Activity')
plt.grid(True, alpha=0.3)
plt.savefig('tmp_scripts/mean_vs_peak_activity.png')

# Create a plot comparing mean activity and variability
plt.figure(figsize=(8, 6))
plt.scatter(mean_activity, std_activity, alpha=0.7)

# Add labels for top neurons
for neuron_id in top_neurons:
    plt.annotate(f"{neuron_id}", 
                (mean_activity[neuron_id], std_activity[neuron_id]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9)

plt.xlabel('Mean Fluorescence')
plt.ylabel('Std Dev of Fluorescence')
plt.title('Relationship Between Mean Activity and Variability')
plt.grid(True, alpha=0.3)
plt.savefig('tmp_scripts/mean_vs_variability.png')
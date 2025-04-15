'''
This script analyzes correlations between different ROIs in the calcium imaging dataset.
It will:
1. Calculate correlation coefficients between ROIs based on their fluorescence signals
2. Visualize the correlation matrix
3. Plot co-activation patterns over time for selected ROIs
'''

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

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
fluorescence_data = fluorescence.data[:]  # Get all data
num_rois = fluorescence_data.shape[1]
print(f"Fluorescence data shape: {fluorescence_data.shape}")

# Calculate the correlation matrix
print("Calculating correlation matrix...")
correlation_matrix = np.zeros((num_rois, num_rois))

for i in range(num_rois):
    for j in range(num_rois):
        if i == j:
            correlation_matrix[i, j] = 1.0  # Perfect correlation with itself
        else:
            # Calculate Pearson correlation
            corr, _ = pearsonr(fluorescence_data[:, i], fluorescence_data[:, j])
            correlation_matrix[i, j] = corr

# Create a DataFrame for the correlation matrix
roi_ids = [f"ROI {i}" for i in range(num_rois)]
corr_df = pd.DataFrame(correlation_matrix, index=roi_ids, columns=roi_ids)

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
plt.title('Correlation Matrix of ROI Fluorescence Signals')
plt.tight_layout()
plt.savefig('tmp_scripts/roi_correlation_matrix.png')

# Find pairs of ROIs with high correlation (absolute value > 0.5)
high_corr_pairs = []
for i in range(num_rois):
    for j in range(i+1, num_rois):  # Only look at upper triangle to avoid duplicates
        if abs(correlation_matrix[i, j]) > 0.5:
            high_corr_pairs.append((i, j, correlation_matrix[i, j]))

print(f"Found {len(high_corr_pairs)} pairs of ROIs with high correlation (abs > 0.5)")

# Sort pairs by absolute correlation value, highest first
high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

# Print the top 10 highly correlated pairs
print("\nTop 10 highly correlated ROI pairs:")
for i, (roi1, roi2, corr) in enumerate(high_corr_pairs[:10]):
    print(f"{i+1}. ROI {roi1} & ROI {roi2}: {corr:.4f}")

# Plot the fluorescence traces for the top 3 correlated pairs
if high_corr_pairs:
    num_pairs_to_plot = min(3, len(high_corr_pairs))
    fig, axes = plt.subplots(num_pairs_to_plot, 1, figsize=(12, 12), sharex=True)
    
    # If there's only one pair, axes won't be an array
    if num_pairs_to_plot == 1:
        axes = [axes]
    
    # Create a time vector (assuming constant sampling rate)
    time = np.arange(fluorescence_data.shape[0]) / fluorescence.rate
    
    for i, (roi1, roi2, corr) in enumerate(high_corr_pairs[:num_pairs_to_plot]):
        ax = axes[i]
        ax.plot(time, fluorescence_data[:, roi1], label=f'ROI {roi1}')
        ax.plot(time, fluorescence_data[:, roi2], label=f'ROI {roi2}')
        ax.set_title(f'Correlation: {corr:.4f} between ROI {roi1} & ROI {roi2}')
        ax.set_ylabel('Fluorescence')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig('tmp_scripts/correlated_roi_pairs.png')

# Create a scatter plot of fluorescence values for the top correlated pair
if high_corr_pairs:
    roi1, roi2, corr = high_corr_pairs[0]  # Top correlated pair
    plt.figure(figsize=(8, 8))
    plt.scatter(fluorescence_data[:, roi1], fluorescence_data[:, roi2], alpha=0.5)
    plt.xlabel(f'ROI {roi1} Fluorescence')
    plt.ylabel(f'ROI {roi2} Fluorescence')
    plt.title(f'Scatter Plot of ROI {roi1} vs ROI {roi2}, Correlation: {corr:.4f}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('tmp_scripts/top_pair_scatter.png')

# Plot the event amplitude data for comparison
print("\nAccessing event amplitude data...")
event_amplitude = nwb.processing["ophys"].data_interfaces["EventAmplitude"]
event_data = event_amplitude.data[:]
print(f"Event amplitude data shape: {event_data.shape}")

# Plot event amplitude for highly correlated ROIs
if high_corr_pairs:
    roi1, roi2, corr = high_corr_pairs[0]  # Top correlated pair
    
    plt.figure(figsize=(12, 6))
    
    # Create a time vector (assuming constant sampling rate)
    time = np.arange(event_data.shape[0]) / event_amplitude.rate
    
    plt.plot(time, event_data[:, roi1], label=f'ROI {roi1}')
    plt.plot(time, event_data[:, roi2], label=f'ROI {roi2}')
    plt.title(f'Event Amplitude for Correlated ROIs {roi1} & {roi2}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Event Amplitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('tmp_scripts/correlated_events.png')

print("Script completed successfully!")
"""
This script analyzes calcium activity patterns in the dataset and 
creates visualizations of fluorescence traces, event amplitude data,
and ROI correlations.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory for plots
os.makedirs('explore/', exist_ok=True)

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/628c87ee-c3e1-44f3-b4b4-54aa67a0f6e4/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get data from the NWB file
one_photon_series = nwb.acquisition["OnePhotonSeries"]
plane_segmentation = nwb.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
roi_response_series = nwb.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['RoiResponseSeries']
event_amplitude = nwb.processing['ophys'].data_interfaces['EventAmplitude']

# Get dimensions
num_rois = len(plane_segmentation.id.data[:])
print(f"Number of ROIs: {num_rois}")
print(f"OnePhotonSeries shape: {one_photon_series.data.shape}")
print(f"Fluorescence data shape: {roi_response_series.data.shape}")
print(f"Event amplitude data shape: {event_amplitude.data.shape}")

# Overlay ROIs on sample frame (fixing previous error)
frame_idx = one_photon_series.data.shape[0] // 2
sample_frame = one_photon_series.data[frame_idx, :, :]

# Get ROI masks
roi_masks = []
for i in range(num_rois):
    roi_mask = np.array(plane_segmentation.image_mask[i])
    roi_masks.append(roi_mask)
roi_masks_array = np.array(roi_masks)

# Create a figure
plt.figure(figsize=(12, 10))

# Show the sample frame
plt.imshow(sample_frame, cmap='gray')
plt.title(f'Sample Frame (Frame #{frame_idx}) with ROI Contours')

# Add contours for each ROI
colors = plt.cm.tab10(np.linspace(0, 1, num_rois))
for i in range(num_rois):
    mask = roi_masks_array[i]
    if np.max(mask) > 0:
        threshold = 0.5 * np.max(mask)
        binary_mask = mask > threshold
        # Use plt.contour without passing a colormap so no colorbar is needed
        plt.contour(binary_mask, levels=[0.5], colors=[colors[i]], linewidths=2)

# Add a legend
handles = [plt.Line2D([0], [0], color=colors[i], lw=2) for i in range(num_rois)]
plt.legend(handles, [f'ROI {i}' for i in range(num_rois)], loc='upper right', bbox_to_anchor=(1.15, 1))

plt.savefig('explore/sample_frame_with_roi_contours_fixed.png')
plt.close()

# Get the fluorescence data and event amplitude data
fluorescence_data = roi_response_series.data[:]
event_data = event_amplitude.data[:]

# Calculate mean fluorescence for each ROI
mean_fluorescence = np.mean(fluorescence_data, axis=0)
plt.figure(figsize=(12, 6))
plt.bar(range(num_rois), mean_fluorescence, color=colors)
plt.xlabel('ROI ID')
plt.ylabel('Mean Fluorescence (a.u.)')
plt.title('Mean Fluorescence by ROI')
plt.xticks(range(num_rois))
for i, v in enumerate(mean_fluorescence):
    plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
plt.savefig('explore/mean_fluorescence_by_roi.png')
plt.close()

# Calculate mean event amplitude for each ROI
mean_events = np.mean(event_data, axis=0)
plt.figure(figsize=(12, 6))
plt.bar(range(num_rois), mean_events, color=colors)
plt.xlabel('ROI ID')
plt.ylabel('Mean Event Amplitude (a.u.)')
plt.title('Mean Event Amplitude by ROI')
plt.xticks(range(num_rois))
for i, v in enumerate(mean_events):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.savefig('explore/mean_event_amplitude_by_roi.png')
plt.close()

# Calculate event frequency (non-zero events) for each ROI
# Define threshold for what constitutes an event
event_threshold = np.mean(event_data) + np.std(event_data)
event_counts = np.sum(event_data > event_threshold, axis=0)
event_frequency = event_counts / event_data.shape[0]  # Events per frame

plt.figure(figsize=(12, 6))
plt.bar(range(num_rois), event_frequency, color=colors)
plt.xlabel('ROI ID')
plt.ylabel('Event Frequency (events/frame)')
plt.title(f'Event Frequency by ROI (threshold: {event_threshold:.3f})')
plt.xticks(range(num_rois))
for i, v in enumerate(event_frequency):
    plt.text(i, v + 0.001, f"{v:.4f}", ha='center')
plt.savefig('explore/event_frequency_by_roi.png')
plt.close()

# Create a heatmap of calcium events over time
subset_length = 1000  # First 1000 time points
plt.figure(figsize=(14, 8))
plt.imshow(event_data[:subset_length, :].T, 
           aspect='auto', 
           interpolation='none',
           cmap='viridis')
plt.colorbar(label='Event Amplitude')
plt.xlabel('Time (frames)')
plt.ylabel('ROI ID')
plt.yticks(range(num_rois))
plt.title('Calcium Event Amplitude Heatmap (First 1000 frames)')
plt.savefig('explore/calcium_event_heatmap.png')
plt.close()

# Calculate correlation between ROIs
correlation_matrix = np.corrcoef(fluorescence_data.T)
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(num_rois))
plt.yticks(range(num_rois))
plt.xlabel('ROI ID')
plt.ylabel('ROI ID')

# Add correlation values as text
for i in range(num_rois):
    for j in range(num_rois):
        plt.text(j, i, f"{correlation_matrix[i, j]:.2f}", 
                 ha="center", va="center", 
                 color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black",
                 fontsize=8)

plt.title('ROI Fluorescence Correlation Matrix')
plt.savefig('explore/roi_correlation_matrix.png')
plt.close()

# Plot a few individual ROIs fluorescence traces with detected events highlighted
plt.figure(figsize=(15, 10))
time = np.arange(subset_length) / roi_response_series.rate  # Convert to seconds

# Plot for a few selected ROIs
selected_rois = [0, 1, 4, 7]  # Choose a few interesting ROIs
for i, roi_id in enumerate(selected_rois):
    plt.subplot(len(selected_rois), 1, i+1)
    
    # Plot fluorescence trace
    plt.plot(time, fluorescence_data[:subset_length, roi_id], 'b-', label=f'Fluorescence')
    
    # Highlight events above threshold
    events_mask = event_data[:subset_length, roi_id] > event_threshold
    plt.plot(time[events_mask], fluorescence_data[:subset_length, roi_id][events_mask], 
             'ro', label='Detected Events')
    
    plt.xlabel('Time (s)') if i == len(selected_rois)-1 else plt.xticks([])
    plt.ylabel('Fluorescence (a.u.)')
    plt.title(f'ROI {roi_id} Activity')
    plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('explore/selected_roi_activity_with_events.png')
plt.close()

# Create a pairwise scatter plot for a few selected ROIs
plt.figure(figsize=(15, 15))
selected_rois = [0, 1, 4, 7]  # Same ROIs as above for consistency

for i, roi1 in enumerate(selected_rois):
    for j, roi2 in enumerate(selected_rois):
        plt.subplot(len(selected_rois), len(selected_rois), i*len(selected_rois) + j + 1)
        
        if i == j:  # Diagonal elements - show histogram
            plt.hist(fluorescence_data[:, roi1], bins=30, color=colors[roi1], alpha=0.7)
            plt.title(f'ROI {roi1}')
        else:  # Off-diagonal elements - show scatter
            plt.scatter(fluorescence_data[:, roi2], fluorescence_data[:, roi1], 
                       s=1, alpha=0.5, c=colors[roi1])
            # Add correlation coefficient
            corr = np.corrcoef(fluorescence_data[:, roi1], fluorescence_data[:, roi2])[0, 1]
            plt.text(0.05, 0.95, f"r = {corr:.2f}", transform=plt.gca().transAxes, 
                    fontsize=10, va='top')
            
            if j == 0:  # First column
                plt.ylabel(f'ROI {roi1}')
            if i == len(selected_rois)-1:  # Last row
                plt.xlabel(f'ROI {roi2}')

plt.tight_layout()
plt.savefig('explore/pairwise_roi_activity.png')
plt.close()

print("\nAnalysis complete. Results saved to the explore/ directory.")
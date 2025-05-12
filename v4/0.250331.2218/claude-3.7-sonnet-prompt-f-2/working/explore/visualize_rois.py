"""
This script visualizes the ROI masks and explores the fluorescence/event data
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
print(f"ROI mask shape: {np.array(plane_segmentation.image_mask[0]).shape}")

# Sample frame for overlay
frame_idx = one_photon_series.data.shape[0] // 2
sample_frame = one_photon_series.data[frame_idx, :, :]
print(f"Sample frame shape: {sample_frame.shape}")

# Create a combined ROI mask for visualization
# Use the maximum for each pixel across all ROIs
roi_masks = []
for i in range(num_rois):
    roi_mask = np.array(plane_segmentation.image_mask[i])
    roi_masks.append(roi_mask)

# Convert to numpy array for easier manipulation
roi_masks_array = np.array(roi_masks)
print(f"All ROI masks shape: {roi_masks_array.shape}")

# Create a combined mask image (maximum across all ROIs)
combined_mask = np.max(roi_masks_array, axis=0)
print(f"Combined mask shape: {combined_mask.shape}")

# Plot the combined ROI mask
plt.figure(figsize=(10, 8))
plt.imshow(combined_mask, cmap='viridis')
plt.colorbar(label='ROI Mask Value (Max across ROIs)')
plt.title('Combined ROI Masks (Max Value)')
plt.savefig('explore/combined_roi_masks.png')

# Create individual color-coded ROI visualization
plt.figure(figsize=(12, 10))
# Start with a blank canvas
roi_visualization = np.zeros(combined_mask.shape + (3,))  # RGB image

# Assign each ROI a different color
colors = plt.cm.tab10(np.linspace(0, 1, num_rois))
for i in range(num_rois):
    # Normalize the mask to 0-1 range
    normalized_mask = roi_masks_array[i] / np.max(roi_masks_array[i]) if np.max(roi_masks_array[i]) > 0 else roi_masks_array[i]
    
    # Add this ROI's contribution to each RGB channel
    for c in range(3):  # RGB channels
        roi_visualization[:, :, c] += normalized_mask * colors[i, c]

# Normalize the visualization to ensure values are in 0-1 range
roi_visualization = np.clip(roi_visualization, 0, 1)

# Plot the color-coded ROIs
plt.imshow(roi_visualization)
plt.title('Color-coded ROI Masks')

# Add a legend
handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(num_rois)]
plt.legend(handles, [f'ROI {i}' for i in range(num_rois)], loc='upper right', bbox_to_anchor=(1.15, 1))

plt.savefig('explore/color_coded_roi_masks.png')

# Overlay ROIs on the sample frame
plt.figure(figsize=(12, 10))
plt.imshow(sample_frame, cmap='gray')

# Create contours for each ROI at multiple levels for better visualization
# Since ROI masks aren't binary, use several threshold levels
for i in range(num_rois):
    # Use the 50% threshold of the max value for this ROI
    mask = roi_masks_array[i]
    if np.max(mask) > 0:
        threshold = 0.5 * np.max(mask)
        # Create a binary mask at this threshold
        binary_mask = mask > threshold
        # Plot contour
        plt.contour(binary_mask, levels=[0.5], colors=[colors[i]], linewidths=2)

plt.title(f'Sample Frame with ROI Contours (Frame #{frame_idx})')
plt.colorbar(label='Fluorescence (a.u.)')

# Add a legend
handles = [plt.Line2D([0], [0], color=colors[i], lw=2) for i in range(num_rois)]
plt.legend(handles, [f'ROI {i}' for i in range(num_rois)], loc='upper right', bbox_to_anchor=(1.15, 1))

plt.savefig('explore/sample_frame_with_roi_contours.png')

# Analyze fluorescence and event data
fluorescence_data = roi_response_series.data[:]  # Get all fluorescence data
event_data = event_amplitude.data[:]  # Get all event data

print(f"Fluorescence data shape: {fluorescence_data.shape}")
print(f"Event data shape: {event_data.shape}")

# Plot mean fluorescence for each ROI
mean_fluorescence = np.mean(fluorescence_data, axis=0)
plt.figure(figsize=(10, 6))
plt.bar(range(num_rois), mean_fluorescence)
plt.xlabel('ROI ID')
plt.ylabel('Mean Fluorescence (a.u.)')
plt.title('Mean Fluorescence by ROI')
plt.xticks(range(num_rois))
plt.savefig('explore/mean_fluorescence_by_roi.png')

# Plot mean event amplitude for each ROI
mean_events = np.mean(event_data, axis=0)
plt.figure(figsize=(10, 6))
plt.bar(range(num_rois), mean_events)
plt.xlabel('ROI ID')
plt.ylabel('Mean Event Amplitude (a.u.)')
plt.title('Mean Event Amplitude by ROI')
plt.xticks(range(num_rois))
plt.savefig('explore/mean_event_amplitude_by_roi.png')

# Plot a heatmap of calcium events over time
# Use a subset of time points for better visualization
time_subset = 1000  # First 1000 time points
roi_subset = num_rois  # All ROIs
time_points = np.arange(time_subset)
plt.figure(figsize=(12, 8))
plt.imshow(event_data[:time_subset, :roi_subset].T, 
           aspect='auto', 
           interpolation='none',
           cmap='viridis')
plt.colorbar(label='Event Amplitude')
plt.xlabel('Time (frames)')
plt.ylabel('ROI ID')
plt.yticks(range(roi_subset))
plt.title('Calcium Event Amplitude Heatmap (First 1000 frames)')
plt.savefig('explore/calcium_event_heatmap.png')

# Calculate correlation between ROIs
correlation_matrix = np.corrcoef(fluorescence_data.T)
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(num_rois))
plt.yticks(range(num_rois))
plt.xlabel('ROI ID')
plt.ylabel('ROI ID')
plt.title('ROI Fluorescence Correlation Matrix')
plt.savefig('explore/roi_correlation_matrix.png')

print("Visualization complete. Results saved to the explore/ directory.")
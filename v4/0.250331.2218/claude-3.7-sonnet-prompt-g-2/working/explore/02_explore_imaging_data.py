"""
This script explores the calcium imaging data in the NWB file.
It visualizes:
1. A sample frame from the raw imaging data
2. ROI masks overlaid on a sample frame
3. Fluorescence traces for a few ROIs
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure aesthetics
sns.set_theme()

# Load the NWB file
print("Loading NWB file...")
url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the data
one_photon_series = nwb.acquisition['OnePhotonSeries']
ophys = nwb.processing['ophys']
fluorescence = ophys.data_interfaces['Fluorescence']
roi_response_series = fluorescence.roi_response_series['RoiResponseSeries']
image_segmentation = ophys.data_interfaces['ImageSegmentation']
plane_seg = image_segmentation.plane_segmentations['PlaneSegmentation']
event_amplitude = ophys.data_interfaces['EventAmplitude']

# Get a sample frame from the raw imaging data (frame 1000)
print("Getting sample frame...")
frame_idx = 1000
sample_frame = one_photon_series.data[frame_idx, :, :]
print(f"Sample frame shape: {sample_frame.shape}")

# Plot the sample frame
plt.figure(figsize=(10, 8))
plt.imshow(sample_frame, cmap='gray')
plt.colorbar(label='Intensity')
plt.title(f'Raw Imaging Data (Frame {frame_idx})')
plt.savefig('sample_frame.png', dpi=150, bbox_inches='tight')
plt.close()

# Get ROI masks
print("Processing ROI masks...")
roi_masks = []
for i in range(len(plane_seg.id)):
    mask = plane_seg['image_mask'][i]
    mask_reshaped = np.reshape(mask, (200, 320)).T  # Reshape to match image dimensions
    roi_masks.append(mask_reshaped)

# Create a visualization of all ROI masks overlaid on the sample frame
plt.figure(figsize=(10, 8))
plt.imshow(sample_frame, cmap='gray')

# Create a colormap for the ROIs
colors = plt.cm.jet(np.linspace(0, 1, len(roi_masks)))

# Overlay ROI masks with transparency
roi_overlay = np.zeros((sample_frame.shape[0], sample_frame.shape[1], 4))
for i, mask in enumerate(roi_masks):
    # Add RGBA values with transparency
    color_mask = np.zeros((sample_frame.shape[0], sample_frame.shape[1], 4))
    color_mask[mask > 0, 0] = colors[i, 0]
    color_mask[mask > 0, 1] = colors[i, 1]
    color_mask[mask > 0, 2] = colors[i, 2]
    color_mask[mask > 0, 3] = 0.5
    roi_overlay += color_mask

plt.imshow(roi_overlay)
plt.title(f'ROIs Overlaid on Frame {frame_idx}')
plt.savefig('roi_overlay.png', dpi=150, bbox_inches='tight')
plt.close()

# Alternative visualization using a heatmap of all masks
combined_masks = np.zeros((sample_frame.shape[0], sample_frame.shape[1]))
for mask in roi_masks:
    combined_masks = np.maximum(combined_masks, mask)

plt.figure(figsize=(10, 8))
plt.imshow(sample_frame, cmap='gray')
plt.imshow(combined_masks, alpha=0.7, cmap='hot')
plt.title(f'ROI Mask Heatmap on Frame {frame_idx}')
plt.colorbar(label='Mask Value')
plt.savefig('roi_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Get fluorescence traces for a few ROIs
print("Extracting fluorescence traces...")
num_frames = roi_response_series.data.shape[0]
time_vector = np.arange(num_frames) / roi_response_series.rate  # Time in seconds

# Select a subset of ROIs to visualize (first 5)
roi_indices = np.arange(5)
selected_rois = roi_response_series.data[:, roi_indices]
roi_ids = plane_seg.id.data[roi_indices]

plt.figure(figsize=(15, 10))
for i, roi_idx in enumerate(roi_indices):
    plt.plot(time_vector, selected_rois[:, i], label=f'ROI {roi_ids[i]}')

plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (a.u.)')
plt.title('Fluorescence Traces for Selected ROIs')
plt.legend()
plt.grid(True)
plt.savefig('fluorescence_traces_short.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot a longer segment of data for one ROI
roi_idx = 0
plt.figure(figsize=(15, 5))
plt.plot(time_vector, selected_rois[:, roi_idx])
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (a.u.)')
plt.title(f'Fluorescence Trace for ROI {roi_ids[roi_idx]} (Full Recording)')
plt.grid(True)
plt.savefig('single_roi_full_trace.png', dpi=150, bbox_inches='tight')
plt.close()

# Compare fluorescence and event amplitude for one ROI
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(time_vector, roi_response_series.data[:, roi_idx])
plt.title(f'Fluorescence Trace for ROI {roi_ids[roi_idx]}')
plt.ylabel('Fluorescence (a.u.)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_vector, event_amplitude.data[:, roi_idx], color='red')
plt.title(f'Event Amplitude for ROI {roi_ids[roi_idx]}')
plt.xlabel('Time (s)')
plt.ylabel('Event Amplitude (a.u.)')
plt.grid(True)
plt.tight_layout()
plt.savefig('fluorescence_vs_events.png', dpi=150, bbox_inches='tight')
plt.close()

print("Finished generating visualizations!")
"""
This script explores the basic structure of the NWB file and visualizes:
1. Sample frames from the calcium imaging data
2. The fluorescence traces for the ROIs
3. The segmentation masks for the ROIs
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

# Print basic information about the file
print("NWB File Information:")
print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject Species: {nwb.subject.species}")
print(f"Subject Sex: {nwb.subject.sex}")
print(f"Subject Age: {nwb.subject.age}")

# Get data interfaces
print("\nData Interfaces:")
for name, interface in nwb.processing['ophys'].data_interfaces.items():
    print(f" - {name}: {type(interface).__name__}")

# Get information about the one photon series
one_photon_series = nwb.acquisition["OnePhotonSeries"]
print("\nOne Photon Series Information:")
print(f"Shape: {one_photon_series.data.shape}")
print(f"Data Type: {one_photon_series.data.dtype}")
print(f"Frame Rate: {one_photon_series.rate} Hz")
print(f"Unit: {one_photon_series.unit}")

# Get information about ROIs
plane_segmentation = nwb.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
roi_count = len(plane_segmentation.id.data[:])
print(f"\nNumber of ROIs: {roi_count}")

# Get fluorescence data
roi_response_series = nwb.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['RoiResponseSeries']
print(f"Fluorescence Data Shape: {roi_response_series.data.shape}")

# Get event amplitude data
event_amplitude = nwb.processing['ophys'].data_interfaces['EventAmplitude']
print(f"Event Amplitude Data Shape: {event_amplitude.data.shape}")

# Plot a sample frame from the one photon series
plt.figure(figsize=(10, 8))
# Sample the middle of the recording (avoid potential artifacts at beginning/end)
frame_idx = one_photon_series.data.shape[0] // 2
sample_frame = one_photon_series.data[frame_idx, :, :]
plt.imshow(sample_frame, cmap='gray')
plt.colorbar(label='Fluorescence (a.u.)')
plt.title(f'Sample Frame (Frame #{frame_idx})')
plt.savefig('explore/sample_frame.png')

# Plot fluorescence traces for a few ROIs
plt.figure(figsize=(15, 8))
# Create time array based on frame rate
time = np.arange(roi_response_series.data.shape[0]) / roi_response_series.rate
# Plot first 5 ROIs (or all if less than 5)
num_roi_to_plot = min(5, roi_count)
for i in range(num_roi_to_plot):
    # Plot a subset of the data (first 1000 points) to make visualization clearer
    subset_length = min(1000, len(time))
    plt.plot(time[:subset_length], roi_response_series.data[:subset_length, i], label=f'ROI {i}')
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (a.u.)')
plt.title(f'Fluorescence Traces (First {subset_length} frames)')
plt.legend()
plt.savefig('explore/fluorescence_traces.png')

# Visualize ROI masks
plt.figure(figsize=(10, 8))
# Create an empty image with the same dimensions as the one photon series
roi_mask_image = np.zeros((one_photon_series.data.shape[1], one_photon_series.data.shape[2]))

# Combine all ROI masks into one image (color-coded)
for i in range(roi_count):
    # Get the mask for this ROI
    roi_mask = plane_segmentation.image_mask[i]
    # Reshape to match image dimensions
    mask_image = np.array(roi_mask).reshape(one_photon_series.data.shape[1], one_photon_series.data.shape[2])
    # Add this mask to the image with a label
    roi_mask_image[mask_image > 0] = i + 1  # Add 1 to avoid 0 (background)

# Plot the combined mask
plt.imshow(roi_mask_image, cmap='nipy_spectral')
plt.colorbar(label='ROI ID')
plt.title(f'All ROI Masks ({roi_count} ROIs)')
plt.savefig('explore/roi_masks.png')

# Plot sample frame with ROI contours
plt.figure(figsize=(12, 10))
plt.imshow(sample_frame, cmap='gray')
# Add contours of each ROI
for i in range(roi_count):
    roi_mask = plane_segmentation.image_mask[i]
    mask_image = np.array(roi_mask).reshape(one_photon_series.data.shape[1], one_photon_series.data.shape[2])
    # Plot contour for this ROI
    if np.max(mask_image) > 0:  # Only if mask has non-zero values
        plt.contour(mask_image, levels=[0.5], colors=['r', 'g', 'b', 'y', 'c', 'm', 'w'][i % 7], linewidths=1.5)
plt.title(f'Sample Frame with ROI Contours (Frame #{frame_idx})')
plt.savefig('explore/sample_frame_with_contours.png')

print("Exploration complete. Results saved to the explore/ directory.")
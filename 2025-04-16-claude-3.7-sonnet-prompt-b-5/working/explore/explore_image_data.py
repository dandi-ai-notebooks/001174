"""
This script explores the image data in the NWB file, including:
1. Getting a sample frame from the raw image data
2. Visualizing the ROI masks
3. Plotting the raw image frame with ROI masks overlaid
"""

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import h5py
import remfile

# Load
url = "https://api.dandiarchive.org/api/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"Session description: {nwb.session_description}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Date: {nwb.session_start_time.date()}")

# Get a sample frame from the raw image data
print("Getting a sample frame from the raw image data...")
frame_index = 1000  # Choose a frame from the middle of the recording
sample_frame = nwb.acquisition["OnePhotonSeries"].data[frame_index, :, :]
print(f"Sample frame shape: {sample_frame.shape}")
print(f"Sample frame data type: {sample_frame.dtype}")
print(f"Sample frame min: {np.min(sample_frame)}, max: {np.max(sample_frame)}")

# Get the image masks for all ROIs
print("Getting image masks...")
image_masks = nwb.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"].image_mask[:]
print(f"Image masks shape: {image_masks.shape}")

# Check mask dimensions by loading one mask
mask = image_masks[0]
print(f"Single mask shape: {mask.shape}")

# Plot the sample frame
plt.figure(figsize=(10, 8))
plt.imshow(sample_frame, cmap='gray')
plt.colorbar(label='Fluorescence intensity')
plt.title(f'Raw image frame {frame_index}')
plt.savefig('explore/raw_image_frame.png')
plt.close()

# Plot a few of the image masks with their original dimensions
plt.figure(figsize=(12, 10))
for i in range(min(9, len(image_masks))):  # Plot up to 9 masks
    plt.subplot(3, 3, i + 1)
    plt.imshow(image_masks[i], cmap='hot')
    plt.title(f'ROI {i}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('explore/roi_masks.png')
plt.close()

# Investigate the dimension mismatch
print(f"One Photon Series dimensions: {nwb.acquisition['OnePhotonSeries'].data.shape[1:3]}")
print(f"Image mask dimensions: {image_masks[0].shape}")

# Create a composite image showing all ROIs
plt.figure(figsize=(12, 10))
# Plot ROI masks alone
mask_overlay = np.zeros((292, 179))
for i in range(len(image_masks)):
    mask_overlay = np.maximum(mask_overlay, image_masks[i])  # Take the maximum value for overlapping masks

plt.imshow(mask_overlay, cmap='hot')
plt.colorbar(label='ROI mask intensity')
plt.title('Composite of all ROI masks')
plt.savefig('explore/composite_roi_masks.png')
plt.close()

# Since the dimensions don't match exactly, let's visualize them separately
# Create a figure showing both the raw image and the ROI masks side by side
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(sample_frame, cmap='gray')
plt.title('Raw image frame')
plt.colorbar(label='Fluorescence intensity')

plt.subplot(1, 2, 2)
plt.imshow(mask_overlay, cmap='hot')
plt.title('ROI masks composite')
plt.colorbar(label='ROI mask intensity')

plt.tight_layout()
plt.savefig('explore/image_and_masks.png')
plt.close()

print("Exploration complete. Check the saved PNG files.")
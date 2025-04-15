'''
This script explores the ROI (region of interest) masks in the dataset.
It visualizes individual ROI masks without attempting to combine them.
'''

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/b4e6bbf7-0564-4628-b8f0-680fd9b8d4ea/download/"
print(f"Loading NWB file from: {url}")

remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the image segmentation data
print("Accessing image segmentation data...")
image_seg = nwb.processing["ophys"].data_interfaces["ImageSegmentation"]
plane_seg = image_seg.plane_segmentations["PlaneSegmentation"]

# Get number of ROIs
num_rois = len(plane_seg.id[:])
print(f"Found {num_rois} ROIs")

# Extract ROI masks
roi_masks = []
for i in range(num_rois):
    mask = plane_seg['image_mask'][i]
    roi_masks.append(mask)
    print(f"ROI {i} mask shape: {mask.shape}")

# Plot individual ROI masks
num_to_plot = min(21, num_rois)  # Plot all ROIs (up to 21)
rows = (num_to_plot + 2) // 3  # Calculate number of rows needed (3 columns)
fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 4))
axes = axes.flatten()

for i in range(num_to_plot):
    mask = roi_masks[i]
    axes[i].imshow(mask, cmap='viridis', interpolation='none')
    axes[i].set_title(f'ROI {i} (shape: {mask.shape})')
    axes[i].axis('off')
    
# Hide any unused subplots
for i in range(num_to_plot, len(axes)):
    axes[i].set_visible(False)
    
plt.tight_layout()
plt.savefig('tmp_scripts/all_roi_masks.png')

# For completeness, let's also extract a raw image frame to get a sense of the data
print("\nExtracting a sample frame from the one photon series...")
one_photon_series = nwb.acquisition["OnePhotonSeries"]
frame_shape = one_photon_series.data.shape[1:]
print(f"Frame shape: {frame_shape}")

# Extract a single frame
sample_frame = one_photon_series.data[0, :, :]
print(f"Sample frame shape: {sample_frame.shape}, min: {np.min(sample_frame)}, max: {np.max(sample_frame)}")

# Visualize the sample frame
plt.figure(figsize=(10, 8))
plt.imshow(sample_frame, cmap='gray')
plt.title('Sample Raw Frame from One Photon Series')
plt.colorbar(label='Intensity')
plt.axis('off')
plt.tight_layout()
plt.savefig('tmp_scripts/sample_raw_frame.png')

print("Script completed successfully!")
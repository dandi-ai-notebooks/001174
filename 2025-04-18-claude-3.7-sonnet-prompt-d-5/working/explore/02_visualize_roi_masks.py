"""
This script extracts and visualizes the ROI masks from the NWB file.
ROI masks represent the spatial footprint of each cell that was detected
during calcium imaging.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the plane segmentation object that contains ROI information
ophys = nwb.processing["ophys"]
image_seg = ophys.data_interfaces["ImageSegmentation"]
plane_seg = image_seg.plane_segmentations["PlaneSegmentation"]

# Get the number of ROIs
num_rois = len(plane_seg.id)
print(f"Number of ROIs: {num_rois}")

# Get the dimensions of the masks
first_mask = plane_seg.image_mask[0]
mask_height, mask_width = first_mask.shape
print(f"Mask dimensions: {mask_height} x {mask_width}")

# Create a figure to display all ROI masks
plt.figure(figsize=(12, 10))

# Plot each individual ROI mask
for i in range(min(num_rois, 6)):  # Plot at most 6 ROIs
    plt.subplot(2, 3, i+1)
    mask = plane_seg.image_mask[i]
    plt.imshow(mask, cmap='viridis')
    plt.title(f"ROI {i}")
    plt.colorbar()

plt.tight_layout()
plt.savefig("explore/roi_masks_individual.png")

# Create a figure to display all ROIs overlaid
plt.figure(figsize=(10, 8))

# Create a combined mask where each ROI has a different color
combined_mask = np.zeros((mask_height, mask_width, 3))

# Assign a different color to each ROI
colors = [
    [1, 0, 0],    # Red
    [0, 1, 0],    # Green
    [0, 0, 1],    # Blue
    [1, 1, 0],    # Yellow
    [1, 0, 1],    # Magenta
    [0, 1, 1],    # Cyan
]

# Add each ROI to the combined mask
for i in range(min(num_rois, len(colors))):
    mask = plane_seg.image_mask[i]
    for c in range(3):  # RGB channels
        combined_mask[:, :, c] += mask * colors[i][c]

# Clip values to [0, 1] range
combined_mask = np.clip(combined_mask, 0, 1)

plt.imshow(combined_mask)
plt.title(f"All {num_rois} ROIs")
plt.colorbar()
plt.tight_layout()
plt.savefig("explore/roi_masks_combined.png")

# Create a heatmap of all ROIs superimposed
plt.figure(figsize=(10, 8))
all_masks = np.zeros((mask_height, mask_width))

# Add all ROIs to create a heatmap
for i in range(num_rois):
    mask = plane_seg.image_mask[i]
    all_masks = np.maximum(all_masks, mask)  # Take the max value at each pixel

plt.imshow(all_masks, cmap='hot')
plt.title(f"Heatmap of all {num_rois} ROIs")
plt.colorbar()
plt.tight_layout()
plt.savefig("explore/roi_masks_heatmap.png")

print("ROI mask visualization completed.")
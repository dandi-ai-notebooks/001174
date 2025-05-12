"""
This script properly visualizes the ROI masks by transforming them to match the imaging data.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import cv2

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
image_segmentation = ophys.data_interfaces['ImageSegmentation']
plane_seg = image_segmentation.plane_segmentations['PlaneSegmentation']

# Get a sample frame from the raw imaging data as background
frame_idx = 1000
sample_frame = one_photon_series.data[frame_idx, :, :]
print(f"Sample frame shape: {sample_frame.shape}")

# Get mask information
mask_0 = plane_seg['image_mask'][0]
print(f"Mask shape: {mask_0.shape}")

# The masks need to be resized to match the imaging data
# Let's create a visualization of individual masks
num_rois = min(9, len(plane_seg.id))  # Show up to 9 ROIs
roi_ids = plane_seg.id.data[:num_rois]

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for i in range(num_rois):
    mask = plane_seg['image_mask'][i]
    
    # Resize mask to match the imaging data dimensions
    resized_mask = cv2.resize(mask, (sample_frame.shape[1], sample_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Display on a subplot
    axes[i].imshow(sample_frame, cmap='gray')
    masked_data = np.ma.masked_where(resized_mask == 0, resized_mask)
    axes[i].imshow(masked_data, cmap='hot', alpha=0.7)
    axes[i].set_title(f'ROI {roi_ids[i]}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('individual_roi_masks.png', dpi=150, bbox_inches='tight')
plt.close()

# Create a combined visualization with all ROIs on one image
plt.figure(figsize=(12, 10))
plt.imshow(sample_frame, cmap='gray')

# Create a colormap for the ROIs
colors = plt.cm.jet(np.linspace(0, 1, len(plane_seg.id)))

# Overlay all ROI masks with transparency and different colors
for i in range(len(plane_seg.id)):
    mask = plane_seg['image_mask'][i]
    resized_mask = cv2.resize(mask, (sample_frame.shape[1], sample_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create a masked array for this ROI
    masked_data = np.ma.masked_where(resized_mask == 0, resized_mask)
    
    # Plot with the color from our colormap
    plt.imshow(masked_data, cmap=ListedColormap([colors[i]]), alpha=0.5, interpolation='none')

plt.title('All ROIs Overlaid on Image')
plt.colorbar(label='ROI Mask Value')
plt.savefig('all_roi_masks.png', dpi=150, bbox_inches='tight')
plt.close()

# Create a heatmap-style visualization of all ROIs
combined_mask = np.zeros_like(sample_frame, dtype=float)

for i in range(len(plane_seg.id)):
    mask = plane_seg['image_mask'][i]
    resized_mask = cv2.resize(mask, (sample_frame.shape[1], sample_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    combined_mask = np.maximum(combined_mask, resized_mask)

plt.figure(figsize=(12, 10))
plt.imshow(sample_frame, cmap='gray')
plt.imshow(combined_mask, cmap='hot', alpha=0.7)
plt.title('ROI Mask Heatmap')
plt.colorbar(label='Mask Value')
plt.savefig('roi_mask_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print("Visualizations saved successfully!")
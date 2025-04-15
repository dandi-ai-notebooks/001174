'''
This script explores the spatial distribution of ROIs (regions of interest) in the dataset.
It will:
1. Extract the image masks for each ROI
2. Visualize the ROI masks individually
3. Create a combined visualization of all ROIs
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

# Get image shape from the one photon series
one_photon_series = nwb.acquisition["OnePhotonSeries"]
frame_width = one_photon_series.data.shape[2]  # Width dimension
frame_height = one_photon_series.data.shape[1]  # Height dimension
print(f"Frame dimensions: {frame_height}x{frame_width}")

# Extract ROI masks
roi_masks = []
for i in range(num_rois):
    mask = plane_seg['image_mask'][i]
    roi_masks.append(mask)
    
# Plot individual ROI masks for a subset of ROIs
num_to_plot = min(9, num_rois)  # Plot up to 9 ROIs
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

for i in range(num_to_plot):
    mask = roi_masks[i]
    axes[i].imshow(mask, cmap='viridis')
    axes[i].set_title(f'ROI {i}')
    axes[i].axis('off')
    
# Hide any unused subplots
for i in range(num_to_plot, len(axes)):
    axes[i].set_visible(False)
    
plt.tight_layout()
plt.savefig('tmp_scripts/individual_roi_masks.png')

# Check the actual dimensions of the masks
mask_shapes = [mask.shape for mask in roi_masks]
print(f"Mask shapes: {mask_shapes[:3]}... (showing first 3)")

# Create a combined visualization of all ROIs
# Each ROI will have a different color
plt.figure(figsize=(10, 8))

# Since ROI masks may have different dimensions than the frame,
# we need to visualize them individually and can't simply overlay them
combined_mask = np.zeros((frame_height, frame_width))
for mask in roi_masks:
    # Create a binary version of the mask (0 or 1 values)
    binary_mask = (mask > 0).astype(float)
    # Add this binary mask to the combined mask
    # This way we'll see the overlap as higher intensity
    combined_mask += binary_mask

plt.imshow(combined_mask, cmap='viridis', interpolation='none')
plt.title(f'All {num_rois} ROIs Combined')
plt.colorbar(label='ROI Mask Value')
plt.axis('off')
plt.tight_layout()
plt.savefig('tmp_scripts/combined_roi_masks.png')

# Instead of trying to color by ROI ID, which is difficult with different sized masks,
# let's create a multi-color overlay where each ROI has a different color
plt.figure(figsize=(10, 8))

# Create an RGB image
rgb_mask = np.zeros((frame_height, frame_width, 3))

# Assign each ROI a different color
import colorsys
for i, mask in enumerate(roi_masks):
    # Get a distinct color for this ROI using HSV color space
    # (which we convert to RGB)
    hue = i / num_rois
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    
    # Create a binary version of the mask
    binary_mask = (mask > 0).astype(float)
    
    # Add this colored mask to the RGB image
    rgb_mask[:, :, 0] += binary_mask * r * 0.5  # Reduce intensity to allow overlap visibility
    rgb_mask[:, :, 1] += binary_mask * g * 0.5
    rgb_mask[:, :, 2] += binary_mask * b * 0.5

# Clip values to [0, 1] range for proper display
rgb_mask = np.clip(rgb_mask, 0, 1)

plt.imshow(rgb_mask)
plt.title(f'All {num_rois} ROIs (Colored by ROI ID)')
plt.colorbar(label='ROI ID')
plt.axis('off')
plt.tight_layout()
plt.savefig('tmp_scripts/colored_roi_masks.png')

print("Script completed successfully!")
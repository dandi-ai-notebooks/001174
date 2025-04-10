# This script explores the ROIs (regions of interest) representing detected neurons

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

# Get the plane segmentation containing the ROIs
plane_seg = nwb.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
print(f"Number of ROIs: {len(plane_seg.id.data)}")
print(f"Columns available: {plane_seg.colnames}")

# Get the image masks for all ROIs
image_masks = []
for i in range(len(plane_seg.id.data)):
    mask = plane_seg['image_mask'][i]
    image_masks.append(mask)
    
image_masks = np.array(image_masks)
print(f"Image masks shape: {image_masks.shape}")

# Create a combined image showing all ROIs
combined_mask = np.zeros(image_masks[0].shape)
for mask in image_masks:
    # Use maximum to combine masks
    combined_mask = np.maximum(combined_mask, mask)

# Create a figure showing each individual ROI (show a subset if there are many)
num_rois_to_show = min(16, len(image_masks))
rows = int(np.ceil(np.sqrt(num_rois_to_show)))
cols = int(np.ceil(num_rois_to_show / rows))

plt.figure(figsize=(15, 15))
for i in range(num_rois_to_show):
    plt.subplot(rows, cols, i+1)
    plt.imshow(image_masks[i], cmap='hot')
    plt.title(f"ROI {i}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('tmp_scripts/individual_rois.png')

# Load a sample calcium imaging frame as background
frame = nwb.acquisition["OnePhotonSeries"].data[0]

# Create a figure showing all ROIs on the background image
plt.figure(figsize=(10, 8))
plt.imshow(frame, cmap='gray', alpha=1.0)
plt.imshow(combined_mask, cmap='hot', alpha=0.7)
plt.colorbar(label='ROI Mask Value')
plt.title("All ROIs overlaid on background")
plt.savefig('tmp_scripts/all_rois_overlay.png')

# Also create a figure with just all ROIs combined
plt.figure(figsize=(10, 8))
plt.imshow(combined_mask, cmap='hot')
plt.colorbar(label='ROI Mask Value')
plt.title("Combined ROI masks")
plt.savefig('tmp_scripts/combined_roi_masks.png')

# Print information about the ROIs
roi_areas = [np.sum(mask > 0) for mask in image_masks]
print(f"Average ROI area: {np.mean(roi_areas):.2f} pixels")
print(f"Min ROI area: {np.min(roi_areas):.2f} pixels")
print(f"Max ROI area: {np.max(roi_areas):.2f} pixels")
# This script loads ROI masks (PlaneSegmentation.image_mask) from the NWB file.
# It creates a heatmap of all ROI masks combined, and an example single ROI mask plot.
# Output images: explore/roi_masks_heatmap.png and explore/roi_mask_example.png

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
EXAMPLE_ROI_IDX = 0  # Show the first ROI's mask

# Load NWB file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

plane_seg = nwb.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
roi_masks = plane_seg["image_mask"][:]  # shape (40, 320, 200); float, 0..1

# Heatmap of all ROI masks (maximum intensity projection)
roi_heatmap = np.max(roi_masks, axis=0)

plt.figure(figsize=(6, 6))
plt.imshow(roi_heatmap, cmap='hot')
plt.colorbar(label='Mask overlap (max projection)')
plt.title("ROI Masks Heatmap (All ROIs, max projection)")
plt.axis('off')
plt.tight_layout()
plt.savefig("explore/roi_masks_heatmap.png", dpi=200)
plt.close()

# Example single ROI mask
plt.figure(figsize=(6, 6))
plt.imshow(roi_masks[EXAMPLE_ROI_IDX], cmap='gray')
plt.title(f"Example ROI Mask (ROI {EXAMPLE_ROI_IDX})")
plt.axis('off')
plt.tight_layout()
plt.savefig("explore/roi_mask_example.png", dpi=200)
plt.close()

print("Done. ROI mask images saved to explore/.")
# This script loads ROI masks from the NWB file and generates a heatmap summarizing all masks using np.max projection.
# The goal is to visualize spatial distribution and extent of the segmented cells.
# The plot will be saved as roi_masks_heatmap.png.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwb = io.read()

plane_segmentation = nwb.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation']

# Retrieve ROI masks into numpy array stack shape (num_rois, height, width)
roi_masks = np.array([np.array(mask) for mask in plane_segmentation['image_mask'].data])

# Compute a heatmap projection (max per pixel)
heatmap = np.max(roi_masks, axis=0)

plt.figure(figsize=(6,6))
plt.imshow(heatmap, cmap='hot')
plt.title('Summary ROI Mask Heatmap (max proj)')
plt.colorbar(label='ROI presence (0-1)')
plt.axis('off')
plt.tight_layout()
plt.savefig('tmp_scripts/roi_masks_heatmap.png')
# no plt.show()

io.close()
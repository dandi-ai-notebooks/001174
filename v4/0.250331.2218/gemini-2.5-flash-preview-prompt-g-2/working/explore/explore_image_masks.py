# This script loads and visualizes the image masks from the PlaneSegmentation

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os

# Load
url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get ImageMask data
image_masks = nwb.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"].image_mask

# Stack the image masks and take the maximum projection for visualization
# Taking a subset of masks if there are too many
num_masks = len(image_masks)
masks_to_plot = np.array([image_masks[i][:] for i in range(num_masks)])

# Create a maximum projection across all masks
max_projection = np.max(masks_to_plot, axis=0)

plt.figure(figsize=(8, 8))
plt.imshow(max_projection, cmap='gray')
plt.title(f'Maximum Projection of {num_masks} Image Masks')
plt.axis('off')

# Save the plot to a file
if not os.path.exists('explore'):
    os.makedirs('explore')
plt.savefig('explore/image_masks_max_projection.png')
plt.close()

print(f"Maximum projection of image masks plot saved to explore/image_masks_max_projection.png")
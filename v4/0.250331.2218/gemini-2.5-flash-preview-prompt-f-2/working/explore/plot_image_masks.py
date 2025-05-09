# This script loads and plots the superimposed image masks for all ROIs.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define the NWB file URL
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"

# Load the NWB file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the PlaneSegmentation and image_mask data
image_segmentation_module = nwb.processing["ophys"]["ImageSegmentation"]
plane_segmentation = image_segmentation_module.plane_segmentations["PlaneSegmentation"]

# Load all image masks
image_masks = plane_segmentation.image_mask[:]

# Superimpose the masks using np.max
# Reshape masks to (n_rois, height, width) if not already
if image_masks.ndim == 2:
    # Assuming masks are stored as flattened arrays in the dataset
    # Need to get dimensions from OnePhotonSeries or ImagingPlane
    # Based on previous inspection, OnePhotonSeries.data shape is (6026, 1280, 800)
    # Assuming the masks correspond to the spatial dimensions (1280, 800)
     height, width = nwb.acquisition["OnePhotonSeries"].data.shape[1:]
     n_rois = image_masks.shape[0]
     image_masks = image_masks.reshape(n_rois, height, width)

superimposed_mask = np.max(image_masks, axis=0)

# Plot the superimposed masks as a heatmap
plt.figure(figsize=(10, 8))
sns.set_theme() # Use seaborn theme for better aesthetics (except for images)
plt.imshow(superimposed_mask, cmap='viridis')
plt.title('Superimposed Image Masks for All ROIs')
plt.xlabel('Width')
plt.ylabel('Height')
plt.colorbar(label='Maximum Fluorescence')

# Save the plot to a file
plt.savefig('explore/image_masks.png')
plt.close()

# Close the NWB file (optional)
io.close()
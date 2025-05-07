# This script loads an NWB file and plots the image masks for all ROIs.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
# No seaborn styling for image plots
# import seaborn as sns
# sns.set_theme()


# Load NWB file
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get plane segmentation (which contains image masks)
plane_segmentation = nwb.processing['ophys']['ImageSegmentation']['PlaneSegmentation']
print(f"Number of ROIs: {len(plane_segmentation.id)}")
image_masks = plane_segmentation['image_mask'][:] # Load all image masks

# Determine the shape of the imaging plane from the first mask if available
if len(image_masks) > 0:
    mask_shape = image_masks[0].shape
    print(f"Shape of individual image mask: {mask_shape}")

    # Create a composite image by taking the maximum projection of all masks
    composite_mask = np.max(image_masks, axis=0)

    plt.figure(figsize=(8, 8))
    plt.imshow(composite_mask, cmap='viridis', interpolation='nearest')
    plt.title(f"All ROI Image Masks (Max Projection, {len(image_masks)} ROIs)")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.colorbar(label="Max Mask Value")
    plt.savefig("explore/roi_masks.png")
    plt.close()
    print("ROI masks plot saved to explore/roi_masks.png")
else:
    print("No image masks found to plot.")

io.close()
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Script to explore the image masks

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the image masks
processing = nwb.processing
ophys = processing["ophys"]
data_interfaces = ophys.data_interfaces
ImageSegmentation = data_interfaces["ImageSegmentation"]
plane_segmentations = ImageSegmentation["PlaneSegmentation"]
image_masks = plane_segmentations.image_mask

# Plot the image masks for the first 10 ROIs
num_rois = 10

# Get the dimensions of the image masks
first_image_mask = image_masks[0]
num_rows = first_image_mask.shape[0]
num_cols = first_image_mask.shape[1]

# Create a figure and axes
fig, axes = plt.subplots(1, num_rois, figsize=(20, 2))

# Plot the image masks
for i in range(num_rois):
    image_mask = image_masks[i]
    axes[i].imshow(image_mask, cmap='gray')
    axes[i].axis('off')

# Set the title of the figure
plt.suptitle(f'Image Masks for First {num_rois} ROIs')

# Save the plot to a file
plt.savefig("explore/image_masks.png")
plt.close()
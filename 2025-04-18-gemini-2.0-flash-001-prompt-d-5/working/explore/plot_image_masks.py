"""
This script loads the image masks for each ROI and plots them superimposed on each other.
"""
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get the image masks
plane_segmentation = nwb.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
image_masks = plane_segmentation.image_mask[:]

# Plot the image masks
plt.figure(figsize=(8, 8))
plt.imshow(np.max(image_masks, axis=0), cmap='gray')
plt.title("Image Masks")
plt.savefig("explore/image_masks.png")
plt.close()
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Script to plot image masks from an NWB file

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the image masks
plane_segmentation = nwb.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
image_masks = plane_segmentation['image_mask']
num_rois = len(image_masks)

# Plot the image masks superimposed
plt.figure(figsize=(8, 8))
all_masks = np.zeros(image_masks[0].shape, dtype=np.float32) # use float32

for i in range(num_rois):
    all_masks = np.maximum(all_masks, image_masks[i])

plt.imshow(all_masks, cmap='viridis')
plt.colorbar(label="Max Mask Value")
plt.title("Superimposed Image Masks")
plt.tight_layout()
plt.savefig("tmp_scripts/image_masks.png")
plt.close()
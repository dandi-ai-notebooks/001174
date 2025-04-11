import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f, mode='r')
nwb = io.read()

# Extract image masks
plane_segmentation = nwb.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
image_masks = plane_segmentation["image_mask"]

# Superimpose image masks
superimposed_masks = np.max(image_masks[:], axis=0)

# Generate heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(superimposed_masks, cmap="viridis")
plt.title("Superimposed Image Masks")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("tmp_scripts/image_masks.png")
plt.close()
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Script to load the NWB file and plot the image masks
try:
    # Load the NWB file
    url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()

    # Get image masks
    ophys = nwb.processing["ophys"]
    data_interfaces = ophys.data_interfaces
    plane_segmentation = data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
    image_masks = plane_segmentation.image_mask[:]

    # Plot the image masks using a heatmap
    plt.figure(figsize=(8, 6))

    # Superimpose all image masks
    superimposed_mask = np.max(image_masks, axis=0)

    plt.imshow(superimposed_mask, cmap='viridis')
    plt.colorbar(label="Intensity")
    plt.title("Superimposed Image Masks")
    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")

    plt.tight_layout()
    plt.savefig("explore/image_masks.png")
    print("Image masks plot saved to explore/image_masks.png")

except Exception as e:
    print(f"Error: {e}")
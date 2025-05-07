# explore_script_2.py
# Goal: Load NWB file and plot ROI image masks.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns # Not needed for image plots

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"
print(f"Loading NWB file from: {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get PlaneSegmentation
plane_segmentation = nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]
image_masks = plane_segmentation["image_mask"] # This is a VectorData of H5_dataset
num_rois = len(plane_segmentation.id) # or image_masks.shape[0] if it were a simple array
print(f"Number of ROIs: {num_rois}")

if num_rois > 0:
    # Determine dimensions of the image masks (they should all be the same)
    # We need to access the first mask to get its shape
    # image_mask is a VectorData, where each element is a 2D array (the mask)
    # For h5py datasets within VectorData, we need to read them individually.
    
    # Assuming all masks have the same shape, get shape from the first one
    # Must load the mask data to get its shape
    first_mask_data = image_masks[0] 
    mask_shape_y, mask_shape_x = first_mask_data.shape 
    print(f"Shape of individual image masks: ({mask_shape_y}, {mask_shape_x})")

    all_masks_array = np.zeros((num_rois, mask_shape_y, mask_shape_x), dtype=first_mask_data.dtype)
    for i in range(num_rois):
        all_masks_array[i, :, :] = image_masks[i][:,:] # Read data for each mask

    # Create a max projection of all masks
    max_projection = np.max(all_masks_array, axis=0)

    print("Plotting superimposed ROI masks...")
    plt.figure(figsize=(8, 8))
    plt.imshow(max_projection, cmap='viridis', interpolation='nearest', aspect='auto') # Changed aspect to auto
    plt.title("Superimposed ROI Image Masks (Max Projection)")
    plt.xlabel("X-pixels")
    plt.ylabel("Y-pixels")
    plt.colorbar(label="Max mask value")
    plt.savefig("explore/roi_masks_superimposed.png")
    plt.close()
    print("Saved roi_masks_superimposed.png")
else:
    print("No ROIs found in PlaneSegmentation.")

io.close()
print("Exploration script 2 finished.")
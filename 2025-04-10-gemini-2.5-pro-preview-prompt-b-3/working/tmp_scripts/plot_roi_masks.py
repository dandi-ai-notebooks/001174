# Script to load and plot the image masks (spatial footprints) of ROIs
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Define the NWB file URL
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"

# Load the NWB file using remfile and h5py
print(f"Loading NWB file from {url}...")
try:
    file = remfile.File(url)
    f = h5py.File(file, 'r')
    io = pynwb.NWBHDF5IO(file=f, mode='r')
    nwb = io.read()
    print("NWB file loaded successfully.")

    # Access the PlaneSegmentation table which contains the image masks
    plane_segmentation = nwb.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]

    # Load the image masks. This loads all masks into memory.
    # The shape is expected to be (num_rois, height, width)
    print("Loading image masks...")
    image_masks_data = plane_segmentation["image_mask"][:]
    num_rois, height, width = image_masks_data.shape
    print(f"Loaded {num_rois} masks with shape {height}x{width}.")

    # Create a max projection of all masks to visualize overlay
    max_projection = np.max(image_masks_data, axis=0)

    print("Plotting max projection of image masks...")
    # Plot the max projection
    plt.figure(figsize=(8, 8))
    # Do not use seaborn style for images
    img = plt.imshow(max_projection, cmap='viridis', interpolation='nearest', origin='lower')
    plt.title(f"Maximum Projection of {num_rois} ROI Masks")
    plt.xlabel("X Pixels")
    plt.ylabel("Y Pixels")
    plt.colorbar(img, label='Max Mask Weight')

    # Save the plot
    plot_filename = "tmp_scripts/roi_masks.png"
    print(f"Saving plot to {plot_filename}...")
    plt.savefig(plot_filename)
    plt.close() # Close the plot to free memory
    print("Plot saved.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Ensure resources are closed
    try:
        io.close()
    except Exception as e:
        print(f"Error closing NWB IO: {e}")
    try:
        f.close() # h5py file
    except Exception as e:
        print(f"Error closing HDF5 file: {e}")
    print("Script finished.")
# Objective: Plot all ROI image masks superimposed to visualize their spatial distribution.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
# No seaborn for this image plot

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwbfile = io.read()

# Access ImageSegmentation data
try:
    plane_segmentation = nwbfile.processing['ophys']['ImageSegmentation']['PlaneSegmentation']
    image_masks = plane_segmentation['image_mask'][:] # This is a list/array of 2D masks

    if not image_masks.any(): # Check if image_masks is not empty or all zeros
        print("Image masks are empty or all zeros.")
    else:
        # Ensure all masks have the same shape, or find common dimensions
        # Assuming all masks are for the same imaging plane and thus same dimensions
        # If image_masks is a 3D array (num_masks, height, width)
        if image_masks.ndim == 3:
            # Create a composite image by taking the maximum projection
            composite_mask = np.max(image_masks, axis=0)
        # If image_masks is a list of 2D arrays (ragged array if loaded directly from h5py)
        elif isinstance(image_masks, (list, h5py.VirtualLayout)) or (isinstance(image_masks, np.ndarray) and image_masks.dtype == 'object'):
            # Find the max dimensions
            max_h = 0
            max_w = 0
            valid_masks = []
            for mask_data in image_masks:
                if mask_data is not None and mask_data.ndim == 2:
                    valid_masks.append(mask_data)
                    if mask_data.shape[0] > max_h:
                        max_h = mask_data.shape[0]
                    if mask_data.shape[1] > max_w:
                        max_w = mask_data.shape[1]
            
            if not valid_masks:
                print("No valid 2D masks found in image_mask dataset.")
                composite_mask = None
            else:
                composite_mask = np.zeros((max_h, max_w))
                for mask_data in valid_masks:
                    # Pad masks if they are smaller than max dimensions (though unlikely for standard NWB)
                    padded_mask = np.zeros((max_h, max_w))
                    padded_mask[:mask_data.shape[0], :mask_data.shape[1]] = mask_data
                    composite_mask = np.maximum(composite_mask, padded_mask)
        else:
            print(f"Unexpected image_mask data structure: shape {image_masks.shape}, dtype {image_masks.dtype}")
            composite_mask = None

        if composite_mask is not None and composite_mask.any():
            plt.figure(figsize=(8, 8))
            plt.imshow(composite_mask, cmap='hot', interpolation='nearest') # Use hot colormap for intensity
            plt.colorbar(label='Max mask value')
            plt.title('Superimposed ROI Image Masks')
            plt.xlabel('X pixels')
            plt.ylabel('Y pixels')
            plt.savefig('explore/roi_masks_superimposed.png')
            plt.close()
            print("Saved plot to explore/roi_masks_superimposed.png")
        elif composite_mask is not None and not composite_mask.any():
            print("Composite mask is all zeros, nothing to plot.")
        else:
            # Already printed error if composite_mask is None
            pass
            

except KeyError as e:
    print(f"Could not find expected data: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

io.close()
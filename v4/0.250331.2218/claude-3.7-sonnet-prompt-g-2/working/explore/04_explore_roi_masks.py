"""
This script explores the ROI masks in the NWB file to understand their format
and create visualizations of the masks.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure aesthetics
sns.set_theme()

# Load the NWB file
print("Loading NWB file...")
url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the data
one_photon_series = nwb.acquisition['OnePhotonSeries']
ophys = nwb.processing['ophys']
image_segmentation = ophys.data_interfaces['ImageSegmentation']
plane_seg = image_segmentation.plane_segmentations['PlaneSegmentation']

# Get a sample frame from the raw imaging data for background
frame_idx = 1000
sample_frame = one_photon_series.data[frame_idx, :, :]
print(f"Sample frame shape: {sample_frame.shape}")

# Examine the first ROI mask to understand the format
first_mask = plane_seg['image_mask'][0]
print(f"First mask type: {type(first_mask)}")
print(f"First mask shape/length: {len(first_mask)}")

# Try to print the first few elements to understand the structure
print("\nFirst 10 elements of the first mask:")
print(first_mask[:10])

# Check if we have multiple masks or just one
num_rois = len(plane_seg.id)
print(f"\nNumber of ROIs: {num_rois}")

# Examine the structure of the mask data more carefully
print("\nExamining mask structure...")
mask_lengths = [len(plane_seg['image_mask'][i]) for i in range(min(5, num_rois))]
print(f"Length of first 5 masks: {mask_lengths}")

# Try to interpret the mask structure
# The masks might be sparse representations with pixel indices and values
print("\nInterpreting mask format...")
first_mask_min = np.min(first_mask) if len(first_mask) > 0 else "N/A"
first_mask_max = np.max(first_mask) if len(first_mask) > 0 else "N/A"
print(f"Min value in first mask: {first_mask_min}")
print(f"Max value in first mask: {first_mask_max}")

# Based on inspection, let's try to reconstruct the first few ROI masks into 2D form
# We'll assume the masks are in (row, col, value) format or similar
# If that doesn't work, we'll investigate other formats

print("\nAttempting to understand mask format by printing first few elements:")
for i in range(min(3, len(first_mask))):
    if isinstance(first_mask[i], np.ndarray):
        print(f"Element {i}: shape={first_mask[i].shape}, value={first_mask[i]}")
    else:
        print(f"Element {i}: type={type(first_mask[i])}, value={first_mask[i]}")

# Let's print the shape of the first mask array if it's multidimensional
if hasattr(first_mask, 'shape') and len(first_mask.shape) > 1:
    print(f"\nFirst mask has shape: {first_mask.shape}")
else:
    print("\nFirst mask is not a multidimensional array")

# Attempt to visualize masks based on pixel coordinates and values
# If the masks are sparse (x,y) coordinates with values, we need to reconstruct them
print("\nAttempting to visualize masks...")

# Try different mask interpretations
try:
    # Attempt 1: Try reshaping directly if the mask is a flat array of pixel values
    img_height, img_width = sample_frame.shape
    if len(first_mask) == img_height * img_width:
        # If mask is a flattened 2D array
        mask_2d = first_mask.reshape(img_height, img_width)
        plt.figure(figsize=(10, 8))
        plt.imshow(mask_2d, cmap='viridis')
        plt.colorbar(label='Mask Value')
        plt.title('ROI Mask (Reshaped Directly)')
        plt.savefig('mask_direct_reshape.png', dpi=150, bbox_inches='tight')
        plt.close()
except Exception as e:
    print(f"Attempt 1 failed: {e}")

try:
    # Attempt 2: If mask contains (y, x, value) triplets or similar
    mask_sparse = first_mask
    # Convert sparse representation to 2D image if it makes sense
    # This depends on the exact format, which we're investigating
    if len(mask_sparse) % 3 == 0:  # If divisible by 3, might be (y,x,value) triplets
        num_points = len(mask_sparse) // 3
        mask_2d = np.zeros(sample_frame.shape)
        for i in range(num_points):
            y, x, value = mask_sparse[i*3], mask_sparse[i*3+1], mask_sparse[i*3+2]
            if 0 <= int(y) < img_height and 0 <= int(x) < img_width:
                mask_2d[int(y), int(x)] = value
        plt.figure(figsize=(10, 8))
        plt.imshow(mask_2d, cmap='viridis')
        plt.colorbar(label='Mask Value')
        plt.title('ROI Mask (From Y,X,Value Triplets)')
        plt.savefig('mask_triplets.png', dpi=150, bbox_inches='tight')
        plt.close()
except Exception as e:
    print(f"Attempt 2 failed: {e}")

# If the mask is in some other format, we need to find a different way to interpret it
# Print the data type and values more comprehensively
print("\nDetailed analysis of mask values:")
mask_0 = plane_seg['image_mask'][0]
mask_0_array = np.array(mask_0)
print(f"Mask 0 data type: {mask_0_array.dtype}")
print(f"Mask 0 unique values count: {len(np.unique(mask_0_array))}")
print(f"Mask 0 first 5 unique values: {np.unique(mask_0_array)[:5]}")

# Create a combined visualization of all masks
all_masks = np.zeros(sample_frame.shape)
for i in range(num_rois):
    # Here we need to adapt based on what we learn about the mask format
    mask = plane_seg['image_mask'][i]
    # For now, just print info about each mask
    print(f"Mask {i}: length={len(mask)}")
    
    # As we understand the format better, we'll update this part
    # to properly combine all masks
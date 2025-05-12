"""
This script examines the ROI masks to understand their shape and structure
"""

import pynwb
import h5py
import remfile
import numpy as np

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/628c87ee-c3e1-44f3-b4b4-54aa67a0f6e4/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the plane segmentation
plane_segmentation = nwb.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
roi_count = len(plane_segmentation.id.data[:])

# Examine the shape of the ROI masks
print(f"Number of ROIs: {roi_count}")

for i in range(min(3, roi_count)):  # Just examine the first few ROIs
    roi_mask = plane_segmentation.image_mask[i]
    print(f"ROI {i} mask type: {type(roi_mask)}")
    print(f"ROI {i} mask shape or length: {len(roi_mask)}")
    
    # Try to determine the mask dimensions
    if hasattr(roi_mask, 'shape'):
        print(f"ROI {i} mask shape: {roi_mask.shape}")
    
    # Convert to numpy array to examine
    mask_array = np.array(roi_mask)
    print(f"ROI {i} as numpy array shape: {mask_array.shape}")
    
    # Check if we can determine the original dimensions
    length = len(mask_array)
    # Try to find factors close to 320x200
    possible_heights = []
    for h in range(200, 250):  # Try heights around expected value
        if length % h == 0:
            w = length // h
            possible_heights.append((h, w))
    
    print(f"Possible dimensions (height, width) for mask of length {length}: {possible_heights}")
    
    # Get some sample values to confirm it's a mask (should be 0 or 1)
    if len(mask_array) > 0:
        print(f"Min value: {np.min(mask_array)}, Max value: {np.max(mask_array)}")
        print(f"Unique values: {np.unique(mask_array)}")
    
    print("-" * 50)
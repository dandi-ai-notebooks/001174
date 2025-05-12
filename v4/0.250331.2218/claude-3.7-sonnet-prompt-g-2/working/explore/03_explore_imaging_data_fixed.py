"""
This script explores the calcium imaging data in the NWB file.
It visualizes:
1. A sample frame from the raw imaging data
2. ROI masks in their original format
3. Fluorescence traces for a few ROIs
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
fluorescence = ophys.data_interfaces['Fluorescence']
roi_response_series = fluorescence.roi_response_series['RoiResponseSeries']
image_segmentation = ophys.data_interfaces['ImageSegmentation']
plane_seg = image_segmentation.plane_segmentations['PlaneSegmentation']
event_amplitude = ophys.data_interfaces['EventAmplitude']

# Get a sample frame from the raw imaging data (frame 1000)
print("Getting sample frame...")
frame_idx = 1000
sample_frame = one_photon_series.data[frame_idx, :, :]
print(f"Sample frame shape: {sample_frame.shape}")

# Plot the sample frame
plt.figure(figsize=(10, 8))
plt.imshow(sample_frame, cmap='gray')
plt.colorbar(label='Intensity')
plt.title(f'Raw Imaging Data (Frame {frame_idx})')
plt.savefig('sample_frame.png', dpi=150, bbox_inches='tight')
plt.close()

# Examine the format of ROI masks
print("Examining ROI masks format...")
num_rois = len(plane_seg.id)
print(f"Number of ROIs: {num_rois}")

# Check what the mask looks like for the first ROI
mask_0 = plane_seg['image_mask'][0]
print(f"Type of mask: {type(mask_0)}")
print(f"Shape or length of mask: {len(mask_0) if hasattr(mask_0, '__len__') else 'N/A'}")

# Get the shape of the raw imaging data to understand dimensions
img_shape = sample_frame.shape
print(f"Image dimensions: {img_shape}")

# Extract and plot fluorescence traces
print("Extracting fluorescence traces...")
num_frames = roi_response_series.data.shape[0]
time_vector = np.arange(num_frames) / roi_response_series.rate  # Time in seconds

# Select a subset of ROIs to visualize (first 5)
roi_indices = np.arange(5)
selected_rois = roi_response_series.data[:, roi_indices]
roi_ids = plane_seg.id.data[roi_indices]

plt.figure(figsize=(15, 10))
for i, roi_idx in enumerate(roi_indices):
    plt.plot(time_vector, selected_rois[:, i], label=f'ROI {roi_ids[i]}')

plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (a.u.)')
plt.title('Fluorescence Traces for Selected ROIs')
plt.legend()
plt.grid(True)
plt.savefig('fluorescence_traces_short.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot a longer segment of data for one ROI
roi_idx = 0
plt.figure(figsize=(15, 5))
plt.plot(time_vector, selected_rois[:, roi_idx])
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (a.u.)')
plt.title(f'Fluorescence Trace for ROI {roi_ids[roi_idx]} (Full Recording)')
plt.grid(True)
plt.savefig('single_roi_full_trace.png', dpi=150, bbox_inches='tight')
plt.close()

# Compare fluorescence and event amplitude for one ROI
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(time_vector, roi_response_series.data[:, roi_idx])
plt.title(f'Fluorescence Trace for ROI {roi_ids[roi_idx]}')
plt.ylabel('Fluorescence (a.u.)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_vector, event_amplitude.data[:, roi_idx], color='red')
plt.title(f'Event Amplitude for ROI {roi_ids[roi_idx]}')
plt.xlabel('Time (s)')
plt.ylabel('Event Amplitude (a.u.)')
plt.grid(True)
plt.tight_layout()
plt.savefig('fluorescence_vs_events.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot a shorter segment of data to see more detail
segment_start = 2000
segment_len = 500
segment_time = time_vector[segment_start:segment_start+segment_len]
segment_fluo = roi_response_series.data[segment_start:segment_start+segment_len, roi_idx]
segment_events = event_amplitude.data[segment_start:segment_start+segment_len, roi_idx]

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(segment_time, segment_fluo)
plt.title(f'Fluorescence Trace for ROI {roi_ids[roi_idx]} (Segment)')
plt.ylabel('Fluorescence (a.u.)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(segment_time, segment_events, color='red')
plt.title(f'Event Amplitude for ROI {roi_ids[roi_idx]} (Segment)')
plt.xlabel('Time (s)')
plt.ylabel('Event Amplitude (a.u.)')
plt.grid(True)
plt.tight_layout()
plt.savefig('fluorescence_vs_events_segment.png', dpi=150, bbox_inches='tight')
plt.close()

print("Finished generating visualizations!")
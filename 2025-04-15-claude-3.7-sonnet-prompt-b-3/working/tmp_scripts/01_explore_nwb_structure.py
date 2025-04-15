'''
This script explores the basic structure of the NWB file, focusing on:
1. The general organization of the data
2. Details about the calcium imaging data
3. Basic statistics of the fluorescence data
'''

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/b4e6bbf7-0564-4628-b8f0-680fd9b8d4ea/download/"
print(f"Loading NWB file from: {url}")

remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the file
print("\n===== BASIC INFORMATION =====")
print(f"Session description: {nwb.session_description}")
print(f"Subject: {nwb.subject.subject_id}, Species: {nwb.subject.species}, Sex: {nwb.subject.sex}, Age: {nwb.subject.age}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Available processing modules: {list(nwb.processing.keys())}")
print(f"Available acquisition data: {list(nwb.acquisition.keys())}")

# Get information about the calcium imaging data
print("\n===== CALCIUM IMAGING DATA =====")
one_photon_series = nwb.acquisition["OnePhotonSeries"]
print(f"Imaging rate: {one_photon_series.rate} Hz")
print(f"Data shape: {one_photon_series.data.shape}")
print(f"Data type: {one_photon_series.data.dtype}")
print(f"Unit: {one_photon_series.unit}")

# Get information about the ROIs (Regions of Interest)
print("\n===== ROI INFORMATION =====")
plane_seg = nwb.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
num_rois = len(plane_seg.id[:])
print(f"Number of ROIs: {num_rois}")

# Get fluorescence data
print("\n===== FLUORESCENCE DATA =====")
fluorescence = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["RoiResponseSeries"]
print(f"Fluorescence data shape: {fluorescence.data.shape}")
print(f"Fluorescence data type: {fluorescence.data.dtype}")
print(f"Fluorescence rate: {fluorescence.rate} Hz")

# Calculate some basic statistics for fluorescence
# Load a sample of the data to avoid memory issues
sample_size = 1000
sample_data = fluorescence.data[:sample_size, :]
print(f"Sample data shape: {sample_data.shape}")

# Calculate statistics
mean_per_roi = np.mean(sample_data, axis=0)
std_per_roi = np.std(sample_data, axis=0)
max_per_roi = np.max(sample_data, axis=0)
min_per_roi = np.min(sample_data, axis=0)

print(f"Mean fluorescence per ROI (first 5): {mean_per_roi[:5]}")
print(f"Std fluorescence per ROI (first 5): {std_per_roi[:5]}")
print(f"Min fluorescence per ROI (first 5): {min_per_roi[:5]}")
print(f"Max fluorescence per ROI (first 5): {max_per_roi[:5]}")

# Plot mean fluorescence for each ROI
plt.figure(figsize=(10, 6))
plt.bar(np.arange(len(mean_per_roi)), mean_per_roi)
plt.xlabel('ROI Index')
plt.ylabel('Mean Fluorescence')
plt.title('Mean Fluorescence by ROI')
plt.tight_layout()
plt.savefig('tmp_scripts/mean_fluorescence_by_roi.png')

# Plot fluorescence traces for a few ROIs
plt.figure(figsize=(12, 8))
for i in range(min(5, fluorescence.data.shape[1])):  # First 5 ROIs
    plt.plot(sample_data[:, i], label=f'ROI {i}')
plt.xlabel('Time Sample Index')
plt.ylabel('Fluorescence')
plt.title('Fluorescence Traces for Sample ROIs')
plt.legend()
plt.tight_layout()
plt.savefig('tmp_scripts/fluorescence_traces.png')

# Check for event amplitude data
print("\n===== EVENT AMPLITUDE DATA =====")
event_amplitude = nwb.processing["ophys"].data_interfaces["EventAmplitude"]
print(f"Event amplitude data shape: {event_amplitude.data.shape}")
print(f"Event amplitude data type: {event_amplitude.data.dtype}")

# Plot event amplitude data for a few ROIs
sample_event_data = event_amplitude.data[:sample_size, :]
plt.figure(figsize=(12, 8))
for i in range(min(5, event_amplitude.data.shape[1])):  # First 5 ROIs
    plt.plot(sample_event_data[:, i], label=f'ROI {i}')
plt.xlabel('Time Sample Index')
plt.ylabel('Event Amplitude')
plt.title('Event Amplitude Traces for Sample ROIs')
plt.legend()
plt.tight_layout()
plt.savefig('tmp_scripts/event_amplitude_traces.png')

print("Script completed successfully!")
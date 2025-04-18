"""
This script explores the basic information and structure of the NWB file,
including metadata, available data types, and dimensions of the data.
"""

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd

# URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic file information
print("=" * 80)
print("BASIC NWB FILE INFORMATION")
print("=" * 80)
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Subject: {nwb.subject.subject_id} ({nwb.subject.species}, {nwb.subject.sex}, age: {nwb.subject.age})")
print()

# Examine device information
print("=" * 80)
print("DEVICE INFORMATION")
print("=" * 80)
for name, device in nwb.devices.items():
    print(f"Device: {name}")
    print(f"Description: {device.description}")
    print(f"Manufacturer: {device.manufacturer}")
print()

# Examine imaging plane information
print("=" * 80)
print("IMAGING PLANE INFORMATION")
print("=" * 80)
for name, plane in nwb.imaging_planes.items():
    print(f"Imaging Plane: {name}")
    print(f"Description: {plane.description}")
    print(f"Excitation wavelength: {plane.excitation_lambda} nm")
    print(f"Imaging rate: {plane.imaging_rate} Hz")
    print(f"Indicator: {plane.indicator}")
    print(f"Location: {plane.location}")
print()

# Examine one-photon series information
print("=" * 80)
print("ONE-PHOTON SERIES INFORMATION")
print("=" * 80)
one_photon_series = nwb.acquisition["OnePhotonSeries"]
print(f"Starting time: {one_photon_series.starting_time}")
print(f"Rate: {one_photon_series.rate} Hz")
print(f"Description: {one_photon_series.description}")
print(f"Unit: {one_photon_series.unit}")
print(f"Data shape: {one_photon_series.data.shape}")
print(f"Data type: {one_photon_series.data.dtype}")
print()

# Examine ophys processing module
print("=" * 80)
print("OPHYS PROCESSING INFORMATION")
print("=" * 80)
ophys = nwb.processing["ophys"]
print(f"Description: {ophys.description}")
print("Data interfaces:")
for name in ophys.data_interfaces:
    print(f"  - {name}")
print()

# Examine image segmentation
print("=" * 80)
print("IMAGE SEGMENTATION INFORMATION")
print("=" * 80)
image_seg = ophys.data_interfaces["ImageSegmentation"]
plane_seg = image_seg.plane_segmentations["PlaneSegmentation"]
print(f"Number of ROIs: {len(plane_seg.id)}")
print(f"Columns available: {plane_seg.colnames}")

# Examine ROI masks
print("\nChecking dimensions of ROI masks:")
for i in range(min(5, len(plane_seg.id))):
    mask = plane_seg.image_mask[i]
    if isinstance(mask, (h5py.Dataset, np.ndarray)):
        if hasattr(mask, 'shape'):
            print(f"  ROI {i} mask shape: {mask.shape}")

# Examine fluorescence data
print("\n" + "=" * 80)
print("FLUORESCENCE DATA INFORMATION")
print("=" * 80)
fluor = ophys.data_interfaces["Fluorescence"]
roi_response = fluor.roi_response_series["RoiResponseSeries"]
print(f"Number of time points: {roi_response.data.shape[0]}")
print(f"Number of ROIs: {roi_response.data.shape[1]}")
print(f"Sampling rate: {roi_response.rate} Hz")
print(f"Duration: {roi_response.data.shape[0] / roi_response.rate:.2f} seconds")
print(f"Duration: {roi_response.data.shape[0] / roi_response.rate / 60:.2f} minutes")

# Examine event amplitude data
print("\n" + "=" * 80)
print("EVENT AMPLITUDE INFORMATION")
print("=" * 80)
event_amp = ophys.data_interfaces["EventAmplitude"] 
print(f"Number of time points: {event_amp.data.shape[0]}")
print(f"Number of ROIs: {event_amp.data.shape[1]}")
print(f"Sampling rate: {event_amp.rate} Hz")
print(f"Duration: {event_amp.data.shape[0] / event_amp.rate:.2f} seconds")
print(f"Duration: {event_amp.data.shape[0] / event_amp.rate / 60:.2f} minutes")
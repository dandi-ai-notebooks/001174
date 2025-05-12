"""
This script explores the structure of the NWB file to understand 
what data is available for analysis. It prints information about the metadata,
data types, and datasets contained in the file.
"""

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the file
print("=" * 80)
print("NWB FILE BASIC INFO")
print("=" * 80)
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"File creation date: {nwb.file_create_date}")

# Print subject information
print("\n" + "=" * 80)
print("SUBJECT INFO")
print("=" * 80)
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")

# Print acquisition information
print("\n" + "=" * 80)
print("ACQUISITION INFO")
print("=" * 80)
for name, data_interface in nwb.acquisition.items():
    print(f"\nAcquisition: {name}")
    print(f"Type: {type(data_interface).__name__}")
    print(f"Description: {data_interface.description}")
    if hasattr(data_interface, 'data'):
        print(f"Data shape: {data_interface.data.shape}")
        print(f"Data type: {data_interface.data.dtype}")
    if hasattr(data_interface, 'rate'):
        print(f"Sampling rate: {data_interface.rate} Hz")
    if hasattr(data_interface, 'unit'):
        print(f"Unit: {data_interface.unit}")

# Print processing information
print("\n" + "=" * 80)
print("PROCESSING INFO")
print("=" * 80)
for module_name, module in nwb.processing.items():
    print(f"\nProcessing module: {module_name}")
    print(f"Description: {module.description}")
    print("\nData interfaces:")
    for interface_name, interface in module.data_interfaces.items():
        print(f"\n  Interface: {interface_name}")
        print(f"  Type: {type(interface).__name__}")
        
        # Handle different types of interfaces
        if hasattr(interface, 'roi_response_series'):
            print(f"  Contains ROI response series: {', '.join(interface.roi_response_series.keys())}")
            for series_name, series in interface.roi_response_series.items():
                print(f"    Series {series_name} shape: {series.data.shape}, dtype: {series.data.dtype}")
                print(f"    Sampling rate: {series.rate} Hz")
                print(f"    Unit: {series.unit}")
            
        elif hasattr(interface, 'plane_segmentations'):
            print(f"  Contains plane segmentations: {', '.join(interface.plane_segmentations.keys())}")
            for seg_name, seg in interface.plane_segmentations.items():
                print(f"    Segmentation {seg_name} has {len(seg.id)} ROIs")
                print(f"    Columns: {seg.colnames}")
                
        elif hasattr(interface, 'data'):
            print(f"  Data shape: {interface.data.shape}")
            print(f"  Data type: {interface.data.dtype}")
            print(f"  Sampling rate: {interface.rate} Hz") if hasattr(interface, 'rate') else None
            print(f"  Unit: {interface.unit}") if hasattr(interface, 'unit') else None

# Print information about ROIs
print("\n" + "=" * 80)
print("ROI INFO")
print("=" * 80)
# Get plane segmentation
ophys = nwb.processing['ophys']
image_segmentation = ophys.data_interfaces['ImageSegmentation']
plane_seg = image_segmentation.plane_segmentations['PlaneSegmentation']
print(f"Number of ROIs: {len(plane_seg.id)}")
print(f"ROI column names: {plane_seg.colnames}")

# Print information about the imaging plane
print("\n" + "=" * 80)
print("IMAGING PLANE INFO")
print("=" * 80)
imaging_plane = nwb.imaging_planes['ImagingPlane']
print(f"Description: {imaging_plane.description}")
print(f"Excitation wavelength: {imaging_plane.excitation_lambda} nm")
print(f"Imaging rate: {imaging_plane.imaging_rate} Hz")
print(f"Indicator: {imaging_plane.indicator}")
print(f"Location: {imaging_plane.location}")

# Print device information
print("\n" + "=" * 80)
print("DEVICE INFO")
print("=" * 80)
for name, device in nwb.devices.items():
    print(f"Device: {name}")
    print(f"Description: {device.description}")
    print(f"Manufacturer: {device.manufacturer}")
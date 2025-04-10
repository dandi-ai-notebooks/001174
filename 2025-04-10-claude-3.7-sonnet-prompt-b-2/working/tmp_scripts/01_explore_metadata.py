# This script explores the basic metadata of the NWB file

import pynwb
import h5py
import remfile

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

# Print basic metadata
print("===== NWB File Metadata =====")
print(f"Session Description: {nwb.session_description}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"File Create Date: {nwb.file_create_date}")

# Subject information
print("\n===== Subject Information =====")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")

# Device information
print("\n===== Device Information =====")
print(f"Device Description: {nwb.devices['Miniscope'].description}")
print(f"Manufacturer: {nwb.devices['Miniscope'].manufacturer}")

# Imaging parameters
print("\n===== Imaging Information =====")
print(f"Imaging Rate: {nwb.imaging_planes['ImagingPlane'].imaging_rate} Hz")
print(f"Excitation Wavelength: {nwb.imaging_planes['ImagingPlane'].excitation_lambda} nm")

# Data dimensions
one_photon_series = nwb.acquisition["OnePhotonSeries"]
frames_shape = one_photon_series.data.shape
print("\n===== Data Dimensions =====")
print(f"One Photon Series: {frames_shape} (frames, height, width)")

# Processed data
print("\n===== Processed Data =====")
print("Fluorescence data shape:", nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["RoiResponseSeries"].data.shape)
print("Event Amplitude data shape:", nwb.processing["ophys"].data_interfaces["EventAmplitude"].data.shape)

# Number of ROIs
num_rois = nwb.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"].id.data.shape[0]
print(f"Number of ROIs (cells): {num_rois}")
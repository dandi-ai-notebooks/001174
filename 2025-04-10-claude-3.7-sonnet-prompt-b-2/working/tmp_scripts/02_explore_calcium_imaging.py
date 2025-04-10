# This script explores the calcium imaging data

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

# Access the calcium imaging data
one_photon_series = nwb.acquisition["OnePhotonSeries"]
print(f"Data shape: {one_photon_series.data.shape}")
print(f"Data rate: {one_photon_series.rate} Hz")

# Extract a few frames for visualization (avoid loading too much data)
# We'll extract frames at different time points
frame_indices = [0, 100, 1000, 2000]  # Frames at different time points
frames = np.array([one_photon_series.data[i] for i in frame_indices])

print(f"Extracted frames shape: {frames.shape}")
print(f"Data type: {frames.dtype}")
print(f"Min value: {np.min(frames)}")
print(f"Max value: {np.max(frames)}")
print(f"Mean value: {np.mean(frames)}")

# Plot the frames
plt.figure(figsize=(15, 10))
for i, idx in enumerate(frame_indices):
    plt.subplot(2, 2, i+1)
    plt.imshow(frames[i], cmap='gray')
    plt.colorbar()
    plt.title(f"Frame {idx} (t = {idx/10:.1f}s)")

plt.tight_layout()
plt.savefig('tmp_scripts/calcium_imaging_frames.png')

# Plot the mean image (average over the frames)
plt.figure(figsize=(8, 6))
mean_frame = np.mean(frames, axis=0)
plt.imshow(mean_frame, cmap='gray')
plt.colorbar()
plt.title("Mean Image")
plt.tight_layout()
plt.savefig('tmp_scripts/calcium_imaging_mean.png')
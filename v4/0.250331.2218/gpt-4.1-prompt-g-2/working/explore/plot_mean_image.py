# This script computes the mean image over the first 100 frames from the OnePhotonSeries data in the NWB file,
# and saves it as a PNG. This approach, using a subset, avoids overloaded remote streaming and is usually
# sufficient for visualizing biological structure in calcium imaging data.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

one_photon = nwb.acquisition['OnePhotonSeries']
data = one_photon.data

# Compute mean image over the first 100 frames
N = 100
mean_image = np.mean(data[:N, :, :], axis=0)

plt.figure(figsize=(8, 5))
plt.imshow(mean_image, cmap='gray', aspect='auto')
plt.title(f'Mean Image (First {N} Calcium Imaging Frames)')
plt.colorbar(label='Fluorescence (a.u.)')
plt.tight_layout()
plt.savefig('explore/mean_image.png')
plt.close()
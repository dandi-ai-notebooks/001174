# This script loads a representative frame from the OnePhotonSeries data in the NWB file
# and saves it as a PNG image. The script will help judge whether the calcium imaging data
# can be visualized effectively for inclusion in the notebook.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

one_photon = nwb.acquisition['OnePhotonSeries']
frame0 = one_photon.data[0, :, :]  # First frame

plt.figure(figsize=(8, 5))
plt.imshow(frame0, cmap='gray', aspect='auto')
plt.title('First Frame of Calcium Imaging Data')
plt.colorbar(label='Fluorescence (a.u.)')
plt.tight_layout()
plt.savefig('explore/sample_frame.png')
plt.close()
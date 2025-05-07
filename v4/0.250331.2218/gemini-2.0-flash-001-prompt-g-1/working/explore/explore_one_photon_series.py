import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Script to explore the OnePhotonSeries data

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the OnePhotonSeries data
acquisition = nwb.acquisition
OnePhotonSeries = acquisition["OnePhotonSeries"]
data = OnePhotonSeries.data

# Plot the first 100 frames of the OnePhotonSeries data
num_frames = 100
fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i in range(10):
    axes[i].imshow(data[i], cmap='gray')
    axes[i].axis('off')
plt.suptitle(f'First {num_frames} frames of OnePhotonSeries data')

# Save plot to a file
plt.savefig("explore/one_photon_series.png")
plt.close()
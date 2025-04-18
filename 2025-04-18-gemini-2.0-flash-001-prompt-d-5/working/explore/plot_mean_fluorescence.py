"""
This script loads the OnePhotonSeries data from the NWB file and plots the mean fluorescence over time for a subset of the data.
"""
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get the OnePhotonSeries data
one_photon_series = nwb.acquisition["OnePhotonSeries"]
data = one_photon_series.data
num_frames = data.shape[0]
width = data.shape[1]
height = data.shape[2]

# Calculate the mean fluorescence for each frame
mean_fluorescence = np.mean(data[:1000, :, :], axis=(1, 2))

# Create a time vector
time = np.arange(0, len(mean_fluorescence)) / one_photon_series.imaging_plane.imaging_rate

# Plot the mean fluorescence over time
plt.figure(figsize=(10, 5))
plt.plot(time, mean_fluorescence)
plt.xlabel("Time (s)")
plt.ylabel("Mean Fluorescence")
plt.title("Mean Fluorescence Over Time")
plt.savefig("explore/mean_fluorescence.png")
plt.close()
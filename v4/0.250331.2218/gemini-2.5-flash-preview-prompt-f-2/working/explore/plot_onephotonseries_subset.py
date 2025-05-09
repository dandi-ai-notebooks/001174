# This script loads a single frame from the OnePhotonSeries data and plots it.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Define the NWB file URL
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"

# Load the NWB file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the OnePhotonSeries data
one_photon_series = nwb.acquisition["OnePhotonSeries"]

# Load a single frame (e.g., the first frame)
# Use array slicing to load only the first frame
frame = one_photon_series.data[0, :, :]

# Plot the frame
plt.figure(figsize=(10, 8))
plt.imshow(frame, cmap='gray')
plt.title('First Frame of OnePhotonSeries Data')
plt.xlabel('Width')
plt.ylabel('Height')
plt.colorbar(label='Fluorescence')

# Save the plot to a file in the explore directory
plt.savefig('explore/first_frame.png')
plt.close()

# Close the NWB file (optional, but good practice)
io.close()
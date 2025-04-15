# Exploratory Script 1
# This script loads and visualizes data from the NWB file. 
# Focus is on One-photon imaging data (first frame) and EventAmplitude data.

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import h5py
import remfile

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Plot the first frame of the OnePhotonSeries
one_photon_data = nwb.acquisition['OnePhotonSeries'].data
first_frame = one_photon_data[0, :, :]

plt.imshow(first_frame, cmap='gray')
plt.title('First Frame of OnePhotonSeries')
plt.savefig('tmp_scripts/first_frame.png')

# Plot EventAmplitude data
event_amplitude_data = nwb.processing['ophys'].data_interfaces['EventAmplitude'].data
mean_event_amp = np.mean(event_amplitude_data, axis=0)

plt.figure()
plt.plot(mean_event_amp)
plt.title('Mean Event Amplitude')
plt.xlabel('Unit')
plt.ylabel('Amplitude (fluorescence)')
plt.savefig('tmp_scripts/mean_event_amplitude.png')

# Clean up
io.close()
remote_file.close()
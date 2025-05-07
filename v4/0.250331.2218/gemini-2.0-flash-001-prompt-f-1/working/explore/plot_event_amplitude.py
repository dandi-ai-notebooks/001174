import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Script to load the NWB file and plot the EventAmplitude data
try:
    # Load the NWB file
    url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()

    # Get EventAmplitude data
    ophys = nwb.processing["ophys"]
    data_interfaces = ophys.data_interfaces
    event_amplitude = data_interfaces["EventAmplitude"]
    event_amplitude_data = event_amplitude.data[:]

    # Plot the EventAmplitude data for the first 3 ROIs
    num_rois = min(3, event_amplitude_data.shape[1])  # Plot only the first 3 ROIs if available
    time = np.linspace(0, event_amplitude_data.shape[0] / event_amplitude.rate, event_amplitude_data.shape[0])
    plt.figure(figsize=(10, 5 * num_rois))
    for i in range(num_rois):
        plt.subplot(num_rois, 1, i + 1)
        plt.plot(time, event_amplitude_data[:, i])
        plt.xlabel("Time (s)")
        plt.ylabel("Fluorescence")
        plt.title(f"EventAmplitude - ROI {i}")

    plt.tight_layout()
    plt.savefig("explore/event_amplitude.png")
    print("EventAmplitude plot saved to explore/event_amplitude.png")

except Exception as e:
    print(f"Error: {e}")
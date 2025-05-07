# explore_script_1.py
# Goal: Load NWB file, print basic info, and plot fluorescence traces for ROIs.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"
print(f"Loading NWB file from: {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Added mode='r'
nwb = io.read()

print(f"NWBFile Identifier: {nwb.identifier}")
print(f"Session Description: {nwb.session_description}")
print(f"Session Start Time: {nwb.session_start_time}")

# OnePhotonSeries
ops_data = nwb.acquisition["OnePhotonSeries"].data
print(f"OnePhotonSeries data shape: {ops_data.shape}") # (frames, y, x)

# Fluorescence (RoiResponseSeries)
rrs = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries"]
rrs_data = rrs.data
print(f"RoiResponseSeries data shape: {rrs_data.shape}") # (time, rois)
print(f"Number of ROIs: {rrs_data.shape[1]}")
print(f"Number of timepoints: {rrs_data.shape[0]}")

# Plot fluorescence traces for all ROIs
if rrs_data.shape[0] > 0 and rrs_data.shape[1] > 0:
    print("Plotting fluorescence traces...")
    sns.set_theme()
    plt.figure(figsize=(15, 5))
    # Load a subset of timepoints if too many, e.g., first 1000
    num_timepoints_to_plot = min(1000, rrs_data.shape[0])
    
    # Create time vector
    time_vector = np.arange(num_timepoints_to_plot) / rrs.rate

    for i in range(rrs_data.shape[1]):
        plt.plot(time_vector, rrs_data[:num_timepoints_to_plot, i], label=f'ROI {i+1}')
    
    plt.xlabel(f"Time (s) - showing first {num_timepoints_to_plot / rrs.rate:.2f} seconds")
    plt.ylabel("Fluorescence")
    plt.title(f"Fluorescence Traces (First {num_timepoints_to_plot} timepoints)")
    #plt.legend() # Avoid legend if too many ROIs, can make it cluttered
    plt.grid(True)
    plt.savefig("explore/fluorescence_traces.png")
    plt.close()
    print("Saved fluorescence_traces.png")
else:
    print("No ROI data to plot or no timepoints.")

io.close() # Close the NWBHDF5IO object
print("Exploration script 1 finished.")
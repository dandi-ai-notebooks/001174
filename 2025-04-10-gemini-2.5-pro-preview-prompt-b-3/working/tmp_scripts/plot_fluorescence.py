# Script to load and plot fluorescence traces for a few ROIs
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define the NWB file URL
url = "https://api.dandiarchive.org/api/assets/193fee16-550e-4a8f-aab8-2383f6d57a03/download/"

# Load the NWB file using remfile and h5py
print(f"Loading NWB file from {url}...")
try:
    file = remfile.File(url)
    f = h5py.File(file, 'r')
    io = pynwb.NWBHDF5IO(file=f, mode='r')
    nwb = io.read()
    print("NWB file loaded successfully.")

    # Access the Fluorescence data
    fluorescence_data = nwb.processing["ophys"].data_interfaces["Fluorescence"].roi_response_series["RoiResponseSeries"]
    data = fluorescence_data.data
    rate = fluorescence_data.rate
    starting_time = fluorescence_data.starting_time

    # Select ROIs and time points
    num_rois_to_plot = 3
    num_time_points = 1000 # Load only the first 1000 time points
    roi_indices = list(range(num_rois_to_plot))
    roi_ids = fluorescence_data.rois.table.id[roi_indices] # Get actual ROI IDs

    print(f"Loading fluorescence data for the first {num_rois_to_plot} ROIs and first {num_time_points} time points...")
    selected_data = data[:num_time_points, roi_indices]

    # Calculate timestamps for the selected data points
    selected_timestamps = starting_time + np.arange(num_time_points) / rate

    print("Plotting fluorescence traces...")
    # Plot the data
    sns.set_theme()
    plt.figure(figsize=(15, 5))
    for i, roi_id in enumerate(roi_ids):
        plt.plot(selected_timestamps, selected_data[:, i], label=f"ROI {roi_id}")

    plt.title(f"Fluorescence Traces (First {num_time_points} points) for ROIs {', '.join(map(str, roi_ids))}")
    plt.xlabel("Time (s)")
    plt.ylabel("Fluorescence")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_filename = "tmp_scripts/fluorescence_traces.png"
    print(f"Saving plot to {plot_filename}...")
    plt.savefig(plot_filename)
    plt.close() # Close the plot to free memory
    print("Plot saved.")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Ensure resources are closed
    try:
        io.close()
    except Exception as e:
        print(f"Error closing NWB IO: {e}")
    try:
        f.close() # h5py file
    except Exception as e:
        print(f"Error closing HDF5 file: {e}")
    print("Script finished.")
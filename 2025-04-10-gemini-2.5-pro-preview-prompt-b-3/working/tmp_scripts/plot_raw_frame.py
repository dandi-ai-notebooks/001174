# Script to load and display a single frame from the raw OnePhotonSeries data
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

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

    # Access the OnePhotonSeries data
    one_photon_series = nwb.acquisition["OnePhotonSeries"]
    raw_data = one_photon_series.data

    # Check dimensions
    num_frames, height, width = raw_data.shape
    print(f"Raw data shape: ({num_frames}, {height}, {width})")

    # Select a frame index (e.g., the first frame)
    frame_index = 0
    print(f"Loading frame {frame_index}...")
    # Load the selected frame
    # Note: This loads the full frame (1280x800) into memory
    frame_data = raw_data[frame_index, :, :]
    print("Frame loaded.")

    print("Plotting raw frame...")
    # Plot the frame
    plt.figure(figsize=(10, 8))
    # Do not use seaborn style for images
    plt.imshow(frame_data, cmap='gray', aspect='auto')
    plt.title(f"Raw Imaging Data - Frame {frame_index}")
    plt.xlabel("X Pixels")
    plt.ylabel("Y Pixels")
    plt.colorbar(label='Pixel Intensity (uint16)')

    # Save the plot
    plot_filename = "tmp_scripts/raw_frame.png"
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
# Script to plot a single frame from the OnePhotonSeries raw data
# Goal: Visualize an example raw image frame from the calcium imaging recording.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

def plot_one_photon_series_frame():
    url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
    
    remote_f = None
    try:
        remote_f = remfile.File(url)
        with h5py.File(remote_f, 'r') as h5_file:
            with pynwb.NWBHDF5IO(file=h5_file, mode='r', load_namespaces=True) as io:
                nwb = io.read()

                one_photon_series = nwb.acquisition.get("OnePhotonSeries")
                if one_photon_series is None:
                    print("OnePhotonSeries not found in acquisition.")
                    return
                
                # Data is typically (frames, height, width) or (frames, width, height)
                # From nwb-file-info: OnePhotonSeries.data # (Dataset) shape (6041, 320, 200); dtype uint16
                # So, height=320, width=200
                
                if one_photon_series.data.shape[0] > 0:
                    # Select a middle frame for visualization
                    frame_index = one_photon_series.data.shape[0] // 2
                    frame_data = one_photon_series.data[frame_index, :, :]
                    
                    plt.style.use('default') # Reset style for image plot
                    plt.figure(figsize=(8, 10)) # Aspect ratio based on 200x320
                    plt.imshow(frame_data, cmap='gray', aspect='auto') # Use 'auto' or calculate from shape
                    plt.title(f"Raw Imaging Data - Frame {frame_index}")
                    plt.xlabel("X pixel")
                    plt.ylabel("Y pixel")
                    plt.colorbar(label="Intensity (uint16)")
                    
                    output_path = "explore/raw_onephotonseries_frame.png"
                    plt.savefig(output_path)
                    print(f"Plot saved to {output_path}")
                    plt.close()
                else:
                    print("OnePhotonSeries data is empty.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if remote_f:
            remote_f.close()

if __name__ == "__main__":
    plot_one_photon_series_frame()
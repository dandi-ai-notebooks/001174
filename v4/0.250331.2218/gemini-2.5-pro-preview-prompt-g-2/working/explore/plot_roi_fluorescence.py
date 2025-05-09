# Script to plot fluorescence traces for a few ROIs
# Goal: Visualize fluorescence activity of selected ROIs over time.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_roi_fluorescence_traces():
    sns.set_theme()

    # Load NWB file
    url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
    
    remote_f = None
    try:
        remote_f = remfile.File(url) # Open remfile
        with h5py.File(remote_f, 'r') as h5_file: # Use h5py as context manager
            with pynwb.NWBHDF5IO(file=h5_file, mode='r', load_namespaces=True) as io: # Use NWBHDF5IO as context manager
                nwb = io.read()

                roi_response_series = nwb.processing["ophys"].get_data_interface("Fluorescence").get_roi_response_series("RoiResponseSeries")
                
                data = roi_response_series.data[:]
                rate = roi_response_series.rate
                num_frames = data.shape[0]
                num_rois_total = data.shape[1]

                timestamps = np.arange(num_frames) / rate

                num_rois_to_plot = min(3, num_rois_total)
                if num_rois_to_plot == 0:
                    print("No ROIs found in RoiResponseSeries.")
                    return

                roi_ids = roi_response_series.rois.table.id[:]
                
                plt.figure(figsize=(15, 7))
                for i in range(num_rois_to_plot):
                    roi_id = roi_ids[i]
                    plt.plot(timestamps, data[:, i], label=f"ROI {roi_id}")
                
                plt.xlabel("Time (s)")
                plt.ylabel("Fluorescence")
                plt.title(f"Fluorescence Traces for First {num_rois_to_plot} ROIs")
                plt.legend()
                plt.grid(True)
                
                output_path = "explore/roi_fluorescence_traces.png"
                plt.savefig(output_path)
                print(f"Plot saved to {output_path}")
                plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if remote_f:
            remote_f.close() # Explicitly close remfile.File

if __name__ == "__main__":
    plot_roi_fluorescence_traces()
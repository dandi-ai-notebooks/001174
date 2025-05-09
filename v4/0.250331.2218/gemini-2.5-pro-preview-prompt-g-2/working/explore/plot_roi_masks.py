# Script to plot ROI image masks
# Goal: Visualize the spatial footprints (image masks) of individual ROIs and a composite of all ROIs.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns # Not strictly needed if not using its themes for all plots

def plot_roi_image_masks():
    url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
    
    remote_f = None
    try:
        remote_f = remfile.File(url)
        with h5py.File(remote_f, 'r') as h5_file:
            with pynwb.NWBHDF5IO(file=h5_file, mode='r', load_namespaces=True) as io:
                nwb = io.read()

                plane_segmentation = nwb.processing["ophys"].get_data_interface("ImageSegmentation").get_plane_segmentation("PlaneSegmentation")
                
                image_masks_data = plane_segmentation["image_mask"] # VectorData
                roi_ids = plane_segmentation.id[:]
                num_rois = len(roi_ids)

                if num_rois == 0:
                    print("No ROIs found in PlaneSegmentation.")
                    return

                # Plot first few individual masks
                num_masks_to_plot_individually = min(3, num_rois)
                
                # Reset to default matplotlib style for image plots (no seaborn grid)
                plt.style.use('default')

                for i in range(num_masks_to_plot_individually):
                    mask = image_masks_data[i] # This is a 2D numpy array
                    roi_id = roi_ids[i]
                    
                    plt.figure(figsize=(6, 5))
                    plt.imshow(mask, cmap='viridis', interpolation='nearest')
                    plt.title(f"Image Mask for ROI {roi_id}")
                    plt.colorbar(label="Intensity")
                    plt.xlabel("X pixel")
                    plt.ylabel("Y pixel")
                    
                    output_path = f"explore/roi_mask_individual_{roi_id}.png"
                    plt.savefig(output_path)
                    print(f"Plot saved to {output_path}")
                    plt.close()

                # Create and plot a composite image of all masks
                if num_rois > 0:
                    # Assuming all masks have the same shape, get shape from the first mask
                    mask_shape = image_masks_data[0].shape
                    composite_mask = np.zeros(mask_shape, dtype=np.float32)
                    
                    for i in range(num_rois):
                        mask = image_masks_data[i]
                        # Ensure mask values are combined appropriately, e.g., max projection
                        composite_mask = np.maximum(composite_mask, mask)
                    
                    plt.figure(figsize=(8, 6))
                    # For the composite, seaborn theme might be fine or use a specific cmap
                    # sns.set_theme() # Optional: apply seaborn theme for this plot
                    plt.imshow(composite_mask, cmap='hot', interpolation='nearest') # 'hot' or 'afmhot' can be good
                    plt.title(f"Superimposed Image Masks for All {num_rois} ROIs (Max Projection)")
                    plt.colorbar(label="Max Intensity")
                    plt.xlabel("X pixel")
                    plt.ylabel("Y pixel")
                    
                    output_path_all = "explore/roi_masks_all_superimposed.png"
                    plt.savefig(output_path_all)
                    print(f"Plot saved to {output_path_all}")
                    plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if remote_f:
            remote_f.close()

if __name__ == "__main__":
    plot_roi_image_masks()
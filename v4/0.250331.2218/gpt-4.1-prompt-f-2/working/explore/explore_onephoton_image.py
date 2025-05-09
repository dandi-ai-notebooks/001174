# This script loads the OnePhotonSeries data from the example NWB file in Dandiset 001174.
# It produces a max projection image across the FIRST 500 timepoints (not the entire set, for performance).
# It also produces a sample frame image.
# Plots are saved to PNGs in the explore/ directory.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Parameters
url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
FRAME_IDX = 100  # Show an early frame (valid range 0-499 for this subset)
N_FRAMES_FOR_MAX = 500  # Only use first 500 frames for max projection

# Load NWB file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the OnePhotonSeries object
ophys = nwb.acquisition["OnePhotonSeries"]
data = ophys.data  # h5py.Dataset, shape (6041, 320, 200)
n_frames = min(N_FRAMES_FOR_MAX, data.shape[0])

# Compute max projection over the subset of frames
print(f"Computing max projection image for first {n_frames} frames...")
max_proj = None
for i in range(n_frames):
    frame = data[i]
    if max_proj is None:
        max_proj = frame.astype(np.float32)
    else:
        np.maximum(max_proj, frame, out=max_proj)
    if i % 100 == 0:
        print(f"Processed {i}/{n_frames} frames...")

plt.figure(figsize=(6, 6))
plt.imshow(max_proj, cmap='gray')
plt.title(f"Max Projection (first {n_frames} frames)")
plt.axis('off')
plt.tight_layout()
plt.savefig("explore/onephoton_max_projection.png", dpi=200)
plt.close()

# Plot a sample frame
print(f"Loading and plotting sample frame index {FRAME_IDX}...")
frame = data[FRAME_IDX]

plt.figure(figsize=(6, 6))
plt.imshow(frame, cmap='gray')
plt.title(f"Sample Frame #{FRAME_IDX}")
plt.axis('off')
plt.tight_layout()
plt.savefig("explore/onephoton_sample_frame.png", dpi=200)
plt.close()

print("Done. Images saved to explore/.")
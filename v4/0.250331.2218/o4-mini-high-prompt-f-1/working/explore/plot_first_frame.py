# Script to plot the first frame of OnePhotonSeries data from the Dandiset NWB file
import os
import matplotlib.pyplot as plt
from dandi.dandiapi import DandiAPIClient
import pynwb, h5py, remfile

# Connect to DANDI archive and load NWB
client = DandiAPIClient()
dandiset = client.get_dandiset("001174", "0.250331.2218")
asset_id = "de07db56-e7f3-4809-9972-755c51598e8d"
url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
remote_file = remfile.File(url)
h5file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5file)
nwb = io.read()

# Access OnePhotonSeries and load first frame
ops = nwb.acquisition["OnePhotonSeries"]
frame0 = ops.data[0, :, :]
print("First frame shape:", frame0.shape)

# Plot and save
os.makedirs("explore", exist_ok=True)
plt.figure(figsize=(6, 4))
plt.imshow(frame0, cmap="gray")
plt.colorbar()
plt.title("First OnePhotonSeries Frame")
plt.axis("off")
plt.savefig("explore/frame0.png", dpi=150)
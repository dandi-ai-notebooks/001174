# %% [markdown]
# Exploring Dandiset 001174: Calcium imaging in SMA and M1 of macaques

**Note**: This notebook was AI-generated and has not been fully verified. Use caution when interpreting the code or results.

# %% [markdown]
## Overview

This notebook demonstrates how to explore and analyze Dandiset 001174 (version 0.250331.2218) from the DANDI Archive.  
Dandiset title: *Calcium imaging in SMA and M1 of macaques*  
Dandiset link: https://dandiarchive.org/dandiset/001174/0.250331.2218  

What this notebook covers:
1. Loading Dandiset metadata and listing assets.  
2. Selecting and loading an NWB file.  
3. Summarizing NWB file contents.  
4. Visualizing example data from the NWB file.  
5. Possible next steps for analysis.

# %% [markdown]
## Required Packages

The following packages are assumed to be installed:
- itertools  
- dandi.dandiapi  
- remfile  
- h5py  
- pynwb  
- numpy  
- pandas  
- matplotlib  
- seaborn  

# %% [markdown]
## 1. Load Dandiset Information

# %%
from itertools import islice
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive and retrieve metadata
client = DandiAPIClient()
dandiset = client.get_dandiset("001174", "0.250331.2218")

metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List the first 5 assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
## 2. Load a Selected NWB File

We select the NWB file:
```
sub-Q/sub-Q_ophys.nwb
```
Asset ID: `de07db56-e7f3-4809-9972-755c51598e8d`  
Download URL:  
https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/

# %%
import remfile
import h5py
import pynwb

# Load the remote NWB file
url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, mode='r')
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# %% [markdown]
## 3. NWB File Metadata Summary

# %%
print("Session description:", nwb.session_description)
print("Identifier:", nwb.identifier)
print("Session start time:", nwb.session_start_time)
print("Timestamps reference time:", nwb.timestamps_reference_time)
print("Subject:", nwb.subject.subject_id, "|", nwb.subject.species, "| Age:", nwb.subject.age)

# %% [markdown]
### Contents of the NWB File

```
nwbfile/
├── acquisition
│   └── OnePhotonSeries (shape: {}) 
├── processing
│   └── ophys
│       ├── EventAmplitude (shape: {})
│       └── Fluorescence: RoiResponseSeries (shape: {})
├── devices (Miniscope)
├── imaging_planes (ImagingPlane)
└── subject (Subject)
```

Replace `{}`, `{}`, `{}` with actual shapes below.

# %%
# Extract shapes for the tree summary
ops = nwb.acquisition['OnePhotonSeries']
ea = nwb.processing['ophys'].data_interfaces['EventAmplitude']
rrs = nwb.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['RoiResponseSeries']

print(f"OnePhotonSeries data shape: {ops.data.shape}")
print(f"EventAmplitude shape: {ea.data.shape}")
print(f"RoiResponseSeries shape: {rrs.data.shape}")

# %% [markdown]
### Explore NWB Data Programmatically

You can explore table columns and metadata:

```python
# Example: list ROI table columns
print(ea.rois.table.colnames)
```

# %% [markdown]
## 4. Quick Visualization Examples

# %%
import numpy as np
import matplotlib.pyplot as plt

# 4.1 Plot the first frame of OnePhotonSeries
first_frame = ops.data[0, :, :]
plt.figure(figsize=(6, 4))
plt.imshow(first_frame, cmap='gray')
plt.colorbar(label='Intensity')
plt.title("OnePhotonSeries: First Frame")
plt.xlabel("X pixels")
plt.ylabel("Y pixels")
plt.tight_layout()
plt.show()

# %%
# 4.2 Plot EventAmplitude traces for the first 5 ROIs
import pandas as pd

roi_ids = ea.rois.table.id[:5]
times = np.arange(ea.data.shape[0]) / ea.rate

plt.figure(figsize=(10, 4))
for idx, roi in enumerate(roi_ids):
    plt.plot(times, ea.data[:, idx], label=f"ROI {roi}")
plt.legend(loc='upper right')
plt.xlabel("Time (s)")
plt.ylabel("Event Amplitude (fluorescence)")
plt.title("EventAmplitude for First 5 ROIs")
plt.tight_layout()
plt.show()

# %%
# 4.3 Heatmap of maximum projection of all ROI masks
masks = ea.rois.table.image_mask[:]  # shape (#ROIs, X, Y)
max_mask = np.max(masks, axis=0)

plt.figure(figsize=(5, 5))
plt.imshow(max_mask, cmap='hot')
plt.colorbar(label='Max mask value')
plt.title("Max Projection of ROI Masks")
plt.tight_layout()
plt.show()

# %% [markdown]
## Link to Neurosift

Explore this NWB file interactively on Neurosift:  
https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/&dandisetId=001174&dandisetVersion=0.250331.2218

# %% [markdown]
## 5. Summary and Future Directions

This notebook showed how to:
- Access Dandiset metadata and list assets.
- Load an NWB file remotely using DANDI APIs.
- Summarize NWB file structure and extract dataset shapes.
- Visualize sample frames, event amplitude traces, and ROI masks.

Possible next steps:
- Dive deeper into cell co-activity analysis.
- Correlate fluorescence with behavioral timestamps.
- Build interactive dashboards (e.g., using Plotly Dash).
- Apply advanced signal processing on fluorescence traces.
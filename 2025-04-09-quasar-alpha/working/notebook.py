# %% [markdown]
# # Calcium imaging in SMA and M1 of macaques (DANDI:001174)
#
# **Disclaimer:** This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Users should be cautious when interpreting results or code.
#
# This notebook explores a dataset of calcium imaging recordings in the supplementary motor area (SMA) and primary motor cortex (M1) of macaques during spontaneous activity and behavior. It guides you through loading the data, visualizing segmentation ROIs, and inspecting example calcium signals.
#
# **Dataset metadata:**
# - **DANDI ID:** 001174
# - **Title:** Calcium imaging in SMA and M1 of macaques
# - **Description:** Study of neuron activity via genetically encoded calcium indicators (GCaMP6f) imaged with GRIN-lens microendoscopes during reaching/rest.
# - **Species:** Macaca mulatta
# - **Methods:** One-photon miniaturized microscopy, segmentation of ROIs, extraction of calcium traces.
# - **Keywords:** one-photon imaging, NHPs, GCaMP, deep layer neurons
# - **License:** CC-BY-4.0
# - **Citation:**
#   Galvan *et al* (2025). Calcium imaging in SMA and M1 of macaques. DANDI Archive. Version draft. https://dandiarchive.org/dandiset/001174/draft

# %% [markdown]
# ## Setup
#
# Below, we use the DANDI API to list assets in this dataset.

# %%
from dandi.dandiapi import DandiAPIClient
client = DandiAPIClient()
dandiset = client.get_dandiset("001174")
assets = list(dandiset.get_assets())
assets[:5]  # Show first 5 assets

# %% [markdown]
# ## Choosing an NWB file
#
# Here we focus on a smaller example file: `sub-Q/sub-Q_ophys.nwb`.

# %% [markdown]
# ## Loading an NWB file remotely
#
# The following code loads the NWB file directly from the DANDI download URL using `remfile`, `h5py`, and `pynwb`. This is efficient for large remote files.

# %%
import pynwb
import h5py
import remfile

url = "https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwb = io.read()

# Basic metadata
print(f"Session description: {nwb.session_description}")
print(f"Subject ID: {nwb.subject.subject_id}, species: {nwb.subject.species}, sex: {nwb.subject.sex}")
print(f"Start time: {nwb.session_start_time}")
print(f"NWB identifier: {nwb.identifier}")

# %% [markdown]
# ## Exploring the segmentation masks
#
# Below is a heatmap summary projection of all ROI masks, showing spatial distributions of segmented cells.

# %% [markdown]
# ![](tmp_scripts/roi_masks_heatmap.png)
#
# The max projection illustrates many discrete ROIs distributed across the imaging field with minimal background noise, consistent with high-quality segmentation.

# %% [markdown]
# ## Example fluorescence traces and event amplitudes
#
# Here we display example fluorescence and event amplitude traces from five randomly selected ROIs, illustrating calcium activity dynamics.

# %% [markdown]
# ![](tmp_scripts/fluor_events_examples.png)
#
# The fluorescence signals exhibit clear calcium transients, corresponding well with punctuated increases in event amplitudes. Variability across ROIs is visible and typical for neural data.

# %% [markdown]
# ## Next steps
#
# This notebook provides starting points for exploring calcium imaging data:
# - Examine event amplitudes and fluorescence traces across conditions and ROIs
# - Quantify co-activity or correlation between ROIs
# - Extract features or epochs related to behavioral events (if available)
# - Overlays or mask-based plots with original imaging data
#
# **Remember:** This notebook is an AI-generated draft. Scripts should be adapted and verified carefully.

# %% [markdown]
# ## Closing
#
# The analyses here demonstrate how to access and visualize the NWB-formatted calcium imaging data. Full, rigorous analyses would involve additional quality control, statistical testing, and deeper neuroscientific questions.
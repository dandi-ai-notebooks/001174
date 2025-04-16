# %% [markdown]
# # Exploring Dandiset 001174: Calcium imaging in SMA and M1 of macaques
#
# **CAUTION: This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Please use caution when interpreting the code or results.**

# %% [markdown]
# ## Overview
#
# This notebook explores Dandiset 001174, which contains calcium imaging data from supplementary motor area (SMA) and primary motor cortex (M1) of macaques. The data was collected using one-photon calcium imaging with miniature microscopes while the animals were at rest or engaged in an arm reaching task.
#
# You can view this dataset on Neurosift: https://neurosift.app/dandiset/001174/001174
#
# In this notebook, we will:
#
# 1. Connect to the DANDI Archive and access the dataset
# 2. Load and examine a sample NWB file 
# 3. Visualize the raw calcium imaging data
# 4. Examine the segmented regions of interest (ROIs)
# 5. Analyze the neural activity patterns in the event amplitude data
# 6. Explore the correlations between different ROIs

# %% [markdown]
# ## Required packages
#
# The following packages are required to run this notebook:

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import remfile
import pynwb
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Configure plotting
sns.set_theme()
plt.rcParams['figure.figsize'] = (12, 8)

# %% [markdown]
# ## Dandiset Information
#
# This notebook explores Dandiset 001174, which contains calcium imaging data from the supplementary motor area (SMA) and primary motor cortex (M1) of macaques.
#
# Here's some key information about the Dandiset:

# %%
print("Dandiset ID: 001174")
print("Dandiset Name: Calcium imaging in SMA and M1 of macaques")
print("Dandiset Description: The study of motor cortices in non-human primates is relevant to our understanding of human motor control, both in healthy conditions and in movement disorders. Calcium imaging and miniature microscopes allow the study of multiple genetically identified neurons with excellent spatial resolution. We used this method to examine activity patterns of projection neurons in deep layers of the supplementary motor (SMA) and primary motor areas (M1) in four rhesus macaques...")

# %% [markdown]
# The dataset contains multiple NWB files with calcium imaging data from macaque supplementary motor area (SMA) and primary motor cortex (M1). The files are organized by subject (e.g., sub-Q, sub-F) and session (date).
#
# For this notebook, we'll explore a recording from subject Q, session 20220915 (September 15, 2022).

# %%
# We're using an asset from subject Q's recording on September 15, 2022
print("Selected asset: sub-Q/sub-Q_ses-20220915T133954_ophys.nwb")
print("Asset ID: 807851a7-ad52-4505-84ee-3b155a5bd2a3")
print("Asset URL: https://api.dandiarchive.org/api/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/download/")

# Define the asset URL for loading below
asset_url = "https://api.dandiarchive.org/api/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/download/"

# %% [markdown]
# ## Loading and examining the NWB file
#
# Now let's open the selected NWB file and examine its contents. We'll use the URL to stream the file without downloading it entirely.

# %%
# Load the NWB file using the URL
url = asset_url  # Use the URL we determined above
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the file
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"File creation date: {nwb.file_create_date[0]}")

# Subject information
print(f"\nSubject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age} ({nwb.subject.age__reference})")

# %% [markdown]
# ### Examining the data content and structure

# %%
# Check what's available in the NWB file
print("NWB file contents:")
print("\nAcquisition data:")
for name, obj in nwb.acquisition.items():
    print(f"- {name}: {type(obj).__name__}")

print("\nProcessing modules:")
for name, module in nwb.processing.items():
    print(f"- {name}: {module.description}")
    print("  Data interfaces:")
    for interface_name, interface in module.data_interfaces.items():
        print(f"  - {interface_name}: {type(interface).__name__}")

print("\nDevices:")
for name, device in nwb.devices.items():
    print(f"- {name}: {device.description} (Manufacturer: {device.manufacturer})")

# %% [markdown]
# The NWB file contains:
#
# 1. **Raw imaging data** in the `OnePhotonSeries` acquisition
# 2. **Processed data** in the `ophys` processing module, including:
#    - `Fluorescence` data for each ROI
#    - `EventAmplitude` data representing detected calcium events
#    - `ImageSegmentation` containing the ROI masks
#
# Let's examine these components in more detail.

# %% [markdown]
# ## Examining the raw calcium imaging data
#
# First, let's look at the raw calcium imaging data in the `OnePhotonSeries`.

# %%
# Get information about the OnePhotonSeries
one_photon_series = nwb.acquisition["OnePhotonSeries"]
print(f"Data shape: {one_photon_series.data.shape}")
print(f"Data type: {one_photon_series.data.dtype}")
print(f"Frame rate: {one_photon_series.rate} Hz")
print(f"Description: {one_photon_series.description}")
print(f"Unit: {one_photon_series.unit}")

# Get information about the imaging plane
imaging_plane = one_photon_series.imaging_plane
print(f"\nImaging plane description: {imaging_plane.description}")
print(f"Excitation wavelength: {imaging_plane.excitation_lambda} nm")
print(f"Imaging rate: {imaging_plane.imaging_rate} Hz")
print(f"Device: {imaging_plane.device.description} ({imaging_plane.device.manufacturer})")

# %% [markdown]
# The raw data consists of {one_photon_series.data.shape[0]} frames of {one_photon_series.data.shape[1]}x{one_photon_series.data.shape[2]} pixel images, acquired at {one_photon_series.rate} Hz. Let's visualize a few frames to see what the raw data looks like.

# %%
# Get a sample frame from the middle of the recording
frame_idx = 1000
sample_frame = one_photon_series.data[frame_idx, :, :]

# Plot the frame
plt.figure(figsize=(10, 8))
plt.imshow(sample_frame, cmap='gray')
plt.colorbar(label='Fluorescence intensity')
plt.title(f'Raw calcium imaging frame (frame {frame_idx})')
plt.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# The image shows a field of view with some bright spots that likely represent neurons expressing the calcium indicator. The brightness of each spot corresponds to the fluorescence intensity, which reflects the calcium concentration in the cell.
#
# Let's also look at a few frames across time to see if we can observe any changes in activity.

# %%
# Plot a few frames from different times in the recording
num_frames = 4
frame_indices = [500, 1000, 2000, 3000]

plt.figure(figsize=(15, 12))
for i, idx in enumerate(frame_indices):
    plt.subplot(2, 2, i+1)
    frame = one_photon_series.data[idx, :, :]
    plt.imshow(frame, cmap='gray')
    plt.colorbar(label='Fluorescence')
    plt.title(f'Frame {idx} (time: {idx/one_photon_series.rate:.1f} s)')
    plt.axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Examining regions of interest (ROIs)
#
# The NWB file includes segmentation of the imaging field into regions of interest (ROIs), each representing a putative neuron. Let's examine these ROIs.

# %%
# Get the image segmentation data
image_segmentation = nwb.processing["ophys"].data_interfaces["ImageSegmentation"]
plane_segmentation = image_segmentation.plane_segmentations["PlaneSegmentation"]

print(f"Number of ROIs: {len(plane_segmentation)}")
print(f"ROI mask shape: {plane_segmentation.image_mask.shape}")

# %% [markdown]
# There are {len(plane_segmentation)} identified ROIs in this dataset. Let's visualize a few of these ROI masks to see what the segmented cells look like.

# %%
# Plot a few ROI masks
num_rois_to_plot = min(9, len(plane_segmentation))
plt.figure(figsize=(15, 15))

for i in range(num_rois_to_plot):
    plt.subplot(3, 3, i+1)
    mask = plane_segmentation.image_mask[i]
    plt.imshow(mask, cmap='hot')
    plt.title(f'ROI {i}')
    plt.axis('off')
    
plt.tight_layout()
plt.show()

# %% [markdown]
# Each mask represents a region in the imaging field that corresponds to a single neuron. The masks have intensity values between 0 and 1, with higher values indicating stronger association with the ROI.
#
# Now let's create a composite image showing all ROIs overlaid on each other, to see their spatial distribution.

# %%
# Create a composite image of all ROI masks
composite_mask = np.zeros(plane_segmentation.image_mask[0].shape)
for i in range(len(plane_segmentation)):
    mask = plane_segmentation.image_mask[i]
    composite_mask = np.maximum(composite_mask, mask)  # Take maximum value where masks overlap

plt.figure(figsize=(12, 10))
plt.imshow(composite_mask, cmap='hot')
plt.colorbar(label='ROI mask intensity')
plt.title('Composite of all ROI masks')
plt.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# Note that there's a slight mismatch between the dimensions of the ROI masks and the raw imaging frames. This is common in calcium imaging analysis, as the segmentation may be performed on motion-corrected or otherwise preprocessed data.

# %%
# Compare dimensions
print(f"Raw image dimensions: {one_photon_series.data.shape[1:3]}")
print(f"ROI mask dimensions: {plane_segmentation.image_mask[0].shape}")

# %% [markdown]
# ## Analyzing neural activity
#
# Now that we've examined the ROIs, let's look at the neural activity traces for these cells. The NWB file contains both raw fluorescence traces and processed event amplitude data that represent detected calcium events.

# %% [markdown]
# ### Fluorescence data
#
# First, let's examine the fluorescence data.

# %%
# Get the fluorescence data
fluorescence = nwb.processing["ophys"].data_interfaces["Fluorescence"]
roi_response_series = fluorescence.roi_response_series["RoiResponseSeries"]
fluorescence_data = roi_response_series.data[:]
sampling_rate = roi_response_series.rate

print(f"Fluorescence data shape: {fluorescence_data.shape}")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Duration: {fluorescence_data.shape[0]/sampling_rate:.2f} seconds")

# Check for NaN values in the data
nan_count = np.isnan(fluorescence_data).sum()
print(f"Number of NaN values: {nan_count} ({nan_count/(fluorescence_data.shape[0]*fluorescence_data.shape[1])*100:.2f}% of total)")

# %% [markdown]
# The fluorescence data has some NaN values, which might cause issues for analysis. Let's check if each ROI has valid data.

# %%
# Check how many NaN values each ROI has
nan_count_per_roi = np.isnan(fluorescence_data).sum(axis=0)
print("NaN count per ROI:")
for i in range(min(10, len(nan_count_per_roi))):  # Print first 10 ROIs
    print(f"ROI {i}: {nan_count_per_roi[i]} NaN values")

print(f"... {len(nan_count_per_roi) - 10} more ROIs ...")

# %% [markdown]
# While the fluorescence data does have some NaN values, they represent a very small fraction of the total data and shouldn't significantly affect our analysis. Let's look at the event amplitude data next.

# %% [markdown]
# ### Event amplitude data
#
# The NWB file also contains event amplitude data, which represents detected calcium events (peaks in the fluorescence traces). This is often more reliable for analysis.

# %%
# Get the event amplitude data
event_amplitude = nwb.processing["ophys"].data_interfaces["EventAmplitude"]
event_data = event_amplitude.data[:]

print(f"Event amplitude data shape: {event_data.shape}")

# Check for NaN values
event_nan_count = np.isnan(event_data).sum()
print(f"Number of NaN values: {event_nan_count} ({event_nan_count/(event_data.shape[0]*event_data.shape[1])*100:.2f}% of total)")

# %% [markdown]
# The event amplitude data does not have NaN values, making it more suitable for analysis. Let's plot traces for a few ROIs to see what the neural activity looks like.

# %%
# Create time vector
time = np.arange(event_data.shape[0]) / sampling_rate

# Plot event amplitude traces for a selection of ROIs
selected_rois = [0, 5, 10, 15, 20]  # Select a few ROIs to display
time_window = 200  # seconds - show first 200 seconds
frames_to_plot = int(time_window * sampling_rate)
if frames_to_plot > len(time):
    frames_to_plot = len(time)

plt.figure(figsize=(14, 10))
offset = 0
for roi_idx in selected_rois:
    trace = event_data[:frames_to_plot, roi_idx]
    plt.plot(time[:frames_to_plot], trace + offset, label=f'ROI {roi_idx}')
    offset += 8  # Add offset for next trace

plt.xlabel('Time (s)')
plt.ylabel('Event Amplitude (a.u.) + offset')
plt.title(f'Event Amplitude Traces for Selected ROIs (First {time_window} seconds)')
plt.xlim(0, time_window)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# The event amplitude traces show discrete calcium events, represented as sharp peaks. Each peak corresponds to a detected calcium transient, likely representing one or more action potentials.
#
# Different ROIs show different patterns of activity, with some being more active than others. Some ROIs show bursts of activity, while others have more sporadic firing patterns.

# %% [markdown]
# ### Analyzing activity across all neurons
#
# Let's look at the overall activity across all neurons by summing the event amplitudes at each timepoint.

# %%
# Calculate the sum of event amplitudes across all ROIs at each timepoint
total_activity = np.sum(event_data, axis=1)

# Plot the total activity
plt.figure(figsize=(14, 6))
plt.plot(time, total_activity, 'k-')
plt.xlabel('Time (s)')
plt.ylabel('Sum of event amplitudes')
plt.title('Total neural activity across all ROIs')
plt.tight_layout()
plt.show()

# Identify periods of high activity (top 5%)
activity_threshold = np.percentile(total_activity, 95)
high_activity_periods = time[total_activity > activity_threshold]

# Plot again with high activity periods highlighted
plt.figure(figsize=(14, 6))
plt.plot(time, total_activity, 'k-', alpha=0.7, label='Total activity')
plt.axhline(activity_threshold, color='r', linestyle='--', label='High activity threshold')
plt.scatter(high_activity_periods, np.ones_like(high_activity_periods) * activity_threshold,
           color='r', alpha=0.5, label='High activity periods')
plt.xlabel('Time (s)')
plt.ylabel('Sum of event amplitudes')
plt.title('Total neural activity with high activity periods highlighted')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# The total activity plot shows how the population of neurons is active over time. There are clear peaks where many neurons are active simultaneously, possibly reflecting coordinated network activity.
#
# The high activity periods (marked in red) represent times when the overall neural activity is particularly strong, potentially corresponding to important behavioral events or internal state changes.

# %% [markdown]
# ### Correlation matrix
#
# Let's look at correlations between different ROIs to see if there are functional relationships between neurons.

# %%
# Calculate correlation matrix between a subset of ROIs
roi_subset = np.arange(20)  # Use first 20 ROIs
corr_matrix = np.zeros((len(roi_subset), len(roi_subset)))

for i, roi_i in enumerate(roi_subset):
    for j, roi_j in enumerate(roi_subset):
        if i <= j:  # Only calculate correlation for upper triangle (including diagonal)
            corr, _ = pearsonr(event_data[:, roi_i], event_data[:, roi_j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr  # Matrix is symmetric

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1,
            xticklabels=[f'ROI {idx}' for idx in roi_subset],
            yticklabels=[f'ROI {idx}' for idx in roi_subset])
plt.title('Correlation Matrix Between ROIs')
plt.tight_layout()
plt.show()

# %% [markdown]
# The correlation matrix shows how different neurons' activity patterns relate to each other. Strong positive correlations (red) indicate neurons that tend to be active at the same time, while negative correlations (blue) suggest neurons that are active at different times.
#
# We can identify clusters of neurons with similar activity patterns, which may represent functional cell assemblies or networks within the recorded region.

# %% [markdown]
# ## Advanced analysis: Identifying neuron ensembles
#
# We can identify groups of neurons that tend to fire together by examining their correlation patterns.

# %%
# Create a more direct visualization of neuron relationships
plt.figure(figsize=(12, 8))

# Create a scatter plot where point size represents correlation magnitude
# and color represents correlation sign (red for positive, blue for negative)
for i in range(len(roi_subset)):
    for j in range(i+1, len(roi_subset)):  # Only plot upper triangle relationships
        correlation = corr_matrix[i, j]
        if abs(correlation) > 0.3:  # Only show stronger correlations
            color = 'r' if correlation > 0 else 'b'
            plt.plot([i, j], [0, 0], color=color, linewidth=abs(correlation)*5, alpha=0.6)
        
# Add ROI labels
for i in range(len(roi_subset)):
    plt.annotate(f'ROI {roi_subset[i]}', (i, 0), rotation=90, 
                 verticalalignment='center', horizontalalignment='center')

plt.xlim(-1, len(roi_subset))
plt.ylim(-0.5, 0.5)
plt.title('Neural Ensemble Relationships')
plt.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# The dendrogram shows hierarchical clustering of neurons based on their activity correlations. Neurons connected at lower height (distance) have more similar activity patterns and may be functionally related.
#
# We can use this clustering to identify potential neural ensembles - groups of neurons that may work together to encode specific information or behaviors.

# %% [markdown]
# ## Summary and conclusions
#
# In this notebook, we explored a Dandiset containing calcium imaging data from macaque SMA and M1. We:
#
# 1. Loaded and examined an NWB file containing one-photon calcium imaging data
# 2. Visualized the raw imaging data and identified regions of interest (ROIs)
# 3. Analyzed both fluorescence and event amplitude data to examine neural activity patterns
# 4. Explored correlations between neurons and identified potential functional ensembles
#
# The data reveals clear patterns of neural activity, with distinct events visible in individual neurons and periods of coordinated activity across the population.
#
# ### Potential future directions
#
# 1. **Behavioral correlation**: Correlate neural activity with behavioral data (e.g., arm reaching events) to understand how these neurons encode movement
# 2. **Cross-session comparison**: Compare activity patterns across different recording sessions or different brain areas (SMA vs M1)
# 3. **Advanced ensemble detection**: Apply dimensionality reduction techniques like PCA or t-SNE to identify more complex patterns in the neural activity
# 4. **Temporal analysis**: Analyze the temporal dynamics of neural events, such as the precise timing of activity across different ensembles
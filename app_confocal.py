import streamlit as st
import numpy as np
import zarr

import os
import requests
import shutil
import zipfile

zarr_url = "https://raw.githubusercontent.com/AmSchulte/DRGrat/main/confocal example images/oib_file.zarr"
# Local filename for the downloaded zip
local_zip = "oib_file.zarr.zip"

# Folder to extract the Zarr dataset
zarr_folder = "oib_file.zarr"

# Download once if not already downloaded
if not os.path.exists(zarr_folder):
    print("Downloading Zarr dataset...")
    r = requests.get(zarr_url, stream=True)
    r.raise_for_status()
    with open(local_zip, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete. Extracting...")
    with zipfile.ZipFile(local_zip, "r") as zip_ref:
        zip_ref.extractall(zarr_folder)
    print("Extraction complete.")
else:
    print("Zarr dataset already exists locally.")

# -----------------------------
# Utility functions
# -----------------------------

def normalize_to_uint8(image, bit_depth=16):
    """Normalize an image to uint8 and enhance brightness/contrast."""
    # normalize to 0-1
    image = (image - np.min(image)) / np.ptp(image)
    # scale to 0-255
    image = (image / image.max()) * 255
    # enhance brightness & contrast
    gain = 1.5
    bias = 0.1
    image = np.clip(image * gain + bias, 0, 255)
    return image.astype(np.uint8)

@st.cache_resource
def open_zarr(zarr_path):
    """Open Zarr dataset lazily and cache it."""
    return zarr.open(zarr_path, mode="r")

def load_channels_slice(zarr_data, channel_indices, z_plane):
    """Load selected channels at a given Z-plane as a list of 2D arrays."""
    # Efficient: load all channels at once
    images = zarr_data[channel_indices, z_plane, :, :]
    if len(channel_indices) == 1:
        return [images[0]]
    else:
        return [images[i] for i in range(images.shape[0])]

def colorize_channel(image, channel_idx):
    """Convert a 2D uint8 image into RGB based on channel index."""
    if channel_idx == 0:      # Cyan
        return np.stack([np.zeros_like(image), image, image], axis=-1)
    elif channel_idx == 1:    # Red
        return np.stack([image, np.zeros_like(image), np.zeros_like(image)], axis=-1)
    elif channel_idx == 2:    # Yellow
        return np.stack([image, image, np.zeros_like(image)], axis=-1)
    else:
        return np.stack([image]*3, axis=-1)  # fallback gray

# -----------------------------
# App setup
# -----------------------------

st.title("Confocal Microscopy Viewer - rat DRG")


CHANNELS = ['NF', 'Fabp7', 'Iba1']

# Sidebar controls
st.sidebar.header("Image selection")

channel_selected = st.sidebar.segmented_control(
    "🌈 Channel",
    CHANNELS,
    default=[CHANNELS[0]],
    selection_mode="multi"
)
channel_indices = [CHANNELS.index(c) for c in channel_selected]

# Lazy open Zarr
zarr_data = open_zarr(zarr_folder)

# Z-plane slider
Z_PLANES = list(range(zarr_data.shape[1]))
z_plane = st.sidebar.select_slider("Z-Plane", Z_PLANES, value=Z_PLANES[len(Z_PLANES)//2])

# -----------------------------
# Load and process image
# -----------------------------

# Load selected channels for this Z-plane
images_2D = load_channels_slice(zarr_data, channel_indices, z_plane)

# Normalize and colorize
images_RGB = []
for idx, img in zip(channel_indices, images_2D):
    img_uint8 = normalize_to_uint8(img)
    img_colored = colorize_channel(img_uint8, idx)
    images_RGB.append(img_colored)

# Merge channels into one RGB image
image_RGB = np.clip(np.sum(images_RGB, axis=0), 0, 255).astype(np.uint8)

# Display
st.image(image_RGB)

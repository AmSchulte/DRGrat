import streamlit as st
import numpy as np
import zarr

# read and cache the metadata file

@st.cache_data
    
def normalize_to_uint8(image, bit_depth=16):
    max_val = 2**bit_depth - 1
    image = (image - np.min(image))/np.ptp(image)
    #image = np.clip(image, 0, max_val)
    image = (image / image.max()) * 255
    # increase britness and contrast
    gain = 1.5
    bias = 0.1
    image = np.clip(image * gain + bias, 0, 255)
    return image.astype(np.uint8)

def load_image_from_zarr(zarr_path, channel, z_plane):
    zarr_data = zarr.load(zarr_path)
    image = zarr_data[channel, z_plane, :, :]
    return image

zarr_path = "confocal example images/oib_file.zarr"

CHANNELS = ['NF', 'Fabp7', 'Iba1']
st.sidebar.header("Image selection")
# Channel selection for multiple channels
channel = st.sidebar.segmented_control("🌈 Channel", CHANNELS, default = CHANNELS[0], selection_mode="multi")
channel_numbers = [CHANNELS.index(c) for c in channel]

# get z-planes from zarr file

Z_PLANES = list(range(zarr.load(zarr_path).shape[1]))
z_plane = st.sidebar.select_slider("Z-Plane", Z_PLANES, value=Z_PLANES[len(Z_PLANES)//2])

image_RGB = []
for channel_n in channel_numbers:
    image = load_image_from_zarr(zarr_path, channel_n, z_plane)
    image = normalize_to_uint8(image)
    # if channel_n = 0 make cyan
    # if channel_n = 1 make red
    # if channel_n = 2 make yellow
    if channel_n == 0:
        image = np.stack([np.zeros_like(image), image, image], axis=-1)
    elif channel_n == 1:
        image = np.stack([image, np.zeros_like(image), np.zeros_like(image)], axis=-1)
    elif channel_n == 2:
        image = np.stack([image, image, np.zeros_like(image)], axis=-1)
    # combine to one image with 255 max
    image_RGB.append(image)

image_RGB = np.clip(np.sum(image_RGB, axis=0), 0, 255).astype(np.uint8)

st.title("Confocal Microscopy of rat DRG")
st.image(image_RGB)
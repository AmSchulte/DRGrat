import streamlit as st
import numpy as np
import zarr
from oiffile import OifFile
import requests
from pathlib import Path
import pandas as pd


# read and cache the metadata file


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

@st.cache_resource(show_spinner="Preparing image dataset...")
def download_oib_as_zarr(
    url: str,
    cache_dir="data",
    chunks=(1, 1, 512, 512),
):
    """
    Download OIB once, convert to Zarr once, reuse forever.
    Returns Path to Zarr directory.
    """
    cache_dir = Path(cache_dir)
    oib_dir = cache_dir / "oib"
    zarr_dir = cache_dir / "zarr"

    oib_dir.mkdir(parents=True, exist_ok=True)
    zarr_dir.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1]
    stem = filename.replace(".oib", "")

    oib_path = oib_dir / filename
    zarr_path = zarr_dir / f"{stem}.zarr"

    # ✅ If Zarr already exists, we are done
    if zarr_path.exists():
        return zarr_path

    # ⬇️ Download OIB if needed
    if not oib_path.exists():
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(oib_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    # 🔁 Convert OIB → Zarr
    oib = OifFile(str(oib_path))
    data = oib.asarray()
    z = zarr.open(zarr_path, mode="w", shape=data.shape, chunks=chunks, dtype=data.dtype)
    z[:] = data

    # 🧹 Optional: remove OIB to save space
    oib_path.unlink()

    return zarr_path

@st.cache_data
def load_metadata():
    # Load the metadata file
    metadata = pd.read_excel("metadata_filtered_confocal.xlsx")
    return metadata
# Load the metadata
metadata_filtered_confocal = load_metadata()

BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/944/S-BIAD1944/Files"
oib_file_path = metadata_filtered_confocal['Files'].iloc[1]
BIOIMAGE_URL = f"{BASE_URL}/CCI%20-%20{oib_file_path[6:]}"

zarr_path = download_oib_as_zarr(BIOIMAGE_URL)

zarr_data = zarr.open(zarr_path, mode="r")

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
    image = zarr_data[channel_n, z_plane, :, :]
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
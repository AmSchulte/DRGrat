import streamlit as st
import tifffile as tiff
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from streamlit_image_zoom import image_zoom
import pandas as pd
from skimage.color import hsv2rgb
import matplotlib.pyplot as plt

def load_tiff_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return tiff.imread(BytesIO(response.content))

def load_png_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("L")  # L = Graustufen

def normalize_to_uint8(image, bit_depth=12):
    max_val = 2**bit_depth - 1
    image = (image - np.min(image))/np.ptp(image)
    #image = np.clip(image, 0, max_val)
    image = (image / image.max()) * 255
    # increase britness and contrast
    gain = 1.5
    bias = 0.1
    image = np.clip(image * gain + bias, 0, 255)
    return image.astype(np.uint8)

# read and cache the metadata file

@st.cache_data
def load_metadata():
    # Load the metadata file
    metadata = pd.read_excel("https://raw.githubusercontent.com/AmSchulte/DRGrat/main/data/metadata_filtered.xlsx")
    return metadata
# Load the metadata
metadata_filtered = load_metadata()

# --------------------------
# CONFIGURATION
# --------------------------
BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/944/S-BIAD1944/Files"
MICROSCOPY_TYPES = ["Apotome", "Confocal"]
SEXES = ["males", "females"]
TIMEPOINTS = ["1 week", "5 weeks"]
SIDES = ["ipsilateral", "contralateral"]



# --------------------------
# UI SELECTION
# --------------------------


st.sidebar.header("Image selection")
show_mask = st.sidebar.toggle("Show segmentation mask")

microscopy = st.sidebar.segmented_control("🔬 Mikroscope", MICROSCOPY_TYPES, default=MICROSCOPY_TYPES[0])
sex = st.sidebar.segmented_control("⚥ Sex", SEXES, default=SEXES[0])
timepoint = st.sidebar.segmented_control("🕒 Time after CCI", TIMEPOINTS, default=TIMEPOINTS[0])
side = st.sidebar.segmented_control("📍 Side of injury", SIDES, default=SIDES[0])

DRG_LEVELS = ["L4", "L5"]
if microscopy == "Confocal":
    DRG_LEVELS = ["L4"]
drg_level = st.sidebar.segmented_control("🔘 DRG level", DRG_LEVELS, default=DRG_LEVELS[0])

# Define the rat IDs based on the metadata
metadata_filtered = metadata_filtered[metadata_filtered['Microscope'] == microscopy]
metadata_filtered = metadata_filtered[metadata_filtered['Sex'] == sex[:-1]] 
metadata_filtered = metadata_filtered[metadata_filtered['Time After CCI'] == timepoint]
metadata_filtered = metadata_filtered[metadata_filtered['Side'] == side]
metadata_filtered = metadata_filtered[metadata_filtered['DRG Level'] == drg_level]

STAINING_COMBINATIONS = metadata_filtered["Staining combination"].unique().tolist()


staining = st.sidebar.selectbox("🖌️ Staining", STAINING_COMBINATIONS)

metadata_filtered = metadata_filtered[metadata_filtered['Staining combination'] == staining]
RAT_IDS = metadata_filtered["Rat ID"].unique().tolist()

rat_id = st.sidebar.selectbox("🐀 Rat", RAT_IDS)

CHANNELS = metadata_filtered["Staining"].unique().tolist()
if "atf3" in CHANNELS:
    CHANNELS.append("atf3_all")

# remove channels that contain "png" or "tif" in their name
CHANNELS = [ch for ch in CHANNELS if "png" not in ch and "tif" not in ch]


channel = st.sidebar.segmented_control("🌈 Channel", CHANNELS, default = CHANNELS[0])

if channel == "atf3_all":
    channel = "atf3"
    atf3_all = True
else:
    atf3_all = False
#show_mask = st.sidebar.checkbox("Show segmentation mask", value=True)

metadata_filtered = metadata_filtered[metadata_filtered['Staining'] == channel]
metadata_filtered = metadata_filtered[metadata_filtered['Rat ID'] == rat_id]
# Get the number of images
IMAGE_NUMBER = len(metadata_filtered["Files"])
# make a slider for the amount of images
image_index = st.sidebar.slider("Image number", 1, IMAGE_NUMBER, 1)


image_filename = str(image_index).zfill(4)+".tif"
mask_filename = "masks/"+str(image_index).zfill(4)+".png"

if timepoint == "1 week":
    timepoint = "1W"
elif timepoint == "5 weeks":
    timepoint = "5W"

if side == "ipsilateral":
    side = "IL"
elif side == "contralateral":
    side = "CL"

for file in metadata_filtered['Files'].unique():
    if sex=='females':
        staining_name = file[34:36]+staining[5:]
        break
    else:
        staining_name = file[32:34]+staining[5:]

        

if microscopy == "Confocal":
    image_path = f"{BASE_URL}/CCI%20-%20{microscopy}/F5_{sex}/Tifs_max/{timepoint}/{rat_id}/{side}/{channel}/{image_filename}"
    mask_path = f"{BASE_URL}/CCI%20-%20{microscopy}/F5_{sex}/Tifs_max/{timepoint}/{rat_id}/{side}/{channel}_pred/{mask_filename}"
else:
    if atf3_all:
        image_path = f"{BASE_URL}/CCI%20-%20{microscopy}/{sex.capitalize()}/Tifs_adj/{timepoint}/{staining_name}/{rat_id}/{side}/{drg_level}/{channel}/{image_filename}"
        mask_path = f"{BASE_URL}/CCI%20-%20{microscopy}/{sex.capitalize()}/Tifs_adj/{timepoint}/{staining_name}/{rat_id}/{side}/{drg_level}/{channel}all_new_pred/{mask_filename}"
    else:
        image_path = f"{BASE_URL}/CCI%20-%20{microscopy}/{sex.capitalize()}/Tifs_adj/{timepoint}/{staining_name}/{rat_id}/{side}/{drg_level}/{channel}/{image_filename}"
        mask_path = f"{BASE_URL}/CCI%20-%20{microscopy}/{sex.capitalize()}/Tifs_adj/{timepoint}/{staining_name}/{rat_id}/{side}/{drg_level}/{channel}_pred/{mask_filename}"
    



# TIFF laden und normalisieren
image = load_tiff_from_url(image_path)
image = normalize_to_uint8(image)
img_pil = Image.fromarray(image).convert("RGB")

# import dilation
from skimage.morphology import binary_dilation

# Bild vorbereiten
if show_mask:
    try:
        # Maske laden
        color = (0, 255, 255, 0)

        # FIXME: add all channels
        if channel == "iba1" or channel == "ib4":
            color = (255, 255, 0, 0)
        elif channel == "gfap" or channel == "fabp7" or channel == "atf3":
            color = (255, 0, 0, 0)
        elif channel == "nf":
            color = (0, 255, 255, 0)

        mask = load_png_from_url(mask_path)
        st.header(str(channel.capitalize())+" & mask")

        # Farbliche Maske erstellen (rot) mit Alpha
        color_mask = Image.new("RGBA", img_pil.size, color)  # rot, aber komplett transparent
        mask_np = np.uint8(np.array(mask) > 0)

        # apply binary dilation, then subtract the original mask to get the border
        footprint = np.ones((3, 3), dtype=bool)
        outline = binary_dilation(mask_np, footprint) - mask_np

        # Transparenz setzen (z. B. 128 von 255 = 50 %)
        alpha = 25
        mask_alpha = (mask_np > 0).astype(np.uint8) * alpha
        mask_alpha[outline > 0] = 200  # set the outline to fully opaque
        color_mask_np = np.array(color_mask)
        color_mask_np[..., 3] = mask_alpha  # Alphakanal setzen
        color_mask = Image.fromarray(color_mask_np, mode="RGBA")

        # Originalbild in RGBA konvertieren
        img_rgba = img_pil.convert("RGBA")

        # Overlay erzeugen
        blended = Image.alpha_composite(img_rgba, color_mask)
        image_display = blended
    except:
        st.header(str(channel.capitalize()))
        st.error("No mask image available for this selection.")
        image_display = image
else:
    st.header(str(channel.capitalize()))
    image_display = image  


# Display image with custom settings
image_zoom(image_display, mode="both", size=(700, 700), keep_aspect_ratio=True, zoom_factor=4.0, increment=0.2)


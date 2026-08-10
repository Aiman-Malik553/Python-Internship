"""
data_loader.py
Flickr8k dataset load karne ke liye functions.
"""
import os
import pandas as pd
from PIL import Image


def load_captions(caption_file_path):
    """Caption CSV file ko pandas DataFrame mein load karta hai."""
    captions_df = pd.read_csv(caption_file_path)
    return captions_df


def get_human_captions(captions_df, image_name):
    """Ek specific image ke saare human-written captions return karta hai."""
    return captions_df[captions_df["image"] == image_name]["caption"].tolist()


def load_image(image_folder, image_name):
    """Image file ko PIL Image object ke roop mein load karta hai."""
    img_path = os.path.join(image_folder, image_name)
    return Image.open(img_path).convert("RGB")


def list_all_images(image_folder):
    """Folder mein saari images ke naam list karta hai."""
    return os.listdir(image_folder)

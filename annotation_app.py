import streamlit as st
import os
import glob
from PIL import Image
import shutil

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
RAW_DATA_DIR = "raw_data"
DATASET_DIR = "dataset/train"

CLASSES = {
    "Plastic": ["Bottle", "Cap", "Bag", "Cup", "Container", "Straw", "Cutlery", "Wrapper"],
    "Non-Plastic": ["Glass", "Paper", "Metal", "Organic", "Cardboard"]
}

EMOJIS = {
    "Bottle": "🍼",
    "Cap": "🪙",
    "Bag": "🛍️",
    "Cup": "🥤",
    "Container": "📦",
    "Straw": "🥤",
    "Cutlery": "🍴",
    "Wrapper": "🍬",
    "Glass": "🥛",
    "Paper": "📄",
    "Metal": "🥫",
    "Organic": "🌿",
    "Cardboard": "📦"
}

def init_dirs():
    # Create the structured directories
    for super_class, sub_classes in CLASSES.items():
        for sub_class in sub_classes:
            folder = os.path.join(DATASET_DIR, sub_class.lower())
            os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "trash"), exist_ok=True) # For bad images

def get_unannotated_images():
    # Grab all images from raw_data subfolders
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.webp')
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(RAW_DATA_DIR, "**", ext), recursive=True))
    return images

def move_image(img_path, category):
    dest_dir = os.path.join(DATASET_DIR, category.lower())
    filename = os.path.basename(img_path)
    
    # Handle filename collisions
    base, ext = os.path.splitext(filename)
    counter = 1
    new_path = os.path.join(dest_dir, filename)
    while os.path.exists(new_path):
        new_path = os.path.join(dest_dir, f"{base}_{counter}{ext}")
        counter += 1
        
    shutil.move(img_path, new_path)
    st.session_state.images_left -= 1

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="EcoScan Annotator", layout="wide")

init_dirs()

if 'images_list' not in st.session_state:
    st.session_state.images_list = get_unannotated_images()
    st.session_state.images_left = len(st.session_state.images_list)

st.title("EcoScanIndia: Rapid Annotator")
st.write(f"**Images remaining to annotate:** {st.session_state.images_left}")

# Load the next available image
current_images = get_unannotated_images()

if len(current_images) == 0:
    st.success("🎉 All images have been annotated!")
    st.balloons()
else:
    current_img_path = current_images[0]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        try:
            img = Image.open(current_img_path)
            st.image(img, caption=f"Source: {os.path.basename(os.path.dirname(current_img_path))}", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            if st.button("Skip Corrupt Image"):
                os.remove(current_img_path)
                st.rerun()

    with col2:
        st.subheader("Plastic Classes")
        cols_p = st.columns(3)
        for i, c in enumerate(CLASSES["Plastic"]):
            emoji = EMOJIS.get(c, "🥤")
            if cols_p[i % 3].button(f"{emoji} {c}", use_container_width=True):
                move_image(current_img_path, c)
                st.rerun()
                
        st.markdown("---")
        st.subheader("Non-Plastic Classes")
        cols_np = st.columns(3)
        for i, c in enumerate(CLASSES["Non-Plastic"]):
            emoji = EMOJIS.get(c, "♻️")
            if cols_np[i % 3].button(f"{emoji} {c}", use_container_width=True):
                move_image(current_img_path, c)
                st.rerun()
                
        st.markdown("---")
        if st.button("🗑️ Trash (Irrelevant/Bad Image)", use_container_width=True, type="primary"):
            move_image(current_img_path, "trash")
            st.rerun()

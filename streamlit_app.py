import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Page config
st.set_page_config(page_title="Plastic Detector AI", page_icon="🌍")

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 2)
    )
    # Load the custom weights
    try:
        model.load_state_dict(torch.load('plastic_detector_custom.pth', map_location=torch.device('cpu')))
    except Exception as e:
        st.error(f"Error loading model: {e}")
    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

RECYCLING_TIPS = {
    'water_bottle': '♻️ Rinse, remove cap, crush, recycle in blue bin',
    'plastic_bag': '🛍️ Take to grocery store drop-off - NOT curbside',
    'plastic_cup': '🥤 Check for #5 symbol - often recyclable',
    'plastic_bottle': '♻️ Remove label, rinse, recycle',
    'straw': '🚫 Single-use straws go in trash - use reusable instead',
}

def predict_image(image):
    """Predict if image contains plastic using custom model"""
    # Convert to PIL if not already
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert('RGB')
    else:
        image = image.convert('RGB')
        
    # Preprocess
    img_tensor = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Class 0: non_plastic, Class 1: plastic
    conf_non_plastic = probabilities[0].item()
    conf_plastic = probabilities[1].item()
    
    is_plastic = conf_plastic > conf_non_plastic
    confidence = conf_plastic if is_plastic else conf_non_plastic
    
    return is_plastic, confidence

def display_results(is_plastic, confidence):
    if is_plastic:
        st.error(f"🔴 **PLASTIC DETECTED** (Confidence: {confidence*100:.1f}%)")
        st.info("♻️ Please ensure you recycle plastic items properly!")
        with st.expander("View Specific Recycling Tips"):
            for item, tip in RECYCLING_TIPS.items():
                st.write(f"**{item.replace('_', ' ').title()}**: {tip}")
    else:
        st.success(f"🟢 **NOT PLASTIC** (Confidence: {confidence*100:.1f}%)")
        st.info("✅ Not plastic. Dispose responsibly.")

# UI
st.title("🌍 Plastic Detector AI")
st.markdown("Upload an image or use your webcam to detect plastic items and get recycling tips.")

tab1, tab2 = st.tabs(["📸 Upload Image", "🎥 Live Webcam"])

with tab1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        image = Image.open(uploaded_file)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
        with col2:
            with st.spinner("Analyzing..."):
                is_plastic, confidence = predict_image(image)
                display_results(is_plastic, confidence)

with tab2:
    st.markdown("### Point your camera at an item to detect plastic")
    camera_image = st.camera_input("Take a picture")
    
    if camera_image is not None:
        image = Image.open(camera_image)
        
        with st.spinner("Analyzing..."):
            is_plastic, confidence = predict_image(image)
            display_results(is_plastic, confidence)

st.markdown("---")
st.markdown("💡 **Tip:** Hold items in good lighting for best results!")

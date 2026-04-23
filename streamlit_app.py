import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request

# Page config
st.set_page_config(page_title="Plastic Detector AI", page_icon="🌍")

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Plastic mapping
PLASTIC_ITEMS = {
    'water_bottle': '💧 Water Bottle',
    'plastic_bag': '🛍️ Plastic Bag',
    'plastic_cup': '🥤 Plastic Cup',
    'plastic_bottle': '🍾 Plastic Bottle',
    'straw': '🥤 Plastic Straw',
}

RECYCLING_TIPS = {
    'water_bottle': '♻️ Rinse, remove cap, crush, recycle in blue bin',
    'plastic_bag': '🛍️ Take to grocery store drop-off - NOT curbside',
    'plastic_cup': '🥤 Check for #5 symbol - often recyclable',
    'plastic_bottle': '♻️ Remove label, rinse, recycle',
    'straw': '🚫 Single-use straws go in trash - use reusable instead',
}

@st.cache_data
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as f:
            labels = [line.decode().strip() for line in f.readlines()]
    except:
        labels = [f"class_{i}" for i in range(1000)]
    return labels

labels = load_labels()

def is_plastic_item(prediction_label):
    """Check if predicted item is plastic"""
    label_lower = prediction_label.lower()
    plastic_keywords = ['bottle', 'bag', 'cup', 'straw', 'container', 'wrapper', 'jug']
    return any(keyword in label_lower for keyword in plastic_keywords)

def predict_image(image):
    """Predict if image contains plastic"""
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
    
    # Get top predictions
    top_probs, top_indices = torch.topk(probabilities, 5)
    
    results = []
    is_plastic = False
    plastic_type = None
    
    for i in range(5):
        label = labels[top_indices[i]]
        confidence = top_probs[i].item()
        results.append((label, confidence))
        
        if is_plastic_item(label) and confidence > 0.3:
            is_plastic = True
            for key in PLASTIC_ITEMS:
                if key.replace('_', ' ') in label:
                    plastic_type = key
                    break
    
    return results, is_plastic, plastic_type

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
                results, is_plastic, plastic_type = predict_image(image)
                
                if is_plastic:
                    st.error("🔴 **PLASTIC DETECTED**")
                    tip = RECYCLING_TIPS.get(plastic_type, "♻️ Check local recycling guidelines")
                    st.info(tip)
                else:
                    st.success("🟢 **NOT PLASTIC**")
                    st.info("✅ This item is likely not plastic. Still, always recycle when possible!")
                
                st.write("**Top Predictions:**")
                for label, conf in results[:3]:
                    st.write(f"• {label}: {conf*100:.1f}%")

with tab2:
    st.markdown("### Point your camera at an item to detect plastic")
    camera_image = st.camera_input("Take a picture")
    
    if camera_image is not None:
        image = Image.open(camera_image)
        
        with st.spinner("Analyzing..."):
            results, is_plastic, plastic_type = predict_image(image)
            
            if is_plastic:
                st.error("🔴 **PLASTIC DETECTED**")
                tip = RECYCLING_TIPS.get(plastic_type, "♻️ Check local guidelines")
                st.info(tip)
            else:
                st.success("🟢 **NOT PLASTIC**")
                st.info("✅ Not plastic. Dispose responsibly.")
            
            st.write("**Top Predictions:**")
            for label, conf in results[:3]:
                st.write(f"• {label}: {conf*100:.1f}%")

st.markdown("---")
st.markdown("💡 **Tip:** Hold plastic items in good lighting for best results!")

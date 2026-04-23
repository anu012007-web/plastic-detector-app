import gradio as gr
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
import urllib.request

# Load model
model = models.mobilenet_v2(pretrained=True)
model.eval()

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

def is_plastic_item(prediction_label):
    """Check if predicted item is plastic"""
    label_lower = prediction_label.lower()
    plastic_keywords = ['bottle', 'bag', 'cup', 'straw', 'container', 'wrapper', 'jug']
    return any(keyword in label_lower for keyword in plastic_keywords)

def predict_image(image):
    """Predict if image contains plastic"""
    # Convert to PIL
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Get top predictions
    top_probs, top_indices = torch.topk(probabilities, 5)
    
    # Load ImageNet labels
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as f:
            labels = [line.decode().strip() for line in f.readlines()]
    except:
        labels = [f"class_{i}" for i in range(1000)]
    
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

def process_upload(image):
    """Process uploaded image"""
    if image is None:
        return "No image uploaded", "No prediction", "No tip", None
    
    results, is_plastic, plastic_type = predict_image(image)
    
    # Format results
    pred_text = f"**Top Predictions:**\n"
    for label, conf in results[:3]:
        pred_text += f"• {label}: {conf*100:.1f}%\n"
    
    if is_plastic:
        verdict = "🔴 **PLASTIC DETECTED**"
        tip = RECYCLING_TIPS.get(plastic_type, "♻️ Check local recycling guidelines")
    else:
        verdict = "🟢 **NOT PLASTIC**"
        tip = "✅ This item is likely not plastic. Still, always recycle when possible!"
    
    return verdict, pred_text, tip, image

# Create the app with SIMPLE, READABLE styling
with gr.Blocks(title="Plastic Detector AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌍 Plastic Detector AI
    
    Upload an image or use your webcam to detect plastic items and get recycling tips.
    """)
    
    with gr.Tabs():
        with gr.TabItem("📸 Upload Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="numpy", label="Upload Image")
                    upload_btn = gr.Button("Analyze Image", variant="primary")
                with gr.Column(scale=1):
                    verdict_output = gr.Markdown("**Waiting for image...**")
                    predictions_output = gr.Markdown("")
                    tip_output = gr.Markdown("")
            
            upload_btn.click(
                process_upload,
                inputs=[input_image],
                outputs=[verdict_output, predictions_output, tip_output, input_image]
            )
        
        with gr.TabItem("🎥 Live Webcam"):
            gr.Markdown("### Point your camera at an item to detect plastic")
            webcam = gr.Image(sources=["webcam"], streaming=True, label="Webcam Feed")
            capture_btn = gr.Button("Capture & Analyze", variant="primary")
            
            webcam_verdict = gr.Markdown("**Waiting for capture...**")
            webcam_predictions = gr.Markdown("")
            webcam_tip = gr.Markdown("")
            
            def capture_and_analyze(frame):
                if frame is None:
                    return "No frame captured", "", ""
                results, is_plastic, plastic_type = predict_image(frame)
                pred_text = f"**Top Predictions:**\n"
                for label, conf in results[:3]:
                    pred_text += f"• {label}: {conf*100:.1f}%\n"
                if is_plastic:
                    verdict = "🔴 **PLASTIC DETECTED**"
                    tip = RECYCLING_TIPS.get(plastic_type, "♻️ Check local guidelines")
                else:
                    verdict = "🟢 **NOT PLASTIC**"
                    tip = "✅ Not plastic. Dispose responsibly."
                return verdict, pred_text, tip
            
            capture_btn.click(
                capture_and_analyze,
                inputs=[webcam],
                outputs=[webcam_verdict, webcam_predictions, webcam_tip]
            )
    
    gr.Markdown("---")
    gr.Markdown("💡 **Tip:** Hold plastic items in good lighting for best results!")

# Launch
if __name__ == "__main__":
    demo.launch()

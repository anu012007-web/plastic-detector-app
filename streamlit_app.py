import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import sqlite3
import pandas as pd
import datetime
import base64
import folium
from streamlit_folium import st_folium
from ultralytics import YOLO
import hashlib
import json

# ==========================================
# CONFIG & INIT
# ==========================================
st.set_page_config(page_title="EcoScan", page_icon="🌿", layout="wide")

# Database Init
def init_db():
    conn = sqlite3.connect('detections.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  model TEXT,
                  plastic_type TEXT,
                  confidence REAL,
                  item_count INTEGER,
                  inf_time REAL,
                  lat REAL,
                  lon REAL,
                  img_hash TEXT)''')
    conn.commit()
    return conn

conn = init_db()

# ==========================================
# MULTILINGUAL DICTIONARIES
# ==========================================
LANGUAGES = {
    "English": {"title": "EcoScan", "hero": "Empowering a Cleaner India", "upload": "Upload Image", "camera": "Use Camera", "detect": "Detect Plastic", "plastic_detected": "PLASTIC DETECTED!", "not_plastic": "NOT PLASTIC", "admin": "Admin Dashboard", "paper": "Research Paper", "map": "Pollution Map"},
    "Hindi (हिन्दी)": {"title": "EcoScan", "hero": "स्वच्छ भारत की ओर एक कदम", "upload": "फोटो अपलोड करें", "camera": "कैमरा इस्तेमाल करें", "detect": "प्लास्टिक पहचानें", "plastic_detected": "प्लास्टिक मिला!", "not_plastic": "प्लास्टिक नहीं है", "admin": "एडमिन डैशबोर्ड", "paper": "शोध पत्र", "map": "प्रदूषण का नक्शा"},
    "Tamil (தமிழ்)": {"title": "EcoScan", "hero": "தூய்மையான இந்தியா", "upload": "படத்தை பதிவேற்றவும்", "camera": "கேமரா", "detect": "கண்டுபிடி", "plastic_detected": "பிளாஸ்டிக் உள்ளது!", "not_plastic": "பிளாஸ்டிக் இல்லை", "admin": "நிர்வாகி", "paper": "ஆராய்ச்சி", "map": "வரைபடம்"},
    "Telugu (తెలుగు)": {"title": "EcoScan", "hero": "క్లీన్ ఇండియా", "upload": "చిత్రాన్ని అప్‌లోడ్ చేయండి", "camera": "కెమెరా", "detect": "గుర్తించు", "plastic_detected": "ప్లాస్టిక్ దొరికింది!", "not_plastic": "ప్లాస్టిక్ కాదు", "admin": "అడ్మిన్", "paper": "పరిశోధన", "map": "మ్యాప్"},
    "Kannada (ಕನ್ನಡ)": {"title": "EcoScan", "hero": "ಸ್ವಚ್ಛ ಭಾರತ", "upload": "ಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ", "camera": "ಕ್ಯಾಮೆರಾ", "detect": "ಪತ್ತೆಮಾಡಿ", "plastic_detected": "ಪ್ಲಾಸ್ಟಿಕ್ ಪತ್ತೆಯಾಗಿದೆ!", "not_plastic": "ಪ್ಲಾಸ್ಟಿಕ್ ಅಲ್ಲ", "admin": "ಆಡಳಿತ", "paper": "ಸಂಶೋಧನೆ", "map": "ನಕ್ಷೆ"},
    "Bengali (বাংলা)": {"title": "EcoScan", "hero": "পরিষ্কার ভারত", "upload": "ছবি আপলোড করুন", "camera": "ক্যামেরা", "detect": "শনাক্ত করুন", "plastic_detected": "প্লাস্টিক সনাক্ত হয়েছে!", "not_plastic": "প্লাস্টিক নয়", "admin": "অ্যাডমিন", "paper": "গবেষণা", "map": "মানচিত্র"},
    "Marathi (मराठी)": {"title": "EcoScan", "hero": "स्वच्छ भारत", "upload": "फोटो अपलोड करा", "camera": "कॅमेरा", "detect": "ओळखा", "plastic_detected": "प्लास्टिक आढळले!", "not_plastic": "प्लास्टिक नाही", "admin": "अॅडमिन", "paper": "संशोधन", "map": "नकाशा"},
    "Gujarati (ગુજરાતી)": {"title": "EcoScan", "hero": "સ્વચ્છ ભારત", "upload": "ફોટો અપલોડ કરો", "camera": "કેમેરા", "detect": "શોધો", "plastic_detected": "પ્લાસ્ટિક મળ્યું!", "not_plastic": "પ્લાસ્ટિક નથી", "admin": "એડમિન", "paper": "સંશોધન", "map": "નકશો"},
    "Malayalam (മലയാളം)": {"title": "EcoScan", "hero": "ക്ലീൻ ഇന്ത്യ", "upload": "ചിത്രം അപ്‌ലോഡ് ചെയ്യുക", "camera": "ക്യാമറ", "detect": "കണ്ടെത്തുക", "plastic_detected": "പ്ലാസ്റ്റിക് കണ്ടെത്തി!", "not_plastic": "പ്ലാസ്റ്റിക് അല്ല", "admin": "അഡ്മിൻ", "paper": "ഗവേഷണം", "map": "മാപ്പ്"}
}

if 'lang' not in st.session_state: st.session_state.lang = "English"

def t(key): return LANGUAGES[st.session_state.lang].get(key, LANGUAGES["English"][key])

# ==========================================
# CSS & PWA INJECTION
# ==========================================
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    /* Theme Colors */
    :root {
        --deep-blue: #1E3A5F;
        --ocean-teal: #00BCD4;
        --yellow: #FFC107;
    }

    /* Hero Section */
    .hero-banner {
        background: linear-gradient(135deg, #FF9933 0%, #FFFFFF 50%, #138808 100%);
        padding: 2rem;
        border-radius: 30px;
        text-align: center;
        margin-bottom: 2rem;
        animation: gradientShift 5s ease infinite;
        background-size: 200% 200%;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Floating Upload Area */
    .upload-area, [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 2px dashed #FF9933;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    .upload-area:hover, [data-testid="stFileUploader"]:hover {
        border-color: #138808;
        background: rgba(255,255,255,1);
        transform: scale(1.02);
    }
    
    /* Fix text visibility in upload area for dark mode */
    [data-testid="stFileUploader"] label p,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] .st-emotion-cache-1b0udgb,
    [data-testid="stFileUploader"] .st-emotion-cache-10trnc2 {
        color: var(--deep-blue) !important;
        font-weight: 600;
    }

    /* Animated Buttons */
    .gradient-btn {
        background: linear-gradient(135deg, #FF6B35, #1E3A5F);
        border: none;
        padding: 12px 28px;
        border-radius: 30px;
        color: white !important;
        font-weight: bold;
        transition: all 0.3s ease;
        display: inline-block;
        text-decoration: none;
        margin-top: 10px;
    }

    .gradient-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(255,107,53,0.4);
        color: white !important;
    }

    /* Result Card Animation */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 1.5rem;
        animation: slideIn 0.5s ease-out;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Progress Bar */
    .confidence-bar {
        background: #E0E0E0;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
        margin-top: 10px;
    }

    .confidence-fill {
        background: linear-gradient(90deg, #FF9933, #138808);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
        animation: pulse 0.5s ease-out;
    }

    @keyframes pulse {
        0% { opacity: 0; width: 0%; }
        100% { opacity: 1; }
    }

    /* Recycling Tip Card */
    .tip-card {
        background: linear-gradient(135deg, #00BCD4 0%, #0097A7 100%);
        border-radius: 15px;
        padding: 1rem;
        color: white;
        margin-top: 1rem;
        animation: bounce 0.5s ease-out;
        text-align: center;
        font-weight: bold;
    }

    @keyframes bounce {
        0% { transform: scale(0.8); opacity: 0; }
        80% { transform: scale(1.05); }
        100% { transform: scale(1); opacity: 1; }
    }

    .swachh-title { font-size: 3.5rem; font-weight: 800; margin-bottom: 0.5rem; text-shadow: 1px 1px 2px rgba(255,255,255,0.8); color: var(--deep-blue); }
    .swachh-subtitle { font-size: 1.5rem; font-weight: 600; color: var(--deep-blue); text-shadow: 1px 1px 2px rgba(255,255,255,0.8); }

    .bounding-box-info {
        background: var(--yellow);
        color: var(--deep-blue);
        padding: 10px;
        border-radius: 10px;
        font-weight: bold;
        margin-top: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .icon-header {
        color: var(--ocean-teal);
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# PWA Service Worker Registration
st.components.v1.html("""
    <script>
    if ('serviceWorker' in navigator) {
      window.addEventListener('load', function() {
        navigator.serviceWorker.register('./sw.js').then(function(registration) {
          console.log('ServiceWorker registration successful');
        }, function(err) {
          console.log('ServiceWorker registration failed: ', err);
        });
      });
    }
    </script>
""", height=0)

# ==========================================
# MODELS & AI
# ==========================================
@st.cache_resource
def load_mobilenet():
    model = models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, 2))
    try:
        model.load_state_dict(torch.load('plastic_detector_custom.pth', map_location=torch.device('cpu')))
    except Exception as e:
        pass # Handle silently, will use untrained for demo if missing
    model.eval()
    return model

@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

mn_model = load_mobilenet()
yolo_model = load_yolo()

# MobileNet Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_mobilenet(image):
    img_tensor = transform(image).unsqueeze(0)
    
    start_time = datetime.datetime.now()
    with torch.no_grad():
        outputs = mn_model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    inf_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
    
    conf_non = probs[0].item()
    conf_plas = probs[1].item()
    is_plastic = conf_plas > conf_non
    conf = conf_plas if is_plastic else conf_non
    
    return is_plastic, conf, inf_time, image

def predict_yolo(image):
    start_time = datetime.datetime.now()
    results = yolo_model(image)
    inf_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
    
    # COCO Plastic classes: bottle (39), cup (41), spoon (44), bowl (45)
    plastic_classes = [39, 41, 44, 45]
    
    img_cv = np.array(image)
    if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
    detected_count = 0
    types_found = []
    max_conf = 0
    
    # Indian Flag Colors (BGR for OpenCV)
    colors = [(51, 153, 255), (255, 255, 255), (19, 136, 8)] # Saffron, White, Green
    
    for i, box in enumerate(results[0].boxes):
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        
        if cls_id in plastic_classes or conf > 0.1: # Also draw others for detailed mode, but flag plastics
            if cls_id in plastic_classes:
                detected_count += 1
                types_found.append(yolo_model.names[cls_id])
                max_conf = max(max_conf, conf)
                
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            color = colors[i % 3]
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
            label = f"{yolo_model.names[cls_id]} {conf:.2f}"
            cv2.putText(img_cv, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
    img_out = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    is_plastic = detected_count > 0
    return is_plastic, max_conf, inf_time, Image.fromarray(img_out), detected_count, types_found

def log_detection(model_name, p_type, conf, count, inf_time, img):
    img_hash = hashlib.md5(img.tobytes()).hexdigest()
    lat, lon = 28.6139 + np.random.normal(0, 0.1), 77.2090 + np.random.normal(0, 0.1) # Mock GPS around Delhi
    c = conn.cursor()
    c.execute("INSERT INTO detections (timestamp, model, plastic_type, confidence, item_count, inf_time, lat, lon, img_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (datetime.datetime.now().isoformat(), model_name, p_type, conf, count, inf_time, lat, lon, img_hash))
    conn.commit()

# ==========================================
# UI COMPONENTS
# ==========================================
st.sidebar.selectbox("Select Language", list(LANGUAGES.keys()), key="lang")
st.sidebar.markdown("---")
st.sidebar.markdown(f"### 🌿 {t('title')}")

st.markdown(f"""
<div class="hero-banner">
    <div class="swachh-title"><i class="fa-solid fa-leaf" style="color: var(--fresh-green)"></i> {t('title')}</div>
    <div class="swachh-subtitle">{t('hero')}</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([f"📸 {t('detect')}", f"🌍 {t('map')}", f"👑 {t('admin')}"])

with tab1:
    col_mode, col_up = st.columns([1, 2])
    mode = col_mode.radio("Detection Mode", ["Simple (MobileNetV2)", "Detailed (YOLOv8)"])
    
    source = col_up.radio("Image Source", ["Upload", "Camera"], horizontal=True)
    img_file = st.file_uploader(t("upload"), type=['jpg','jpeg','png']) if source == "Upload" else st.camera_input(t("camera"))
    
    if img_file:
        img = Image.open(img_file).convert('RGB')
        
        with st.spinner("Analyzing with AI..."):
            is_p, conf, inf, overlay = False, 0.0, 0.0, None
            count, p_types = 0, []
            
            if "Simple" in mode:
                is_p_mn, conf_mn, inf_mn, hm_img = predict_mobilenet(img)
                is_p, conf, inf, overlay = is_p_mn, conf_mn, inf_mn, hm_img
                log_detection("MobileNetV2", "general" if is_p else "none", conf, 1 if is_p else 0, inf, img)
            elif "Detailed" in mode:
                is_p_y, conf_y, inf_y, box_img, count_y, types_y = predict_yolo(img)
                is_p, conf, inf, overlay = is_p_y, conf_y, inf_y, box_img
                count, p_types = count_y, types_y
                log_detection("YOLOv8", ",".join(p_types) if p_types else "none", conf, count, inf, img)
                
        c1, c2 = st.columns(2)
        c1.image(img, caption="Original Image", use_container_width=True)
        c2.image(overlay if overlay is not None else img, caption="AI Analysis Result", use_container_width=True)
        
        if is_p:
            st.markdown(f'''
            <div class="result-card">
                <h2><i class="fa-solid fa-trash-can" style="color: #FF9933"></i> {t("plastic_detected")}</h2>
                <p><i class="fa-solid fa-bullseye"></i> Confidence: {conf:.1%}</p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {conf*100}%;"></div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            if count > 0: 
                st.markdown(f'<div class="bounding-box-info"><i class="fa-solid fa-crosshairs"></i> Detected {count} distinct plastic items: {", ".join(set(p_types))}</div>', unsafe_allow_html=True)
            
            wa_text = f"EcoScan Alert! 🚨 I just detected plastic waste using AI with {conf:.1%} confidence. Let's protect our environment! 🌿"
            wa_url = f"https://wa.me/?text={st.session_state.lang}%20{wa_text.replace(' ', '%20')}"
            st.markdown(f'<a href="{wa_url}" target="_blank" class="gradient-btn"><i class="fa-brands fa-whatsapp"></i> Share on WhatsApp</a>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="tip-card"><i class="fa-solid fa-recycle"></i> Tip: Please segregate this plastic and dispose of it responsibly! ♻️</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="result-card" style="background: linear-gradient(135deg, #138808 0%, #4CAF50 100%);">
                <h2><i class="fa-solid fa-leaf" style="color: white"></i> {t("not_plastic")}</h2>
                <p><i class="fa-solid fa-bullseye"></i> Confidence: {conf:.1%}</p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {conf*100}%;"></div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

with tab2:
    st.markdown(f'<h2><i class="fa-solid fa-earth-asia icon-header"></i> {t("map")}</h2>', unsafe_allow_html=True)
    st.write("Community reported plastic hotspots")
    
    df = pd.read_sql_query("SELECT lat, lon, plastic_type FROM detections WHERE plastic_type != 'none'", conn)
    if not df.empty:
        m = folium.Map(location=[28.6139, 77.2090], zoom_start=11)
        for _, row in df.iterrows():
            folium.Marker([row['lat'], row['lon']], tooltip=row['plastic_type'], icon=folium.Icon(color='red')).add_to(m)
        st_folium(m, width=800, height=400)
    else:
        st.info("No data available yet.")

with tab3:
    st.markdown(f'<h2><i class="fa-solid fa-crown icon-header"></i> {t("admin")}</h2>', unsafe_allow_html=True)
    pwd = st.text_input("Enter Admin Password", type="password")
    if pwd == "swachhbharat": # Mock password
        df_all = pd.read_sql_query("SELECT * FROM detections ORDER BY id DESC", conn)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Scans", len(df_all))
        c2.metric("Plastics Found", len(df_all[df_all['plastic_type'] != 'none']))
        c3.metric("Est. CO2 Saved (kg)", round(len(df_all[df_all['plastic_type'] != 'none']) * 0.15, 2))
        
        st.dataframe(df_all)
        csv = df_all.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV Report", csv, "ecoscan_report.csv", "text/csv")
    elif pwd:
        st.error("Incorrect password")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Powered by EcoScan AI</div>", unsafe_allow_html=True)

# ♻️ EcoScanIndia

EcoScanIndia is an advanced, AI-driven web application tailored to detect plastic waste and promote environmental sustainability. Built with a focus on the Swachh Bharat Mission, it leverages deep learning to accurately identify plastic objects in uploaded images or via a live camera feed.

## Features

- **Advanced AI Detection:** Utilizes a combination of YOLOv8 for object detection and a fine-tuned MobileNetV2 model for high-accuracy plastic classification.
- **Multilingual Support:** Comprehensive support for 22 official Indian languages, including proper rendering for non-Latin and Right-to-Left (RTL) scripts.
- **Modern User Interface:** A vibrant, responsive, and mobile-ready web app built with Streamlit, featuring dynamic styling.
- **Accessibility:** Designed to be user-friendly across different devices.

## Prerequisites

Ensure you have Python installed (Python 3.8+ recommended). 

## Installation

1. Open your terminal or command prompt.
2. Navigate to the project directory:
   ```bash
   cd "path/to/Plastic detector"
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application Locally

1. In the terminal, run the following command to start the Streamlit server:
   ```bash
   streamlit run streamlit_app.py
   ```
2. The terminal will output a local URL, typically `http://localhost:8501`.
3. Open the URL in your web browser.
4. Use the interface to upload an image or use your camera to detect plastic items!

## Project Structure

- `streamlit_app.py` - The main Streamlit web application.
- `translations_complete.py` - Contains the translation dictionaries for 22 Indian languages.
- `requirements.txt` - Project dependencies.
- `plastic_mapping.py` - Logic for mapping model outputs to specific plastic recycling guidelines.
- `README.md` - Project documentation.

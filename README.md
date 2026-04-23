# ♻️ Plastic Detector AI

This project is a simple web application that detects if an uploaded image contains a plastic object using a pre-trained **MobileNetV2** artificial intelligence model.

## Prerequisites
Ensure you have Python installed. The project relies on deep learning libraries to classify objects accurately.

## Installation

1. Open your terminal or command prompt.
2. Navigate to this directory (`cd /path/to/plastic_detector`).
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. In the terminal, run the following command:
   ```bash
   python app.py
   ```
2. The terminal will display a local URL, typically `http://127.0.0.1:7860`.
3. Open the URL in your web browser.
4. Upload an image to test whether the primary object is made of plastic!

## File Structure

- `app.py` - Main web app with Gradio interface and model logic.
- `requirements.txt` - Required packages (PyTorch + Gradio + Pillow).
- `plastic_mapping.py` - List of mapped ImageNet classes for plastic items.
- `README.md` - Documentation and setup instructions.

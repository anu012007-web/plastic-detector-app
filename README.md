# ♻️ EcoScanIndia

EcoScanIndia is an advanced, AI-driven web application tailored to detect plastic waste and promote environmental sustainability. Built with a focus on the Swachh Bharat Mission, it leverages deep learning to accurately identify plastic objects in uploaded images or via a live camera feed.

## Features

- **Advanced AI Detection:** Utilizes a combination of YOLOv8 for object detection and a fine-tuned MobileNetV2 model for high-accuracy plastic classification.
- **Multilingual Support:** Comprehensive support for 22 official Indian languages, including proper rendering for non-Latin and Right-to-Left (RTL) scripts.
- **Automatic Rain Shield System:** Integrates Raspberry Pi 4 GPIO hardware control to automatically deploy a protective shield over the robot's camera/electronics when rain is detected, auto-dock the robot, and retract the shield safely once the weather clears.
- **Modern User Interface & Status Widget:** A vibrant, responsive Streamlit dashboard featuring live telemetry feeds (battery, temperature, shield position, and rain status) synchronized dynamically using FastAPI webhooks.
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

## Automatic Rain Shield & Telemetry Setup

The Rain Shield subsystem consists of a hardware controller running on the Raspberry Pi and a central webhook receiver endpoint that stores status telemetry:

### 1. Start the Webhook Receiver API
Run the receiver server locally (or on your central server) to receive and store robot telemetry:
```bash
pip install fastapi uvicorn pydantic
python webhook_receiver.py
```
This runs a FastAPI server at `http://localhost:8000`.

### 2. Run/Simulate the Rain Shield Controller
To run the automated shield monitor (with servo safety checks, debounce button override, and stability timers):
```bash
python rain_shield.py
```
*(On non-Raspberry Pi systems, it automatically falls back to a Mock GPIO simulation mode for testing.)*

### 3. Run Simulated Hardware Tests
Run the unit test suite to verify the state machine behavior (automatic rain cycle, button override, override timeout, and dock events):
```bash
python test_rain_shield.py
```

### 4. Raspberry Pi Autostart Service
Install the `rain_shield.service` file to run the controller automatically at boot:
```bash
sudo cp rain_shield.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rain_shield.service
sudo systemctl start rain_shield.service
```

## Project Structure

- `streamlit_app.py` - The main Streamlit web application with live robot telemetry.
- `rain_shield.py` - The automatic hardware/servo controller for Raspberry Pi 4.
- `webhook_receiver.py` - FastAPI webhook endpoint logging status to the SQLite database.
- `test_rain_shield.py` - Offline simulation test suite for the controller logic.
- `rain_shield.service` - systemd daemon config for automated Raspberry Pi startup.
- `translations_complete.py` - Contains the translation dictionaries for 22 Indian languages.
- `requirements.txt` - Project dependencies.
- `plastic_mapping.py` - Logic for mapping model outputs to specific plastic recycling guidelines.
- `README.md` - Project documentation.

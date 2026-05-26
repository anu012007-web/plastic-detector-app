# EcoScanIndia: Indian Plastic Waste Dataset Documentation

## 1. Dataset Overview
- **Dataset Name**: EcoScanIndia Custom Waste Dataset v1.0
- **Total Images**: [To be filled after collection, e.g., 2,500]
- **Number of Classes**: 11 (6 Plastic, 5 Non-Plastic)
- **Primary Objective**: Fine-tune classification models (MobileNetV2, YOLOv8) specifically for the unique visual characteristics of waste found in Indian urban and rural environments.
- **Collection Date Range**: [Insert Dates]

## 2. Methodology
### 2.1 Data Collection
Images were collected programmatically using `bing-image-downloader` to simulate diverse, real-world search results reflecting Indian street and household contexts. Search queries were localized to Indian cities (e.g., Delhi, Mumbai, Bangalore, Chennai) and specific contexts ("Swachh Bharat", "street garbage", "recycling center").

### 2.2 Data Annotation
A custom Streamlit application (`annotation_app.py`) was developed for rapid, human-verified annotation. Each image was manually inspected and moved into its respective ground-truth class directory. Corrupt, irrelevant, or highly ambiguous images were discarded.

### 2.3 Dataset Split Strategy
The dataset is split using PyTorch's `random_split` functionality to ensure a balanced distribution across the training pipeline:
- **Train (70%)**: Used for calculating loss and updating model weights.
- **Validation (15%)**: Used for hyperparameter tuning and model checkpointing.
- **Test (15%)**: Held out completely during training for final unbiased evaluation reporting.

## 3. Class Distribution
| Category | Class | Total Images | Train | Val | Test |
|----------|-------|--------------|-------|-----|------|
| **Plastic** | Bottle | [0] | [0] | [0] | [0] |
| **Plastic** | Cap | [0] | [0] | [0] | [0] |
| **Plastic** | Bag | [0] | [0] | [0] | [0] |
| **Plastic** | Cup | [0] | [0] | [0] | [0] |
| **Plastic** | Straw | [0] | [0] | [0] | [0] |
| **Plastic** | Cutlery | [0] | [0] | [0] | [0] |
| **Plastic** | Container | [0] | [0] | [0] | [0] |
| **Plastic** | Wrapper | [0] | [0] | [0] | [0] |
| **Non-Plastic** | Glass | [0] | [0] | [0] | [0] |
| **Non-Plastic** | Paper | [0] | [0] | [0] | [0] |
| **Non-Plastic** | Metal | [0] | [0] | [0] | [0] |
| **Non-Plastic** | Organic | [0] | [0] | [0] | [0] |
| **Non-Plastic** | Cardboard| [0] | [0] | [0] | [0] |

## 4. Environmental and Contextual Variances
This dataset specifically captures the extreme variance seen in Indian waste management scenarios:

- **Lighting Conditions**: High variance, ranging from extreme direct sunlight (over-exposure on reflective plastics) to low-light monsoon/evening street conditions.
- **Backgrounds**: Highly cluttered. Waste is rarely isolated; it is often embedded in soil, mixed with organic waste, or partially obscured by street infrastructure.
- **Degradation**: Includes images of physically degraded, crushed, or dirt-covered plastic items (e.g., mud-covered bottles, torn polythene bags).
- **Device Used**: Simulated varied capture devices. Because the data is scraped from web sources (news articles, NGOs, community reports), the visual fidelity mimics a wide array of mobile phone cameras, matching the target deployment platform for the EcoScanIndia app.

## 5. Potential Biases
- **Urban Skew**: Web scraped images heavily favor urban centers (Delhi, Mumbai) where news reporting on waste is more frequent, potentially under-representing rural waste visual characteristics.
- **Visibility Bias**: Extremely small plastics (microplastics, tiny wrapper shreds) are likely under-represented compared to large, highly visible items (bottles, bags).

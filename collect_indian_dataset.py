import os
import time
import requests
import base64
import glob
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# Force UTF-8 encoding for Windows terminals to display emojis correctly
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass # Fallback for older python versions if any

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
OUTPUT_DIR = "raw_data"

CATEGORIES = {
    "plastic_bottles": {
        "queries": ["plastic water bottle India", "empty plastic bottle street"],
        "limit": 500
    },
    "bottle_caps": {
        "queries": ["plastic bottle cap", "cola bottle cap waste"],
        "limit": 250
    },
    "plastic_bags": {
        "queries": ["plastic carry bag India", "plastic bag garbage"],
        "limit": 400
    },
    "plastic_cups": {
        "queries": ["plastic cup waste", "tea stall plastic cup"],
        "limit": 300
    },
    "plastic_straws": {
        "queries": ["plastic straw garbage", "straw waste India"],
        "limit": 250
    },
    "plastic_cutlery": {
        "queries": ["plastic spoon waste", "plastic fork garbage"],
        "limit": 200
    },
    "plastic_containers": {
        "queries": ["plastic container waste", "takeout container"],
        "limit": 250
    },
    "plastic_wrappers": {
        "queries": ["plastic wrapper garbage", "chip packet waste"],
        "limit": 300
    }
}

def download_image(src, save_path):
    try:
        if src.startswith('http'):
            response = requests.get(src, timeout=5)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        elif src.startswith('data:image'):
            # Extract base64 part
            header, encoded = src.split(',', 1)
            data = base64.b64decode(encoded)
            with open(save_path, 'wb') as f:
                f.write(data)
            return True
    except Exception as e:
        pass
    return False

def main():
    print("=" * 50)
    print("🚀 EcoScanIndia: Selenium Image Scraper")
    print("=" * 50)
    
    print("Initializing Selenium WebDriver...")
    # Modern Selenium 4 automatically handles chromedriver downloads!
    chrome_options = Options()
    # chrome_options.add_argument("--headless") # Uncomment to hide the browser window
    chrome_options.add_argument("--log-level=3") # Suppress unnecessary warnings
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
    except Exception as e:
        print(f"Error starting Chrome: {e}")
        print("Please ensure Google Chrome is installed on your computer.")
        return
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for category, config in CATEGORIES.items():
        print(f"\n=========================================")
        print(f"📂 Category: {category} (Target: {config['limit']} images)")
        print(f"=========================================")
        
        folder_path = os.path.join(OUTPUT_DIR, category)
        os.makedirs(folder_path, exist_ok=True)
        
        # Count existing images in the folder to allow resuming
        existing_files = glob.glob(os.path.join(folder_path, "img_*.jpg"))
        downloaded = len(existing_files)
        if downloaded >= config['limit']:
            print(f"Category '{category}' already has {downloaded} images. Skipping.")
            continue
            
        print(f"Found {downloaded} existing images. Resuming download...")
        
        for query in config["queries"]:
            if downloaded >= config["limit"]:
                break
                
            print(f"\n=> Query: '{query}'")
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=isch"
            driver.get(search_url)
            time.sleep(2) # Wait for initial page load
            
            scrolls = 0
            max_scrolls = 30 # Scroll more if needed to reach higher targets
            
            while downloaded < config["limit"] and scrolls < max_scrolls:
                # Scroll down
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2) # Wait for images to load
                
                # Try to click "Show more results" button if it appears
                try:
                    button = driver.find_element(By.CSS_SELECTOR, "input[type='button'][value='Show more results'], input.mye4qd")
                    if button.is_displayed():
                        button.click()
                        time.sleep(2)
                except Exception:
                    pass
                
                # Find all image tags
                images = driver.find_elements(By.TAG_NAME, "img")
                
                for img in images:
                    if downloaded >= config["limit"]:
                        break
                        
                    try:
                        src = img.get_attribute('src')
                        if not src:
                            src = img.get_attribute('data-src')
                            
                        if src and (src.startswith('http') or src.startswith('data:image')):
                            # Skip Google logos and tiny tracking pixels
                            if "logo" in src.lower() or "favicon" in src.lower() or len(src) < 100:
                                continue
                                
                            save_path = os.path.join(folder_path, f"img_{downloaded+1}.jpg")
                            
                            # Only download if we haven't already saved this sequence
                            if not os.path.exists(save_path):
                                if download_image(src, save_path):
                                    downloaded += 1
                                    print(f"  Downloaded {downloaded}/{config['limit']}", end='\r')
                    except Exception:
                        continue
                scrolls += 1
                
            print(f"\nFinished query '{query}'. Category progress: {downloaded}/{config['limit']}")

    driver.quit()
    print("\n" + "=" * 50)
    print("✅ Collection Complete!")
    print(f"All images have been downloaded to the '{OUTPUT_DIR}' directory.")
    print("Next step: Run `streamlit run annotation_app.py` to annotate these images.")

if __name__ == "__main__":
    main()

import os
import requests
import time
from requests.exceptions import RequestException

NUM_IMAGES = 1000
IMG_SIZE = 128
FOLDER = "data/images"
MAX_RETRIES = 3
RETRY_DELAY = 5  

os.makedirs(FOLDER, exist_ok=True)

def download_images():
    for i in range(1, NUM_IMAGES + 1):
        url = f"https://picsum.photos/{IMG_SIZE}/{IMG_SIZE}?random={i}"
        retries = 0
        
        while retries < MAX_RETRIES:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()  
                
                with open(f"{FOLDER}/image_{i}.jpg", "wb") as img_file:
                    img_file.write(response.content)
                print(f"Downloaded {i}/{NUM_IMAGES}")
                break  
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 503:
                    print(f"503 Service Unavailable for image {i}, retry {retries + 1}/{MAX_RETRIES}")
                    retries += 1
                    if retries < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                    continue
                else:
                    print(f"HTTP Error {e.response.status_code} for image {i}: {str(e)}")
                    break
                    
            except RequestException as e:
                print(f"Error downloading image {i}: {str(e)}")
                retries += 1
                if retries < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                continue
                
        else:
            print(f"Failed to download image {i} after {MAX_RETRIES} attempts")

download_images()
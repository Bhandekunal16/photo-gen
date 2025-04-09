from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image_dir = "./data/images"
caption_file = "./data/captions.txt"

with open(caption_file, 'w') as f:
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            raw_image = Image.open(image_path).convert("RGB")
            
            inputs = processor(raw_image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            f.write(f"{filename}|{caption}\n")
            print(f"{filename} → {caption}")

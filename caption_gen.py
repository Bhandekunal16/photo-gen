from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

MODEL_NAME = "Salesforce/blip-image-captioning-base"

processor = BlipProcessor.from_pretrained(MODEL_NAME, use_fast=True)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()

image_dir = "./data/image60px"
caption_file = "./data/captions.txt"

with open(caption_file, "w", encoding="utf-8") as f:

    for filename in sorted(os.listdir(image_dir)):

        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(image_dir, filename)

        try:
            raw_image = Image.open(image_path).convert("RGB")

            inputs = processor(raw_image, return_tensors="pt")

            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=30, num_beams=5, repetition_penalty=1.1
                )

            caption = processor.decode(output[0], skip_special_tokens=True).strip()

            caption = " ".join(caption.split())

            f.write(f"{filename}|{caption}\n")

            print(f"{filename} → {caption}")

        except Exception as e:
            print(f"ERROR: {filename} → {e}")

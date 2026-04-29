import os
import cv2
import numpy as np
from PIL import Image, ImageOps

def preprocess_image(image_path, output_path, size=(384, 384)):
    """
    Standardizes image size using padding to preserve aspect ratio,
    then resizes for ViT ingestion.
    """
    try:
        with Image.open(image_path) as img:
            # 1. Convert to RGB (handles grayscale or CMYK sources)
            img = ImageOps.exif_transpose(img) # Add this
            img = img.convert('RGB')
            
            # 2. Pad to Square (Letterboxing)
            # This adds black borders to the top/bottom or sides
            w, h = img.size
            max_wh = max(w, h)
            hp = (max_wh - w) // 2
            vp = (max_wh - h) // 2
            padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
            img = ImageOps.expand(img, padding, fill=(0, 0, 0))
            
            # 3. Resize to 384x384
            # Using LANCZOS (high-quality downsampling) to preserve artifacts
            img = img.resize(size, Image.Resampling.LANCZOS)
            
            # 4. Save to target directory
            img.save(output_path, quality=95)
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def batch_process(input_root, output_root):
    # Classes: 0 (Real), 1 (Deepfake), 2 (Synthetic)
    categories = ['class_0', 'class_1', 'class_2']
    
    for cat in categories:
        in_dir = os.path.join(input_root, cat)
        out_dir = os.path.join(output_root, cat)
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"Processing {cat}...")
        for filename in os.listdir(in_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                preprocess_image(
                    os.path.join(in_dir, filename),
                    os.path.join(out_dir, filename)
                )

# Example usage:
batch_process('/home/bam20007/ai-augmented-images/data/raw_data', '/home/bam20007/ai-augmented-images/preprocessed_data_v2')
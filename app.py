import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from model_v6_vit_base.vit_model import build_vit_classifier  # Ensure this path is correct
import torchvision.transforms as transforms

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="ViT", layout="wide")
st.title("Deepfake & AI-Generated Image Detector")
st.markdown("---")

@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vit_classifier()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model, device

# 2. SIDEBAR SETTINGS
st.sidebar.header("Calibration")
default_path = "/home/bam20007/ai-augmented-images/checkpoints/final_plesae_work/best_vit_model.pth"
model_path = st.sidebar.text_input("Model Checkpoint", default_path)
bias = st.sidebar.slider("Authentic Bias (Safety Threshold)", 0.0, 2.0, 0.75, 0.05)

# 3. INITIALIZE MODEL
try:
    model, device = load_model(model_path)
    st.sidebar.success(f"Model loaded on {device}!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# 4. FILE UPLOADER
uploaded_file = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert('RGB')
    fixed_img = ImageOps.exif_transpose(raw_img)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(fixed_img, caption="Original Upload", use_container_width=True)

    # 5. FORENSIC PREPROCESSING (MATCHING DATASET.PY)
    # We resize larger then center crop to 384 to strip black bars
    inference_transform = transforms.Compose([
        transforms.Resize(600),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Matching display transform for the heatmap overlay
    display_transform = transforms.Compose([
        transforms.Resize(600),
        transforms.CenterCrop(384)
    ])
    
    cropped_img_for_display = display_transform(fixed_img)
    img_tensor = inference_transform(fixed_img).unsqueeze(0).to(device)

    # 6. INFERENCE
    with torch.no_grad():
        outputs = model(img_tensor, output_attentions=True)
        logits = outputs.logits.clone()
        # Apply the decision threshold bias
        logits[:, 0] += bias
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = torch.argmax(logits, dim=-1).item()

    class_names = ["Authentic", "Deepfake", "AI-Generated"]
    
    # 7. DISPLAY RESULTS
    with col2:
        st.subheader(f"Analysis Result: **{class_names[pred_idx]}**")
        
        # Probability Visuals
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {probs[i]*100:.1f}%")
            st.progress(float(probs[i]))

        st.markdown("---")
        

st.sidebar.markdown("---")
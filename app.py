import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from model_v5_vit_base.vit_model import build_vit_classifier  # Ensure this path is correct
import torchvision.transforms as transforms

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Forensic ViT Detector", layout="wide")
st.title("🛡️ ViT Deepfake & AI-Gen Detector")
st.markdown("---")

@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vit_classifier()
    # map_location ensures we can run on CPU/Login nodes or GPUs seamlessly
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model, device

# 2. SIDEBAR SETTINGS
st.sidebar.header("Calibration")
# Updated to your Step 8 path
default_path = "/home/bam20007/ai-augmented-images/checkpoints/vit_base_16_384_aggressive_regularization/best_vit_model.pth"
model_path = st.sidebar.text_input("Model Checkpoint", default_path)
# Recommended starting bias based on your Eval logs
bias = st.sidebar.slider("Authentic Bias (Safety Threshold)", 0.0, 2.0, 0.75, 0.05)

# 3. INITIALIZE MODEL
try:
    model, device = load_model(model_path)
    st.sidebar.success(f"Model loaded on {device}!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# 4. FILE UPLOADER
uploaded_file = st.file_uploader("Upload an image for forensic analysis...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process Image Orientation (Critical for phone photos)
    raw_img = Image.open(uploaded_file).convert('RGB')
    fixed_img = ImageOps.exif_transpose(raw_img)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(fixed_img, caption="Original Upload", use_container_width=True)

    # 5. FORENSIC PREPROCESSING (MATCHING DATASET.PY)
    # We resize larger then center crop to 384 to strip black bars
    inference_transform = transforms.Compose([
        transforms.Resize(600), # Zoom in much further
        transforms.CenterCrop(384), # This will definitely clear the bars
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Matching display transform for the heatmap overlay
    display_transform = transforms.Compose([
        transforms.Resize(600), # Zoom in much further
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
        
        # 8. ATTENTION HEATMAP (384px ALIGNED)
        st.write("### Forensic Attention Map (Cropped View)")
        st.info("Heatmap is focused on the 384x384 central area to ignore background shortcuts.")
        
        attentions = outputs.attentions[-1]
        # Average heads and reshape to 24x24 (Standard for ViT-Base-384)
        att_map = attentions.mean(dim=1).squeeze(0)[0, 1:].reshape(24, 24).cpu().numpy()
        
        # Normalize and resize to match the 384px cropped display image
        att_resized = cv2.resize(att_map / np.max(att_map), (384, 384))
        
        # Generate Heatmap Color Overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * att_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend Heatmap with the Cropped Face (Linear Blending for clarity)
        cropped_np = np.array(cropped_img_for_display)
        overlay = cv2.addWeighted(cropped_np, 0.6, heatmap_rgb, 0.4, 0)
        
        st.image(overlay, caption="ViT Forensic Focus Area", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("SDP 12: Bear Detection & Forensic Systems")
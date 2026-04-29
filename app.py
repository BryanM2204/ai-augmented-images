import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from model_v3.vit_model import build_vit_classifier
import torchvision.transforms as transforms

# Page Config
st.set_page_config(page_title="ViT Detector", layout="wide")
st.title("Deepfake & AI-Gen Detector")

@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vit_classifier()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model, device

# Sidebar Settings
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model Path", "/home/bam20007/ai-augmented-images/checkpoints/vit_base_16_384_aggressive_regularization/best_vit_model.pth")
bias = st.sidebar.slider("Authentic Bias", 0.0, 2.0, 0.0, 0.1)

# 1. Load Model
try:
    model, device = load_model(model_path)
    st.sidebar.success("Model Loaded!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# 2. File Uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process Image
    raw_img = Image.open(uploaded_file).convert('RGB')
    fixed_img = ImageOps.exif_transpose(raw_img)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(fixed_img, caption="Uploaded Image", use_container_width=True)

    # 3. Inference logic
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(fixed_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor, output_attentions=True)
        logits = outputs.logits.clone()
        logits[:, 0] += bias
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = torch.argmax(logits, dim=-1).item()

    class_names = ["Authentic", "Deepfake", "AI-Generated"]
    
    # 4. Display Results
    with col2:
        st.subheader(f"Prediction: **{class_names[pred_idx]}**")
        
        # Confidence Bars
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {probs[i]*100:.1f}%")
            st.progress(float(probs[i]))

        # 5. Attention Map Overlay
        st.write("### Forensic Attention Map")
        attentions = outputs.attentions[-1]
        att_map = attentions.mean(dim=1).squeeze(0)[0, 1:].reshape(24, 24).cpu().numpy()
        att_resized = cv2.resize(att_map / np.max(att_map), (fixed_img.size[0], fixed_img.size[1]))
        
        heatmap = cv2.applyColorMap(np.uint8(255 * att_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Optional: Smoother blend
        overlay = cv2.addWeighted(np.uint8(fixed_img), 0.6, heatmap_rgb, 0.4, 0)
        st.image(overlay, caption="Forensic Attention Heatmap", use_container_width=True)
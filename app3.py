import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from model_v2.vit_model import build_vit_classifier
import torchvision.transforms as transforms

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Forensic ViT Detector (224px)", layout="wide")
st.title("🛡️ ViT Deepfake & AI-Gen Detector (224px)")

@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Note: Ensure build_vit_classifier() is configured for 224px
    model = build_vit_classifier() 
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model, device

# 2. SIDEBAR SETTINGS
st.sidebar.header("Settings")
# Update this path to your older 224px checkpoint
default_224_path = "/home/bam20007/ai-augmented-images/checkpoints/model_v2_checkpoint/best_vit_model.pth"
model_path = st.sidebar.text_input("Model Path", default_224_path)
bias = st.sidebar.slider("Authentic Bias", 0.0, 2.0, 0.75, 0.1)

# 3. INITIALIZE MODEL
try:
    model, device = load_model(model_path)
    st.sidebar.success("Model Loaded!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# 4. FILE UPLOADER
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert('RGB')
    fixed_img = ImageOps.exif_transpose(raw_img)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(fixed_img, caption="Uploaded Image", use_container_width=True)

    # 5. UPDATED INFERENCE LOGIC (224px)
    # To maintain the "No Bars" forensic strategy: Resize 256 -> CenterCrop 224
    inference_transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    display_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    
    cropped_img_for_display = display_transform(fixed_img)
    img_tensor = inference_transform(fixed_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor, output_attentions=True)
        logits = outputs.logits.clone()
        logits[:, 0] += bias
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = torch.argmax(logits, dim=-1).item()

    class_names = ["Authentic", "Deepfake", "AI-Generated"]
    
    # 6. DISPLAY RESULTS
    with col2:
        st.subheader(f"Prediction: **{class_names[pred_idx]}**")
        
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {probs[i]*100:.1f}%")
            st.progress(float(probs[i]))

        st.markdown("---")

        # 7. UPDATED ATTENTION GRID (14x14)
        st.write("### Forensic Attention Map (224px View)")
        attentions = outputs.attentions[-1]
        
        # 14x14 grid is standard for 224px models
        att_map = attentions.mean(dim=1).squeeze(0)[0, 1:].reshape(14, 14).cpu().numpy()
        
        # Resize to match the 224px display crop
        att_resized = cv2.resize(att_map / np.max(att_map), (224, 224))
        
        heatmap = cv2.applyColorMap(np.uint8(255 * att_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        cropped_np = np.array(cropped_img_for_display)
        overlay = cv2.addWeighted(cropped_np, 0.6, heatmap_rgb, 0.4, 0)
        
        st.image(overlay, caption="ViT Attention (224px Aligned)", use_container_width=True)
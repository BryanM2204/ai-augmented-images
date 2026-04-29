import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os

# Internal import from your project structure
from model_v3.vit_model import build_vit_classifier

def run_inference(image_path, model_path, bias=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Authentic", "Deepfake", "AI-Generated"]

    # 1. Load Model
    model = build_vit_classifier()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    # 2. Load & Fix Rotation
    # This is the most important fix for phone photos
    raw_img = Image.open(image_path).convert('RGB')
    fixed_img = ImageOps.exif_transpose(raw_img) 

    # 3. Preprocessing (No Cropping - Full Frame)
    # We use Resize((224, 224)) to match your training pipeline
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(fixed_img).unsqueeze(0).to(device)

    # 4. Predict with Bias
    with torch.no_grad():
        outputs = model(img_tensor, output_attentions=True)
        logits = outputs.logits.clone()
        logits[:, 0] += bias # Our balanced 1.0 bias
        
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = torch.argmax(logits, dim=-1).item()

    # 5. Attention Map Extraction
    attentions = outputs.attentions[-1] 
    att_map = attentions.mean(dim=1).squeeze(0)[0, 1:].reshape(24, 24).cpu().numpy()    
    # Resize to match the display size
    att_resized = cv2.resize(att_map / np.max(att_map), (fixed_img.size[0], fixed_img.size[1]))
    
    # 6. Heatmap Generation (Fixing Color Space)
    heatmap = cv2.applyColorMap(np.uint8(255 * att_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # Corrects the blue/red inversion
    
    # Overlay logic
    overlay = (np.float32(heatmap_rgb) / 255) * 0.4 + (np.float32(fixed_img) / 255)
    overlay = overlay / np.max(overlay)

    # 7. Final Visualization
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(fixed_img)
    plt.title(f"Full Frame Input\n{os.path.basename(image_path)}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    
    # Confidence breakdown
    res_str = f"PREDICTION: {class_names[pred_idx]}\n"
    res_str += f"Auth: {probs[0]*100:.1f}% | Fake: {probs[1]*100:.1f}% | AI: {probs[2]*100:.1f}%"
    plt.title(res_str, fontsize=10, fontweight='bold')
    plt.axis('off')

    plt.tight_layout()
    save_path = f"full_frame_{os.path.basename(image_path)}"
    plt.savefig(save_path)
    print(f"Prediction: {class_names[pred_idx]} ({probs[pred_idx]*100:.1f}%)")
    plt.show()

if __name__ == "__main__":
    IMG_PATH = "/home/bam20007/ai-augmented-images/image.png" 
    MODEL_PATH = "/home/bam20007/ai-augmented-images/checkpoints/vit_base_16_384_aggressive_regularization/best_vit_model.pth"
    run_inference(IMG_PATH, MODEL_PATH, bias=0.0)
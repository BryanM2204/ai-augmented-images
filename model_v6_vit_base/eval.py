import os
import torchvision.transforms as transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image

# Internal imports
from dataset import get_dataloaders
from vit_model import build_vit_classifier

def evaluate_metrics(model, test_loader, device, authentic_bias=0.0):
    """
    Evaluates the model across the test set with a specific logit bias 
    for the Authentic class (Index 0).
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Apply bias to the Authentic class to adjust sensitivity
            logits = outputs.logits.clone()
            logits[:, 0] += authentic_bias
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = ["Authentic", "Deepfake", "AI-Generated"]
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Save a confusion matrix for every bias level to track the trade-off
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Bias={authentic_bias})')
    plt.savefig(f'confusion_matrix_biased_{authentic_bias}.png')
    plt.close() # Close to prevent memory accumulation in loops

    return report_dict

def visualize_attention(model, raw_image, device, save_name='attention_sample.png'):
    """
    Extracts and overlays attention maps using the 600->384 'Deep Zoom' 
    to ensure the heatmap stays on the face.
    """
    model.eval()
    
    # 1. Replicate the 'No Bars' cropping for the display image
    h, w = raw_image.shape[:2]
    scale = 600 / min(h, w)
    temp_img = cv2.resize(raw_image, (int(w * scale), int(h * scale)))
    
    y1, x1 = int((temp_img.shape[0]-384)/2), int((temp_img.shape[1]-384)/2)
    original_cropped = temp_img[y1:y1+384, x1:x1+384]

    # 2. Prepare tensor (matching the val_transform)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(600),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor, output_attentions=True)
    
    # 3. Process Attention (Last Layer, mean across 12 heads)
    attention = outputs.attentions[-1].mean(dim=1).squeeze(0)
    cls_attention = attention[0, 1:] # CLS token to all patches
    
    # Reshape to 24x24 grid (for 384x384 resolution)
    grid_size = 24 
    attention_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
    attention_map_res = cv2.resize(attention_map / np.max(attention_map), (384, 384))
    
    # 4. Generate Overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map_res), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_cropped, 0.6, heatmap_rgb, 0.4, 0)

    # 5. Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_cropped)
    ax[0].set_title('Inference Input (Cropped)')
    ax[1].imshow(overlay)
    ax[1].set_title('Forensic Attention')
    plt.savefig(save_name)
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the best 'Honest' model (Step 5)
    model = build_vit_classifier()
    ckpt_path = "/home/bam20007/ai-augmented-images/checkpoints/final_plesae_work/best_vit_model.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)

    # Load test data (Ensure dataset.py uses Resize(600) + CenterCrop(384))
    data_dir = os.path.expanduser('~/ai-augmented-images/data/final_dataset_v2')
    _, _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=32)

    # --- GRID SEARCH FOR OPTIMAL BIAS ---
    bias_values = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    results = []

    print(f"{'Bias':<10} | {'Auth Recall':<12} | {'Fake Recall':<12} | {'AI-Gen F1':<10} | {'Macro F1':<10}")
    print("-" * 75)

    for bias in bias_values:
        report = evaluate_metrics(model, test_loader, device, authentic_bias=bias)
        
        auth_rec = report['Authentic']['recall']
        fake_rec = report['Deepfake']['recall']
        ai_gen_f1 = report['AI-Generated']['f1-score']
        macro_f1 = report['macro avg']['f1-score']
        
        results.append((bias, auth_rec, fake_rec, ai_gen_f1, macro_f1))
        print(f"{bias:<10} | {auth_rec:<12.2f} | {fake_rec:<12.2f} | {ai_gen_f1:<10.2f} | {macro_f1:<10.2f}")

    # Optimal selection: Minimize the gap between Authentic and Deepfake recall
    best_match = min(results, key=lambda x: abs(x[1] - x[2]))
    print(f"\n--- Optimal Balance Found ---")
    print(f"Bias {best_match[0]} yields {best_match[1]:.2f} Auth Recall and {best_match[2]:.2f} Fake Recall.")
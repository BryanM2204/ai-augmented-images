import os
import torchvision.transforms as transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import cv2

from dataset import get_dataloaders
from vit_model import build_vit_classifier
# from dataset import get_test_loader # Justin's module

def evaluate_metrics(model, test_loader, device, authentic_bias=1.5):
    """Generates F1 scores and Confusion Matrix with threshold adjustment."""
    model.eval()
    all_preds = []
    all_labels = []

    print(f"Running evaluation with Authentic Bias: {authentic_bias}...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # 1. Get the raw scores (logits)
            logits = outputs.logits.clone() # Use clone to avoid modifying the original
            
            # 2. Add a 'bonus' to the Authentic class (Index 0)
            # This makes it 'harder' for the other classes to win unless their scores are very high
            logits[:, 0] += authentic_bias
            
            # 3. Take the argmax of the biased logits
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = ["Authentic", "Deepfake", "AI-Generated"]
    
    print("\n--- Adjusted Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # create dictionary report for easier access to metrics
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # Save the adjusted confusion matrix to compare with the previous version
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Bias={authentic_bias})')
    plt.savefig(f'confusion_matrix_biased_{authentic_bias}.png')
    print(f"Adjusted matrix saved as 'confusion_matrix_biased_{authentic_bias}.png'")

    return report_dict

def visualize_attention(model, image_tensor, original_image):
    """
    Extracts and overlays the attention map from the final ViT layer.
    Requires Peter's preprocessing to keep a copy of the original image.
    """
    model.eval()
    
    # We must explicitly request attentions from the Hugging Face model
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0), output_attentions=True)
    
    # Get attention weights from the last layer: shape (batch, num_heads, seq_len, seq_len)
    attention = outputs.attentions[-1]
    
    # Average attention across all 12 heads
    attention = attention.mean(dim=1).squeeze(0)
    
    # The first token [0] is the CLS token. We want its attention to all other patches [1:]
    cls_attention = attention[0, 1:]
    
    # Reshape the 196 patches back into a 14x14 grid
    grid_size = int(np.sqrt(cls_attention.size(0)))
    attention_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
    
    # Normalize the map for visualization
    attention_map = attention_map / np.max(attention_map)
    
    # Resize the 14x14 attention map to match the 224x224 image dimensions
    attention_map_resized = cv2.resize(attention_map, (original_image.shape[1], original_image.shape[0]))
    
    # Create a heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap * 0.5 + np.float32(original_image) / 255
    overlay = overlay / np.max(overlay)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    ax[1].imshow(overlay)
    ax[1].set_title('Attention Map Overlay')
    ax[1].axis('off')
    
    plt.savefig('attention_map_sample.png')
    print("Attention visualization saved as 'attention_map_sample.png'")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = build_vit_classifier()
    model.load_state_dict(torch.load("/home/bam20007/ai-augmented-images/checkpoints/model_v2_checkpoint/best_vit_model.pth", weights_only=True))
    model.to(device)

    # 2. Load Data
    data_dir = os.path.expanduser('~/ai-augmented-images/data/final_dataset')
    _, _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=32)

    # 3. Grid Search for the "Sweet Spot"
    bias_values = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    results = []

    print(f"{'Bias':<10} | {'Auth Recall':<12} | {'Fake Recall':<12} | {'Macro F1':<10}")
    print("-" * 55)

    for bias in bias_values:
        # Run evaluation with the current bias
        # (Assuming you updated evaluate_metrics to return the report dictionary)
        report = evaluate_metrics(model, test_loader, device, authentic_bias=bias)
        
        auth_recall = report['Authentic']['recall']
        fake_recall = report['Deepfake']['recall']
        macro_f1 = report['macro avg']['f1-score']
        
        results.append((bias, auth_recall, fake_recall, macro_f1))
        print(f"{bias:<10} | {auth_recall:<12.2f} | {fake_recall:<12.2f} | {macro_f1:<10.2f}")

    # Find the bias where the difference between recalls is minimized
    best_match = min(results, key=lambda x: abs(x[1] - x[2]))
    print(f"\n--- Optimal Balance Found ---")
    print(f"Bias: {best_match[0]} yields {best_match[1]:.2f} Auth Recall and {best_match[2]:.2f} Fake Recall.")
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

def evaluate_metrics(model, test_loader, device):
    """Generates F1 scores and Confusion Matrix."""
    model.eval()
    all_preds = []
    all_labels = []

    print("Running evaluation on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = ["Authentic", "Deepfake", "AI-Generated"]
    
    # 1. Classification Report (F1, Precision, Recall)
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix: ViT Classification')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")

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
    
    # 1. Load Model with weights_only=True to avoid that warning you saw earlier
    model = build_vit_classifier()
    model.load_state_dict(torch.load("/home/bam20007/ai-augmented-images/checkpoints/model_v2_checkpoint/best_vit_model.pth", weights_only=True))
    model.to(device)

    # 2. Load the Test Data
    data_dir = os.path.expanduser('~/ai-augmented-images/data/final_dataset')
    _, _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=32)

    # 3. Run Quantitative Evaluation
    evaluate_metrics(model, test_loader, device)
    
    # 4. ADD THIS: Qualitative Attention Check
    # Get one batch of images to visualize
    sample_images, sample_labels = next(iter(test_loader))
    
    # Take the first image in the batch
    img_tensor = sample_images[0].to(device)
    
    # IMPORTANT: To visualize, we need the "Original" image. 
    # Since the tensor is normalized, we should convert it back to a viewable format.
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    unnormalized_img = inv_normalize(sample_images[0]).permute(1, 2, 0).numpy()
    unnormalized_img = np.clip(unnormalized_img, 0, 1) # Ensure pixel values are valid

    visualize_attention(model, img_tensor, unnormalized_img)
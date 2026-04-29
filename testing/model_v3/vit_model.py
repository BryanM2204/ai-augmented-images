import torch
from transformers import ViTForImageClassification, ViTConfig

def build_vit_classifier(model_ckpt="google/vit-large-patch16-384"):
    """
    Initializes a pre-trained Vision Transformer for 3-class classification.
    
    Classes based on the project proposal:
    0: Authentic (Real images from Defactify / Celeb-DF v2)
    1: Deepfake (Manipulated frames from Celeb-DF v2)
    2: AI-Generated (Fully synthesized from DALL-E 3, Midjourney v6, SD3)
    """
    
    # Define the class mappings as outlined in your proposal
    id2label = {
        0: "Authentic",
        1: "Deepfake",
        2: "AI-Generated"
    }
    label2id = {v: k for k, v in id2label.items()}

    # Load the pre-trained ViT model
    # ignore_mismatched_sizes=True is crucial: it drops the original 1000-node 
    # classification head and initializes a new 3-node head.
    model = ViTForImageClassification.from_pretrained(
        model_ckpt,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        output_attentions=True
    )

    return model
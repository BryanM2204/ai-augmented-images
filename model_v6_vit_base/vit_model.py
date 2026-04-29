from transformers import ViTForImageClassification, ViTConfig

def build_vit_classifier(model_ckpt="google/vit-base-patch16-384"):
    id2label = {0: "Authentic", 1: "Deepfake", 2: "AI-Generated"}
    label2id = {v: k for k, v in id2label.items()}

    # Load config and increase dropout to 0.2
    config = ViTConfig.from_pretrained(model_ckpt)
    config.hidden_dropout_prob = 0.2      # Standard is 0.1
    config.attention_probs_dropout_prob = 0.2 
    config.num_labels = 3
    config.id2label = id2label
    config.label2id = label2id
    config.output_attentions = True

    model = ViTForImageClassification.from_pretrained(
        model_ckpt,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return model
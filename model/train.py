import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os

# Import the model builder you created
from model import build_vit_classifier

# Placeholder for Justin's dataloader implementation
# from dataset import get_dataloaders 

def train_model(epochs=10, batch_size=32, learning_rate=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Initialize Model
    model = build_vit_classifier()
    model.to(device)

    # 2. Load Data (Coordinate with Justin on this import)
    # train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    
    # --- Dummy Loaders for testing the loop before data is ready ---
    # Remove these when dataset.py is integrated
    train_loader = [(torch.randn(batch_size, 3, 224, 224), torch.randint(0, 3, (batch_size,))) for _ in range(10)]
    val_loader = [(torch.randn(batch_size, 3, 224, 224), torch.randint(0, 3, (batch_size,))) for _ in range(5)]
    # -------------------------------------------------------------

    # 3. Optimizer and Loss Function
    # AdamW is highly recommended for Vision Transformers
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Learning Rate Scheduler (Linear Warmup)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1), # 10% of training is warmup
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')
    os.makedirs("saved_models", exist_ok=True)

    # 4. Training Loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        total_train_loss = 0

        # Progress bar for training
        train_bar = tqdm(train_loader, desc="Training")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_bar.set_postfix(loss=(total_train_loss / (train_bar.n + 1)))

        # 5. Validation Loop
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validating")
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs.logits, labels)
                total_val_loss += loss.item()

                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct / total

        print(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

        # 6. Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "saved_models/best_vit_model.pth")
            print(">>> Saved new best model!")

if __name__ == "__main__":
    train_model()
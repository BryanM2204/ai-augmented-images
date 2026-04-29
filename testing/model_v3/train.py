import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pandas as pd
import time

# Internal imports from your team's files
from vit_model import build_vit_classifier
from dataset import get_dataloaders

def save_checkpoint(model, optimizer, scheduler, history, epoch, step, filename="saved_models/checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history
    }
    torch.save(checkpoint, filename)
    pd.DataFrame(history).to_csv("saved_models/training_log.csv", index=False)

def load_checkpoint(model, optimizer, scheduler, filename="saved_models/checkpoint.pth"):
    if os.path.exists(filename):
        print(f"!!! Found checkpoint at {filename}. Resuming...")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['step'], checkpoint.get('history', {
            'step': [], 'train_loss': [], 'val_loss': [], 'val_acc': []
        })
    return 0, 0, {'step': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}

def validate(model, val_loader, criterion, device):
    model.eval()
    total_val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs.logits, labels)
            total_val_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_val_loss / len(val_loader)
    accuracy = correct / total
    model.train()
    return avg_loss, accuracy

def train_model(epochs=10, batch_size=16, learning_rate=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on: {device}")

    # 1. Initialize Architecture & Checkpointing
    model = build_vit_classifier()
    model.gradient_checkpointing_enable() # Lowers VRAM "waterline"
    model.to(device)

    # 2. Accumulation Configuration
    # Effective Batch Size = batch_size * accumulation_steps (16 * 4 = 64)
    accumulation_steps = 4 

    data_dir = os.path.expanduser('~/ai-augmented-images/data/final_dataset_v2')
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=data_dir, 
        batch_size=batch_size, 
        num_workers=16
    )

    # 3. Optimizer & Schedulers
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    
    # IMPORTANT: The scheduler should step based on actual optimizer updates,
    # so we divide total steps by accumulation_steps.
    total_optimizer_steps = (len(train_loader) // accumulation_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_optimizer_steps * 0.1), 
        num_training_steps=total_optimizer_steps
    )

    os.makedirs("saved_models", exist_ok=True)
    start_epoch, start_step, history = load_checkpoint(model, optimizer, scheduler)
    best_val_loss = float('inf') if not history['val_loss'] else min(history['val_loss'])
    running_train_loss = 0.0
    patience = 3
    epochs_no_improve = 0

    # 5. Training Loop
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad() # Initialize once at start of accumulation cycle
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, (images, labels) in enumerate(train_bar):
            if epoch == start_epoch and step < start_step:
                continue

            images, labels = images.to(device), labels.to(device)
            
            # --- FORWARD PASS (Scaled for Accumulation) ---
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                # Divide loss so the average gradient is correct after accumulation
                loss = criterion(outputs.logits, labels) / accumulation_steps
            
            # --- BACKWARD PASS (Grads are summed on .backward()) ---
            scaler.scale(loss).backward()

            # --- OPTIMIZER STEP (Every N steps) ---
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # Clear gradients for the next cycle
                scheduler.step()

            running_train_loss += loss.item() * accumulation_steps
            train_bar.set_postfix(loss=(running_train_loss / (step + 1)))

        # 6. End-of-Epoch Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        avg_train_loss = running_train_loss / len(train_loader)
        
        history['step'].append(epoch + 1) 
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        running_train_loss = 0.0
        save_checkpoint(model, optimizer, scheduler, history, epoch, 0)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "saved_models/best_vit_model.pth")
            print(f"\n[Epoch {epoch+1}] New Best Val Loss: {val_loss:.4f}!")
        else:
            epochs_no_improve += 1
            print(f"\n[Epoch {epoch+1}] Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            break
        start_step = 0

    print("\nTraining Complete!")

if __name__ == "__main__":
    train_model()
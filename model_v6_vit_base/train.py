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

def save_checkpoint(model, optimizer, scheduler, history, epoch, step, filename="vit_base_16_384_aggressive_regularization/checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history
    }
    torch.save(checkpoint, filename)
    pd.DataFrame(history).to_csv("vit_base_16_384_aggressive_regularization/training_log.csv", index=False)

def load_checkpoint(model, optimizer, scheduler, filename="vit_base_16_384_aggressive_regularization/checkpoint.pth"):
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

def train_model(epochs=10, batch_size=32, learning_rate=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on: {device}")

    # 1. Initialize Architecture & Checkpointing
    model = build_vit_classifier()
    model.to(device)

    # 2. Accumulation Configuration
    # Effective Batch Size = batch_size * accumulation_steps (32 * 8 = 256)

    data_dir = os.path.expanduser('~/ai-augmented-images/data/final_dataset_v2')
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=data_dir, 
        batch_size=batch_size, 
        num_workers=16
    )

    # 3. Optimizer & Schedulers
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    scaler = torch.amp.GradScaler('cuda')
    
    # Scheduler steps every batch now
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.2), 
        num_training_steps=total_steps
    )

    os.makedirs("vit_base_16_384_aggressive_regularization", exist_ok=True)
    start_epoch, start_step, history = load_checkpoint(model, optimizer, scheduler)
    best_val_loss = float('inf') if not history['val_loss'] else min(history['val_loss'])
    running_train_loss = 0.0
    patience = 3
    epochs_no_improve = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        running_train_loss = 0.0
        
        for step, (images, labels) in enumerate(train_bar):
            if epoch == start_epoch and step < start_step:
                continue

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs.logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_train_loss += loss.item()
            train_bar.set_postfix(loss=(running_train_loss / (step + 1)))

        # 4. End-of-Epoch Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        avg_train_loss = running_train_loss / len(train_loader)
        
        history['step'].append(epoch + 1) 
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        save_checkpoint(model, optimizer, scheduler, history, epoch + 1, 0)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Fixed pathing to use SAVE_DIR
            torch.save(model.state_dict(), f"vit_base_16_384_aggressive_regularization/best_vit_model.pth")
            print(f"\n[Epoch {epoch+1}] New Best Val Loss: {val_loss:.4f}!")
        else:
            epochs_no_improve += 1
            print(f"\n[Epoch {epoch+1}] No improvement. Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"\nEarly Stopping triggered at Epoch {epoch + 1}.")
            break

    print("\nTraining Complete! Ready for Streamlit visualization.")

if __name__ == "__main__":
    train_model()
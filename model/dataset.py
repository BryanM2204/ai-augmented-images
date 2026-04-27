import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    # Standard ImageNet normalization for ViT
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Training: Add slight forensic augmentations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # Consider adding transforms.RandomHorizontalFlip() for robustness
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Validation/Test: No augmentations, just normalization
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
    test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
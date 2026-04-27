import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32, num_workers=16):
    """
    Standardizes data loading with robust forensic augmentations 
    to prevent the model from 'cheating' via background logos.
    """
    
    # 1. Training: Robust augmentations to fight overfitting
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # Scrambles the logo position so it's not a stationary feature
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # Prevents bias toward specific newsroom color grading
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        # Simulates video compression artifacts common in deepfakes
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Validation/Test: No augmentations, just the standard ViT normalization
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load Datasets
    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
    test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_transform)

    # 4. Wrap in Dataloaders
    # Note: pin_memory=True is critical for the A100 to stay fed
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32, num_workers=16):
    # 1. Training: Force the model to learn from small facial patches
    train_transform = transforms.Compose([
        # Scale (0.3, 1.0) forces the model to look at parts of the face, not the whole frame
        transforms.RandomResizedCrop((384, 384), scale=(0.3, 1.0)), 
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Validation/Test: Strip the "cheating" bars with CenterCrop
    val_transform = transforms.Compose([
        transforms.Resize(448), # Resize so the short side is 448
        transforms.CenterCrop(384), # Crop the middle 384 to remove pillar/letterboxing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
    test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_transform)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    )
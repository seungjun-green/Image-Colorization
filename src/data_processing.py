import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.color import rgb2lab

class CocoColorizationDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.image_dir, self.image_files[idx])

        # Load the RGB image
        img = Image.open(file_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)

        # Convert to LAB color space
        img_np = img.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        img_lab = rgb2lab(img_np).astype('float32')

        # Normalize LAB channels
        L = img_lab[:, :, 0] / 50.0 - 1.0  # Normalize L to [-1, 1]
        AB = img_lab[:, :, 1:] / 128.0     # Normalize AB to [-1, 1]

        # Convert to tensors
        L = torch.tensor(L, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        AB = torch.tensor(AB, dtype=torch.float32).permute(2, 0, 1)  # (2, H, W)

        return L, AB

def get_dataloaders(train_dir, val_dir, batch_size=16, num_workers=8):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to 256x256
        transforms.ToTensor()           # Convert the image to a PyTorch tensor
    ])

    # Create train and validation datasets
    train_dataset = CocoColorizationDataset(image_dir=train_dir, transform=transform)
    val_dataset = CocoColorizationDataset(image_dir=val_dir, transform=transform)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.color import rgb2lab

class CocoColorizationDataset(Dataset):
    def __init__(self, image_dir, transform=None, grayscale_threshold=5, dark_threshold=30):
        self.image_dir = image_dir
        self.transform = transform
        self.grayscale_threshold = grayscale_threshold
        self.dark_threshold = dark_threshold
        self.image_files = []
        
        self.num_gray = 0
        self.num_dark = 0

        # Filter images during initialization
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                file_path = os.path.join(image_dir, filename)
                try:
                    with Image.open(file_path) as img:
                        img = img.convert('RGB')
                        
                        # skip grayscale images
                        if self._is_grayscale(img):
                            self.num_gray += 1
                            continue
                            
                        # skip dark images
                        if self._is_too_dark(img):
                            self.num_dark += 1
                            continue
                            
                        self.image_files.append(filename)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    
        print(f"Skippped: gray: {self.num_gray} | dark: {self.num_dark}")

    def _is_grayscale(self, img):
        small_img = img.resize((64, 64))
        r, g, b = small_img.split()
        
        r = np.array(r, dtype=np.float32)
        g = np.array(g, dtype=np.float32)
        b = np.array(b, dtype=np.float32)

        rg_diff = np.mean(np.abs(r - g))
        gb_diff = np.mean(np.abs(g - b))
        avg_diff = (rg_diff + gb_diff) / 2.0
        
        return avg_diff < self.grayscale_threshold

    def _is_too_dark(self, img):
        small_img = img.resize((64, 64))
        gray_img = small_img.convert('L')
        avg_luminance = np.array(gray_img).mean()
        
        return avg_luminance < self.dark_threshold

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(file_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # convert to LAB color space
        img_np = img.permute(1, 2, 0).numpy()
        img_lab = rgb2lab(img_np).astype('float32')

        # normalize and convert to tensors
        L = torch.tensor(img_lab[:, :, 0] / 50.0 - 1.0, dtype=torch.float32).unsqueeze(0)
        AB = torch.tensor(img_lab[:, :, 1:] / 128.0, dtype=torch.float32).permute(2, 0, 1)

        return L, AB

def get_dataloaders(train_dir, val_dir, batch_size=16, num_workers=8):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_dataset = CocoColorizationDataset(train_dir, transform=transform)
    val_dataset = CocoColorizationDataset(val_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
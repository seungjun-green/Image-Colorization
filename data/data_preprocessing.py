import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.color import rgb2lab
from tqdm import tqdm

class CocoColorizationDataset(Dataset):
    def __init__(self, image_dir, transform=None, grayscale_threshold=5, yellow_threshold=25):
        self.image_dir = image_dir
        self.transform = transform
        self.grayscale_threshold = grayscale_threshold
        self.yellow_threshold = yellow_threshold
        self.image_files = []
        
        self.num_gray = 0
        self.num_yellow = 0
        
        self.bad_images = [] # combined gray images and safia images

        # Filter images during initialization
        for filename in tqdm(os.listdir(image_dir), desc="Filtering images"):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                file_path = os.path.join(image_dir, filename)
                try:
                    with Image.open(file_path) as img:
                        img = img.convert('RGB')
                        
                        # skip grayscale images
                        if self._is_grayscale(img):
                            self.num_gray += 1
                            # self.bad_images.append(filename)
                            continue
                        
                        # skip yellow images
                        if self._has_yellow_tint(img, self.yellow_threshold):
                            self.num_yellow += 1
                            self.bad_images.append(filename)
                            continue
                            
                        self.image_files.append(filename)
                        
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                        
        print(f"Skippped: gray: {self.num_gray}")
        print(f"Skippped: yellow: {self.num_yellow}")
        print(self.bad_images)
        
        # if the file is provided just use that ones, if not do the thing and save a the file

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
    
    def _has_yellow_tint(self, img, yellow_threshold):
        small_img = img.resize((64, 64))
        img_np = np.array(small_img).astype('float32') / 255.0
        img_lab = rgb2lab(img_np)
        b_channel_avg = np.mean(img_lab[:, :, 2])
        return b_channel_avg > yellow_threshold

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

def get_dataloaders(train_dir, val_dir, batch_size=16, num_workers=2):
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


def get_val_dataloader(val_dir, batch_size=16, num_workers=8):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    val_dataset = CocoColorizationDataset(val_dir, transform=transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return val_loader
    
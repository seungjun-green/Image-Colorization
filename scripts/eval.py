import json
import warnings
import torch
import torch.nn.functional as F

import numpy as np
from utils.utils import lab_to_rgb
from data.data_preprocessing import get_val_dataloader
from models import *
from utils.model_utils import *

def log_eval(generator, lpips_model, val_loader, num_batches, device):
    ''' func to do eval during training
    '''
    generator.eval()
    total_lpips = 0.0
    
    with torch.no_grad(): 
        for L, AB in val_loader:
            L = L.to(device)
            AB = AB.to(device)
            
            fake_AB = generator(L)
            fake_img = torch.cat((L, fake_AB), dim=1)
            
            real_img = torch.cat((L, AB), dim=1)
            
            fake_L, fake_AB = fake_img[:, :1, :, :], fake_img[:, 1:, :, :]
            real_L, real_AB = real_img[:, :1, :, :], real_img[:, 1:, :, :]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fake_rgb = lab_to_rgb(fake_L, fake_AB)
                real_rgb = lab_to_rgb(real_L, real_AB)
            
            fake_rgb = np.array(fake_rgb)
            real_rgb = np.array(real_rgb)
            
            fake_rgb_tensor = torch.from_numpy(fake_rgb).to(device).float() 
            real_rgb_tensor = torch.from_numpy(real_rgb).to(device).float() 
            
            if device=="cuda":
                fake_rgb_tensor = fake_rgb_tensor / 255.0
                real_rgb_tensor = real_rgb_tensor / 255.0

            
            fake_rgb_tensor = (fake_rgb_tensor * 2) - 1
            real_rgb_tensor = (real_rgb_tensor * 2) - 1


            batch_lpips = lpips_model(fake_rgb_tensor, real_rgb_tensor).mean().item()
            total_lpips += batch_lpips
            
    return total_lpips / num_batches
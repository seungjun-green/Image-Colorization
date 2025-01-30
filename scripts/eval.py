import json
import warnings
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np
from utils.utils import lab_to_rgb
from data.data_preprocessing import get_val_dataloader
from models import *
from utils.model_utils import *

def calculate_psnr(predicted, ground_truth):
    mse = F.mse_loss(predicted, ground_truth)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(predicted, ground_truth):
    predicted = predicted.permute(0, 2, 3, 1).cpu().numpy()
    ground_truth = ground_truth.permute(0, 2, 3, 1).cpu().numpy()

    ssim_values = []
    for i in range(predicted.shape[0]):
        ssim_val = ssim(
            ground_truth[i],
            predicted[i],
            multichannel=True,
            data_range=1.0,
            win_size=3
        )
        ssim_values.append(ssim_val)

    return np.mean(ssim_values)



def log_eval(generator, val_loader, num_batches, device):
    ''' func to do eval during training
    '''
    generator.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    
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

            batch_psnr = calculate_psnr(fake_rgb, real_rgb)
            total_psnr += batch_psnr

            batch_ssim = calculate_ssim(fake_rgb, real_rgb)
            total_ssim += batch_ssim
            

    return {
        'psnr': total_psnr / num_batches,
        'ssim': total_ssim / num_batches,
    }

def eval_model(config, model_path, device, **kwargs):
    ''' func to eval model from checkpoint
    '''
    with open(config, 'r') as f:
        config = json.load(f)
        
    if 'train_dir' in kwargs:
        config['train_dir'] = kwargs['train_dir']
    if 'val_dir' in kwargs:
        config['val_dir'] = kwargs['val_dir']
        
    generator = load_generator(config['gen_type']).to(device)
    
    if device == "cuda" and torch.cuda.is_available():
        generator.to(device)
        map_location = None 
    else:
        device = "cpu" 
        generator.to(device) 
        map_location = torch.device('cpu')

    generator.load_state_dict(torch.load(model_path, map_location=map_location))

    generator.eval() 
    total_psnr = 0.0
    total_ssim = 0.0
    
    val_loader = get_val_dataloader(
        val_dir=config['val_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    num_batches = len(val_loader)

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

            batch_psnr = calculate_psnr(fake_rgb, real_rgb)
            total_psnr += batch_psnr

            batch_ssim = calculate_ssim(fake_rgb, real_rgb)
            total_ssim += batch_ssim

    return {
        'psnr': total_psnr / num_batches,
        'ssim': total_ssim / num_batches,
    }
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np
from src.utils import lab_to_rgb

def calculate_psnr(predicted, ground_truth):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    Args:
        predicted (torch.Tensor): Predicted image, shape (N, C, H, W).
        ground_truth (torch.Tensor): Ground truth image, shape (N, C, H, W).
    Returns:
        float: PSNR value.
    """
    mse = F.mse_loss(predicted, ground_truth)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(predicted, ground_truth):
    """
    Calculate Structural Similarity Index (SSIM).
    Args:
        predicted (torch.Tensor): Predicted image, shape (N, C, H, W).
        ground_truth (torch.Tensor): Ground truth image, shape (N, C, H, W).
    Returns:
        float: Average SSIM value across all samples.
    """
    predicted = predicted.permute(0, 2, 3, 1).cpu().numpy()  # Convert to (N, H, W, C)
    ground_truth = ground_truth.permute(0, 2, 3, 1).cpu().numpy()

    ssim_values = []
    for i in range(predicted.shape[0]):
        ssim_val = ssim(ground_truth[i], predicted[i], multichannel=True, data_range=1.0)
        ssim_values.append(ssim_val)

    return np.mean(ssim_values)


def eval_model(generator, val_loader, device='cuda'):
    """
    Evaluate the generator on the validation dataset using PSNR and SSIM.
    Args:
        generator (torch.nn.Module): The trained generator model.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (str): Device to run the evaluation on ('cuda' or 'cpu').
    Returns:
        dict: Average PSNR and SSIM scores across the validation set.
    """
    generator.eval()  # Set model to evaluation mode
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():  # Disable gradient calculations for evaluation
        for L, AB in val_loader:
            # Move data to the correct device
            L = L.to(device)
            AB = AB.to(device)

            # Generate predicted AB channels
            fake_AB = generator(L)

            # Concatenate L and AB channels to form full LAB images
            fake_img = torch.cat((L, fake_AB), dim=1)  # Predicted image
            real_img = torch.cat((L, AB), dim=1)      # Ground truth image

            # Convert LAB images to RGB for evaluation
            fake_rgb = lab_to_rgb(fake_img)
            real_rgb = lab_to_rgb(real_img)

            # Compute PSNR for the current batch
            batch_psnr = calculate_psnr(fake_rgb, real_rgb)
            total_psnr += batch_psnr

            # Compute SSIM for the current batch
            batch_ssim = calculate_ssim(fake_rgb, real_rgb)
            total_ssim += batch_ssim

    # Return the average PSNR and SSIM scores
    return {
        'psnr': total_psnr / num_batches,
        'ssim': total_ssim / num_batches,
    }
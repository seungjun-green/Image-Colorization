import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb

def initialize_weights(model):
    """
    Initializes the weights of the model using Xavier initialization for Conv2D and ConvTranspose2D layers.
    BatchNorm2D weights are set to 1 and biases to 0.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

            
class LabToRGB2(nn.Module):
    def __init__(self):
        super(LabToRGB2, self).__init__()
        # Define the transformation matrices as buffers to ensure they're moved with the model
        matrix_Lab_to_XYZ = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ]).float()
        self.register_buffer('matrix_Lab_to_XYZ', matrix_Lab_to_XYZ)

        matrix_XYZ_to_RGB = torch.tensor([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ]).float()
        self.register_buffer('matrix_XYZ_to_RGB', matrix_XYZ_to_RGB)

        # Define the reference white point D65
        self.register_buffer('ref_X', torch.tensor(95.047).float())
        self.register_buffer('ref_Y', torch.tensor(100.000).float())
        self.register_buffer('ref_Z', torch.tensor(108.883).float())

    def forward(self, L, AB):
        # Denormalize L and AB from [-1, 1] to [0, 100] and [-127, 127]
        L_denorm = (L + 1) * 50.0      # Maps [-1, 1] to [0, 100]
        AB_denorm = AB * 127.0         # Maps [-1, 1] to [-127, 127]
        AB_denorm = torch.clamp(AB_denorm, -128.0, 127.0)

        # Separate A and B channels
        a, b = torch.chunk(AB_denorm, 2, dim=1)

        # Normalize L for conversion
        Y = (L_denorm + 16.0) / 116.0
        X = Y + (a / 500.0)
        Z = Y - (b / 200.0)

        # Apply the nonlinear transformation
        epsilon = 0.008856

        X = torch.where(X ** 3 > epsilon, X ** 3, (X - 16.0 / 116.0) / 7.787)
        Y = torch.where(Y ** 3 > epsilon, Y ** 3, (Y - 16.0 / 116.0) / 7.787)
        Z = torch.where(Z ** 3 > epsilon, Z ** 3, (Z - 16.0 / 116.0) / 7.787)

        # Scale by the reference white point
        X = X * self.ref_X
        Y = Y * self.ref_Y
        Z = Z * self.ref_Z

        # Stack XYZ channels
        XYZ = torch.cat([X, Y, Z], dim=1)

        # Normalize XYZ by scaling down by 1/100
        XYZ = XYZ / 100.0

        # Reshape for matrix multiplication
        N, C, H, W = XYZ.shape
        XYZ = XYZ.view(N, 3, -1)

        # Matrix multiplication: XYZ to RGB
        RGB = torch.matmul(self.matrix_XYZ_to_RGB, XYZ)

        # Reshape back to (N, 3, H, W)
        RGB = RGB.view(N, 3, H, W)

        # Clamp the values to [0, 1]
        RGB = torch.clamp(RGB, 0.0, 1.0)

        # Check for NaNs or Infs
        if torch.isnan(RGB).any() or torch.isinf(RGB).any():
            raise ValueError("NaN or Inf detected in RGB output of LabToRGB.")

        return RGB

def lab_to_rgb(L, AB):
    """
    Converts normalized LAB tensors to RGB.
    Args:
        L (torch.Tensor): L channel tensor normalized to [-1, 1], shape (N, 1, H, W).
        AB (torch.Tensor): AB channels tensor normalized to [-1, 1], shape (N, 2, H, W).
    Returns:
        torch.Tensor: RGB image tensor with values in [0, 1], shape (N, 3, H, W).
    """
    # Denormalize L and AB to original LAB range
    L = (L + 1.0) * 50.0  # L channel: [-1, 1] -> [0, 100]
    AB = AB * 128.0       # AB channels: [-1, 1] -> [-128, 128]

    # Concatenate L and AB to form LAB image
    LAB = torch.cat([L, AB], dim=1)  # Shape: (N, 3, H, W)

    # Convert to NumPy and reshape for skimage.lab2rgb
    LAB_np = LAB.permute(0, 2, 3, 1).cpu().detach().numpy()  # (N, H, W, 3)

    # Convert LAB to RGB
    RGB_np = np.stack([lab2rgb(lab) for lab in LAB_np], axis=0)  # (N, H, W, 3)

    # Convert back to torch.Tensor and reorder dimensions
    RGB = torch.from_numpy(RGB_np).permute(0, 3, 1, 2)  # (N, 3, H, W)

    return RGB

def show_examples(generator, example_loader, device='cuda'):
    """
    Displays examples of colorization results.
    Args:
        generator (nn.Module): Trained generator model.
        example_loader (DataLoader): DataLoader for validation examples.
        device (str): Device to perform computations on.
    """
    generator.eval()
    fig, axes = plt.subplots(5, 3, figsize=(12, 20))  # Increased figure size for clarity
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    with torch.no_grad():
        for idx, (L, AB) in enumerate(example_loader):
            if idx >= 5:
                break

            # Move tensors to the correct device
            L = L.to(device)
            AB = AB.to(device)

            fake_AB = generator(L)
            real_rgb = lab_to_rgb(L, AB)
            fake_rgb = lab_to_rgb(L, fake_AB)

            L_np = (L[0, 0].cpu().detach().numpy() + 1) * 50.0
            real_rgb_np = real_rgb[0].permute(1, 2, 0).cpu().detach().numpy()
            fake_rgb_np = fake_rgb[0].permute(1, 2, 0).cpu().detach().numpy()

            # Plot L channel
            axes[idx, 0].imshow(L_np, cmap="gray", vmin=0, vmax=100)
            axes[idx, 0].axis("off")
            if idx == 0:
                axes[idx, 0].set_title("L Channel (Grayscale)")

            # Plot Original Color
            axes[idx, 1].imshow(real_rgb_np)
            axes[idx, 1].axis("off")
            if idx == 0:
                axes[idx, 1].set_title("Original Color (L + AB)")

            # Plot Predicted Color
            axes[idx, 2].imshow(fake_rgb_np)
            axes[idx, 2].axis("off")
            if idx == 0:
                axes[idx, 2].set_title("Predicted Color (L + ^AB)")

    plt.show()
    plt.close(fig)

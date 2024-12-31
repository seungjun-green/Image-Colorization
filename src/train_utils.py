import torch
import torch.nn as nn

# Loss Functions
bce_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

def get_disc_loss(fake_output, real_output):
    """
    Computes the discriminator loss.
    Args:
        fake_output (Tensor): Discriminator output for fake images.
        real_output (Tensor): Discriminator output for real images.
    Returns:
        Tensor: Total loss for the discriminator.
    """
    fake_loss = bce_loss(fake_output, torch.zeros_like(fake_output))
    real_loss = bce_loss(real_output, torch.ones_like(real_output))
    total_loss = fake_loss + real_loss
    return total_loss


def get_gen_loss(fake_output, fake_image, real_image, lambda_l1=100):
    """
    Computes the generator loss.
    Args:
        fake_output (Tensor): Discriminator output for fake images.
        fake_image (Tensor): Generated fake image.
        real_image (Tensor): Real image.
        lambda_l1 (float): Weight for L1 loss.
    Returns:
        Tensor: Total loss for the generator.
    """
    adv_loss = bce_loss(fake_output, torch.ones_like(fake_output))
    l1_recon_loss = l1_loss(fake_image, real_image) * lambda_l1
    total_loss = adv_loss + l1_recon_loss
    return total_loss


def get_optimizers(gen, disc, lr=2e-4, beta1=0.5, beta2=0.999):
    """
    Returns optimizers for the generator and discriminator.
    Args:
        gen (nn.Module): Generator model.
        disc (nn.Module): Discriminator model.
        lr (float): Learning rate for optimizers.
        beta1 (float): Beta1 hyperparameter for Adam.
        beta2 (float): Beta2 hyperparameter for Adam.
    Returns:
        Tuple[torch.optim.Adam, torch.optim.Adam]: Generator and discriminator optimizers.
    """
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta1, beta2))
    return gen_optimizer, disc_optimizer

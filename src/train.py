import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings

from src.models import UNetGenerator, PatchGANDiscriminator
from src.utils import lab_to_rgb, show_examples
from src.data_processing import get_dataloaders

def initialize_weights(model):
    """
    Initializes model weights using Xavier Normal Initialization (as per pix2pix paper).
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)  # Gamma = 1
            nn.init.constant_(m.bias, 0)    # Beta = 0

def train_model(config):
    """
    Main training loop.
    Args:
        config (dict): Configuration dictionary containing hyperparameters and paths.
    """
    device = config['device']

    # Initialize models dynamically
    generator = config['generator']().to(device)
    discriminator = config['discriminator']().to(device)

    # Initialize weights
    if 'initialize_weights' in config and config['initialize_weights']:
        initialize_weights(generator)
        initialize_weights(discriminator)

    # Define loss functions
    gen_loss_fn = config['gen_loss_fn']
    disc_loss_fn = config['disc_loss_fn']
    criterion = nn.BCEWithLogitsLoss()

    # Define optimizers
    gen_optimizer = config['gen_optimizer'](generator.parameters())
    disc_optimizer = config['disc_optimizer'](discriminator.parameters())
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        train_dir=config['train_dir'],
        val_dir=config['val_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Suppress specific warnings
    warnings.filterwarnings("ignore", message="Conversion from CIE-LAB, via XYZ to sRGB color space resulted in")

    # Training loop
    for epoch in range(config['epochs']):
        generator.train()
        discriminator.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch_idx, (L, AB) in progress_bar:
            L = L.to(device)
            AB = AB.to(device)

            # Train Discriminator
            discriminator.zero_grad()
            fake_AB = generator(L)
            real_output = discriminator(L, AB)
            fake_output = discriminator(L, fake_AB.detach())
            disc_loss = disc_loss_fn(fake_output, real_output, criterion)  # Pass the criterion
            disc_loss.backward()
            disc_optimizer.step()

            # Train Generator
            generator.zero_grad()
            fake_output = discriminator(L, fake_AB)
            gen_loss = gen_loss_fn(fake_output, fake_AB, AB, criterion, lambda_l1=config['lambda_l1'])  # Pass extra params
            gen_loss.backward()
            gen_optimizer.step()

            progress_bar.set_postfix(
                D_Loss=f"{disc_loss.item():.4f}",
                G_Loss=f"{gen_loss.item():.4f}"
            )

            # Show examples periodically
            if batch_idx % config['show_interval'] == 0 and batch_idx != 0:
                example_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=1, shuffle=True, num_workers=4)
                show_examples(generator, example_loader, device=device)

                # Save the trained models
                torch.save(generator.state_dict(), config['generator_path'])
                torch.save(discriminator.state_dict(), config['discriminator_path'])
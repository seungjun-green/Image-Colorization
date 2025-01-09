import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
import os

from utils.utils import lab_to_rgb, show_examples
from data.data_preprocessing import get_dataloaders

def initialize_weights(model):
    """ Initializes model weights using Xavier Normal Initialization (as per pix2pix paper). """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)  # Gamma = 1
            nn.init.constant_(m.bias, 0)    # Beta = 0

def train_model(config):
    device = config['device']
    global_min = config['glb_min']

    # initalize models
    generator = config['generator']().to(device)
    discriminator = config['discriminator']().to(device)

    # Initialize weights
    if config['initialize_weights']:
        initialize_weights(generator)
        initialize_weights(discriminator)

    # Define loss functions
    gen_loss_fn = config['gen_loss_fn']
    disc_loss_fn = config['disc_loss_fn']

    # Define optimizers
    gen_optimizer = config['gen_optimizer'](generator.parameters())
    disc_optimizer = config['disc_optimizer'](discriminator.parameters())
    
    # 
    train_loader, val_loader = get_dataloaders(
        train_dir=config['train_dir'],
        val_dir=config['val_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    total_batch = len(train_loader)
    
    warnings.filterwarnings("ignore", message="Conversion from CIE-LAB, via XYZ to sRGB color space resulted in")

    for epoch in range(config['epochs']):
        generator.train()
        discriminator.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch_idx, (L, AB) in progress_bar:
            L = L.to(device)
            AB = AB.to(device)

            ## training disc ##
            discriminator.zero_grad()
            fake_AB = generator(L)
            real_output = discriminator(L, AB)
            fake_output = discriminator(L, fake_AB.detach())
            disc_loss = disc_loss_fn(fake_output, real_output)
            disc_loss.backward()
            disc_optimizer.step()

            ## training gen ##
            generator.zero_grad()
            fake_output = discriminator(L, fake_AB)
            gen_loss = gen_loss_fn(fake_output, fake_AB, AB, lambda_l1=config['lambda_l1'])
            gen_loss.backward()
            gen_optimizer.step()

            progress_bar.set_postfix(
                D_Loss=f"{disc_loss.item():.4f}",
                G_Loss=f"{gen_loss.item():.4f}"
            )

            # show examples
            if (batch_idx % config['show_interval'] == 0 or batch_idx == total_batch-1) and batch_idx != 0:
                example_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=1, shuffle=True, num_workers=4)
                show_examples(generator, example_loader, device=device)

                # checkpoint
                torch.save(generator.state_dict(), os.path.join('checkpoints/gen', f"generator_epoch{epoch}_batch{batch_idx}.pth"))
                torch.save(discriminator.state_dict(), os.path.join('checkpoints/disc', f"discriminator_epoch{epoch}_batch{batch_idx}.pth"))
            
            if gen_loss.item() < global_min:
                # save the model checkpoint whenver the loss reaches a new minimum.
                torch.save(generator.state_dict(), config['generator_path'])
                torch.save(discriminator.state_dict(), config['discriminator_path'])
                global_min = gen_loss.item()
                print(f"New best model saved with (gen)loss: {global_min:.4f}")
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
import os
from utils.utils import lab_to_rgb, show_examples
from data.data_preprocessing import get_dataloaders
from utils.train_utils import *  
from utils.model_utils import initialize_weights
import json

def load_config(json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config


def train_model(gen, disc, config_file):
    config = load_config(config_file)
    
    device = config['device']
    global_min = config['glb_min']
    
    # initalize models
    gen = gen.to(device)
    disc = disc.to(device)

    # weight initalization
    if config['initialize_weights']:
        initialize_weights(gen)
        initialize_weights(disc)

    # Define loss functions
    gen_loss_fn = get_gen_loss()
    disc_loss_fn = get_disc_loss()

    # Define optimizers
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=config.gen_lr, betas=(config.beta1, config.beta2))
    disc_optimizer = torch.optim.Adam(gen.parameters(), lr=config.dic_lr, betas=(config.beta1, config.beta2))
    
    # create the train data loader
    train_loader, val_loader = get_dataloaders(
        train_dir=config['train_dir'],
        val_dir=config['val_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    total_batch = len(train_loader)
    
    warnings.filterwarnings("ignore", message="Conversion from CIE-LAB, via XYZ to sRGB color space resulted in")

    for epoch in range(config['epochs']):
        gen.train()
        disc.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch_idx, (L, AB) in progress_bar:
            L = L.to(device)
            AB = AB.to(device)

            ## training disc ##
            disc.zero_grad()
            fake_AB = gen(L)
            real_output = disc(L, AB)
            fake_output = disc(L, fake_AB.detach())
            disc_loss = disc_loss_fn(fake_output, real_output)
            disc_loss.backward()
            disc_optimizer.step()

            ## training gen ##
            gen.zero_grad()
            fake_output = disc(L, fake_AB)
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
                show_examples(gen, example_loader, device=device)

                # checkpoint
                torch.save(gen.state_dict(), os.path.join('checkpoints/gen', f"gen_epoch{epoch}_batch{batch_idx}.pth"))
                torch.save(disc.state_dict(), os.path.join('checkpoints/disc', f"disc_epoch{epoch}_batch{batch_idx}.pth"))
            
            if gen_loss.item() < global_min:
                # save the model checkpoint whenver the loss reaches a new minimum.
                torch.save(gen.state_dict(), config['generator_path'])
                torch.save(disc.state_dict(), config['discriminator_path'])
                global_min = gen_loss.item()
                print(f"New best model saved with (gen)loss: {global_min:.4f}")
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
import os
import json
from utils.utils import lab_to_rgb, show_examples
from data.data_preprocessing import get_dataloaders
from utils.train_utils import get_gen_loss, get_disc_loss
from utils.model_utils import *

class ImageColorizationTrainer:
    def __init__(self, config_file, **kwargs):
        """
        Initializes the Image Colorization Trainer.

        Args:
            generator_cls (class): Class for the generator model.
            discriminator_cls (class): Class for the discriminator model.
            config_file (str): Path to the configuration JSON file.
        """
        
        os.makedirs('checkpoints/gen', exist_ok=True)
        os.makedirs('checkpoints/disc', exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_file)
        self.device = self.config['device']
        self.global_min = self.config['glb_min']
        self.training_ratio = self.config['training_ratio']
        self.gen_type = self.config['gen_type']
        self.disc_type = self.config['disc_type']
        
        self.train_dir = kwargs.get('train_dir', self.config.get('train_dir', ''))
        self.val_dir = kwargs.get('val_dir', self.config.get('val_dir', ''))

        # Initialize models
        self.generator = load_generator(self.gen_type).to(self.device)
        self.discriminator = load_discriminator(self.disc_type).to(self.device)

        # Initialize weights if specified
        if self.config['initialize_weights']:
            initialize_weights(self.generator)
            initialize_weights(self.discriminator)

        # Define optimizers
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.config['gen_lr'],
            betas=(self.config['beta1'], self.config['beta2']),
        )
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config['dic_lr'],
            betas=(self.config['beta1'], self.config['beta2']),
        )

        # Initialize data loaders
        self.train_loader, self.val_loader = get_dataloaders(
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
        )

    @staticmethod
    def _load_config(json_path):
        """Loads the configuration JSON file."""
        with open(json_path, 'r') as f:
            return json.load(f)

    def train(self):
        """Main training loop."""
        total_batches = len(self.train_loader)
        

        warnings.filterwarnings(
            "ignore",
            message="Conversion from CIE-LAB, via XYZ to sRGB color space resulted in",
        )

        for epoch in range(self.config['epochs']):
            self.generator.train()
            self.discriminator.train()

            progress_bar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch {epoch + 1}/{self.config['epochs']}",
            )

            for batch_idx, (L, AB) in progress_bar:
                L = L.to(self.device)
                AB = AB.to(self.device)

                # Train discriminator
                self.discriminator.zero_grad()
                fake_AB = self.generator(L)
                real_output = self.discriminator(L, AB)
                fake_output = self.discriminator(L, fake_AB.detach())
                disc_loss = get_disc_loss(fake_output, real_output)
                disc_loss.backward()
                self.disc_optimizer.step()

                # Train generator
                for _ in range(self.training_ratio):
                    self.generator.zero_grad()
                    fake_output = self.discriminator(L, fake_AB)
                    gen_loss = get_gen_loss(fake_output, fake_AB, AB, lambda_l1=self.config['lambda_l1'])
                    gen_loss.backward()
                    self.gen_optimizer.step()

                progress_bar.set_postfix(
                    D_Loss=f"{disc_loss.item():.4f}",
                    G_Loss=f"{gen_loss.item():.4f}",
                )

                # Show examples and save checkpoints
                if (batch_idx % self.config['show_interval'] == 0 or batch_idx == total_batches - 1) and batch_idx != 0:
                    self._show_and_save_examples(gen_loss.item(), disc_loss.item(), epoch, batch_idx)

                # Save the best model
                if gen_loss.item() < self.global_min:
                    self._save_best_model(gen_loss.item(), disc_loss.item(), epoch, batch_idx)

    def _show_and_save_examples(self, gen_loss, disc_loss, epoch, batch_idx):
        """Displays and saves example results."""
        example_loader = torch.utils.data.DataLoader(self.val_loader.dataset, batch_size=1, shuffle=True, num_workers=2)
        show_examples(self.generator, example_loader, device=self.device)
        torch.save(self.generator.state_dict(), f"{self.config['generator_path']}/epoch{epoch}_batch{batch_idx}_{gen_loss}.pth")
        torch.save(self.discriminator.state_dict(), f"{self.config['discriminator_path']}/epoch{epoch}_batch{batch_idx}_{disc_loss}.pth")

    def _save_best_model(self, gen_loss, disc_loss, epoch, batch_idx):
        """Saves the best generator and discriminator models."""
        torch.save(self.generator.state_dict(), f"{self.config['generator_path']}/BEST_GEN_epoch{epoch}_batch{batch_idx}_{gen_loss}.pth")
        torch.save(self.discriminator.state_dict(), f"{self.config['discriminator_path']}/BEST_DISC_epoch{epoch}_batch{batch_idx}_{disc_loss}.pth")
        self.global_min = gen_loss
        print(f"New best model saved with (gen) loss: {self.global_min:.4f}")
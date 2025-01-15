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
    def __init__(self, config_file):
        """
        Initializes the Image Colorization Trainer.

        Args:
            generator_cls (class): Class for the generator model.
            discriminator_cls (class): Class for the discriminator model.
            config_file (str): Path to the configuration JSON file.
        """
        # Load configuration
        self.config = self._load_config(config_file)
        self.device = self.config['device']
        self.global_min = self.config['glb_min']

        # Initialize models
        self.generator = load_generator(self.config.gen_type).to(self.device)
        self.discriminator = load_discriminator(self.config.disc_type).to(self.device)

        # Initialize weights if specified
        if self.config['initialize_weights']:
            initialize_weights(self.generator)
            initialize_weights(self.discriminator)

        # Define loss functions
        self.gen_loss_fn = get_gen_loss()
        self.disc_loss_fn = get_disc_loss()

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
            train_dir=self.config['train_dir'],
            val_dir=self.config['val_dir'],
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
                disc_loss = self.disc_loss_fn(fake_output, real_output)
                disc_loss.backward()
                self.disc_optimizer.step()

                # Train generator
                self.generator.zero_grad()
                fake_output = self.discriminator(L, fake_AB)
                gen_loss = self.gen_loss_fn(fake_output, fake_AB, AB, lambda_l1=self.config['lambda_l1'])
                gen_loss.backward()
                self.gen_optimizer.step()

                progress_bar.set_postfix(
                    D_Loss=f"{disc_loss.item():.4f}",
                    G_Loss=f"{gen_loss.item():.4f}",
                )

                # Show examples and save checkpoints
                if (batch_idx % self.config['show_interval'] == 0 or batch_idx == total_batches - 1) and batch_idx != 0:
                    self._show_and_save_examples(epoch, batch_idx, total_batches)

                # Save the best model
                if gen_loss.item() < self.global_min:
                    self._save_best_model(gen_loss.item())

    def _show_and_save_examples(self, epoch, batch_idx, total_batches):
        """Displays and saves example results."""
        example_loader = torch.utils.data.DataLoader(
            self.val_loader.dataset, batch_size=1, shuffle=True, num_workers=4
        )
        show_examples(self.generator, example_loader, device=self.device)

        # Save checkpoint
        os.makedirs('checkpoints/gen', exist_ok=True)
        os.makedirs('checkpoints/disc', exist_ok=True)
        torch.save(self.generator.state_dict(), f"checkpoints/gen/gen_epoch{epoch}_batch{batch_idx}.pth")
        torch.save(self.discriminator.state_dict(), f"checkpoints/disc/disc_epoch{epoch}_batch{batch_idx}.pth")

    def _save_best_model(self, gen_loss):
        """Saves the best generator and discriminator models."""
        torch.save(self.generator.state_dict(), self.config['generator_path'])
        torch.save(self.discriminator.state_dict(), self.config['discriminator_path'])
        self.global_min = gen_loss
        print(f"New best model saved with (gen) loss: {self.global_min:.4f}")
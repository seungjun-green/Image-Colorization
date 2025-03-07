import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
import os
import json
from utils.utils import show_examples
from data.data_preprocessing import get_dataloaders
from utils.train_utils import get_gen_loss, get_disc_loss
from utils.model_utils import *
from scripts.eval import log_eval
from lpips import LPIPS 

class ImageColorizationTrainer:
    def __init__(self, config_file, **kwargs):
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
        self.training_ratio = self.config['training_ratio']
        self.gen_type = self.config['gen_type']
        self.disc_type = self.config['disc_type']
        
        self.generator_path = self.config['generator_path']
        self.discriminator_path = self.config['discriminator_path']

        os.makedirs(self.generator_path, exist_ok=True)
        os.makedirs(self.discriminator_path, exist_ok=True)        
        
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
            num_workers=self.config['num_workers']
        )
        
        # load model for evaluation
        self.eval_model = LPIPS(net='alex').to(self.device)
        
        # early stopping
        self.best_val_score = float('inf') 
        self.early_stopping_patience = self.config['patience']
        self.no_improvement_count = 0  

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
                    stop_training = self._show_and_save_examples(gen_loss.item(), epoch, batch_idx)
                    if stop_training:
                        return
                    self.generator.train()

    def _show_and_save_examples(self, gen_loss, epoch, batch_idx):
        """display some examples and save model weights"""
        # show some examples
        example_loader = torch.utils.data.DataLoader(self.val_loader.dataset, batch_size=1, shuffle=True, num_workers=2)
        show_examples(self.generator, example_loader, device=self.device)
        
        # get the current score on the validation set and print it
        val_score = log_eval(self.generator, self.eval_model, self.val_loader, self.config['device'])
        print(f"Epoch: {epoch+1} Step: {batch_idx+1} | val lpips score: {round(val_score, 4)}, val gen loss: {round(gen_loss, 4)}")
        
        # patience checker, if val score does not get improved 5 times, then stop the training
        if val_score < self.best_val_score:
            self.best_val_score = val_score
            self.no_improvement_count = 0
            torch.save(self.generator.state_dict(), f"{self.config['generator_path']}/epoch{epoch}_batch{batch_idx}_{round(val_score, 5)}.pth")
            return False
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.early_stopping_patience:
                print(f"Early stopping triggered after {self.early_stopping_patience} consecutive logs without improvement.")
                return True
import numpy as np
import os
import pickle as pkl
import copy
import logging
from datetime import datetime
import torch
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from sbi.inference import NPE
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.net_builders import build_nsf
from net_builder import DeepSet

class InferenceModel:
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        hidden_dim_phi: int=44,
        hidden_dim_rho: int=44,
        output_dim: int=128,
        learning_rate: float=5e-4,
        weight_decay: float=0,
        dropout: float=0,
        max_epochs: int=1000,
        min_epochs: int=50,
        stop_after_epochs: int=50,
        device: str='cuda',
        log_progress: bool=True
    ):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_progress = log_progress

        self.max_epochs = max_epochs
        self.min_epoch = min_epochs
        self.stop_after_epochs = stop_after_epochs
        self.terminate = False
        self.epoch = 0
        self.best_val_loss = np.inf
        self.best_model_state_dict = None
        self.cur_plateau = 0
        self.device = device

        self.history = {
            "training_loss": [],
            "validation_loss": []
        }

        perm_inv_net = DeepSet(hidden_dim_phi=hidden_dim_phi, hidden_dim_rho=hidden_dim_rho, output_dim=output_dim)

        dummy_theta, dummy_x = next(iter(self.train_loader))
        dummy_theta, dummy_x = dummy_theta.to(self.device), dummy_x.to(self.device)
        
        print("theta device:", dummy_theta.device)
        print("x device:", dummy_x.device)
        print(dummy_theta.size())
        print(dummy_x.size())
        self.density_estimator = build_nsf(dummy_theta, dummy_x, embedding_net=perm_inv_net, z_score_x='structured', z_score_y='structured', exclude_invalid_y=False, dropout_probability=dropout)
        self.density_estimator.to(self.device)

        self.opt = Adam(list(self.density_estimator.parameters()), lr=learning_rate, weight_decay=weight_decay)

    def train(self):
        if self.log_progress:
            logging.info(f"[{datetime.now()}] Begin Training")

        while not self.terminate:
            self.density_estimator.train()
            loss_sum = 0
            for idx, (theta_batch, x_batch) in enumerate(self.train_loader):
                theta_batch = theta_batch.to(self.device)
                x_batch = x_batch.to(self.device)
                
                self.opt.zero_grad()
                losses = self.density_estimator.loss(theta_batch, condition=x_batch)
                loss = torch.mean(losses)
                loss_sum += losses.sum().item()
                loss.backward()
                self.opt.step()

            train_loss_average = loss_sum / (len(self.train_loader) * self.train_loader.batch_size)
            self.history['training_loss'].append(train_loss_average)

            self.density_estimator.eval()
            val_loss_average = self.compute_nltp()
            self.history['validation_loss'].append(val_loss_average)

            if self.log_progress:
                logging.info(f"[{datetime.now()}] Epoch {self.epoch}: {train_loss_average}, {val_loss_average}")
            print(f"[{datetime.now()}] Epoch {self.epoch}: {train_loss_average}, {val_loss_average}")

            if self.epoch == 0 or val_loss_average < self.best_val_loss:
                self.best_val_loss = val_loss_average
                self.cur_plateau = 0
                self.best_model_state_dict = copy.deepcopy(self.density_estimator.state_dict())
            else:
                self.cur_plateau += 1

            self.epoch += 1

            if (self.cur_plateau >= self.stop_after_epochs and self.epoch >= self.min_epoch) or self.epoch >= self.max_epochs:
                self.terminate = True
        
        self.density_estimator.load_state_dict(self.best_model_state_dict)

    def compute_nltp(self):
        val_loss_sum = 0

        with torch.no_grad():
            for idx, (theta_batch, x_batch) in enumerate(self.val_loader):
                theta_batch = theta_batch.to(self.device)
                x_batch = x_batch.to(self.device)
                # NOTE: calling .loss() is the same as calling -1*.log_prob()
                val_losses = self.density_estimator.loss(theta_batch, x_batch)
                val_loss_sum += val_losses.sum().item()

        # Take mean over all validation samples.
        val_loss = val_loss_sum / (len(self.val_loader) * self.val_loader.batch_size)

        return val_loss

    def save(self, out_path):
        with open(out_path, 'wb') as f:
            pkl.dump(self, f)
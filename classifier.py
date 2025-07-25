import torch
from torch import Tensor, nn
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.nn.functional as F


class DeepSet(nn.Module):
    """DeepSet model for permutation invariance.


    Similar to the sbi.neural_nets.embedding_nets.PermutationInvariantEmbedding class. Some differences include
        - accepts self-attention as the aggregation function instead of sum/mean pooling
        - automatic intialization of phi/rho modules with direct control
    """

    def __init__(
        self,
        aggregation_fn: Optional[str] = "mean",
        aggregation_dim: int = 1,
        input_dim: int = 44,
        hidden_dim_phi: int = 128,
        hidden_dim_rho: int = 128,
        num_layers: int = 2,
        output_dim: int = 20,
        num_heads: int = 4

    ):
        super().__init__()

        assert aggregation_fn in [
            "mean",
            "sum",
            "attention"
        ], "aggregation_fn must be 'mean', 'sum', or 'attention'."
        self.aggregation_fn = aggregation_fn
        self.aggregation_dim = aggregation_dim

        # single-trial feature extractor
        phi_layers = [nn.Linear(input_dim, hidden_dim_phi), nn.ReLU()]
        for _ in range(num_layers - 1):
            phi_layers.append(nn.Linear(hidden_dim_phi, hidden_dim_phi))
            phi_layers.append(nn.ReLU())
            phi_layers.append(nn.LayerNorm(hidden_dim_phi))
        self.phi = nn.Sequential(*phi_layers)

        if self.aggregation_fn == "attention":
            self.attention = nn.MultiheadAttention(embed_dim=hidden_dim_phi, num_heads=num_heads, batch_first=True)
        else:
            self.attention = None

        # output function
        rho_layers = [nn.Linear(hidden_dim_phi + 1, hidden_dim_rho), nn.ReLU()] # +1 to encode number of trials
        for _ in range(num_layers - 2):
            rho_layers.append(nn.Linear(hidden_dim_rho, hidden_dim_rho))
            rho_layers.append(nn.ReLU())
            rho_layers.append(nn.LayerNorm(hidden_dim_rho))
        rho_layers.append(nn.Linear(hidden_dim_rho, output_dim))
        self.rho = nn.Sequential(*rho_layers)



    def forward(self, x: Tensor) -> Tensor:
        """Network forward pass.
        Args:
            x: Input tensor (batch_size, permutation_dim, input_dim)
        Returns:
            Network output (batch_size, output_dim).
        """

        # Get number of trials from non-nan entries
        num_batch, max_num_trials = x.shape[0], x.shape[self.aggregation_dim]
        nan_counts = (
            torch.isnan(x)
            .sum(dim=self.aggregation_dim)  # count nans over trial dimension
            .reshape(-1)[:num_batch]  # counts are the same across data dims
            .unsqueeze(-1)  # make it (batch, 1) to match embeddings below
        )
        # number of non-nan trials
        trial_counts = max_num_trials - nan_counts

        # get nan entries
        is_nan = torch.isnan(x)
        # apply trial net with nan entries replaced with 0
        masked_x = torch.nan_to_num(x, nan=0.0)
        #print(f'masked_x: {masked_x.shape}')
        trial_embeddings = self.phi(masked_x)
        # replace previous nan entries with zeros
        trial_embeddings = trial_embeddings * (~is_nan.all(-1, keepdim=True)).float()

        # Take mean over permutation dimension divide by number of trials
        # (instead of just taking torch.mean) to account for masking.
        if self.aggregation_fn == "mean":
            combined_embedding = (trial_embeddings.sum(dim=self.aggregation_dim) / trial_counts)
        elif self.aggregation_fn == "sum":
            combined_embedding = trial_embeddings.sum(dim=self.aggregation_dim)
        else:
            attn_output, _ = self.attention(trial_embeddings, trial_embeddings, trial_embeddings)  # Self-attention
            combined_embedding = attn_output.mean(dim=self.aggregation_dim)


        assert not torch.isnan(combined_embedding).any(), "NaNs in embedding."

        # add number of trials as additional input
        return self.rho(torch.cat([combined_embedding, trial_counts], dim=1)).           


#embedding_net = DeepSet(hidden_dim_phi=64, hidden_dim_rho=64, output_dim=128)

embedding_net = DeepSet(
    hidden_dim_phi=128,   # increased from 64
    hidden_dim_rho=128,   # increased from 64
    output_dim=64,       # more expressive embedding
)

class DeepSetClassifier(nn.Module):
    def __init__(self, embedding_net, embedding_dim=64, num_classes=2):
        super().__init__()
        self.embedding_net = embedding_net
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.0)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x, return_embedding=False):
        embedded = self.embedding_net(x)  # (B, embedding_dim)
        x = self.fc1(embedded)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)  # raw logits (no softmax yet)
        
        if return_embedding:
            return logits, x  # return logits and last hidden layer
        else:
            probs = F.softmax(logits, dim=-1)
            return probs
    

# ------------------------------
# Training Loop
# ------------------------------
def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for x, labels in dataloader:
        x, labels = x.to(device), labels.to(device)

        pred = model(x, return_embedding=False)  # (B, 44, 3)
        loss = loss_fn(pred.view(-1, 3), labels.view(-1))  # CrossEntropy expects (N, C) and (N,)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds_class = torch.argmax(pred, dim=-1)  # predicted class labels
        all_preds.append(preds_class.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).view(-1).numpy()
    all_labels = torch.cat(all_labels).view(-1).numpy()

    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / len(dataloader)
    return avg_loss, f1, precision, recall
    

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, labels in dataloader:
            x, labels = x.to(device), labels.to(device)
            pred = model(x,return_embedding=False)  # (B, 44, 3)
            loss = loss_fn(pred.view(-1, 3), labels.view(-1))  # CrossEntropy expects (N, C) and (N,)
            total_loss += loss.item()
            preds_class = torch.argmax(pred, dim=-1)  # predicted class labels
            all_preds.append(preds_class.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).view(-1).numpy()
    all_labels = torch.cat(all_labels).view(-1).numpy()

    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    avg_loss = total_loss / len(dataloader)
    return avg_loss, f1, precision, recall

# ------------------------------
# Main
# ------------------------------

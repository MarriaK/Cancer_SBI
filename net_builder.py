from typing import Optional
import torch
from torch import Tensor, nn

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
        hidden_dim_phi: int = 100,
        hidden_dim_rho: int = 100,
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
        return self.rho(torch.cat([combined_embedding, trial_counts], dim=1))
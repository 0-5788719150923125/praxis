import torch
import torch.nn as nn

from ..configuration_praxis import PraxisConfig
from .block import PraxisBlock
from .router import PraxisMixtureOfDepths


class PraxisEncoder(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.config = config
        self.n_dim = config.n_dim
        self.wte = nn.Embedding(config.vocab_size, config.n_emb)
        self.wme = nn.Linear(config.n_emb, config.n_dim, bias=False)
        self.max_pca_k = min(
            config.n_dim, config.n_emb
        )  # Maximum number of principal components
        self.n_factors = config.n_factors
        self.pca = nn.Linear(config.n_dim + self.max_pca_k, config.n_dim, bias=True)

    def forward(self, x):
        # Word token embeddings
        input_embeds = self.wte(x)

        # Linear projection (residual)
        inputs_reduced = self.wme(input_embeds)

        # Calculate pca_k dynamically
        q = min(self.max_pca_k, input_embeds.size(0), input_embeds.size(1)) - 1

        # PCA operation
        if q > 0:
            _, _, v = torch.pca_lowrank(
                input_embeds, q=q, center=True, niter=self.n_factors
            )
            pca_reduced = torch.matmul(input_embeds, v)
        else:
            # Fallback if PCA is not possible
            pca_reduced = torch.zeros(
                *input_embeds.shape[:-1], 0, device=input_embeds.device
            )

        # Pad pca_reduced if necessary
        if pca_reduced.size(-1) < self.max_pca_k:
            padding = torch.zeros(
                *pca_reduced.shape[:-1],
                self.max_pca_k - pca_reduced.size(-1),
                device=pca_reduced.device
            )
            pca_reduced = torch.cat([pca_reduced, padding], dim=-1)

        # Combine linear projection and PCA results
        combined = torch.cat([inputs_reduced, pca_reduced], dim=-1)
        hidden_states = self.pca(combined)

        return dict(hidden_states=hidden_states, aux_loss=0)

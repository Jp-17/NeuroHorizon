"""IDEncoder: Generates unit embeddings from reference window statistical features.

Replaces InfiniteVocabEmbedding for gradient-free cross-session generalization.
New neurons only need their statistical features computed to get embeddings,
no fine-tuning required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IDEncoder(nn.Module):
    """Feed-forward network that generates unit embeddings from reference features.

    Given per-unit statistical features (firing rate, ISI stats, autocorrelation, etc.),
    produces dense unit embeddings compatible with the POYO+ encoder architecture.

    Args:
        ref_dim: Dimension of input reference features (default: 33).
        embedding_dim: Dimension of output embeddings.
        hidden_dim: Hidden layer dimension. If None, uses embedding_dim.
        num_layers: Number of MLP layers (default: 3).
        dropout: Dropout rate between layers (default: 0.1).
    """

    def __init__(
        self,
        ref_dim: int = 33,
        embedding_dim: int = 256,
        hidden_dim: int = None,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ref_dim = ref_dim
        self.embedding_dim = embedding_dim
        hidden_dim = hidden_dim or embedding_dim

        layers = []
        in_dim = ref_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else embedding_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(out_dim))
            if dropout > 0 and i < num_layers - 1:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

        # Cached embeddings for the current batch
        self._cached_embeddings = None

    def compute_embeddings(self, reference_features: torch.Tensor) -> torch.Tensor:
        """Pre-compute unit embeddings from reference features.

        Call this once per session/batch before forward() to cache the embeddings.

        Args:
            reference_features: (n_units, ref_dim) tensor of statistical features.

        Returns:
            (n_units, embedding_dim) tensor of unit embeddings.
        """
        embeddings = self.mlp(reference_features)
        self._cached_embeddings = embeddings
        return embeddings

    def forward(self, unit_index: torch.Tensor) -> torch.Tensor:
        """Look up unit embeddings by index from cached embeddings.

        Args:
            unit_index: Integer tensor of unit indices to look up.

        Returns:
            Tensor of unit embeddings with shape (*unit_index.shape, embedding_dim).
        """
        if self._cached_embeddings is None:
            raise RuntimeError(
                "No cached embeddings. Call compute_embeddings() first."
            )
        return self._cached_embeddings[unit_index]

    def forward_from_features(
        self, reference_features: torch.Tensor, unit_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute embeddings and look up in one step.

        Args:
            reference_features: (n_units, ref_dim) tensor.
            unit_index: Integer tensor of unit indices.

        Returns:
            Tensor of unit embeddings with shape (*unit_index.shape, embedding_dim).
        """
        embeddings = self.compute_embeddings(reference_features)
        return embeddings[unit_index]

    def clear_cache(self):
        """Clear cached embeddings to free memory."""
        self._cached_embeddings = None

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, kl_divergence
from torch.nn.functional import one_hot
import torch.nn.functional as F

from scvi import REGISTRY_KEYS, settings
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import FCLayers


if TYPE_CHECKING:
    from collections.abc import Callable
    from torch.distributions import Distribution

logger = logging.getLogger(__name__)

from torch.special import gammaln

def group_logsumexp(A, B):
    """
    Computes the log-sum-exp of B, grouped by A.
    A is a sparse matrix (must be coalesced already!) with shape (P, G), where P is the number of variables (junctions)
    and G is the number of groups (ATSEs). B is a dense matrix with shape (N, P), where
    N is the number of samples (cells). The result is a dense matrix with shape (N, G).
    """
    idx_p, idx_g = A.indices()              # (2, P), since each var belongs to one group
    idx_p = idx_p.to(B.device)
    idx_g = idx_g.to(B.device)

    N, P = B.shape
    G = A.shape[1]

    # (N, P) → gather relevant group index for each variable
    group_idx = torch.empty(P, dtype=torch.long, device=B.device)
    group_idx[idx_p] = idx_g

    # Get max within each group
    max_B = torch.full((N, G), float('-inf'), device=B.device)
    max_B.scatter_reduce_(1, group_idx.expand(N, -1), B, reduce='amax')

    # Center B by group max
    B_centered = B - max_B.gather(1, group_idx.expand(N, -1))

    exp_B_centered = torch.exp(B_centered)

    # Sum exp over each group
    sum_exp = torch.zeros((N, G), device=B.device)
    sum_exp.scatter_add_(1, group_idx.expand(N, -1), exp_B_centered)

    return torch.log(sum_exp + 1e-8) + max_B

def subtract_group_logsumexp(A, B, group_logsumexp):
    """
    Subtracts the group log-sum-exp from each element in B, grouped by A.
    A is a sparse matrix (must be coalesced already!) with shape (P, G), where P is the number of variables (junctions)
    and G is the number of groups (ATSEs). B is a dense matrix with shape (N, P), where
    N is the number of samples (cells). The result is a dense matrix with shape (N, P).
    """
    idx_p, idx_g = A.indices()
    idx_g = idx_g.to(B.device)
    N = B.size(0)
    return B - group_logsumexp[:, idx_g]

def nbetaln(count, alpha):  
    """
    Computes log(count * beta(count, alpha)) == gammaln(count + 1) + gammaln(alpha) - gammaln(count + alpha). 
    Note this is zero when count == 0, which could potentially be used to improve efficiency.
    """
    return gammaln(alpha) + gammaln(count+1.) - gammaln(alpha + count)

def dirichlet_multinomial_likelihood(
    counts: torch.Tensor,        # (N, P)
    atse_counts: torch.Tensor,   # (N, G)
    junc2atse: torch.sparse.Tensor,  # (P, G)
    alpha: torch.Tensor,         # (N, P)
    mask: torch.Tensor | None = None,  # optional (N, P)
) -> torch.Tensor:
    """
    Computes the batch-averaged Dirichlet–multinomial log-likelihood,
    **masking out** any junctions where mask==0 and any ATSEs where atse_counts==0.

    Returns:
        scalar = mean_over_cells( sum_per_ATSE(LL_atse * ATSE_mask)
                                - sum_per_junction(LL_junc * junc_mask) )
    """
    device = counts.device
    atse_counts = atse_counts.to(device)
    alpha       = alpha.to(device)
    mask        = mask.to(device) if mask is not None else None
    junc2atse   = junc2atse.to(device)
    N, P = counts.shape
    G     = junc2atse.shape[1]

    idx_p, idx_g = junc2atse.indices()
    idx_p = idx_p.to(counts.device)
    idx_g = idx_g.to(counts.device)
    idx_p, idx_g = junc2atse.indices()
    idx_p = idx_p.to(device)
    idx_g = idx_g.to(device)             # each length ≈ P

    true_atse = torch.zeros((N, G), device=device, dtype=atse_counts.dtype)
    true_atse.scatter_reduce_(
        dim=1,
        index=idx_g.expand(N, -1),                    # (N, P)
        src=atse_counts,                              # (N, P)
        reduce="amax",
    )
    atse_counts = true_atse                           # (N, G)

    # 2) compute group-sums of alpha → (N, G)
    alpha_sums = alpha @ junc2atse                    # (N, G)

    # 3) per-ATSE log-likelihoods
    ll_atse = nbetaln(atse_counts, alpha_sums)        # (N, G)
    # mask out any ATSEs with zero count
    atse_mask = (atse_counts > 0).to(ll_atse.dtype)   # (N, G)
    ll_atse = ll_atse * atse_mask                     # zeros where atse_count==0
    # sum per ATSE and then average over cells
    per_atse = ll_atse.sum(dim=1).mean()              # scalar

    # 4) per-junction log-likelihoods
    ll_junc = nbetaln(counts, alpha)                  # (N, P)
    if mask is not None:
        mask_bool = mask.to(torch.bool)               # (N, P)
        ll_junc = ll_junc * mask_bool.to(ll_junc.dtype)
    # sum per junction and then average over cells
    per_junc = ll_junc.sum(dim=1).mean()              # scalar

    # 5) return batch-averaged difference
    return per_atse - per_junc



def binomial_loss_function(
    logits: torch.Tensor,
    junction_counts: torch.Tensor,
    cluster_counts: torch.Tensor,
    n,
    k,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reconstruction Binomial loss function for VAE.
    
    Parameters:
      logits: Reconstructed logits from decoder.
      junction_counts: Junction counts from data.
      cluster_counts: Cluster counts from data.
      mask: Mask to exclude missing values when running partialVAE.
      
    Returns:
      total_loss: Reconstruction loss according to binomial likelihood.
    """

    if mask is not None:
        mask = mask.to(torch.bool)  # ← convert floats/ints to bool
        logits = logits[mask]
        junction_counts = junction_counts[mask]
        cluster_counts = cluster_counts[mask]

    probs = torch.sigmoid(logits)
    log_p = torch.log(probs + 1e-8)
    log_1_p = torch.log(1 - probs + 1e-8)
    log_lik = (
        junction_counts * log_p + (cluster_counts - junction_counts) * log_1_p
    )
    return -log_lik.mean()


def beta_binomial_log_pmf(k, n, alpha, beta):
    """
    Calculate the log probability mass function for the beta-binomial distribution.
    
    Parameters:
      k: Number of successes.
      n: Number of trials.
      alpha: Alpha parameter of beta distribution.
      beta: Beta parameter of beta distribution.
      
    Returns:
      log_pmf: Log of probability mass function.
    """
    return (
        torch.lgamma(n + 1)
        - torch.lgamma(k + 1)
        - torch.lgamma(n - k + 1)
        + torch.lgamma(k + alpha)
        + torch.lgamma(n - k + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + torch.lgamma(alpha + beta)
        - torch.lgamma(n + alpha + beta)
    )


def beta_binomial_loss_function(
    logits: torch.Tensor,
    junction_counts: torch.Tensor,
    cluster_counts: torch.Tensor,
    n: int,
    k: int,
    concentration: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Beta-binomial loss function for VAE.
    
    Parameters:
      logits: Reconstructed logits from decoder.
      junction_counts: Junction counts from data.
      cluster_counts: Cluster counts from data.
      n: Number of samples in dataset.
      k: Number of batches in dataloader.
      concentration: Concentration parameter.
      mask: Optional mask to exclude missing values.
      
    Returns:
      total_loss: Combined reconstruction and KL loss.
    """
    if mask is not None:
        mask = mask.to(torch.bool)  # ← convert floats/ints to bool
        logits = logits[mask]
        junction_counts = junction_counts[mask]
        cluster_counts = cluster_counts[mask]


    probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    alpha = probs * concentration
    beta = (1.0 - probs) * concentration
    log_lik = beta_binomial_log_pmf(
        k=junction_counts, n=cluster_counts, alpha=alpha, beta=beta
    )
    log_lik = log_lik #* (float(n) / float(k))
    return -log_lik.mean()


import torch
from torch import nn
from collections.abc import Iterable
from scvi.nn import FCLayers

#NON FAST VERSIONS (PER CELL LOOPS)
class PartialEncoderEDDI(nn.Module):
    """
    EDDI partial encoder.

    Shapes / notation:
      B=batch, J=junctions, D=code_dim.
    Input:
      x:    (B, J)
      mask: (B, J) 1=observed, 0=missing
    Output:
      mu, logvar: (B, Z)
    """
    def __init__(
        self,
        input_dim: int,                 # J
        code_dim: int,                  # D
        h_hidden_dim: int,
        encoder_hidden_dim: int,
        latent_dim: int,
        dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None,
        n_cont: int = 0,
        inject_covariates: bool = True,
        pool_mode: Literal["mean","sum"] = "mean",
    ):
        super().__init__()
        self.code_dim = code_dim
        self.pool_mode = pool_mode
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates

        # Per-junction embedding table (J, D)
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        # h-layer: [psi (1) | F_j (D)] -> D
        in_dim_h = 1 + code_dim
        self.h_layer = nn.Sequential(
            nn.Linear(in_dim_h, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_hidden_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        # Cell-level encoder MLP → (mu, logvar)
        self.encoder_mlp = FCLayers(
            n_in=code_dim, n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [], n_cont=n_cont,
            n_layers=2, n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False, use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

    def forward(
        self,
        x: torch.Tensor,                # (B, J)
        mask: torch.Tensor,             # (B, J)
        *cat_list: torch.Tensor,
        cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, J = x.shape
        D = self.code_dim
        dev = x.device

        outputs = []
        for b in range(B):
            obs = mask[b].bool().nonzero(as_tuple=True)[0]      # (n_obs,)
            if obs.numel() == 0:
                outputs.append(torch.zeros(D, device=dev))
                continue

            x_obs = x[b, obs]                                   # (n_obs,)
            F_obs = self.feature_embedding[obs]                  # (n_obs, D)
            h_in  = torch.cat([x_obs.unsqueeze(-1), F_obs], dim=1)  # (n_obs, 1+D)
            h_out = self.h_layer(h_in)                          # (n_obs, D)

            pooled = h_out.sum(dim=0)
            c = pooled / obs.numel() if self.pool_mode == "mean" else pooled
            outputs.append(c)

        c = torch.stack(outputs, dim=0)                         # (B, D)
        mu_logvar = self.encoder_mlp(c, *cat_list, cont=cont)   # (B, 2Z)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar

class PartialEncoderEDDIATSE(nn.Module):
    """
    EDDI partial encoder with learnable ATSE embeddings.

    Shapes:
      B=batch, J=junctions, D=code_dim, A=ATSEs, Ae=ATSE emb dim.
    h-layer sees: [psi (1) | F_j (D) | E_atse(j) (Ae)]
    """
    def __init__(
        self,
        input_dim: int, code_dim: int, h_hidden_dim: int,
        encoder_hidden_dim: int, latent_dim: int,
        dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None,
        n_cont: int = 0, inject_covariates: bool = True,
        pool_mode: Literal["mean","sum"] = "mean",
        atse_embedding_dimension: int = 16,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.h_hidden_dim = h_hidden_dim
        self.dropout_rate = dropout_rate
        self.pool_mode = pool_mode
        self.atse_embedding_dimension = int(atse_embedding_dimension)
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates

        # Junction embedding (J, D)
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        # Placeholder h-layer; rebuilt when ATSE embeddings exist
        self.h_layer = self._make_h_layer(n_atse_embed=0)

        # Cell encoder
        self.encoder_mlp = FCLayers(
            n_in=code_dim, n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [], n_cont=n_cont,
            n_layers=2, n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False, use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

        # ATSE mapping / embeddings
        self.register_buffer("junc2atse_sparse", torch.sparse_coo_tensor(size=(input_dim, 0)))
        self.register_buffer("atse_index_per_j", torch.zeros(input_dim, dtype=torch.long))
        self.n_atse = 0
        self.atse_embedding: nn.Parameter | None = None

    def _make_h_layer(self, n_atse_embed: int) -> nn.Sequential:
        in_dim = 1 + self.code_dim + n_atse_embed
        return nn.Sequential(
            nn.Linear(in_dim, self.h_hidden_dim),
            nn.LayerNorm(self.h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.h_hidden_dim, self.code_dim),
            nn.LayerNorm(self.code_dim),
            nn.ReLU(),
        )

    @torch.no_grad()
    def register_junc2atse(self, junc2atse: torch.sparse.Tensor):
        """
        Register (J × A) mapping; build `atse_index_per_j` and ATSE embedding table.
        """
        j2a = junc2atse.coalesce().to(self.feature_embedding.device)
        self.register_buffer("junc2atse_sparse", j2a)
        J = self.feature_embedding.shape[0]

        # Dense J→ATSE index
        idx_p, idx_g = j2a.indices()
        if idx_g.numel() == 0:
            self.n_atse = 0
            self.atse_embedding = None
            self.atse_index_per_j.zero_()
            self.h_layer = self._make_h_layer(n_atse_embed=0)
            return

        self.n_atse = int(idx_g.max().item()) + 1
        atse_idx = torch.zeros(J, dtype=torch.long, device=j2a.device)
        atse_idx[idx_p] = idx_g
        self.register_buffer("atse_index_per_j", atse_idx)

        # Embeddings (A, Ae) and rebuild h-layer
        self.atse_embedding = nn.Parameter(
            torch.randn(self.n_atse, self.atse_embedding_dimension, device=j2a.device)
        )
        self.h_layer = self._make_h_layer(n_atse_embed=self.atse_embedding_dimension)

    def forward(
        self,
        x: torch.Tensor, mask: torch.Tensor,
        *cat_list: torch.Tensor, cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.atse_embedding is None:
            raise RuntimeError("Call `register_junc2atse(...)` before forward.")
        B, J = x.shape
        D = self.code_dim
        dev = x.device


        outputs = []
        for b in range(B):
            obs = mask[b].bool().nonzero(as_tuple=True)[0]
            if obs.numel() == 0:
                outputs.append(torch.zeros(D, device=dev))
                continue

            x_obs = x[b, obs]                         # (n_obs,)
            F_obs = self.feature_embedding[obs]       # (n_obs, D)
            Ae_obs = self.atse_embedding[self.atse_index_per_j[obs]]  # (n_obs, Ae)

            h_in  = torch.cat([x_obs.unsqueeze(-1), F_obs, Ae_obs], dim=1)  # (n_obs, 1+D+Ae)
            h_out = self.h_layer(h_in)                                       # (n_obs, D)

            pooled = h_out.sum(dim=0)
            c = pooled / obs.numel() if self.pool_mode == "mean" else pooled
            outputs.append(c)

        c = torch.stack(outputs, dim=0)                # (B, D)
        mu_logvar = self.encoder_mlp(c, *cat_list, cont=cont)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar

class PartialEncoderWeightedSumEDDIMultiWeight(nn.Module):
    """
    EDDI encoder with W weighted-sum heads.

    Shapes:
      B=batch, J=junctions, D=code_dim, W=heads.
    """
    def __init__(
        self,
        input_dim: int, code_dim: int, h_hidden_dim: int,
        encoder_hidden_dim: int, latent_dim: int,
        num_weight_vectors: int = 4, dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None, n_cont: int = 0,
        inject_covariates: bool = True,
        temperature_value: float = 1.0, temperature_fixed: bool = True,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.num_weight_vectors = num_weight_vectors
        self.temperature_value = float(temperature_value)
        self.temperature_fixed = bool(temperature_fixed)
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates

        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        in_dim_h = 1 + code_dim
        self.h_layer = nn.Sequential(
            nn.Linear(in_dim_h, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_hidden_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(code_dim, code_dim // 2),
            nn.ReLU(),
            nn.Linear(code_dim // 2, self.num_weight_vectors),
        )

        self.combiner = nn.Identity() if self.num_weight_vectors == 1 else nn.Sequential(
            nn.Linear(self.num_weight_vectors * code_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        self.encoder_mlp = FCLayers(
            n_in=code_dim, n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [], n_cont=n_cont,
            n_layers=2, n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False, use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

    def forward(
        self,
        x: torch.Tensor, mask: torch.Tensor,
        *cat_list: torch.Tensor, cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, J = x.shape
        D, W = self.code_dim, self.num_weight_vectors
        dev = x.device


        outputs = []
        for b in range(B):
            obs = mask[b].bool().nonzero(as_tuple=True)[0]
            if obs.numel() == 0:
                outputs.append(torch.zeros(D, device=dev))
                continue

            x_obs = x[b, obs]                       # (n_obs,)
            F_obs = self.feature_embedding[obs]     # (n_obs, D)
            h_in  = torch.cat([x_obs.unsqueeze(-1), F_obs], dim=1)   # (n_obs, 1+D)
            h_out = self.h_layer(h_in)                                 # (n_obs, D)

            raw_gates = self.gate_net(h_out)                           # (n_obs, W)
            scale = self.temperature_value if self.temperature_fixed else (float(obs.numel())) ** -0.5
            logits = (raw_gates * scale).clamp(min=-10.0, max=10.0)
            weights = torch.softmax(logits, dim=0)                     # (n_obs, W)

            head_sums = (weights.T.unsqueeze(-1) * h_out.unsqueeze(0)).sum(dim=1)  # (W, D)
            combined = self.combiner(head_sums.reshape(-1))                           # (D,)
            outputs.append(combined)

        c = torch.stack(outputs, dim=0)                  # (B, D)
        mu_logvar = self.encoder_mlp(c, *cat_list, cont=cont)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar

class PartialEncoderWeightedSumEDDIMultiWeightATSE(nn.Module):
    """
    EDDI encoder with W weighted-sum heads and ATSE embeddings.

    h-layer: [psi | F_j] -> D
    Gate input: [h_out (D) | E_atse(j) (Ae)] -> W
    """
    def __init__(
        self,
        input_dim: int, code_dim: int, h_hidden_dim: int,
        encoder_hidden_dim: int, latent_dim: int,
        num_weight_vectors: int = 4, dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None, n_cont: int = 0,
        inject_covariates: bool = True,
        temperature_value: float = 1.0, temperature_fixed: bool = True,
        atse_embedding_dimension: int = 16,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.num_weight_vectors = num_weight_vectors
        self.temperature_value = float(temperature_value)
        self.temperature_fixed = bool(temperature_fixed)
        self.atse_embedding_dimension = int(atse_embedding_dimension)
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates

        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        in_dim_h = 1 + code_dim
        self.h_layer = nn.Sequential(
            nn.Linear(in_dim_h, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_hidden_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        in_dim_gate = code_dim + self.atse_embedding_dimension
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim_gate, max(in_dim_gate // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(in_dim_gate // 2, 1), self.num_weight_vectors),
        )

        self.combiner = nn.Identity() if self.num_weight_vectors == 1 else nn.Sequential(
            nn.Linear(self.num_weight_vectors * code_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        self.encoder_mlp = FCLayers(
            n_in=code_dim, n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [], n_cont=n_cont,
            n_layers=2, n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False, use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

        # ATSE mapping/embeddings
        self.register_buffer("junc2atse_sparse", torch.sparse_coo_tensor(size=(input_dim, 0)))
        self.register_buffer("atse_index_per_j", torch.zeros(input_dim, dtype=torch.long))
        self.n_atse = 0
        self.atse_embedding: nn.Parameter | None = None

    @torch.no_grad()
    def register_junc2atse(self, junc2atse: torch.sparse.Tensor):
        j2a = junc2atse.coalesce().to(self.feature_embedding.device)
        self.register_buffer("junc2atse_sparse", j2a)
        J = self.feature_embedding.shape[0]

        idx_p, idx_g = j2a.indices()
        if idx_g.numel() == 0:
            self.n_atse = 0
            self.atse_embedding = None
            self.atse_index_per_j.zero_()
            return

        self.n_atse = int(idx_g.max().item()) + 1
        atse_idx = torch.zeros(J, dtype=torch.long, device=j2a.device)
        atse_idx[idx_p] = idx_g
        self.register_buffer("atse_index_per_j", atse_idx)

        self.atse_embedding = nn.Parameter(
            torch.randn(self.n_atse, self.atse_embedding_dimension, device=j2a.device)
        )

    def forward(
        self,
        x: torch.Tensor, mask: torch.Tensor,
        *cat_list: torch.Tensor, cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.atse_embedding is None:
            raise RuntimeError("Call `register_junc2atse(...)` before forward.")
        B, J = x.shape
        D, W = self.code_dim, self.num_weight_vectors
        dev = x.device


        outputs = []
        for b in range(B):
            obs = mask[b].bool().nonzero(as_tuple=True)[0]
            if obs.numel() == 0:
                outputs.append(torch.zeros(D, device=dev))
                continue

            x_obs = x[b, obs]                      # (n_obs,)
            F_obs = self.feature_embedding[obs]    # (n_obs, D)
            h_in  = torch.cat([x_obs.unsqueeze(-1), F_obs], dim=1)  # (n_obs, 1+D)
            h_out = self.h_layer(h_in)                                 # (n_obs, D)

            Ae_obs = self.atse_embedding[self.atse_index_per_j[obs]]   # (n_obs, Ae)
            gate_in = torch.cat([h_out, Ae_obs], dim=1)                # (n_obs, D+Ae)
            raw_gates = self.gate_net(gate_in)                         # (n_obs, W)

            scale = self.temperature_value if self.temperature_fixed else (float(obs.numel())) ** -0.5
            logits = (raw_gates * scale).clamp(min=-10.0, max=10.0)
            weights = torch.softmax(logits, dim=0)                     # (n_obs, W)

            head_sums = (weights.T.unsqueeze(-1) * h_out.unsqueeze(0)).sum(dim=1)  # (W, D)
            combined = self.combiner(head_sums.reshape(-1))                           # (D,)
            outputs.append(combined)

        c = torch.stack(outputs, dim=0)                      # (B, D)
        mu_logvar = self.encoder_mlp(c, *cat_list, cont=cont)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar

#FAST VERSIONS (BATCHING)

class PartialEncoderEDDIFast(nn.Module):
    """
    Fast EDDI partial encoder, batched.

    Shapes / notation:
      B = batch size (cells)
      J = number of junctions
      D = code_dim
      Z = latent_dim

    """
    def __init__(
        self,
        input_dim: int,                  # J
        code_dim: int,                   # D
        h_hidden_dim: int,
        encoder_hidden_dim: int,
        latent_dim: int,                 # Z
        dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None,
        n_cont: int = 0,
        inject_covariates: bool = True,
        pool_mode: Literal["mean", "sum"] = "mean",
    ):
        super().__init__()
        self.code_dim = code_dim
        self.pool_mode = pool_mode
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates

        # Per-junction embedding table (J, D)
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        # h-layer: [psi (1) | F_j (D)] -> D
        in_dim_h = 1 + code_dim
        self.h_layer = nn.Sequential(
            nn.Linear(in_dim_h, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_hidden_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        # Cell-level encoder MLP → (mu, logvar)
        self.encoder_mlp = FCLayers(
            n_in=code_dim, n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [], n_cont=n_cont,
            n_layers=2, n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False, use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

    def forward(
        self,
        x: torch.Tensor,                 # (B, J)
        mask: torch.Tensor,              # (B, J)
        *cat_list: torch.Tensor,
        cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, J = x.shape
        D = self.code_dim

        mask_bool = mask.bool()                      # (B, J)
        F = self.feature_embedding                   # (J, D)

        # Prepare h-layer inputs
        x_exp = x.unsqueeze(-1)                      # (B, J, 1)
        F_exp = F.unsqueeze(0).expand(B, J, D)       # (B, J, D)
        h_in  = torch.cat([x_exp, F_exp], dim=-1)    # (B, J, 1 + D)

        # Apply h-layer in a batched way
        h_out = self.h_layer(h_in.view(B * J, -1)).view(B, J, D)  # (B, J, D)

        # Mask and pool across junctions
        h_masked = h_out * mask_bool.unsqueeze(-1)   # (B, J, D)
        pooled = h_masked.sum(dim=1)                 # (B, D)
        if self.pool_mode == "mean":
            denom = mask_bool.sum(dim=1, keepdim=True).clamp(min=1)
            c = pooled / denom                       # (B, D)
        else:
            c = pooled

        # Final per-cell projection → (mu, logvar)
        mu_logvar = self.encoder_mlp(c, *cat_list, cont=cont)  # (B, 2Z)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar


class PartialEncoderEDDIFaster(nn.Module):
    """
    Fast EDDI partial encoder, batched.

    Shapes / notation:
      B = batch size (cells)
      J = number of junctions
      D = code_dim
      Z = latent_dim

    """
    def __init__(
        self,
        input_dim: int,                  # J
        code_dim: int,                   # D
        h_hidden_dim: int,
        encoder_hidden_dim: int,
        latent_dim: int,                 # Z
        dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None,
        n_cont: int = 0,
        inject_covariates: bool = True,
        pool_mode: Literal["mean", "sum"] = "mean",
        max_nobs: int = -1,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.pool_mode = pool_mode
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates
        self.max_nobs = max_nobs

        # Per-junction embedding table (J, D)
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        # h-layer: [psi (1) | F_j (D)] -> D
        in_dim_h = 1 + code_dim

        

        self.h_layer = nn.Sequential(
            nn.Linear(in_dim_h, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_hidden_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        # Cell-level encoder MLP → (mu, logvar)
        self.encoder_mlp = FCLayers(
            n_in=code_dim, n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [], n_cont=n_cont,
            n_layers=2, n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False, use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

    def forward(
    self,
    x: torch.Tensor,                 # (B, J)
    mask: torch.Tensor,              # (B, J), 1=observed, 0=missing
    *cat_list: torch.Tensor,
    cont: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Observed-only path:
        - Gather (b, j) pairs where mask==1
        - Run h_layer only on those rows
        - Sum (or mean) per cell
        - Encode to (mu, logvar)
        """
        B, J = x.shape
        D = self.code_dim
        device = x.device
        dtype  = x.dtype

        # Ensure boolean mask (no grad)
        mask_bool = mask.bool()

        # Find observed indices (vectors of shape (N_obs,))
        b_idx, j_idx = mask_bool.nonzero(as_tuple=True)
        N_obs = b_idx.numel()
        F_j = self.feature_embedding

        # Early-exit if nothing is observed
        if N_obs == 0:
            pooled = torch.zeros(B, D, device=device, dtype=dtype)
            mu_logvar = self.encoder_mlp(pooled, *cat_list, cont=cont)  # (B, 2Z)
            mu, logvar = mu_logvar.chunk(2, dim=-1)
            return mu, logvar

        # Precompute a normalized copy once (avoids re-normalizing the same rows repeatedly)
        # Do this where F_j is defined / available before the if/else:
        F_j_norm = F.normalize(F_j, p=2, dim=1, eps=1e-8)   # (J, D)

        if (self.max_nobs < 0) or (N_obs <= self.max_nobs):
            # ---- Original (no chunking) path ----
            x_obs = x[b_idx, j_idx].unsqueeze(1)                  # (N_obs, 1)
            F_obs = F_j_norm.index_select(0, j_idx)               # (N_obs, D), already L2-normalized

            F_obs_scaled = F_obs * x_obs                          # broadcast scale by usage ratio
            h_in  = torch.cat([x_obs, F_obs_scaled], dim=1)       # (N_obs, 1 + D)
            h_obs = self.h_layer(h_in)                            # (N_obs, D)

            pooled = torch.zeros(B, D, device=device, dtype=h_obs.dtype)
            pooled.index_add_(0, b_idx, h_obs)

        else:
            # ---- Chunked path: same result, steadier memory ----
            pooled = None
            for start in range(0, N_obs, self.max_nobs):
                end = min(start + self.max_nobs, N_obs)

                bi = b_idx[start:end]                             # (n,)
                jj = j_idx[start:end]                             # (n,)

                x_chunk = x[bi, jj].unsqueeze(1)                  # (n, 1)
                F_chunk = F_j_norm.index_select(0, jj)            # (n, D), already L2-normalized

                F_chunk_scaled = F_chunk * x_chunk                # scale by usage ratio
                h_in  = torch.cat([x_chunk, F_chunk_scaled], dim=1)  # (n, 1 + D)
                h_out = self.h_layer(h_in)                           # (n, D)

                if pooled is None:
                    pooled = torch.zeros(B, D, device=device, dtype=h_out.dtype)
                pooled.index_add_(0, bi, h_out)


        # Mean pooling if requested
        if self.pool_mode == "mean":
            # counts per cell (B,), keep at least 1 to avoid division by zero
            counts = torch.bincount(b_idx, minlength=B).to(pooled.dtype).view(B, 1).clamp_min_(1)
            pooled = pooled / counts

        # Final per-cell projection → (mu, logvar)
        mu_logvar = self.encoder_mlp(pooled, *cat_list, cont=cont)  # (B, 2Z)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar

#FAST VERSIONS (BATCHING)
class PartialEncoderEDDIATSEFast(nn.Module):
    """
    Fast EDDI + ATSE encoder (batched) with cached `atse_index_per_j`.

    h-layer sees: [psi (1) | F_j (D) | E_atse(j) (Ae)], applied to all (B×J) then masked/pool.
    """
    def __init__(
        self,
        input_dim: int, code_dim: int, h_hidden_dim: int,
        encoder_hidden_dim: int, latent_dim: int,
        dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None, n_cont: int = 0,
        inject_covariates: bool = True,
        pool_mode: Literal["mean","sum"] = "mean",
        atse_embedding_dimension: int = 16,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.h_hidden_dim = h_hidden_dim
        self.dropout_rate = dropout_rate
        self.pool_mode = pool_mode
        self.atse_embedding_dimension = int(atse_embedding_dimension)
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates

        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))
        self.h_layer = self._make_h_layer(n_atse_embed=0)

        self.encoder_mlp = FCLayers(
            n_in=code_dim, n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [], n_cont=n_cont,
            n_layers=2, n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False, use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

        self.register_buffer("junc2atse_sparse", torch.sparse_coo_tensor(size=(input_dim, 0)))
        self.register_buffer("atse_index_per_j", torch.zeros(input_dim, dtype=torch.long))
        self.n_atse = 0
        self.atse_embedding: nn.Parameter | None = None

    def _make_h_layer(self, n_atse_embed: int) -> nn.Sequential:
        in_dim = 1 + self.code_dim + n_atse_embed
        return nn.Sequential(
            nn.Linear(in_dim, self.h_hidden_dim),
            nn.LayerNorm(self.h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.h_hidden_dim, self.code_dim),
            nn.LayerNorm(self.code_dim),
            nn.ReLU(),
        )

    @torch.no_grad()
    def register_junc2atse(self, junc2atse: torch.sparse.Tensor):
        j2a = junc2atse.coalesce().to(self.feature_embedding.device)
        self.register_buffer("junc2atse_sparse", j2a)
        J = self.feature_embedding.shape[0]

        idx_p, idx_g = j2a.indices()
        if idx_g.numel() == 0:
            self.n_atse = 0
            self.atse_embedding = None
            self.atse_index_per_j.zero_()
            self.h_layer = self._make_h_layer(n_atse_embed=0)
            return

        self.n_atse = int(idx_g.max().item()) + 1
        atse_idx = torch.zeros(J, dtype=torch.long, device=j2a.device)
        atse_idx[idx_p] = idx_g
        self.register_buffer("atse_index_per_j", atse_idx)

        self.atse_embedding = nn.Parameter(
            torch.randn(self.n_atse, self.atse_embedding_dimension, device=j2a.device)
        )
        self.h_layer = self._make_h_layer(n_atse_embed=self.atse_embedding_dimension)

    def forward(
        self,
        x: torch.Tensor, mask: torch.Tensor,
        *cat_list: torch.Tensor, cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.atse_embedding is None:
            raise RuntimeError("Call `register_junc2atse(...)` before forward.")
        B, J = x.shape
        D = self.code_dim

        mask_bool = mask.bool()                                     # (B, J)
        F = self.feature_embedding                                   # (J, D)

        x_exp = x.unsqueeze(-1)                                      # (B, J, 1)
        F_exp = F.unsqueeze(0).expand(B, J, D)                       # (B, J, D)
        Ae   = self.atse_embedding[self.atse_index_per_j]            # (J, Ae)
        Ae_exp = Ae.unsqueeze(0).expand(B, J, -1)                    # (B, J, Ae)

        h_in  = torch.cat([x_exp, F_exp, Ae_exp], dim=-1)            # (B, J, 1+D+Ae)
        h_out = self.h_layer(h_in.view(B * J, -1)).view(B, J, D)     # (B, J, D)

        masked_h = h_out * mask_bool.unsqueeze(-1)
        pooled = masked_h.sum(dim=1)                                 # (B, D)
        if self.pool_mode == "mean":
            denom = mask_bool.sum(dim=1, keepdim=True).clamp(min=1)
            c = pooled / denom
        else:
            c = pooled

        mu_logvar = self.encoder_mlp(c, *cat_list, cont=cont)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar

class PartialEncoderEDDIATSEFaster(nn.Module):
    """
    Fast EDDI + ATSE encoder (batched) with cached `atse_index_per_j`.

    h-layer sees: [psi (1) | F_j (D) | E_atse(j) (Ae)], applied to all (B×J) then masked/pool.
    """
    def __init__(
        self,
        input_dim: int, code_dim: int, h_hidden_dim: int,
        encoder_hidden_dim: int, latent_dim: int,
        dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None, n_cont: int = 0,
        inject_covariates: bool = True,
        pool_mode: Literal["mean","sum"] = "mean",
        atse_embedding_dimension: int = 16,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.h_hidden_dim = h_hidden_dim
        self.dropout_rate = dropout_rate
        self.pool_mode = pool_mode
        self.atse_embedding_dimension = int(atse_embedding_dimension)
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates

        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))
        self.h_layer = self._make_h_layer(n_atse_embed=0)

        self.encoder_mlp = FCLayers(
            n_in=code_dim, n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [], n_cont=n_cont,
            n_layers=2, n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False, use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

        self.register_buffer("junc2atse_sparse", torch.sparse_coo_tensor(size=(input_dim, 0)))
        self.register_buffer("atse_index_per_j", torch.zeros(input_dim, dtype=torch.long))
        self.n_atse = 0
        self.atse_embedding: nn.Parameter | None = None

    def _make_h_layer(self, n_atse_embed: int) -> nn.Sequential:
        in_dim = 1 + self.code_dim + n_atse_embed
        return nn.Sequential(
            nn.Linear(in_dim, self.h_hidden_dim),
            nn.LayerNorm(self.h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.h_hidden_dim, self.code_dim),
            nn.LayerNorm(self.code_dim),
            nn.ReLU(),
        )

    @torch.no_grad()
    def register_junc2atse(self, junc2atse: torch.sparse.Tensor):
        j2a = junc2atse.coalesce().to(self.feature_embedding.device)
        self.register_buffer("junc2atse_sparse", j2a)
        J = self.feature_embedding.shape[0]

        idx_p, idx_g = j2a.indices()
        if idx_g.numel() == 0:
            self.n_atse = 0
            self.atse_embedding = None
            self.atse_index_per_j.zero_()
            self.h_layer = self._make_h_layer(n_atse_embed=0)
            return

        self.n_atse = int(idx_g.max().item()) + 1
        atse_idx = torch.zeros(J, dtype=torch.long, device=j2a.device)
        atse_idx[idx_p] = idx_g
        self.register_buffer("atse_index_per_j", atse_idx)

        self.atse_embedding = nn.Parameter(
            torch.randn(self.n_atse, self.atse_embedding_dimension, device=j2a.device)
        )
        self.h_layer = self._make_h_layer(n_atse_embed=self.atse_embedding_dimension)

    def forward(
        self,
        x: torch.Tensor, mask: torch.Tensor,
        *cat_list: torch.Tensor, cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.atse_embedding is None:
            raise RuntimeError("Call `register_junc2atse(...)` before forward.")
        B, J = x.shape
        D = self.code_dim
        device, dtype = x.device, x.dtype

        b_idx, j_idx = mask.bool().nonzero(as_tuple=True)
        N_obs = b_idx.numel()

        if N_obs == 0:
            pooled = torch.zeros(B, D, device=device, dtype=dtype)
            mu_logvar = self.encoder_mlp(pooled, *cat_list, cont=cont)
            mu, logvar = mu_logvar.chunk(2, dim=-1)
            return mu, logvar

        x_obs  = x[b_idx, j_idx].unsqueeze(1)                 # (N_obs, 1)
        F_obs  = self.feature_embedding.index_select(0, j_idx) # (N_obs, D)
        Ae_obs = self.atse_embedding[self.atse_index_per_j[j_idx]]  # (N_obs, Ae)

        h_in  = torch.cat([x_obs, F_obs, Ae_obs], dim=1)      # (N_obs, 1+D+Ae)
        h_obs = self.h_layer(h_in)                             # (N_obs, D)

        pooled = torch.zeros(B, D, device=device, dtype=h_obs.dtype)
        pooled.index_add_(0, b_idx, h_obs)

        if self.pool_mode == "mean":
            counts = torch.bincount(b_idx, minlength=B).to(pooled.dtype).view(B, 1).clamp_min_(1)
            pooled = pooled / counts

        mu_logvar = self.encoder_mlp(pooled, *cat_list, cont=cont)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar

class PartialEncoderWeightedSumEDDIMultiWeightFast(nn.Module):
    """
    Fast EDDI encoder with W weighted-sum heads (no ATSE), batched.

    Shapes / notation:
      B = batch size (cells)
      J = number of junctions
      D = code_dim
      W = number of heads / weight vectors
      Z = latent_dim

    """
    def __init__(
        self,
        input_dim: int,                  # J
        code_dim: int,                   # D
        h_hidden_dim: int,
        encoder_hidden_dim: int,
        latent_dim: int,                 # Z
        num_weight_vectors: int = 4,     # W
        dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None,
        n_cont: int = 0,
        inject_covariates: bool = True,
        temperature_value: float = 1.0,
        temperature_fixed: bool = True,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.num_weight_vectors = num_weight_vectors
        self.temperature_value = float(temperature_value)
        self.temperature_fixed = bool(temperature_fixed)
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates

        # Per-junction embedding table (J, D)
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        # h-layer: [psi (1) | F_j (D)] -> D
        in_dim_h = 1 + code_dim
        self.h_layer = nn.Sequential(
            nn.Linear(in_dim_h, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_hidden_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        # Gate network on per-junction codes: D -> W
        self.gate_net = nn.Sequential(
            nn.Linear(code_dim, code_dim // 2),
            nn.ReLU(),
            nn.Linear(code_dim // 2, self.num_weight_vectors),
        )

        # Combine W head vectors → D
        self.combiner = nn.Identity() if self.num_weight_vectors == 1 else nn.Sequential(
            nn.Linear(self.num_weight_vectors * code_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        # Cell-level encoder MLP → (mu, logvar)
        self.encoder_mlp = FCLayers(
            n_in=code_dim, n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [], n_cont=n_cont,
            n_layers=2, n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False, use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

    def forward(
        self,
        x: torch.Tensor,                 # (B, J)
        mask: torch.Tensor,              # (B, J)
        *cat_list: torch.Tensor,
        cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, J = x.shape
        D, W = self.code_dim, self.num_weight_vectors


        mask_bool = mask.bool()                          # (B, J)
        F = self.feature_embedding                       # (J, D)

        # h-layer inputs/outputs
        x_exp = x.unsqueeze(-1)                          # (B, J, 1)
        F_exp = F.unsqueeze(0).expand(B, J, D)           # (B, J, D)
        h_in  = torch.cat([x_exp, F_exp], dim=-1)        # (B, J, 1 + D)
        h_out = self.h_layer(h_in.view(B * J, -1)).view(B, J, D)  # (B, J, D)

        # Gate logits per junction/head
        raw_gates = self.gate_net(h_out)                 # (B, J, W)

        # Temperature / neighbor scaling
        if self.temperature_fixed:
            logits = raw_gates * self.temperature_value
        else:
            n_obs = mask_bool.sum(dim=1).clamp(min=1).float()   # (B,)
            logits = raw_gates * n_obs.rsqrt().view(B, 1, 1)

        # Mask unobserved junctions before softmax so they get zero weight
        logits = logits.masked_fill(~mask_bool.unsqueeze(-1), float("-1e9"))  # (B, J, W)

        # Softmax across junctions (dim=1)
        weights = torch.softmax(logits, dim=1)           # (B, J, W)

        # Also mask codes so unseen junctions never contribute numerically
        h_masked = h_out * mask_bool.unsqueeze(-1)       # (B, J, D)

        # Weighted sums per head: (B, W, D)
        head_sums = (weights.transpose(1, 2).unsqueeze(-1) * h_masked.unsqueeze(1)).sum(dim=2)

        # Combine heads → (B, D)
        combined = head_sums.squeeze(1) if W == 1 else self.combiner(head_sums.reshape(B, W * D))

        # Final per-cell projection → (mu, logvar)
        mu_logvar = self.encoder_mlp(combined, *cat_list, cont=cont)  # (B, 2Z)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar
    
class PartialEncoderWeightedSumEDDIMultiWeightFaster(nn.Module):
    """
    Fast EDDI encoder with W weighted-sum heads (no ATSE), batched.

    Shapes / notation:
      B = batch size (cells)
      J = number of junctions
      D = code_dim
      W = number of heads / weight vectors
      Z = latent_dim

    """
    def __init__(
        self,
        input_dim: int,                  # J
        code_dim: int,                   # D
        h_hidden_dim: int,
        encoder_hidden_dim: int,
        latent_dim: int,                 # Z
        num_weight_vectors: int = 4,     # W
        dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None,
        n_cont: int = 0,
        inject_covariates: bool = True,
        temperature_value: float = 1.0,
        temperature_fixed: bool = True,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.num_weight_vectors = num_weight_vectors
        self.temperature_value = float(temperature_value)
        self.temperature_fixed = bool(temperature_fixed)
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates

        # Per-junction embedding table (J, D)
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        # h-layer: [psi (1) | F_j (D)] -> D
        in_dim_h = 1 + code_dim
        self.h_layer = nn.Sequential(
            nn.Linear(in_dim_h, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_hidden_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        # Gate network on per-junction codes: D -> W
        self.gate_net = nn.Sequential(
            nn.Linear(code_dim, code_dim // 2),
            nn.ReLU(),
            nn.Linear(code_dim // 2, self.num_weight_vectors),
        )

        # Combine W head vectors → D
        self.combiner = nn.Identity() if self.num_weight_vectors == 1 else nn.Sequential(
            nn.Linear(self.num_weight_vectors * code_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        # Cell-level encoder MLP → (mu, logvar)
        self.encoder_mlp = FCLayers(
            n_in=code_dim, n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [], n_cont=n_cont,
            n_layers=2, n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False, use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

    def forward(
        self,
        x: torch.Tensor, mask: torch.Tensor,
        *cat_list: torch.Tensor, cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, J = x.shape
        D, W = self.code_dim, self.num_weight_vectors
        device, dtype = x.device, x.dtype

        b_idx, j_idx = mask.bool().nonzero(as_tuple=True)
        N_obs = b_idx.numel()

        # If nothing observed: zeros -> MLP
        if N_obs == 0:
            combined = torch.zeros(B, D, device=device, dtype=dtype)
            mu_logvar = self.encoder_mlp(combined, *cat_list, cont=cont)
            mu, logvar = mu_logvar.chunk(2, dim=-1)
            return mu, logvar

        # Gather observed inputs
        x_obs = x[b_idx, j_idx].unsqueeze(1)                  # (N_obs, 1)
        F_obs = self.feature_embedding.index_select(0, j_idx) # (N_obs, D)

        # Per-row code
        h_in  = torch.cat([x_obs, F_obs], dim=1)              # (N_obs, 1+D)
        h_obs = self.h_layer(h_in)                             # (N_obs, D)

        # Gate logits per observed row/head
        raw_gates = self.gate_net(h_obs)                       # (N_obs, W)

        # Temperature scaling
        if self.temperature_fixed:
            logits = raw_gates * self.temperature_value
        else:
            counts = torch.bincount(b_idx, minlength=B).clamp_min(1).float()  # (B,)
            scale_row = counts.rsqrt()[b_idx].unsqueeze(1)                     # (N_obs, 1)
            logits = raw_gates * scale_row

        # --- Grouped softmax across observed rows within each cell ---
        # 1) per-cell amax for stability
        max_logits = torch.full((B, W), float("-inf"), device=device, dtype=logits.dtype)
        max_logits.scatter_reduce_(0, b_idx.view(-1, 1).expand(-1, W), logits, reduce="amax")
        logits_centered = logits - max_logits[b_idx]        # (N_obs, W)

        # 2) exp and per-cell sums
        exp_logits = torch.exp(logits_centered)             # (N_obs, W)
        sum_exp = torch.zeros(B, W, device=device, dtype=exp_logits.dtype)
        sum_exp.index_add_(0, b_idx, exp_logits)            # (B, W)

        weights = exp_logits / (sum_exp[b_idx] + 1e-12)     # (N_obs, W)

        # Weighted h by head, then scatter-add per cell
        # (N_obs, W, D)
        h_weighted = weights.unsqueeze(-1) * h_obs.unsqueeze(1)

        # Accumulate to (B, W*D) in one index_add, then reshape -> (B, W, D)
        pooled_flat = torch.zeros(B, W * D, device=device, dtype=h_obs.dtype)
        pooled_flat.index_add_(0, b_idx, h_weighted.reshape(N_obs, W * D))
        pooled = pooled_flat.view(B, W, D)                  # (B, W, D)

        # Combine heads
        combined = pooled.squeeze(1) if W == 1 else self.combiner(pooled.reshape(B, W * D))

        # Final projection
        mu_logvar = self.encoder_mlp(combined, *cat_list, cont=cont)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar

class PartialEncoderWeightedSumEDDIMultiWeightATSEFast(nn.Module):
    """
    Fast weighted-sum EDDI + ATSE (batched) with cached `atse_index_per_j`.

    h-layer: [psi | F_j] -> D
    gate:    [h_out (D) | E_atse(j) (Ae)] -> W
    """
    def __init__(
        self,
        input_dim: int, code_dim: int, h_hidden_dim: int,
        encoder_hidden_dim: int, latent_dim: int,
        num_weight_vectors: int = 4, dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None, n_cont: int = 0,
        inject_covariates: bool = True,
        temperature_value: float = 1.0, temperature_fixed: bool = True,
        atse_embedding_dimension: int = 16,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.num_weight_vectors = num_weight_vectors
        self.temperature_value = float(temperature_value)
        self.temperature_fixed = bool(temperature_fixed)
        self.atse_embedding_dimension = int(atse_embedding_dimension)
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates

        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        in_dim_h = 1 + code_dim
        self.h_layer = nn.Sequential(
            nn.Linear(in_dim_h, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_hidden_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        in_dim_gate = code_dim + self.atse_embedding_dimension
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim_gate, max(in_dim_gate // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(in_dim_gate // 2, 1), self.num_weight_vectors),
        )

        self.combiner = nn.Identity() if self.num_weight_vectors == 1 else nn.Sequential(
            nn.Linear(self.num_weight_vectors * code_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        self.encoder_mlp = FCLayers(
            n_in=code_dim, n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [], n_cont=n_cont,
            n_layers=2, n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False, use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

        self.register_buffer("junc2atse_sparse", torch.sparse_coo_tensor(size=(input_dim, 0)))
        self.register_buffer("atse_index_per_j", torch.zeros(input_dim, dtype=torch.long))
        self.n_atse = 0
        self.atse_embedding: nn.Parameter | None = None

    @torch.no_grad()
    def register_junc2atse(self, junc2atse: torch.sparse.Tensor):
        j2a = junc2atse.coalesce().to(self.feature_embedding.device)
        self.register_buffer("junc2atse_sparse", j2a)
        J = self.feature_embedding.shape[0]

        idx_p, idx_g = j2a.indices()
        if idx_g.numel() == 0:
            self.n_atse = 0
            self.atse_embedding = None
            self.atse_index_per_j.zero_()
            return

        self.n_atse = int(idx_g.max().item()) + 1
        atse_idx = torch.zeros(J, dtype=torch.long, device=j2a.device)
        atse_idx[idx_p] = idx_g
        self.register_buffer("atse_index_per_j", atse_idx)

        self.atse_embedding = nn.Parameter(
            torch.randn(self.n_atse, self.atse_embedding_dimension, device=j2a.device)
        )

    def forward(
        self,
        x: torch.Tensor, mask: torch.Tensor,
        *cat_list: torch.Tensor, cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.atse_embedding is None:
            raise RuntimeError("Call `register_junc2atse(...)` before forward.")
        B, J = x.shape
        D, W = self.code_dim, self.num_weight_vectors

        mask_bool = mask.bool()                        # (B, J)
        F = self.feature_embedding                     # (J, D)

        x_exp  = x.unsqueeze(-1)                       # (B, J, 1)
        F_exp  = F.unsqueeze(0).expand(B, J, D)        # (B, J, D)
        h_in   = torch.cat([x_exp, F_exp], dim=-1)     # (B, J, 1+D)
        h_out  = self.h_layer(h_in.view(B * J, -1)).view(B, J, D)  # (B, J, D)

        Ae     = self.atse_embedding[self.atse_index_per_j]        # (J, Ae)
        Ae_exp = Ae.unsqueeze(0).expand(B, J, -1)                  # (B, J, Ae)
        gate_in = torch.cat([h_out, Ae_exp], dim=-1)               # (B, J, D+Ae)
        raw_gates = self.gate_net(gate_in)                         # (B, J, W)

        if self.temperature_fixed:
            logits = raw_gates * self.temperature_value
        else:
            n_obs = mask_bool.sum(dim=1).clamp(min=1).float()      # (B,)
            logits = raw_gates * n_obs.rsqrt().view(B, 1, 1)

        logits = logits.masked_fill(~mask_bool.unsqueeze(-1), float("-1e9"))
        weights = torch.softmax(logits, dim=1)                     # (B, J, W)

        h_masked = h_out * mask_bool.unsqueeze(-1)                 # (B, J, D)
        head_sums = (weights.transpose(1, 2).unsqueeze(-1) * h_masked.unsqueeze(1)).sum(dim=2)  # (B, W, D)

        combined = head_sums.squeeze(1) if W == 1 else self.combiner(head_sums.reshape(B, W * D))
        mu_logvar = self.encoder_mlp(combined, *cat_list, cont=cont)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar
    
class PartialEncoderWeightedSumEDDIMultiWeightATSEFaster(nn.Module):
    """
    Fast weighted-sum EDDI + ATSE (batched) with cached `atse_index_per_j`.

    h-layer: [psi | F_j] -> D
    gate:    [h_out (D) | E_atse(j) (Ae)] -> W
    """
    def __init__(
        self,
        input_dim: int, code_dim: int, h_hidden_dim: int,
        encoder_hidden_dim: int, latent_dim: int,
        num_weight_vectors: int = 4, dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None, n_cont: int = 0,
        inject_covariates: bool = True,
        temperature_value: float = 1.0, temperature_fixed: bool = True,
        atse_embedding_dimension: int = 16,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.num_weight_vectors = num_weight_vectors
        self.temperature_value = float(temperature_value)
        self.temperature_fixed = bool(temperature_fixed)
        self.atse_embedding_dimension = int(atse_embedding_dimension)
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates

        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        in_dim_h = 1 + code_dim
        self.h_layer = nn.Sequential(
            nn.Linear(in_dim_h, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_hidden_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        in_dim_gate = code_dim + self.atse_embedding_dimension
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim_gate, max(in_dim_gate // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(in_dim_gate // 2, 1), self.num_weight_vectors),
        )

        self.combiner = nn.Identity() if self.num_weight_vectors == 1 else nn.Sequential(
            nn.Linear(self.num_weight_vectors * code_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        self.encoder_mlp = FCLayers(
            n_in=code_dim, n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [], n_cont=n_cont,
            n_layers=2, n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False, use_layer_norm=True,
            inject_covariates=inject_covariates,
        )

        self.register_buffer("junc2atse_sparse", torch.sparse_coo_tensor(size=(input_dim, 0)))
        self.register_buffer("atse_index_per_j", torch.zeros(input_dim, dtype=torch.long))
        self.n_atse = 0
        self.atse_embedding: nn.Parameter | None = None

    @torch.no_grad()
    def register_junc2atse(self, junc2atse: torch.sparse.Tensor):
        j2a = junc2atse.coalesce().to(self.feature_embedding.device)
        self.register_buffer("junc2atse_sparse", j2a)
        J = self.feature_embedding.shape[0]

        idx_p, idx_g = j2a.indices()
        if idx_g.numel() == 0:
            self.n_atse = 0
            self.atse_embedding = None
            self.atse_index_per_j.zero_()
            return

        self.n_atse = int(idx_g.max().item()) + 1
        atse_idx = torch.zeros(J, dtype=torch.long, device=j2a.device)
        atse_idx[idx_p] = idx_g
        self.register_buffer("atse_index_per_j", atse_idx)

        self.atse_embedding = nn.Parameter(
            torch.randn(self.n_atse, self.atse_embedding_dimension, device=j2a.device)
        )

    def forward(
        self,
        x: torch.Tensor, mask: torch.Tensor,
        *cat_list: torch.Tensor, cont: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.atse_embedding is None:
            raise RuntimeError("Call `register_junc2atse(...)` before forward.")
        B, J = x.shape
        D, W = self.code_dim, self.num_weight_vectors
        device, dtype = x.device, x.dtype

        b_idx, j_idx = mask.bool().nonzero(as_tuple=True)
        N_obs = b_idx.numel()

        if N_obs == 0:
            combined = torch.zeros(B, D, device=device, dtype=dtype)
            mu_logvar = self.encoder_mlp(combined, *cat_list, cont=cont)
            mu, logvar = mu_logvar.chunk(2, dim=-1)
            return mu, logvar

        # Observed inputs
        x_obs  = x[b_idx, j_idx].unsqueeze(1)                  # (N_obs, 1)
        F_obs  = self.feature_embedding.index_select(0, j_idx) # (N_obs, D)
        h_in   = torch.cat([x_obs, F_obs], dim=1)              # (N_obs, 1+D)
        h_obs  = self.h_layer(h_in)                             # (N_obs, D)

        Ae_obs = self.atse_embedding[self.atse_index_per_j[j_idx]]  # (N_obs, Ae)
        gate_in = torch.cat([h_obs, Ae_obs], dim=1)                 # (N_obs, D+Ae)
        raw_gates = self.gate_net(gate_in)                           # (N_obs, W)

        # Temperature scaling
        if self.temperature_fixed:
            logits = raw_gates * self.temperature_value
        else:
            counts = torch.bincount(b_idx, minlength=B).clamp_min(1).float()  # (B,)
            scale_row = counts.rsqrt()[b_idx].unsqueeze(1)                     # (N_obs, 1)
            logits = raw_gates * scale_row

        # Grouped softmax per cell (over observed rows)
        max_logits = torch.full((B, W), float("-inf"), device=device, dtype=logits.dtype)
        max_logits.scatter_reduce_(0, b_idx.view(-1, 1).expand(-1, W), logits, reduce="amax")
        logits_centered = logits - max_logits[b_idx]            # (N_obs, W)

        exp_logits = torch.exp(logits_centered)                 # (N_obs, W)
        sum_exp = torch.zeros(B, W, device=device, dtype=exp_logits.dtype)
        sum_exp.index_add_(0, b_idx, exp_logits)                # (B, W)
        weights = exp_logits / (sum_exp[b_idx] + 1e-12)         # (N_obs, W)

        # Weighted sums -> (B, W, D)
        h_weighted = weights.unsqueeze(-1) * h_obs.unsqueeze(1) # (N_obs, W, D)
        pooled_flat = torch.zeros(B, W * D, device=device, dtype=h_obs.dtype)
        pooled_flat.index_add_(0, b_idx, h_weighted.reshape(N_obs, W * D))
        pooled = pooled_flat.view(B, W, D)

        # Combine heads -> (B, D)
        combined = pooled.squeeze(1) if W == 1 else self.combiner(pooled.reshape(B, W * D))

        mu_logvar = self.encoder_mlp(combined, *cat_list, cont=cont)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar

class LinearDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        n_cat_list: Iterable[int] | None = None,
        n_cont: int = 0,
    ):
        super().__init__()
        # track covariates exactly as in encoder
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont     = n_cont
        total_cov       = sum(self.n_cat_list) + self.n_cont

        # single linear layer: input = [z, covariates]
        self.linear = nn.Linear(latent_dim + total_cov, output_dim)

    def forward(
        self,
        z: torch.Tensor,              # (B, latent_dim)
        *cat_list: torch.Tensor,      # each (B,1)
        cont: torch.Tensor | None = None,  # (B, n_cont)
    ) -> torch.Tensor:
        
        # if self.n_cat_list:
        #     batch_idx = cat_list[0]
        #     print("▶ LinearDecoder: batch_index unique →", torch.unique(batch_idx).tolist())
        # if cont is not None:
        #     print("▶ LinearDecoder: cont_covs shape →", cont.shape)

        # rebuild the same covariate concatenation
        covs: list[torch.Tensor] = []
        if cont is not None:
            covs.append(cont)
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            # one-hot encode each categorical and append
            oh = F.one_hot(cat.squeeze(-1), n_cat).float()
            covs.append(oh)
        # final input to linear: (B, latent_dim + total_cov)
        inp = torch.cat([z, *covs], dim=1) if covs else z
        return self.linear(inp)

class PARTIALVAE(BaseModuleClass):
    """
    Partial Variational autoencoder module for splicing data in scvi-tools.

    Parameters
    ----------
    n_input : int
        Number of splicing features (junctions).
    n_batch : int
        Number of batches; 0 = no batch correction.
    n_labels : int
        Number of labels; 0 = no label correction.
    n_hidden : int
        Hidden size for internal representations (unused directly here).
    n_latent : int
        Dimension of the latent space Z.
    n_continuous_cov : int
        Number of continuous covariates to include.
    n_cats_per_cov : list[int] | None
        Number of categories for each categorical covariate 
        (e.g. `[n_batch] + other_cats`).
    dropout_rate : float
        Dropout rate applied in encoder/decoder.
    splice_likelihood : {"binomial","beta_binomial", "dirichlet_multinomial"}
        Which reconstruction loss to use.
    latent_distribution : {"normal","ln"}
        Latent prior type.
    encode_covariates : bool
        Whether to concatenate covariates to **encoder** inputs.
    deeply_inject_covariates : bool
        Whether to concatenate covariates to **decoder** inputs.
    batch_representation : {"one-hot","embedding"}
        How to treat the batch covariate internally.
    use_batch_norm : {"none","encoder","decoder","both"}
        Where to apply batch normalization.
    use_layer_norm : {"none","encoder","decoder","both"}
        Where to apply layer normalization.
    extra_payload_autotune : bool
        Return extra payload for autotune (advanced).
    code_dim : int
        Dimensionality of per-feature embeddings in the encoder.
    h_hidden_dim : int
        Hidden size for the shared “h” layer in the encoder.
    encoder_hidden_dim : int
        Hidden size for the final encoder MLP.
    learn_concentration : bool
        If True, learn Beta‐Binomial concentration parameter.
    """
    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_continuous_cov: int = 0,
        n_cats_per_cov: list[int] | None = None,
        dropout_rate: float = 0.1,
        splice_likelihood: Literal["binomial", "beta_binomial"] = "beta_binomial",
        latent_distribution: Literal["normal", "ln"] = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        batch_representation: Literal["one-hot", "embedding"] = "one-hot",
        use_batch_norm: Literal["none", "encoder", "decoder", "both"] = "both",
        use_layer_norm: Literal["none", "encoder", "decoder", "both"] = "none",
        extra_payload_autotune: bool = False,
        code_dim: int = 16,
        h_hidden_dim: int = 64,
        encoder_hidden_dim: int = 128,
        learn_concentration: bool = True,
        n_heads: int = 4,
        ff_dim: int | None = None,
        num_transformer_layers: int = 2,
        encoder_type: Literal[
            "PartialEncoderEDDI",
            "PartialEncoderEDDIATSE",
            "PartialEncoderWeightedSumEDDIMultiWeight",
            "PartialEncoderWeightedSumEDDIMultiWeightATSE",
        ] = "PartialEncoderEDDI",
        pool_mode: Literal["mean","sum"] = "mean",
        num_weight_vectors: int = 4,
        temperature_value: float = -1.0, #if temperature_value is set to -1, then the median number of observations is used as the fixed temperature value.
        temperature_fixed: bool = True, #if temperature is fixed, it is fixed to the value of temperature_value
        forward_style: Literal["per-cell", "batched", "scatter"] = "batched",
        atse_embedding_dimension: int = 16,
        max_nobs: int = -1,
        ):
        super().__init__()
        self.encoder_type = encoder_type

        # Store high-level flags
        self.encode_covariates = encode_covariates
        self.deeply_inject_covariates = deeply_inject_covariates

        # Reconstruction / prior settings
        self.splice_likelihood = splice_likelihood
        self.latent_distribution = latent_distribution


        self.learn_concentration = learn_concentration
        self.extra_payload_autotune = extra_payload_autotune

        # AnnData dimensions
        self.input_dim = n_input
        self.n_continuous_cov = n_continuous_cov
        self.n_cats_per_cov = n_cats_per_cov or []

        cat_list = [n_batch] + list(n_cats_per_cov) if n_cats_per_cov is not None else []
        encoder_cat_list = cat_list if encode_covariates else None

        # ——— shared concentration parameter ϕ ———
        # initialize to log(100) ≈ 4.6
        if learn_concentration:
            self.log_concentration = nn.Parameter(torch.tensor(4.6))
        else:
            self.log_concentration = None

        # Instantiate encoder + decoder, passing covariate specs

        print(f"Using code_dim={code_dim}!")
        print(f"Using latent_dim={n_latent}")
        print(f"Forward style: {forward_style}")

        # instantiate the requested encoder
        if forward_style == "per-cell":
            if encoder_type == "PartialEncoderEDDI":
                print(f"Using EDDI Partial Encoder")
                self.encoder = PartialEncoderEDDI(
                    input_dim=n_input,
                    code_dim=code_dim,
                    h_hidden_dim=h_hidden_dim,
                    encoder_hidden_dim=encoder_hidden_dim,
                    latent_dim=n_latent,
                    dropout_rate=dropout_rate,
                    n_cat_list=encoder_cat_list,
                    n_cont=n_continuous_cov,
                    inject_covariates=encode_covariates,
                    pool_mode=pool_mode,
                )

            elif encoder_type == "PartialEncoderEDDIATSE":
                print("Using EDDI + ATSE Partial Encoder")
                self.encoder = PartialEncoderEDDIATSE(
                    input_dim=n_input,
                    code_dim=code_dim,
                    h_hidden_dim=h_hidden_dim,
                    encoder_hidden_dim=encoder_hidden_dim,
                    latent_dim=n_latent,
                    dropout_rate=dropout_rate,
                    n_cat_list=encoder_cat_list,
                    n_cont=n_continuous_cov,
                    inject_covariates=encode_covariates,
                    pool_mode=pool_mode,
                    atse_embedding_dimension=atse_embedding_dimension
                )
            
            elif encoder_type == "PartialEncoderWeightedSumEDDIMultiWeight":
                print("Using PartialEncoderWeightedSumEDDIMultiWeight")
                self.encoder = PartialEncoderWeightedSumEDDIMultiWeight(
                    input_dim=n_input,
                    code_dim=code_dim,
                    h_hidden_dim=h_hidden_dim,
                    encoder_hidden_dim=encoder_hidden_dim,
                    latent_dim=n_latent,
                    dropout_rate=dropout_rate,
                    n_cat_list=encoder_cat_list,
                    n_cont=n_continuous_cov,
                    inject_covariates=encode_covariates,
                    num_weight_vectors = num_weight_vectors,
                    temperature_value=temperature_value,
                    temperature_fixed=temperature_fixed,
                )

            elif encoder_type == "PartialEncoderWeightedSumEDDIMultiWeightATSE":
                print("Using PartialEncoderWeightedSumEDDIMultiWeightATSE")
                self.encoder = PartialEncoderWeightedSumEDDIMultiWeightATSE(
                    input_dim=n_input,
                    code_dim=code_dim,
                    h_hidden_dim=h_hidden_dim,
                    encoder_hidden_dim=encoder_hidden_dim,
                    latent_dim=n_latent,
                    dropout_rate=dropout_rate,
                    n_cat_list=encoder_cat_list,
                    n_cont=n_continuous_cov,
                    inject_covariates=encode_covariates,
                    num_weight_vectors = num_weight_vectors,
                    temperature_value=temperature_value,
                    temperature_fixed=temperature_fixed,
                    atse_embedding_dimension=atse_embedding_dimension
                )
        elif forward_style == "batched":
            if encoder_type == "PartialEncoderEDDI":
                print(f"Using EDDI Partial Encoder Fast")
                self.encoder = PartialEncoderEDDIFast(
                    input_dim=n_input,
                    code_dim=code_dim,
                    h_hidden_dim=h_hidden_dim,
                    encoder_hidden_dim=encoder_hidden_dim,
                    latent_dim=n_latent,
                    dropout_rate=dropout_rate,
                    n_cat_list=encoder_cat_list,
                    n_cont=n_continuous_cov,
                    inject_covariates=encode_covariates,
                    pool_mode=pool_mode,
                )

            elif encoder_type == "PartialEncoderEDDIATSE":
                print("Using EDDI + ATSE Partial Encoder Fast")
                self.encoder = PartialEncoderEDDIATSEFast(
                    input_dim=n_input,
                    code_dim=code_dim,
                    h_hidden_dim=h_hidden_dim,
                    encoder_hidden_dim=encoder_hidden_dim,
                    latent_dim=n_latent,
                    dropout_rate=dropout_rate,
                    n_cat_list=encoder_cat_list,
                    n_cont=n_continuous_cov,
                    inject_covariates=encode_covariates,
                    pool_mode=pool_mode,
                    atse_embedding_dimension=atse_embedding_dimension
                )
            
            elif encoder_type == "PartialEncoderWeightedSumEDDIMultiWeight":
                print("Using PartialEncoderWeightedSumEDDIMultiWeight Fast")
                self.encoder = PartialEncoderWeightedSumEDDIMultiWeightFast(
                    input_dim=n_input,
                    code_dim=code_dim,
                    h_hidden_dim=h_hidden_dim,
                    encoder_hidden_dim=encoder_hidden_dim,
                    latent_dim=n_latent,
                    dropout_rate=dropout_rate,
                    n_cat_list=encoder_cat_list,
                    n_cont=n_continuous_cov,
                    inject_covariates=encode_covariates,
                    num_weight_vectors = num_weight_vectors,
                    temperature_value=temperature_value,
                    temperature_fixed=temperature_fixed,
                )

            elif encoder_type == "PartialEncoderWeightedSumEDDIMultiWeightATSE":
                print("Using PartialEncoderWeightedSumEDDIMultiWeightATSE Fast")
                self.encoder = PartialEncoderWeightedSumEDDIMultiWeightATSEFast(
                    input_dim=n_input,
                    code_dim=code_dim,
                    h_hidden_dim=h_hidden_dim,
                    encoder_hidden_dim=encoder_hidden_dim,
                    latent_dim=n_latent,
                    dropout_rate=dropout_rate,
                    n_cat_list=encoder_cat_list,
                    n_cont=n_continuous_cov,
                    inject_covariates=encode_covariates,
                    num_weight_vectors = num_weight_vectors,
                    temperature_value=temperature_value,
                    temperature_fixed=temperature_fixed,
                    atse_embedding_dimension=atse_embedding_dimension
                )
        elif forward_style == "scatter":
            if encoder_type == "PartialEncoderEDDI":
                print(f"Using EDDI Partial Encoder Faster")
                self.encoder = PartialEncoderEDDIFaster(
                    input_dim=n_input,
                    code_dim=code_dim,
                    h_hidden_dim=h_hidden_dim,
                    encoder_hidden_dim=encoder_hidden_dim,
                    latent_dim=n_latent,
                    dropout_rate=dropout_rate,
                    n_cat_list=encoder_cat_list,
                    n_cont=n_continuous_cov,
                    inject_covariates=encode_covariates,
                    pool_mode=pool_mode,
                    max_nobs = max_nobs,
                )

            elif encoder_type == "PartialEncoderEDDIATSE":
                print("Using EDDI + ATSE Partial Encoder Faster")
                self.encoder = PartialEncoderEDDIATSEFaster(
                    input_dim=n_input,
                    code_dim=code_dim,
                    h_hidden_dim=h_hidden_dim,
                    encoder_hidden_dim=encoder_hidden_dim,
                    latent_dim=n_latent,
                    dropout_rate=dropout_rate,
                    n_cat_list=encoder_cat_list,
                    n_cont=n_continuous_cov,
                    inject_covariates=encode_covariates,
                    pool_mode=pool_mode,
                    atse_embedding_dimension=atse_embedding_dimension,
                )

            elif encoder_type == "PartialEncoderWeightedSumEDDIMultiWeight":
                print("Using PartialEncoderWeightedSumEDDIMultiWeight Faster")
                self.encoder = PartialEncoderWeightedSumEDDIMultiWeightFaster(
                    input_dim=n_input,
                    code_dim=code_dim,
                    h_hidden_dim=h_hidden_dim,
                    encoder_hidden_dim=encoder_hidden_dim,
                    latent_dim=n_latent,
                    dropout_rate=dropout_rate,
                    n_cat_list=encoder_cat_list,
                    n_cont=n_continuous_cov,
                    inject_covariates=encode_covariates,
                    num_weight_vectors=num_weight_vectors,
                    temperature_value=temperature_value,
                    temperature_fixed=temperature_fixed,
                )

            elif encoder_type == "PartialEncoderWeightedSumEDDIMultiWeightATSE":
                print("Using PartialEncoderWeightedSumEDDIMultiWeightATSE Faster")
                self.encoder = PartialEncoderWeightedSumEDDIMultiWeightATSEFaster(
                    input_dim=n_input,
                    code_dim=code_dim,
                    h_hidden_dim=h_hidden_dim,
                    encoder_hidden_dim=encoder_hidden_dim,
                    latent_dim=n_latent,
                    dropout_rate=dropout_rate,
                    n_cat_list=encoder_cat_list,
                    n_cont=n_continuous_cov,
                    inject_covariates=encode_covariates,
                    num_weight_vectors=num_weight_vectors,
                    temperature_value=temperature_value,
                    temperature_fixed=temperature_fixed,
                    atse_embedding_dimension=atse_embedding_dimension,
                )

    
        else:
            raise ValueError(f"Unknown encoder_type={encoder_type!r}")
            
        if latent_distribution == "ln":
            self.encoder.z_transformation = nn.Softmax(dim=-1)
        else:
            self.encoder.z_transformation = lambda x: x

        self.decoder = LinearDecoder(
            latent_dim=n_latent,
            output_dim=n_input,
            n_cat_list=cat_list,
            n_cont=n_continuous_cov,
        )

    @auto_move_data
    def _get_inference_input(self, tensors):
        return {
            MODULE_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY],        # your junction ratios
            "mask":           tensors.get(REGISTRY_KEYS.PSI_MASK_KEY),
            "batch_index":    tensors[REGISTRY_KEYS.BATCH_KEY],     # integer batch code
            "cat_covs":       tensors.get(REGISTRY_KEYS.CAT_COVS_KEY),  # stacked one‐hot columns
            "cont_covs":      tensors.get(REGISTRY_KEYS.CONT_COVS_KEY), # numeric covariates
        }

    @auto_move_data
    def inference(
        self,
        x,
        mask,
        batch_index,
        cat_covs=None,
        cont_covs=None,
        n_samples=1,
    ) -> dict[str, torch.Tensor]:
        # 1) prepare encoder inputs (concatenate continuous covariates if requested)
        if cont_covs is not None and self.encode_covariates:
            x_input = torch.cat([x, cont_covs], dim=-1)
        else:
            x_input = x

        # 2) split out any one-hot categorical covariates
        if cat_covs is not None and self.encode_covariates:
            cat_list = torch.split(cat_covs, 1, dim=1)
        else:
            cat_list = ()

        # 3) run encoder → returns (mu, raw_logvar)
        #    we still pass batch_index as the first “category”
        mu, raw_logvar = self.encoder(
            x_input, mask, batch_index, *cat_list, cont=cont_covs
        )

        # --- ensure logvar is in a safe range ---
        # prevents under/overflow in exp()
        logvar = torch.clamp(raw_logvar, min=-10.0, max=10.0)

        # 4) build the posterior Normal
        var   = torch.exp(logvar)         # variance = exp(logvar) > 0
        sigma = torch.sqrt(var)           # standard deviation
        qz    = Normal(mu, sigma)

        # 5) sample (rsample so that gradients flow)
        if n_samples == 1:
            z = qz.rsample()
        else:
            z = qz.sample((n_samples,))

        # 6) apply any “z_transformation” (e.g. identity or softmax)
        z = self.encoder.z_transformation(z)

        return {
            "z":    z,      # (B, Z) or (n_samples, B, Z)
            "qz_m": mu,     # posterior means
            "qz_v": var,    # **variance**, not logvar
            "x":    x,
        }


    def _get_generative_input(self, tensors, inference_outputs):
        return {
            MODULE_KEYS.Z_KEY:   inference_outputs["z"],
            "qz_m":              inference_outputs["qz_m"],      # <<< add this
            "batch_index":       tensors[REGISTRY_KEYS.BATCH_KEY],
            "cat_covs":          tensors.get(REGISTRY_KEYS.CAT_COVS_KEY),
            "cont_covs":         tensors.get(REGISTRY_KEYS.CONT_COVS_KEY),
        }

    @auto_move_data
    def generative(
        self,
        z,
        qz_m,
        batch_index,
        cat_covs=None,
        cont_covs=None,
        use_z_mean=False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # if you want to use the posterior mean:
        if use_z_mean:
            z = qz_m

        # again split cat_covs → tuple
        if cat_covs is not None:
            cat_list = torch.split(cat_covs, 1, dim=1)
        else:
            cat_list = ()

        # prepare decoder input by concatenating cont_covs if desired
        decoder_input = (
            torch.cat([z, cont_covs], dim=-1)
            if (cont_covs is not None and self.deeply_inject_covariates)
            else z
        )

        # finally call your decoder, passing batch_index first:
        reconstruction = self.decoder(decoder_input, batch_index, *cat_list, cont=cont_covs)

        return {"reconstruction": reconstruction}   # now shape (J,)



    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
        kl_weight: float | torch.Tensor = 1.0,
    ) -> LossOutput:
        # 1) unpack data
        x = tensors[REGISTRY_KEYS.X_KEY]                      # (B, D) input ratios (unused here)
        mask = tensors.get(REGISTRY_KEYS.PSI_MASK_KEY, None)  # (B, D) binary
        junc = tensors["junction_counts"]                     # (B, D) successes
        clus = tensors["cluster_counts"]                      # (B, D) trials

        # 2) reconstruction
        p = generative_outputs["reconstruction"]  # (B, D) Bernoulli probabilities
        if self.splice_likelihood == "binomial":
            reconst = binomial_loss_function(
                logits=p,
                junction_counts=junc,
                cluster_counts=clus,
                n=x.numel(),
                k=x.shape[0],
                mask=mask,
            )

        elif self.splice_likelihood == "beta_binomial":
            concentration = torch.exp(self.log_concentration)
            reconst = beta_binomial_loss_function(
                logits=p,
                junction_counts=junc,
                cluster_counts=clus,
                n=x.numel(),
                k=x.shape[0],
                concentration=concentration,
                mask=mask,
            )

        elif self.splice_likelihood == "dirichlet_multinomial":
            # --- prepare inputs ---
            # junc:        (B,D)  = counts per junction per cell
            # clus:        (B,D)  = repeated ATSE totals per junction per cell
            # self.junc2atse: sparse (D, G)

            # decoder output p is currently a “logit” tensor of shape (B,D)
            logits = p

            # we need atse_counts of shape (B, D), which you already have in `clus`.
            atse_counts = clus

            # concentration scalar fixed at 10.0 for now
            
            concentration = torch.exp(self.log_concentration)

            # --- turn logits into per‐group softmax‐logits ---
            # group_logsumexp and subtract_group_logsumexp expect (N,P)→(N,G)
            lse = group_logsumexp(self.junc2atse, logits)            # → (N,G)
            softmax_logits = subtract_group_logsumexp(
                self.junc2atse, logits, lse
            )                                                           # → (N,P)

            # build alpha = conc * exp(softmax_logits)
            alpha = concentration * torch.exp(softmax_logits)

            # compute the DM log‐likelihood
            ll = dirichlet_multinomial_likelihood(
                counts=junc,
                atse_counts=atse_counts,
                junc2atse=self.junc2atse,
                alpha=alpha,
                mask = mask,
            )
            # negative mean‐log‐lik
            reconst = -ll

        else:
            raise ValueError(f"Unknown splice_likelihood={self.splice_likelihood}")

        # reconst is a scalar: -E_q [ log p(x|z) ]

        # 3) KL divergence
        mu = inference_outputs["qz_m"]   # (B, Z)
        var = inference_outputs["qz_v"]  # (B, Z) = σ²
        sigma = torch.sqrt(var)          # (B, Z)
        qz = Normal(mu, sigma)
        pz = Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        kl_local = kl_divergence(qz, pz).sum(dim=1)  # (B,) sum over latent dims

        # 4) total ELBO
        #    average KL over batch, add to reconst
        total = reconst + kl_weight * kl_local.mean()

        # 5) wrap in LossOutput
        return LossOutput(
            loss=total,
            reconstruction_loss=reconst,
            kl_local=kl_local,
            n_obs_minibatch=x.shape[0],
        )

    @torch.inference_mode()
    def sample(self, tensors: Dict[str, torch.Tensor], n_samples: int = 1) -> torch.Tensor:
        _, gen_out = self.forward(tensors, compute_loss=False)
        return gen_out["reconstruction"].cpu() 
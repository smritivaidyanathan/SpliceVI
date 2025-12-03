from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Literal, Dict, Any, Sequence

import numpy as np
import torch
from torch.distributions import Normal
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._constants import ADATA_MINIFY_TYPE

from scvi.data.fields import (
    LayerField,
    CategoricalObsField,
    NumericalObsField,
    NumericalJointObsField,
)
from scvi.model.base import (
    BaseModelClass,
    UnsupervisedTrainingMixin,
    VAEMixin,
)
from partialvae import PARTIALVAE
from scvi.utils import setup_anndata_dsp

from scvi.train import TrainRunner, TrainingPlan
from scvi.utils._docstrings import de_dsp, devices_dsp, setup_anndata_dsp
from scvi.train._callbacks import SaveBestState

from scvi.model._utils import (
    use_distributed_sampler,
)

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)
import pandas as pd 

from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sp


class EDDISPLICE(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    _module_cls = PARTIALVAE

    def __init__(
        self,
        adata: AnnData | None = None,
        code_dim: int = 16,
        h_hidden_dim: int = 64,
        encoder_hidden_dim: int = 128,
        encoder_n_layers: int = 2,
        latent_dim: int = 10,
        dropout_rate: float = 0.1,
        learn_concentration: bool = True,
        splice_likelihood: Literal["binomial", "beta_binomial", "dirichlet_multinomial"] = "dirichlet_multinomial",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        initialize_embeddings_from_pca: bool = True,
        pool_mode: Literal["mean", "sum"] = "mean",
        max_nobs: int = -1,
        latent_distribution: Literal["normal", "ln"] = "normal",
        **kwargs,
    ):
        """
        Initialize an EDDISPLICE model wrapping a PARTIALVAE with the
        `PartialEncoderEDDIFaster` encoder and a linear decoder.

        This model learns a low-dimensional latent representation of junction-level
        splicing usage from an AnnData object set up via
        :meth:`EDDISPLICE.setup_anndata`. The encoder uses per-junction embeddings
        plus an EDDI-style partial observation mechanism that only aggregates
        observed junctions per cell. The reconstruction head supports binomial,
        beta-binomial, or Dirichlet–multinomial likelihoods.

        Parameters
        ----------
        adata
            AnnData object with layers/fields registered using
            :meth:`EDDISPLICE.setup_anndata`. If ``None``, the module is
            initialized lazily at training time.
        code_dim
            Dimensionality of the per-junction embedding table used by the encoder.
        h_hidden_dim
            Hidden size of the shared "h" MLP that processes each observed
            junction embedding + PSI value before pooling across junctions.
        encoder_hidden_dim
            Hidden size of the final encoder MLP that maps the pooled
            per-cell representation to the mean and variance of the latent
            Gaussian.
        encoder_n_layers
            Number of hidden layers in the encoder MLP (passed to
            :class:`scvi.nn.FCLayers`).
        latent_dim
            Dimensionality of the latent space :math:`z`.
        dropout_rate
            Dropout probability applied in the encoder networks.
        learn_concentration
            If ``True``, learn a global concentration parameter used in the
            beta-binomial or Dirichlet–multinomial reconstruction likelihood.
        splice_likelihood
            Reconstruction likelihood for junction counts. One of:

            * ``"binomial"``: simple binomial over junction vs. cluster counts.
            * ``"beta_binomial"``: beta-binomial with learned concentration.
            * ``"dirichlet_multinomial"``: hierarchical Dirichlet–multinomial
              over ATSE-level totals using a junction→ATSE mapping.
        encode_covariates
            If ``True``, concatenates observed covariates (batch and any extra
            covariates registered via ``setup_anndata``) into the encoder input.
        deeply_inject_covariates
            If ``True``, passes covariates into the decoder so that reconstruction
            can be conditioned on batch / covariates as well as :math:`z`.
        initialize_embeddings_from_pca
            If ``True`` and ``adata`` is provided, initialize the per-junction
            embedding table with a truncated SVD (PCA) fit to the registered
            splicing layer instead of random initialization.
        pool_mode
            How to aggregate per-junction representations into a per-cell vector:

            * ``"mean"``: average over observed junctions per cell.
            * ``"sum"``: sum over observed junctions per cell.
        max_nobs
            Maximum number of observed (cell, junction) pairs processed in a
            single chunk inside the encoder. If negative, disables chunking.
            Useful for controlling memory when masks are dense.
        latent_distribution
            Form of the latent prior/approximate posterior. ``"normal"`` uses
            a standard Gaussian; ``"ln"`` applies a softmax transformation to
            the latent samples to produce simplex-valued representations.
        **kwargs
            Additional keyword arguments forwarded to :class:`PARTIALVAE`. These
            are rarely needed in routine use and are kept for extensibility.

        Notes
        -----
        * For ``splice_likelihood="dirichlet_multinomial"``, a junction→ATSE
          mapping is constructed from ``adata.var["event_id"]`` and stored as a
          sparse tensor ``module.junc2atse``.
        * Per-junction embeddings can optionally be initialized from PCA for
          faster and more stable training on high-dimensional splicing matrices.
        """
        super().__init__(adata)

        self._module_kwargs: dict[str, Any] = dict(
            code_dim=code_dim,
            h_hidden_dim=h_hidden_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_n_layers=encoder_n_layers,
            n_latent=latent_dim,
            dropout_rate=dropout_rate,
            learn_concentration=learn_concentration,
            splice_likelihood=splice_likelihood,
            encode_covariates=encode_covariates,
            deeply_inject_covariates=deeply_inject_covariates,
            pool_mode=pool_mode,
            max_nobs=max_nobs,
            latent_distribution=latent_distribution,
            **kwargs,
        )

        self._model_summary_string = (
            "EDDISPLICE PartialVAE with "
            f"h_hidden_dim={h_hidden_dim}, "
            f"encoder_hidden_dim={encoder_hidden_dim}, "
            f"encoder_n_layers={encoder_n_layers}, "
            f"latent_dim={latent_dim}, "
            f"learn_concentration={learn_concentration}, "
            f"splice_likelihood={splice_likelihood}, "
            f"encode_covariates={encode_covariates}, "
            f"deeply_inject_covariates={deeply_inject_covariates}, "
            f"initialize_embeddings_from_pca={initialize_embeddings_from_pca}, "
            f"pool_mode={pool_mode}, "
            f"max_nobs={max_nobs}."
        )

        if self._module_init_on_train:
            self.module = None
            warnings.warn(
                "Model initialized without adata; will init on train.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        else:
            if self.adata is not None:
                if initialize_embeddings_from_pca:
                    self.init_feature_embedding_from_adata()

                # Only needed for Dirichlet–multinomial
                if splice_likelihood == "dirichlet_multinomial":
                    self.init_junc2atse()

        self.init_params_ = self._get_init_params(locals())



    def make_junc2atse(self, atse_labels):
        print("Making Junc2Atse...")
        num_junctions = len(atse_labels)
        atse_labels = atse_labels.astype('category')
        row_indices = torch.arange(num_junctions, dtype=torch.long)
        col_indices = torch.tensor(atse_labels.cat.codes.values)

        return torch.sparse_coo_tensor(
            indices=torch.stack([row_indices, col_indices]),
            values=torch.ones(len(row_indices), dtype=torch.float32),
            size=(num_junctions, len(atse_labels.cat.categories))
        ).coalesce()

    def init_junc2atse(self) -> None:
        # Do nothing if already initialized
        if hasattr(self.module, "junc2atse"):
            return

        clus_layer = self.adata_manager.data_registry["cluster_counts"].attr_key
        cluster_counts = self.adata.layers[clus_layer]  # not used but sanity-checks registry
        atse_labels = self.adata.var["event_id"]

        j2a = self.make_junc2atse(atse_labels)
        self.module.junc2atse = j2a.coalesce().to(self.module.device)


    def init_feature_embedding_from_adata(self) -> None:
        layer = self.adata_manager.data_registry[REGISTRY_KEYS.X_KEY].attr_key
        X = self.adata.layers[layer]
        # keep X sparse:
        if not sp.issparse(X):
            X = sp.csr_matrix(X)

        # fit a (code_dim)-component SVD on the sparse X
        svd = TruncatedSVD(n_components=self.module.encoder.code_dim,
                        algorithm="randomized")
        svd.fit(X)

        # components_.shape == (n_components, n_vars)
        comps = svd.components_.T  # (n_vars, code_dim)

        with torch.no_grad():
            self.module.encoder.feature_embedding.copy_(
                torch.as_tensor(comps, dtype=self.module.encoder.feature_embedding.dtype)
            )

        
    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int = 200,
        lr: float = 1e-4, # 0.001
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float | None = None,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 512,
        weight_decay: float = 1e-3, 
        eps: float = 1e-8,
        early_stopping: bool = True,
        save_best: bool = True,
        check_val_every_n_epoch: int | None = None,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = 10,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_min: float = 0.0,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Trains the model using amortized variational inference on splicing data.

        Parameters
        ----------
        max_epochs
            Number of epochs to train over.
        lr
            Learning rate for optimization.
        accelerator, devices
            Hardware acceleration options.
        train_size, validation_size
            Proportions for splitting the data.
        shuffle_set_split
            Whether to shuffle indices before splitting.
        batch_size
            Minibatch size for training.
        weight_decay, eps
            Optimizer hyperparameters.
        early_stopping, save_best
            Early stopping and checkpointing options.
        check_val_every_n_epoch
            Frequency of validation checks.
        n_steps_kl_warmup, n_epochs_kl_warmup
            KL warmup parameters.
        datasplitter_kwargs, plan_kwargs, **kwargs
            Additional options for data splitting, training plan, and trainer.
        """
        update_dict = {
            "lr": lr,
            "weight_decay": weight_decay,
            "eps": eps,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
            "optimizer": "AdamW",
            "reduce_lr_on_plateau": reduce_lr_on_plateau,
            "lr_factor": lr_factor,
            "lr_patience": lr_patience,
            "lr_threshold": lr_threshold,
            "lr_min": lr_min,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        datasplitter_kwargs = datasplitter_kwargs or {}

        if save_best:
            warnings.warn(
                "`save_best` is deprecated in v1.2 and will be removed in v1.3. "
                "Please use `enable_checkpointing` instead. See "
                "https://github.com/scverse/scvi-tools/issues/2568 for more details.",
                DeprecationWarning,
                stacklevel=settings.warnings_stacklevel,
            )
            if "callbacks" not in kwargs:
                kwargs["callbacks"] = []
            kwargs["callbacks"].append(SaveBestState(monitor="reconstruction_loss_validation"))

        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            distributed_sampler=use_distributed_sampler(kwargs.get("strategy", None)),
            batch_size=batch_size,
            **datasplitter_kwargs,
        )

        training_plan = self._training_plan_cls(self.module, **plan_kwargs)

        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            early_stopping_monitor="reconstruction_loss_validation",
            early_stopping_patience=10,
            **kwargs,
        )

        return runner()


    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        junc_ratio_layer: str,
        junc_counts_layer: str,
        cluster_counts_layer: str,
        psi_mask_layer: str,
        batch_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """
        Set up AnnData for EddiSplice.

        Parameters
        ----------
        adata
            AnnData to register.
        junc_ratio_layer
            Layer with junction usage ratios (X input).
        junc_counts_layer
            Layer with junction counts (successes).
        cluster_counts_layer
            Layer with total cluster counts (trials).
        psi_mask_layer
            Layer with binary mask (1=observed, 0=missing) per junction.
        batch_key
            Column in obs for batch.
        size_factor_key
            If provided, registers size factor but unused for splicing.
        categorical_covariate_keys
        continuous_covariate_keys
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            # Ratios as input
            LayerField(REGISTRY_KEYS.X_KEY, junc_ratio_layer, is_count_data=False),
            # Counts for likelihood
            LayerField("junction_counts", junc_counts_layer, is_count_data=True),
            LayerField("cluster_counts", cluster_counts_layer, is_count_data=True),
            # Mask layer for partial VAE
            LayerField(REGISTRY_KEYS.PSI_MASK_KEY, psi_mask_layer, is_count_data=False),
            # batch covariate
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
        ]
        # optional size factor (unused)
        if size_factor_key is not None:
            anndata_fields.append(
                NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False)
            )
        # additional covariates if desired
        if categorical_covariate_keys:
            anndata_fields.append(
                CategoricalObsField(REGISTRY_KEYS.CAT_COVS_KEY, None)
            )
        if continuous_covariate_keys:
            anndata_fields.append(
                NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, None)
            )
        # register
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    # add at top of file alongside np, torch, etc.

    # … inside your EDDISPLICE class …

    @torch.inference_mode()
    def get_normalized_splicing(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        use_z_mean: bool = True,
        n_samples: int = 1,
        batch_size: int | None = None,
        return_numpy: bool = False,
        silent: bool = True,
    ) -> np.ndarray | pd.DataFrame:
        """
        Return the decoded splicing probabilities p_nj = sigmoid(decoder_logits).

        Parameters
        ----------
        adata
            AnnData for inference (defaults to the one used at init).
        indices
            Which cells to pull (default: all).
        use_z_mean
            If True, run generative with use_z_mean=True.
        n_samples
            How many posterior samples to draw (passed to inference).
        batch_size
            Mini-batch size (defaults to scvi.settings.batch_size).
        return_numpy
            If True, returns a (n_cells, n_junctions) numpy array;
            otherwise returns a DataFrame with var_names as columns.
        silent
            If False, shows a little progress info.

        Returns
        -------
        Array or DataFrame of shape (cells, junctions) of decoded probabilities.
        """
        adata = self._validate_anndata(adata)
        # pick all cells if nothing specified
        if indices is None:
            indices = np.arange(adata.n_obs)
        # build loader
        dl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        all_spls = []
        for tensors in dl:
            # 1) inference
            inf_inputs = self.module._get_inference_input(tensors)
            inf_out = self.module.inference(**inf_inputs, n_samples=n_samples)
            # 2) generative
            gen_inputs = self.module._get_generative_input(tensors, inf_out)
            gen_out = self.module.generative(**gen_inputs, use_z_mean=use_z_mean)
            logits = gen_out["reconstruction"]  # (batch, n_junctions)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_spls.append(probs)
        # concat back
        spls = np.concatenate(all_spls, axis=0)  # (n_cells, n_junctions)
        if return_numpy:
            return spls
        # otherwise DataFrame
        cols = adata.var_names
        idx = adata.obs_names[indices]
        return pd.DataFrame(spls, index=idx, columns=cols)


    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        Return latent embeddings of splicing VAE.

        Parameters
        ----------
        adata
            AnnData for inference (defaults to init adata).
        indices
            Cell indices to use.
        give_mean
            If True, use posterior mean; else sample.
        batch_size
            Batch size.

        Returns
        -------
        Array of shape (cells, latent_dim).
        """
        if not self.is_trained_:
            raise RuntimeError("Train the model before extracting latent.")
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        latents = []
        for tensors in scdl:
            inf_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inf_inputs)
            qz_m = outputs["qz_m"]  # Underlying Normal distribution
            z = outputs["z"]
            if give_mean:
                z = qz_m
            latents.append(z.cpu())
        return torch.cat(latents).numpy()

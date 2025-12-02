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
    """
    Variational autoencoder for splicing data (SCpliceVAE).

    Parameters
    ----------
    adata
        AnnData object registered via :meth:`~EddiSplice.setup_anndata`.
    code_dim
        Dimensionality of feature embeddings.
    h_hidden_dim
        Hidden size for shared PointNet-like layer.
    encoder_hidden_dim
        Hidden size for encoder MLP.
    latent_dim
        Dimensionality of latent space.
    decoder_hidden_dim
        Hidden size for decoder layers.
    dropout_rate
        Dropout probability.
    learn_concentration
        Whether to learn Beta-Binomial or Dirichlet Multinomial concentration.
    encode_covariates
        If True, concatenates obs‐level covariates to each encoder/decoder input.
    deeply_inject_covariates
        If True, injects covariates at every hidden layer (rather than just the first).

    """
    _module_cls = PARTIALVAE

    def __init__(
        self,
        adata: AnnData | None = None,
        code_dim: int = 16,
        h_hidden_dim: int = 64,
        encoder_hidden_dim: int = 128,
        latent_dim: int = 10,
        dropout_rate: float = 0.1, 
        learn_concentration: bool = True,
        splice_likelihood: Literal["binomial", "beta_binomial", "dirichlet_multinomial"] = "beta_binomial",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        initialize_embeddings_from_pca: bool = True,
        num_transformer_layers: int = 2,
        encoder_type: Literal[
            "PartialEncoderWeightedSumEDDIMultiWeight",
            "PartialEncoderWeightedSumEDDIMultiWeightATSE",
            "PartialEncoderEDDI",
            "PartialEncoderEDDIATSE",
        ] = "PartialEncoderEDDI",
        pool_mode: Literal["mean","sum"]="mean",
        num_weight_vectors: int = 4,
        temperature_value: float = -1.0, #if temperature_value is set to -1, then the median number of observations is used as the fixed temperature value.
        temperature_fixed: bool = True, #if temperature is fixed, it is fixed to the value of temperature_value
        forward_style: Literal["per-cell", "batched", "scatter"] = "batched",
        max_nobs: int = -1,
        **kwargs,
    ):
        super().__init__(adata)

        # 1) Build the kwargs we’ll pass through to the PARTIALVAE
        #    including your two new flags.
        self._module_kwargs: dict[str, Any] = dict(
            code_dim=code_dim,
            h_hidden_dim=h_hidden_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            n_latent=latent_dim,
            dropout_rate=dropout_rate,
            learn_concentration=learn_concentration,
            splice_likelihood=splice_likelihood,
            encode_covariates=encode_covariates,
            deeply_inject_covariates=deeply_inject_covariates,
            num_transformer_layers=num_transformer_layers,
            encoder_type = encoder_type,
            pool_mode=pool_mode,
            num_weight_vectors = num_weight_vectors,
            temperature_value=temperature_value,
            temperature_fixed=temperature_fixed,
            forward_style=forward_style,
            max_nobs = max_nobs,
            **kwargs,
        )

        # 2) Summary string (optional)
        self._model_summary_string = (
            f"EddiSplice PartialVAE with "
            f"h_hidden_dim={h_hidden_dim}, "
            f"encoder_hidden_dim={encoder_hidden_dim}, "
            f"latent_dim={latent_dim}, "
            f"learn_concentration={learn_concentration}, "
            f"splice_likelihood={splice_likelihood}, "
            f"encode_covariates={encode_covariates}, "
            f"deeply_inject_covariates={deeply_inject_covariates}, "
            f"initialize_embeddings_from_pca={initialize_embeddings_from_pca}, "
            f"encoder_type={encoder_type}, num_weight_vectors={num_weight_vectors}, "
            f"temperature_value={temperature_value}, temperature_fixed={temperature_fixed}, forward_style={forward_style}, pool_mode={pool_mode}, max_nobs = {max_nobs}."
        )


        # 3) If we only initialize module at train time, bail out now
        if self._module_init_on_train:
            self.module = None
            warnings.warn(
                "Model initialized without adata; will init on train.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        else:
            # 4) Compute how many covariates we have
            n_batch = self.summary_stats.n_batch
            n_cont = self.summary_stats.get("n_extra_continuous_covs", 0)
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry:
                n_cats = (
                    self.adata_manager
                        .get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)
                        .n_cats_per_key
                )
            else:
                n_cats = []

            # 5) Instantiate your PARTIALVAE, passing everything in
            self.module = self._module_cls(
                n_input=self.summary_stats.n_vars,
                n_batch=n_batch,
                n_continuous_cov=n_cont,
                n_cats_per_cov=n_cats,
                **self._module_kwargs,
            )

            if self.adata is not None:
                if initialize_embeddings_from_pca:
                    self.init_feature_embedding_from_adata()
                if splice_likelihood == "dirichlet_multinomial":
                    self.init_junc2atse()
                    if "gnn" in encoder_type.lower(): #NOTE: THE GNN WILL FOR NOW ONLY WORK IF DM LIKELIHOOD IS USED!!!
                        self._setup_junction_gnn_edges()
                    if "atse" in encoder_type.lower():
                        print("Registering junc2ATSE")
                        self.module.encoder.register_junc2atse(self.module.junc2atse)
                    if "multi" in encoder_type.lower():
                        print("Multi in encoder type")
                        if self.module.encoder.temperature_value < 0.0:
                            print("Resetting temperature value.")
                            self.set_median_obs_temperature()
                            
                self.module.num_junctions=len(self.adata.var)


        self.init_params_ = self._get_init_params(locals())


    def set_median_obs_temperature(self):
        M = self.adata.layers["psi_mask"]
        if sp.issparse(M):
            # per-row nonzeros, memory-light
            counts = M.tocsr().getnnz(axis=1)
            median_obs = np.median(counts)
        else:
            # works for bool or 0/1 dense
            median_obs = np.median(np.count_nonzero(M, axis=1))

        T = 1.0 / max(np.sqrt(float(median_obs)), 1.0)
        print(f"Resetting temperature_value to 1/sqrt(median n_obs). median(n_obs)={median_obs}, T={T:.6f}")
        self.module.encoder.temperature_value= T  # python float is fine
        
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
        clus_layer = self.adata_manager.data_registry["cluster_counts"].attr_key
        cluster_counts = self.adata.layers[clus_layer]
        atse_labels = self.adata.var["event_id"]
        j2a = self.make_junc2atse(atse_labels)
        self.module.junc2atse = j2a.coalesce().to(self.module.device)

    def _setup_junction_gnn_edges(self) -> None:
        """
        Build a J×J junction–junction edge_index from
        the module’s junc2atse sparse tensor and
        attach it to the encoder as `edge_index`.
        """
        print("Setting up junction GNN edges...")
        # coalesced sparse COO (J, G) J = num junctions, G = num ATSEs
        j2a = self.module.junc2atse.coalesce()
        J, G = j2a.shape
        idx_j, idx_g = j2a.indices()  # each length ≈ J

        # group junctions by ATSE
        groups: dict[int, list[int]] = {}
        for j, g in zip(idx_j.tolist(), idx_g.tolist()):
            groups.setdefault(g, []).append(j)

        # fully connect all junctions within each group
        edge_list: list[list[int]] = []
        for js in groups.values():
            for i in js:
                for j in js:
                    if i != j:
                        edge_list.append([i, j])

        E = len(edge_list)
        avg_deg = E / J
        print(f"→ Total directed edges: {E}")
        print(f"→ Number of junctions:   {J}")
        print(f"→ Average edges per junction: {avg_deg:.2f}")

        edge_index = (
            torch.tensor(edge_list, dtype=torch.long)
                 .t()
                 .contiguous()
                 .to(self.module.device)
        )
        self.module.encoder.edge_index = edge_index
        print("Set up junction GNN edges!")

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
        batch_size: int = 512, #bigger batch size
        weight_decay: float = 1e-3, #remove weight decay
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

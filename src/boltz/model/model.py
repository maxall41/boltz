import gc
import random
from typing import Any, Optional

import torch
import torch._dynamo
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MeanMetric
import torch.nn.functional as F
import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.data.feature.symmetry import (
    minimum_lddt_symmetry_coords,
    minimum_symmetry_coords,
)
from boltz.model.loss.confidence import confidence_loss
from boltz.model.loss.distogram import distogram_loss
from boltz.model.loss.validation import (
    compute_pae_mae,
    compute_pde_mae,
    compute_plddt_mae,
    factored_lddt_loss,
    factored_token_lddt_dist_loss,
    weighted_minimum_rmsd,
)
from boltz.model.modules.encoders import RelativePositionEncoder
from boltz.model.modules.trunk import (
    InputEmbedder,
    MSAModule,
    PairformerModule,
)
from boltz.model.modules.utils import ExponentialMovingAverage
from boltz.model.optim.scheduler import AlphaFoldLRScheduler


class Boltz1(LightningModule):
    def __init__(  # noqa: PLR0915, C901
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        num_bins: int,
        training_args: dict[str, Any],
        validation_args: dict[str, Any],
        embedder_args: dict[str, Any],
        msa_args: dict[str, Any],
        pairformer_args: dict[str, Any],
        score_model_args: dict[str, Any],
        diffusion_process_args: dict[str, Any],
        diffusion_loss_args: dict[str, Any],
        confidence_model_args: dict[str, Any],
        atom_feature_dim: int = 128,
        confidence_prediction: bool = False,
        confidence_imitate_trunk: bool = False,
        alpha_pae: float = 0.0,
        structure_prediction_training: bool = True,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        compile_pairformer: bool = False,
        compile_structure: bool = False,
        compile_confidence: bool = False,
        nucleotide_rmsd_weight: float = 5.0,
        ligand_rmsd_weight: float = 10.0,
        no_msa: bool = False,
        no_atom_encoder: bool = False,
        ema: bool = False,
        ema_decay: float = 0.999,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        predict_args: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.structure_prediction_training = structure_prediction_training

        self.save_hyperparameters()

        self.lddt = nn.ModuleDict()
        self.disto_lddt = nn.ModuleDict()
        self.complex_lddt = nn.ModuleDict()
        if confidence_prediction:
            self.top1_lddt = nn.ModuleDict()
            self.iplddt_top1_lddt = nn.ModuleDict()
            self.ipde_top1_lddt = nn.ModuleDict()
            self.pde_top1_lddt = nn.ModuleDict()
            self.ptm_top1_lddt = nn.ModuleDict()
            self.iptm_top1_lddt = nn.ModuleDict()
            self.ligand_iptm_top1_lddt = nn.ModuleDict()
            self.protein_iptm_top1_lddt = nn.ModuleDict()
            self.avg_lddt = nn.ModuleDict()
            self.plddt_mae = nn.ModuleDict()
            self.pde_mae = nn.ModuleDict()
            self.pae_mae = nn.ModuleDict()
        for m in const.out_types + ["pocket_ligand_protein"]:
            self.lddt[m] = MeanMetric()
            self.disto_lddt[m] = MeanMetric()
            self.complex_lddt[m] = MeanMetric()
            if confidence_prediction:
                self.top1_lddt[m] = MeanMetric()
                self.iplddt_top1_lddt[m] = MeanMetric()
                self.ipde_top1_lddt[m] = MeanMetric()
                self.pde_top1_lddt[m] = MeanMetric()
                self.ptm_top1_lddt[m] = MeanMetric()
                self.iptm_top1_lddt[m] = MeanMetric()
                self.ligand_iptm_top1_lddt[m] = MeanMetric()
                self.protein_iptm_top1_lddt[m] = MeanMetric()
                self.avg_lddt[m] = MeanMetric()
                self.pde_mae[m] = MeanMetric()
                self.pae_mae[m] = MeanMetric()
        for m in const.out_single_types:
            if confidence_prediction:
                self.plddt_mae[m] = MeanMetric()
        self.rmsd = MeanMetric()
        self.best_rmsd = MeanMetric()

        self.train_confidence_loss_logger = MeanMetric()
        self.train_confidence_loss_dict_logger = nn.ModuleDict()
        for m in [
            "plddt_loss",
            "resolved_loss",
            "pde_loss",
            "pae_loss",
            "rel_plddt_loss",
            "rel_pde_loss",
            "rel_pae_loss",
        ]:
            self.train_confidence_loss_dict_logger[m] = MeanMetric()

        self.ema = None
        self.use_ema = ema
        self.ema_decay = ema_decay

        self.training_args = training_args
        self.validation_args = validation_args
        self.diffusion_loss_args = diffusion_loss_args
        self.predict_args = predict_args

        self.nucleotide_rmsd_weight = nucleotide_rmsd_weight
        self.ligand_rmsd_weight = ligand_rmsd_weight

        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.is_pairformer_compiled = False

        # Input projections
        s_input_dim = (
            token_s + 2 * const.num_tokens + 1 + len(const.pocket_contact_info)
        )
        self.s_init = nn.Linear(s_input_dim, token_s, bias=False)
        self.z_init_1 = nn.Linear(s_input_dim, token_z, bias=False)
        self.z_init_2 = nn.Linear(s_input_dim, token_z, bias=False)

        # Input embeddings
        full_embedder_args = {
            "atom_s": atom_s,
            "atom_z": atom_z,
            "token_s": token_s,
            "token_z": token_z,
            "atoms_per_window_queries": atoms_per_window_queries,
            "atoms_per_window_keys": atoms_per_window_keys,
            "atom_feature_dim": atom_feature_dim,
            "no_atom_encoder": no_atom_encoder,
            **embedder_args,
        }
        self.input_embedder = InputEmbedder(**full_embedder_args)
        self.rel_pos = RelativePositionEncoder(token_z)
        self.token_bonds = nn.Linear(1, token_z, bias=False)

        # Normalization layers
        self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        # Recycling projections
        self.s_recycle = nn.Linear(token_s, token_s, bias=False)
        self.z_recycle = nn.Linear(token_z, token_z, bias=False)
        init.gating_init_(self.s_recycle.weight)
        init.gating_init_(self.z_recycle.weight)

        # Pairwise stack
        self.no_msa = no_msa
        if not no_msa:
            self.msa_module = MSAModule(
                token_z=token_z,
                s_input_dim=s_input_dim,
                **msa_args,
            )
        self.pairformer_module = PairformerModule(token_s, token_z, **pairformer_args)
        self.prediction_head = nn.Sequential(
            nn.Linear(66048, 5012),
            nn.Mish(),
            nn.Linear(5012, 1024),
            nn.Mish(),
            nn.Linear(1024, 512),
            nn.Mish(),
            nn.Linear(512, 1),
        )

        if compile_pairformer:
            # Big models hit the default cache limit (8)
            self.is_pairformer_compiled = True
            torch._dynamo.config.cache_size_limit = 512
            torch._dynamo.config.accumulated_cache_size_limit = 512
            self.pairformer_module = torch.compile(
                self.pairformer_module,
                dynamic=False,
                fullgraph=False,
            )

        # Remove grad from weights they are not trained for ddp
        if not structure_prediction_training:
            for name, param in self.named_parameters():
                if name.split(".")[0] != "confidence_module":
                    param.requires_grad = False

    def forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: int = 0,
        num_sampling_steps: Optional[int] = None,
        multiplicity_diffusion_train: int = 1,
        diffusion_samples: int = 1,
        run_confidence_sequentially: bool = False,
    ) -> (Tensor, Tensor):
        # Compute input embeddings
        with torch.set_grad_enabled(
            self.training and self.structure_prediction_training
        ):
            s_inputs = self.input_embedder(feats)

            # Initialize the sequence and pairwise embeddings
            s_init = self.s_init(s_inputs)
            z_init = (
                self.z_init_1(s_inputs)[:, :, None]
                + self.z_init_2(s_inputs)[:, None, :]
            )
            relative_position_encoding = self.rel_pos(feats)
            z_init = z_init + relative_position_encoding
            z_init = z_init + self.token_bonds(feats["token_bonds"].float())

            # Perform rounds of the pairwise stack
            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)

            # Compute pairwise mask
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

            for i in range(recycling_steps + 1):
                with torch.set_grad_enabled(self.training and (i == recycling_steps)):
                    # Fixes an issue with unused parameters in autocast
                    if (
                        self.training
                        and (i == recycling_steps)
                        and torch.is_autocast_enabled()
                    ):
                        torch.clear_autocast_cache()

                    # Apply recycling
                    s = s_init + self.s_recycle(self.s_norm(s))
                    z = z_init + self.z_recycle(self.z_norm(z))

                    # Compute pairwise stack
                    if not self.no_msa:
                        z = z + self.msa_module(z, s_inputs, feats)

                    # Revert to uncompiled version for validation
                    if self.is_pairformer_compiled and not self.training:
                        pairformer_module = self.pairformer_module._orig_mod  # noqa: SLF001
                    else:
                        pairformer_module = self.pairformer_module

                    s, z = pairformer_module(s, z, mask=mask, pair_mask=pair_mask)
            s = torch.mean(s, dim=2)
            z = torch.flatten(torch.mean(z, dim=1), start_dim=1)
            sz = torch.cat((s, z), dim=-1)
            out = self.prediction_head(sz)
            return out

    def training_step(self, batch, batch_idx: int) -> Tensor:
        # Sample recycling steps
        recycling_steps = random.randint(0, self.training_args.recycling_steps)

        # Compute the forward pass
        out = self(
            feats=batch,
            recycling_steps=recycling_steps,
            num_sampling_steps=-1,
            multiplicity_diffusion_train=-1,
            diffusion_samples=-1,
        )
        label = torch.tensor(batch["label"], device=out.device).unsqueeze(dim=0)
        # print(out, label)
        loss = F.binary_cross_entropy_with_logits(out, label)
        loss = loss.cpu().item()
        self.log("train/loss", loss)
        self.training_log()
        return loss

    def training_log(self):
        self.log("train/grad_norm", self.gradient_norm(self), prog_bar=False)
        self.log("train/param_norm", self.parameter_norm(self), prog_bar=False)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False)

        self.log(
            "train/grad_norm_msa_module",
            self.gradient_norm(self.msa_module),
            prog_bar=False,
        )
        self.log(
            "train/param_norm_msa_module",
            self.parameter_norm(self.msa_module),
            prog_bar=False,
        )

        self.log(
            "train/grad_norm_pairformer_module",
            self.gradient_norm(self.pairformer_module),
            prog_bar=False,
        )
        self.log(
            "train/param_norm_pairformer_module",
            self.parameter_norm(self.pairformer_module),
            prog_bar=False,
        )

    def gradient_norm(self, module) -> float:
        # Only compute over parameters that are being trained
        parameters = filter(lambda p: p.requires_grad, module.parameters())
        parameters = filter(lambda p: p.grad is not None, parameters)
        norm = torch.tensor([p.grad.norm(p=2) ** 2 for p in parameters]).sum().sqrt()
        return norm

    def parameter_norm(self, module) -> float:
        # Only compute over parameters that are being trained
        parameters = filter(lambda p: p.requires_grad, module.parameters())
        norm = torch.tensor([p.norm(p=2) ** 2 for p in parameters]).sum().sqrt()
        return norm

    def validation_step(self, batch, batch_idx: int):
        # Sample recycling steps
        recycling_steps = random.randint(0, self.training_args.recycling_steps)

        # Compute the forward pass
        out = self(
            feats=batch,
            recycling_steps=recycling_steps,
            num_sampling_steps=-1,
            multiplicity_diffusion_train=-1,
            diffusion_samples=-1,
        )

        loss = F.binary_cross_entropy(out, batch["label"])
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        try:
            out = self(
                batch,
                recycling_steps=self.predict_args["recycling_steps"],
                num_sampling_steps=-1,
                diffusion_samples=-1,
            )
            return out

        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return {"exception": True}
            raise {"exception": True}

    def configure_optimizers(self):
        """Configure the optimizer."""
        if self.structure_prediction_training:
            parameters = [p for p in self.parameters() if p.requires_grad]
        else:
            parameters = [
                p for p in self.confidence_module.parameters() if p.requires_grad
            ]

        optimizer = torch.optim.Adam(
            parameters,
            betas=(self.training_args.adam_beta_1, self.training_args.adam_beta_2),
            eps=self.training_args.adam_eps,
            lr=self.training_args.base_lr,
        )
        if self.training_args.lr_scheduler == "af3":
            scheduler = AlphaFoldLRScheduler(
                optimizer,
                base_lr=self.training_args.base_lr,
                max_lr=self.training_args.max_lr,
                warmup_no_steps=self.training_args.lr_warmup_no_steps,
                start_decay_after_n_steps=self.training_args.lr_start_decay_after_n_steps,
                decay_every_n_steps=self.training_args.lr_decay_every_n_steps,
                decay_factor=self.training_args.lr_decay_factor,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        return optimizer

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.use_ema:
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )
        if self.use_ema:
            if self.ema.compatible(checkpoint["ema"]["shadow_params"]):
                self.ema.load_state_dict(checkpoint["ema"], device=torch.device("cpu"))
            else:
                print("EMA not compatible with checkpoint, skipping...")
        elif "ema" in checkpoint:
            self.load_state_dict(checkpoint["ema"]["shadow_params"], strict=False)

    def on_train_start(self):
        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )
        elif self.use_ema:
            self.ema.to(self.device)

    def on_train_epoch_start(self) -> None:
        if self.use_ema:
            self.ema.restore(self.parameters())

    def prepare_eval(self) -> None:
        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )

        if self.use_ema:
            self.ema.store(self.parameters())
            self.ema.copy_to(self.parameters())

    def on_validation_start(self):
        self.prepare_eval()

    def on_predict_start(self) -> None:
        self.prepare_eval()

    def on_test_start(self) -> None:
        self.prepare_eval()

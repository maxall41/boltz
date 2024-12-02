import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fire
import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from omegaconf import OmegaConf, listconfig
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from boltz.data.module.training import BoltzTrainingDataModule, DataConfig
from boltz.main import check_inputs, download, process_inputs
from torch.distributed.fsdp import CPUOffload


@dataclass
class TrainConfig:
    """Train configuration.

    Attributes
    ----------
    data : DataConfig
        The data configuration.
    model : ModelConfig
        The model configuration.
    output : str
        The output directory.
    trainer : Optional[dict]
        The trainer configuration.
    resume : Optional[str]
        The resume checkpoint.
    pretrained : Optional[str]
        The pretrained model.
    wandb : Optional[dict]
        The wandb configuration.
    disable_checkpoint : bool
        Disable checkpoint.
    matmul_precision : Optional[str]
        The matmul precision.
    find_unused_parameters : Optional[bool]
        Find unused parameters.
    save_top_k : Optional[int]
        Save top k checkpoints.
    validation_only : bool
        Run validation only.
    debug : bool
        Debug mode.
    strict_loading : bool
        Fail on mismatched checkpoint weights.
    load_confidence_from_trunk: Optional[bool]
        Load pre-trained confidence weights from trunk.

    """

    data: DataConfig
    model: LightningModule
    output: str
    trainer: Optional[dict] = None
    resume: Optional[str] = None
    pretrained: Optional[str] = None
    wandb: Optional[dict] = None
    disable_checkpoint: bool = False
    matmul_precision: Optional[str] = None
    find_unused_parameters: Optional[bool] = False
    save_top_k: Optional[int] = 1
    validation_only: bool = False
    debug: bool = False
    strict_loading: bool = True
    load_confidence_from_trunk: Optional[bool] = False


def train(raw_config: str, data_dir: str, out_dir: str, sample: bool) -> None:  # noqa: C901, PLR0912, PLR0915
    """Run training.

    Parameters
    ----------
    raw_config : str
        The input yaml configuration.
    args : list[str]
        Any command line overrides.

    """
    cache_in = "~/.boltz"

    # Set cache path
    cache = Path(cache_in).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Create output directories
    data = Path(data_dir).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    download(cache)

    # Load CCD
    ccd_path = cache / "ccd.pkl"
    with ccd_path.open("rb") as file:
        ccd = pickle.load(file)  # noqa: S301

    # Check if data is a directory
    data = check_inputs(data, out_dir, False)
    processed = process_inputs(data, out_dir, ccd, sample=sample)

    # Load the configuration
    raw_config = omegaconf.OmegaConf.load(raw_config)

    # Instantiate the task
    cfg = hydra.utils.instantiate(raw_config)
    cfg = TrainConfig(**cfg)

    # Set matmul precision
    if cfg.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)

    # Create trainer dict
    trainer = cfg.trainer
    if trainer is None:
        trainer = {}

    # Flip some arguments in debug mode
    devices = trainer.get("devices", 1)

    wandb = cfg.wandb
    if cfg.debug:
        if isinstance(devices, int):
            devices = 1
        elif isinstance(devices, (list, listconfig.ListConfig)):
            devices = [devices[0]]
        trainer["devices"] = devices
        cfg.data.num_workers = 0
        if wandb:
            wandb = None

    # Create objects
    data_config = DataConfig(**cfg.data)
    data_config.datasets[0].target_dir = processed.targets_dir
    data_config.datasets[0].msa_dir = processed.msa_dir
    data_config.datasets[0].manifest_path = processed.manifest
    if len(data_config.datasets) > 1:
        raise Exception("More than one dataset!")
    data_module = BoltzTrainingDataModule(data_config)
    model_module = cfg.model

    # Create checkpoint callback
    callbacks = []
    dirpath = cfg.output
    if not cfg.disable_checkpoint:
        mc = ModelCheckpoint(
            monitor="val/lddt",
            save_top_k=cfg.save_top_k,
            save_last=True,
            mode="max",
            every_n_epochs=1,
        )
        callbacks = [mc]

    # Create wandb logger
    loggers = []
    if wandb:
        wdb_logger = WandbLogger(
            group=wandb["name"],
            save_dir=cfg.output,
            project=wandb["project"],
            entity=wandb["entity"],
            log_model=False,
        )
        loggers.append(wdb_logger)
        # Save the config to wandb

        @rank_zero_only
        def save_config_to_wandb() -> None:
            config_out = Path(wdb_logger.experiment.dir) / "run.yaml"
            with Path.open(config_out, "w") as f:
                OmegaConf.save(raw_config, f)
            wdb_logger.experiment.save(str(config_out))

        save_config_to_wandb()

    trainer = pl.Trainer(
        default_root_dir=str(dirpath),
        strategy="deepspeed_stage_2_offload",
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=not cfg.disable_checkpoint,
        reload_dataloaders_every_n_epochs=1,
        **trainer,
    )

    if not cfg.strict_loading:
        model_module.strict_loading = False

    if cfg.resume is not None:
        checkpoint = torch.load(cfg.resume)
        model_module.load_state_dict(
            checkpoint["state_dict"], strict=cfg.strict_loading
        )

    if cfg.validation_only:
        trainer.validate(
            model_module,
            datamodule=data_module,
        )
    else:
        trainer.fit(
            model_module,
            datamodule=data_module,
        )


if __name__ == "__main__":
    fire.Fire(train)

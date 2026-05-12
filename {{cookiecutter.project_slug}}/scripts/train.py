#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: {{cookiecutter.author_name}}, {{cookiecutter.author_email}}
#
#    Copyright (C) {{cookiecutter.project_year}}  {{cookiecutter.author_name}}

# System modules
import logging

# External modules
import hydra
from omegaconf import DictConfig

# Internal modules


main_logger = logging.getLogger(__name__)


def train_task(cfg: DictConfig) -> None:
    # Import within main loop to speed up training on jean zay
    from omegaconf import OmegaConf
    import wandb
    from hydra.utils import instantiate
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import WandbLogger
    
    from {{cookiecutter.project_slug}}.modules.train_module import TrainingModule

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    torch.use_deterministic_algorithms(mode=False, warn_only=True)
    torch.set_float32_matmul_precision("medium")
    torch._dynamo.config.suppress_errors = True
    
    main_logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: pl.LightningDataModule = instantiate(cfg.data)
    data_module.setup("fit")

    main_logger.info(f"Instantiating model <{cfg['train_module']._target_}>")
    model: TrainingModule = instantiate(cfg["train_module"])
    model.hparams["batch_size"] = cfg.batch_size
    if cfg.compile:
        # Compile for training
        model.network = torch.compile(model.network, mode="reduce-overhead")

    if OmegaConf.select(cfg, "callbacks") is not None:
        callbacks = []
        for _, callback_cfg in cfg.callbacks.items():
            curr_callback: pl.callbacks.Callback = instantiate(callback_cfg)
            callbacks.append(curr_callback)
    else:
        callbacks = None

    training_logger = None
    if OmegaConf.select(cfg, "logger") is not None:
        try:
            training_logger = instantiate(cfg.logger)
        except Exception:
            # Needed for restart on jean zay
            # Set wandb to offline
            training_logger = instantiate(
                cfg.logger, mode="offline", offline=True, log_model=False
            )

    if isinstance(training_logger, WandbLogger):
        main_logger.info("Watch gradients and parameters of model")
        training_logger.watch(model, log="all", log_freq=100)

    main_logger.info("Instantiating trainer")
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=training_logger
    )

    main_logger.info("Starting training")
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.ckpt_path)
    main_logger.info("Training finished")
    wandb.finish()
    
    
@hydra.main(
    version_base=None, config_path='../configs/', config_name='train'
)
def main_train(cfg: DictConfig) -> None:
    try:
        train_task(cfg)
    except MemoryError:
        pass


if __name__ == "__main__":
    main_train()
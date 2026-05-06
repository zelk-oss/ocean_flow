#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: {{cookiecutter.author_name}}, {{cookiecutter.author_email}}
#
#    Copyright (C) {{cookiecutter.project_year}}  {{cookiecutter.author_name}}

r'''Environment setup for the forecast pipeline.'''

# System modules
import logging

# External modules
import distributed
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig


main_logger = logging.getLogger(__name__)


__all__ = ["setup_environment", "initialize_client"]


def setup_environment(
        cfg: DictConfig,
        worker_rank: int = 0,
) -> None:
    r'''
    Set up the runtime environment for forecasting.

    Configures logging, random seeds, and torch settings
    for inference.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing:

        - logging_level : str
            Logging level (e.g. ``"INFO"``).
        - seed : int, optional
            Random seed for reproducibility.
    worker_rank : int, optional
        Worker rank offset added to seed. Default is 0.
    '''
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging_level),
        format=(
            '%(asctime)s - %(name)s'
            ' - %(levelname)s - %(message)s'
        ),
    )

    if cfg.get("seed") is not None:
        actual_seed = cfg.seed + worker_rank
        torch.manual_seed(actual_seed)
        np.random.seed(actual_seed)

    # Set torch matmul precision and cuDNN benchmark
    torch.set_float32_matmul_precision("medium")
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    main_logger.info(
        "Environment configured for forecasting"
    )


def initialize_client(
        cfg: DictConfig,
        dp_rank: int = 0,
) -> distributed.Client:
    r'''
    Initialize the Dask distributed client.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing:

        - dask.scheduler : str, list of str, or None
            Dask scheduler address(es). If a list,
            selects by dp_rank. If None, a local
            cluster is created.
        - dask.n_workers : int
            Number of workers for the local cluster.
        - dask.dashboard_address : str or None
            Dashboard address.
    dp_rank : int, optional
        Data-parallel rank for scheduler selection.
        Default is 0.

    Returns
    -------
    distributed.Client
        The initialized Dask distributed client.
    '''
    scheduler = cfg.dask.scheduler
    if isinstance(scheduler, (list, ListConfig)):
        address = scheduler[dp_rank]
        main_logger.info(
            "Connecting to Dask scheduler at "
            "%s (dp_rank=%d)", address, dp_rank,
        )
        client = distributed.Client(address)
    elif scheduler is not None:
        main_logger.info(
            "Connecting to Dask scheduler at "
            "%s", scheduler,
        )
        client = distributed.Client(scheduler)
    else:
        main_logger.info(
            "Creating local Dask cluster with "
            "%d workers", cfg.dask.n_workers,
        )
        cluster = distributed.LocalCluster(
            n_workers=cfg.dask.n_workers,
            threads_per_worker=1,
            dashboard_address=cfg.dask.dashboard_address,
            processes=False,
        )
        client = distributed.Client(cluster)
    main_logger.info("Dask client initialized successfully")
    return client

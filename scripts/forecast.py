#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Chiara Zelco, chiara.zelco@ens.psl.eu
#
#    Copyright (C) 2026  Chiara Zelco

r'''Hydra entry point for the forecast pipeline.

Splits the forecast workflow into a global phase (single
process, pre-launch validation) and a local phase
(per-worker execution via Fabric).
'''

# System modules
import logging
import os
from typing import Any, List, Tuple

# External modules
import hydra
import lightning.fabric
import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig

# Internal modules
from ocean_flow.forecast.checkpoint import (
    load_forecast_model,
)
from ocean_flow.forecast.config import (
    generate_forecast_configs,
)
from ocean_flow.forecast.environment import (
    setup_environment,
    initialize_client,
)
from ocean_flow.forecast.runner import (
    initialize_io,
    run_forecast,
)
from ocean_flow.forecast.output import (
    OutputWriter,
)
from ocean_flow.forecast.restart import (
    check_written_regions,
    filter_forecast_configs,
)
from ocean_flow.forecast.validation import (
    validate_restart_config,
    validate_initial_conditions,
    validate_auxiliary,
    validate_forcing,
    validate_checkpoint,
    create_output_store,
    validate_output_store,
    validate_dask_addresses,
)


main_logger = logging.getLogger(__name__)


def _run_global_phase(
        cfg: DictConfig,
) -> Tuple[List[Any], int]:
    r'''
    Run global validation and config generation.

    Validates stores and checkpoint, creates or validates
    the output store, generates forecast configs, and
    validates dask addresses. Runs in the main process
    before ``fabric.launch()``.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.

    Returns
    -------
    tuple of (list, int)
        Forecast configs and number of DP workers.
    '''
    validate_restart_config(
        cfg.io.restart,
        cfg.io.recreate_store,
    )
    validate_initial_conditions(
        cfg.io.data_path,
        cfg.io.state_variables,
    )
    validate_auxiliary(
        cfg.io.auxiliary_path,
        cfg.io.auxiliary_variables,
    )
    validate_forcing(
        cfg.io.forcing_path,
        cfg.io.forcing_variables,
        cfg.io.data_path,
    )
    validate_checkpoint(cfg.ckpt_path)

    n_dp_workers = (
        len(cfg.trainer.devices)
        if isinstance(cfg.trainer.devices, (list, tuple, ListConfig))
        else int(cfg.trainer.devices)
    )

    init_times = pd.date_range(
        start=cfg.init_start,
        end=cfg.init_end,
        freq=cfg.init_freq,
    )
    lead_times = pd.timedelta_range(
        start=cfg.step_freq,
        end=cfg.lead_time,
        freq=cfg.step_freq,
    )
    ens_mems = np.arange(cfg.ensemble_size)

    if cfg.io.restart and os.path.exists(
        cfg.io.store_path
    ):
        validate_output_store(
            store_path=cfg.io.store_path,
            state_variables=cfg.io.state_variables,
            init_times=init_times,
            lead_times=lead_times,
            ens_mems=ens_mems,
        )
        forecast_configs = list(generate_forecast_configs(
            cfg, dp_world_size=n_dp_workers,
        ))
        written_regions = check_written_regions(
            cfg.io.store_path,
            cfg.io.state_variables,
        )
        output_writer = OutputWriter(
            data_path=cfg.io.data_path,
            state_variables=cfg.io.state_variables,
            store_path=cfg.io.store_path,
            init_times=init_times,
            lead_times=lead_times,
            ens_mems=ens_mems,
        )
        forecast_configs = filter_forecast_configs(
            forecast_configs,
            output_writer,
            written_regions,
        )
    else:
        create_output_store(
            data_path=cfg.io.data_path,
            state_variables=cfg.io.state_variables,
            store_path=cfg.io.store_path,
            init_times=init_times,
            lead_times=lead_times,
            ens_mems=ens_mems,
            n_store_freq=cfg.n_store_freq,
            recreate=cfg.io.recreate_store,
        )
        forecast_configs = list(generate_forecast_configs(
            cfg, dp_world_size=n_dp_workers,
        ))

    validate_dask_addresses(
        cfg.dask.scheduler,
        n_dp_workers,
    )

    return forecast_configs, n_dp_workers


def run_local_forecast(
        fabric: lightning.fabric.Fabric,
        cfg: DictConfig,
        forecast_configs: List[Any],
) -> None:
    r'''
    Per-worker forecast execution.

    Runs on each Fabric worker after ``fabric.launch()``.
    Sets up environment, initializes dask client, creates
    IO objects, loads model, and runs the forecast loop.

    Parameters
    ----------
    fabric : lightning.fabric.Fabric
        Fabric instance for this worker.
    cfg : DictConfig
        Full Hydra configuration.
    forecast_configs : list
        Forecast configurations from global phase.
    '''
    dp_rank = fabric.global_rank
    dp_world_size = fabric.world_size
    worker_rank = fabric.global_rank

    setup_environment(cfg, worker_rank=worker_rank)

    with initialize_client(
        cfg, dp_rank=dp_rank,
    ) as client:
        input_reader, output_writer = (
            initialize_io(cfg)
        )
        model = load_forecast_model(cfg, fabric)
        run_forecast(
            client=client,
            model=model,
            input_reader=input_reader,
            output_writer=output_writer,
            forecast_configs=forecast_configs,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            n_prefetch_init=cfg.dask.n_prefetch_init,
            n_prefetch_forcing=cfg.dask.n_prefetch_forcing,
        )

    fabric.barrier()

    if fabric.global_rank == 0:
        written = check_written_regions(
            cfg.io.store_path,
            cfg.io.state_variables,
        )
        if written:
            expected_last = len(output_writer.lead_times) - 1
            incomplete = {
                k: v for k, v in written.items()
                if v < expected_last
            }
            if incomplete:
                raise RuntimeError(
                    "Output store has incomplete "
                    f"regions: {incomplete}"
                )
        main_logger.info(
            "Completeness check passed",
        )


def _launch_forecast(cfg: DictConfig) -> None:
    r'''
    Create Fabric and launch the forecast pipeline.

    Runs the global phase, then creates a Fabric instance
    from the trainer config and launches per-worker
    forecast execution.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration. Must contain
        ``trainer.accelerator``, ``trainer.devices``,
        ``trainer.precision``, and ``trainer.strategy``.
    '''
    forecast_configs, n_dp_workers = (
        _run_global_phase(cfg)
    )
    fabric = lightning.fabric.Fabric(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        strategy=cfg.trainer.strategy,
    )
    fabric.launch(
        run_local_forecast, cfg, forecast_configs,
    )


@hydra.main(
    version_base=None,
    config_path='../configs/',
    config_name='forecast',
)
def main_forecast(cfg: DictConfig) -> None:
    r'''
    Hydra entry point for the forecast pipeline.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.
    '''
    _launch_forecast(cfg)


if __name__ == "__main__":  # pragma: no cover
    main_forecast()

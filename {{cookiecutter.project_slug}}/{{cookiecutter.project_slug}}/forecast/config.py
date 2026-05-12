#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: {{cookiecutter.author_name}}, {{cookiecutter.author_email}}
#
#    Copyright (C) {{cookiecutter.project_year}}  {{cookiecutter.author_name}}

# System modules
import itertools
import logging
from dataclasses import dataclass
from typing import Iterable, Iterator

# External modules
from omegaconf import DictConfig

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

# Internal modules


main_logger = logging.getLogger(__name__)


__all__ = [
    "ForecastConfig",
    "generate_forecast_configs",
]


def _total_init_ens_pairs(cfg: DictConfig) -> int:
    r'''Return the total number of (init_time, ensemble) pairs.

    Parameters
    ----------
    cfg : DictConfig
        Forecast configuration with init_start, init_end,
        init_freq, and ensemble_size fields.

    Returns
    -------
    int
        Number of init-time x ensemble-member pairs.
    '''
    init_times = pd.date_range(
        start=cfg.init_start,
        end=cfg.init_end,
        freq=cfg.init_freq,
    )
    return len(init_times) * cfg.ensemble_size


@dataclass
class ForecastConfig:
    r'''
    Configuration dataclass for the forecasting pipeline.

    Attributes
    ----------
    init_times : pd.DatetimeIndex
        Initialization times for this batch of forecasts.
    lead_times : pd.TimedeltaIndex
        Full sequence of lead times (timedeltas from init_time) to
        produce for each forecast.
    ens_mems : ArrayLike
        Ensemble member indices for each element in the batch.
    n_store_freq : int
        Number of lead-time steps to advance and store per iteration
        (controls chunking via :meth:`get_leadtime_iterator`).
    '''
    init_times: pd.DatetimeIndex
    lead_times: pd.TimedeltaIndex
    ens_mems: ArrayLike
    n_store_freq: int

    def get_leadtime_iterator(self) -> Iterator[pd.TimedeltaIndex]:
        r'''
        Iterate over the lead times in chunks of size n_store_freq.

        Yields
        ------
        pd.TimedeltaIndex
            The current chunk of lead times.
        '''
        for i in range(0, len(self.lead_times), self.n_store_freq):
            lead_time_chunk = self.lead_times[i:i+self.n_store_freq]
            yield lead_time_chunk


def generate_forecast_configs(
    cfg: DictConfig,
    dp_world_size: int = 1,
) -> Iterable[ForecastConfig]:
    r'''
    Generate ForecastConfig objects for the forecast pipeline.

    Produces batched configs from the Cartesian product of
    initialization times and ensemble members. When
    ``dp_world_size > 1``, the per-worker batch size is
    ``max(1, cfg.batch_size // dp_world_size)``.

    Parameters
    ----------
    cfg : DictConfig
        Forecast configuration with init_start, init_end,
        init_freq, lead_time, step_freq, ensemble_size,
        batch_size, and n_store_freq fields.
    dp_world_size : int, optional, default = 1
        Number of data-parallel workers. The global
        ``cfg.batch_size`` is divided by this value to
        obtain the per-worker batch size.

    Yields
    ------
    ForecastConfig
        A ForecastConfig for each batch of forecasts.
    '''
    init_times = pd.date_range(
        start=cfg.init_start,
        end=cfg.init_end,
        freq=cfg.init_freq
    )
    lead_times = pd.timedelta_range(
        start=cfg.step_freq,
        end=cfg.lead_time,
        freq=cfg.step_freq
    )
    ensemble_members = np.arange(cfg.ensemble_size)

    # Cartesian product, init-times-major order
    init_ens_pairs = list(
        itertools.product(init_times, ensemble_members)
    )
    n_total = _total_init_ens_pairs(cfg)
    local_batch_size = max(
        1, cfg.batch_size // dp_world_size
    )

    for i in range(0, n_total, local_batch_size):
        chunk = init_ens_pairs[i:i + local_batch_size]
        batch_init_times = pd.DatetimeIndex([t for t, _ in chunk])
        batch_ens_mems = np.array([m for _, m in chunk], dtype=int)

        # Construct forecast config for the current batch
        yield ForecastConfig(
            init_times=batch_init_times,
            lead_times=lead_times,
            ens_mems=batch_ens_mems,
            n_store_freq=cfg.n_store_freq
        )

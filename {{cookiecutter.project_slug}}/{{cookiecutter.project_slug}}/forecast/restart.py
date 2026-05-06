#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Anonymous, anonymous@example.com
#
#    Copyright (C) 2026  Anonymous

r'''Restart logic for resuming forecasts from zarr.'''

# System modules
import logging
from typing import Dict, List, Tuple

# External modules
import numpy as np
import xarray as xr

# Internal modules
from .config import ForecastConfig
from .output import OutputWriter


main_logger = logging.getLogger(__name__)


__all__ = [
    "check_written_regions",
    "filter_forecast_configs",
]


def _get_contiguous_last_index(
    data_var: xr.DataArray,
    init_idx: int,
    ens_idx: int,
) -> int:
    r'''
    Find last contiguous non-NaN lead time index.

    Parameters
    ----------
    data_var : xr.DataArray
        Variable array with dims including init_time,
        lead_time, and ensemble.
    init_idx : int
        Index along init_time dimension.
    ens_idx : int
        Index along ensemble dimension.

    Returns
    -------
    int
        Last contiguous non-NaN lead time index, or -1
        if fully NaN.
    '''
    sliced = data_var.isel(
        init_time=init_idx, ensemble=ens_idx
    )
    # Reduce over all dims except lead_time
    spatial_dims = [
        d for d in sliced.dims if d != "lead_time"
    ]
    not_nan = ~np.isnan(sliced)
    if spatial_dims:
        all_valid = not_nan.all(dim=spatial_dims)
    else:  # pragma: no cover
        all_valid = not_nan
    valid = all_valid.values
    if not valid[0]:
        return -1
    # Find first False (NaN) position
    false_positions = np.where(~valid)[0]
    if len(false_positions) == 0:
        return len(valid) - 1
    return int(false_positions[0] - 1)


def check_written_regions(
    store_path: str,
    state_variables: List[str],
) -> Dict[Tuple[int, int], int]:
    r'''
    Inspect zarr store for written (non-NaN) regions.

    Opens the zarr store and checks the first state
    variable to determine which (init_time, ensemble)
    pairs have been written and to what lead time index.

    Parameters
    ----------
    store_path : str
        Path to the forecast zarr store.
    state_variables : list of str
        State variable names; the first is inspected.

    Returns
    -------
    dict of (int, int) to int
        Mapping from (init_time_idx, ens_idx) to last
        contiguous non-NaN lead time index. Returns -1
        for fully NaN slices.
    '''
    ds = xr.open_zarr(store_path, consolidated=False)
    var_name = state_variables[0]
    data_var = ds[var_name]

    n_init = data_var.sizes["init_time"]
    n_ens = data_var.sizes["ensemble"]

    result: Dict[Tuple[int, int], int] = {}
    for init_idx in range(n_init):
        for ens_idx in range(n_ens):
            result[(init_idx, ens_idx)] = (
                _get_contiguous_last_index(
                    data_var, init_idx, ens_idx
                )
            )
    return result


def filter_forecast_configs(
    forecast_configs: List[ForecastConfig],
    output_writer: OutputWriter,
    written_regions: Dict[Tuple[int, int], int],
) -> List[ForecastConfig]:
    r'''
    Filter and trim forecast configs based on written data.

    Skips fully-written configs and trims partially-written
    configs to start from the first missing lead time.

    Parameters
    ----------
    forecast_configs : list of ForecastConfig
        Original forecast configurations.
    output_writer : OutputWriter
        The output writer with init_times, ens_mems, and
        lead_times attributes.
    written_regions : dict of (int, int) to int
        Mapping from (init_time_idx, ens_idx) to last
        contiguous non-NaN lead time index.

    Returns
    -------
    list of ForecastConfig
        Filtered and possibly trimmed configurations.
    '''
    n_leads = len(output_writer.lead_times)
    filtered: List[ForecastConfig] = []

    for config in forecast_configs:
        pairs_status = []
        for it, em in zip(
            config.init_times, config.ens_mems
        ):
            init_idx = int(
                np.where(
                    output_writer.init_times == it
                )[0][0]
            )
            ens_idx = int(
                np.where(
                    output_writer.ens_mems == em
                )[0][0]
            )
            key = (init_idx, ens_idx)
            last_written = written_regions.get(key, -1)
            pairs_status.append(last_written)

        # Check if all pairs are fully written
        if all(s >= n_leads - 1 for s in pairs_status):
            continue

        # Find minimum restart index across all pairs
        min_restart = min(s + 1 for s in pairs_status)

        if min_restart > 0:
            trimmed_leads = config.lead_times[min_restart:]
            main_logger.warning(
                "Restart: trimming config to lead time "
                "index %d (skipping %d written steps)",
                min_restart,
                min_restart,
            )
            filtered.append(
                ForecastConfig(
                    init_times=config.init_times,
                    lead_times=trimmed_leads,
                    ens_mems=config.ens_mems,
                    n_store_freq=config.n_store_freq,
                )
            )
        else:
            filtered.append(config)

    return filtered

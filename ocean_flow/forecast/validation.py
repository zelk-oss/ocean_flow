#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Chiara Zelco, chiara.zelco@ens.psl.eu
#
#    Copyright (C) 2026  Chiara Zelco

r'''Global validators for the forecast pipeline.

Provides seven public functions that validate inputs, outputs,
and configuration before launching distributed forecast workers.
All validation runs in the main process before ``fabric.launch()``.
'''

# System modules
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Union

# External modules
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from numpy.typing import ArrayLike
from omegaconf import ListConfig

# Internal modules


main_logger = logging.getLogger(__name__)


__all__ = [
    'validate_restart_config',
    'validate_initial_conditions',
    'validate_auxiliary',
    'validate_forcing',
    'validate_checkpoint',
    'validate_output_store',
    'create_output_store',
    'validate_dask_addresses',
]


def validate_restart_config(
        restart: bool,
        recreate_store: bool,
) -> None:
    r'''
    Validate that restart and recreate_store are not both True.

    Parameters
    ----------
    restart : bool
        Whether to restart from an existing output store.
    recreate_store : bool
        Whether to delete and recreate the output store.

    Raises
    ------
    ValueError
        If both restart and recreate_store are True.
    '''
    if restart and recreate_store:
        raise ValueError(
            "Cannot set both io.restart=True and "
            "io.recreate_store=True: restarting from "
            "an existing store conflicts with "
            "recreating it."
        )


def validate_initial_conditions(
        data_path: str,
        state_variables: List[str],
) -> None:
    r'''
    Validate that the initial-condition zarr store exists
    and contains the required state variables.

    Parameters
    ----------
    data_path : str
        Path to the zarr store with initial conditions.
    state_variables : list of str
        Variable names that must exist in the store.

    Raises
    ------
    FileNotFoundError
        If the zarr store does not exist.
    ValueError
        If a required variable is missing.
    '''
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Initial condition store not found: "
            f"{data_path}"
        )
    ds = xr.open_zarr(data_path, consolidated=False)
    for var in state_variables:
        if var not in ds.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in "
                f"initial condition store. "
                f"Found: {list(ds.data_vars)}"
            )


def validate_auxiliary(
        auxiliary_path: Optional[str],
        auxiliary_variables: Optional[List[str]],
) -> None:
    r'''
    Validate auxiliary netCDF path and variables.

    If both arguments are ``None``, returns immediately.
    If exactly one is set, raises ``ValueError``. If both
    are set, opens the netCDF and checks variables exist.

    Parameters
    ----------
    auxiliary_path : str or None
        Path to the auxiliary netCDF file.
    auxiliary_variables : list of str or None
        Variable names that must exist in the file.

    Raises
    ------
    ValueError
        If only one of path/variables is provided, or
        if a required variable is missing.
    FileNotFoundError
        If the netCDF file does not exist.
    '''
    if auxiliary_path is None and auxiliary_variables is None:
        return
    if auxiliary_path is not None and auxiliary_variables is None:
        raise ValueError(
            "auxiliary_path is set but "
            "auxiliary_variables is None"
        )
    if auxiliary_path is None and auxiliary_variables is not None:
        raise ValueError(
            "auxiliary_variables is set but "
            "auxiliary_path is None"
        )
    if not Path(auxiliary_path).exists():
        raise FileNotFoundError(
            f"Auxiliary file not found: "
            f"{auxiliary_path}"
        )
    with xr.open_dataset(auxiliary_path) as ds:
        for var in auxiliary_variables:
            if var not in ds.data_vars:
                raise ValueError(
                    f"Variable '{var}' not found in "
                    f"auxiliary file. "
                    f"Found: {list(ds.data_vars)}"
                )


def validate_forcing(
        forcing_path: Optional[str],
        forcing_variables: Optional[List[str]],
        data_path: str,
) -> None:
    r'''
    Validate forcing variables exist in the data store.

    If ``forcing_variables`` is ``None``, returns immediately.
    Otherwise opens the zarr store at ``forcing_path`` (or
    ``data_path`` if ``forcing_path`` is ``None``) and checks
    that the variables exist.

    Parameters
    ----------
    forcing_path : str or None
        Path to a separate forcing zarr store, or ``None``
        to use ``data_path``.
    forcing_variables : list of str or None
        Variable names that must exist.
    data_path : str
        Fallback path to the initial-condition zarr store.

    Raises
    ------
    ValueError
        If a required variable is missing.
    '''
    if forcing_variables is None:
        return
    path = forcing_path if forcing_path is not None else data_path
    ds = xr.open_zarr(path, consolidated=False)
    for var in forcing_variables:
        if var not in ds.data_vars:
            raise ValueError(
                f"Forcing variable '{var}' not found. "
                f"Found: {list(ds.data_vars)}"
            )


def validate_checkpoint(
        ckpt_path: Optional[str],
) -> None:
    r'''
    Validate that the checkpoint file exists.

    If ``ckpt_path`` is ``None``, returns immediately.

    Parameters
    ----------
    ckpt_path : str or None
        Path to the model checkpoint file.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    '''
    if ckpt_path is None:
        return
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}"
        )


def validate_output_store(
        store_path: str,
        state_variables: List[str],
        init_times: pd.DatetimeIndex,
        lead_times: pd.TimedeltaIndex,
        ens_mems: ArrayLike,
) -> None:
    r'''
    Validate that an output zarr store has correct shape.

    Opens the store and checks that each state variable has
    the expected dimension sizes for init_time, lead_time,
    and ensemble.

    Parameters
    ----------
    store_path : str
        Path to the output zarr store.
    state_variables : list of str
        State variable names to validate.
    init_times : pd.DatetimeIndex
        Expected initialization times.
    lead_times : pd.TimedeltaIndex
        Expected lead times.
    ens_mems : array-like
        Expected ensemble member indices.

    Raises
    ------
    ValueError
        If a variable is missing or any dimension size does
        not match.
    '''
    ds = xr.open_zarr(store_path, consolidated=False)
    expected = {
        'init_time': len(init_times),
        'lead_time': len(lead_times),
        'ensemble': len(np.asarray(ens_mems)),
    }
    for var_name in state_variables:
        if var_name not in ds:
            raise ValueError(
                f"Variable '{var_name}' not found "
                f"in output store"
            )
        var_data = ds[var_name]
        for dim_name, expected_len in expected.items():
            if dim_name not in var_data.dims:
                raise ValueError(
                    f"Dimension '{dim_name}' not found "
                    f"in variable '{var_name}'"
                )
            actual_len = var_data.sizes[dim_name]
            if actual_len != expected_len:
                raise ValueError(
                    f"Dimension mismatch for variable "
                    f"'{var_name}' dimension '{dim_name}': "
                    f"expected {expected_len}, got {actual_len}"
                )


def create_output_store(
        data_path: str,
        state_variables: List[str],
        store_path: str,
        init_times: pd.DatetimeIndex,
        lead_times: pd.TimedeltaIndex,
        ens_mems: ArrayLike,
        n_store_freq: int,
        recreate: bool = True,
) -> None:
    r'''
    Create a NaN-filled zarr output store.

    Uses the reference zarr at ``data_path`` to determine
    spatial dimensions and dtypes. If ``recreate`` is
    ``False`` and the store already exists, skips creation.

    Parameters
    ----------
    data_path : str
        Path to the reference zarr store.
    state_variables : list of str
        Variable names to create in the output store.
    store_path : str
        Path for the output zarr store.
    init_times : pd.DatetimeIndex
        Initialization times for the output store.
    lead_times : pd.TimedeltaIndex
        Lead times for the output store.
    ens_mems : array-like
        Ensemble member indices.
    n_store_freq : int
        Chunk size along the lead_time dimension.
    recreate : bool, optional, default = True
        If ``True``, overwrite any existing store.
    '''
    store_exists = Path(store_path).exists()
    if not recreate and store_exists:
        return

    if store_exists and recreate:
        shutil.rmtree(store_path)

    ens_array = np.asarray(ens_mems)
    if ens_array.ndim == 0:
        ens_array = np.arange(int(ens_array.item()))
    ref_ds = xr.open_zarr(data_path, consolidated=False)
    try:
        spatial_dims = [
            dim for dim in ref_ds.dims
            if dim not in ['time', 'ensemble']
        ]

        coords = {}
        for dim in spatial_dims:
            if dim in ref_ds.coords:
                coords[dim] = ref_ds.coords[dim].values
        coords['init_time'] = init_times
        coords['lead_time'] = lead_times
        coords['ensemble'] = ens_array

        xr.Dataset(coords=coords).to_zarr(
            store_path, mode='w',
            consolidated=False,
        )

        store = zarr.open(store_path, mode='a')
        for var in state_variables:
            ref_var = ref_ds[var]
            valid_indices = [
                k for k, d in enumerate(ref_var.dims)
                if d in spatial_dims
            ]
            ref_dims = [
                ref_var.dims[k] for k in valid_indices
            ]
            ref_shape = [
                ref_var.shape[k] for k in valid_indices
            ]
            ref_dtype = ref_var.dtype

            shape = (
                len(init_times),
                len(lead_times),
                len(ens_array),
                *ref_shape,
            )
            chunks = (
                1,
                n_store_freq,
                1,
                *ref_shape,
            )
            zarr_var = store.create_dataset(
                var,
                shape=shape,
                chunks=chunks,
                dtype=ref_dtype,
                fill_value=np.nan,
                overwrite=True,
            )
            zarr_var.attrs['_ARRAY_DIMENSIONS'] = [
                'init_time',
                'lead_time',
                'ensemble',
                *ref_dims,
            ]

        zarr.consolidate_metadata(store_path)
    finally:
        ref_ds.close()


def validate_dask_addresses(
        scheduler: Union[str, List[str], None],
        n_dp_workers: int,
) -> None:
    r'''
    Validate dask scheduler address configuration.

    If ``scheduler`` is a list, its length must match
    ``n_dp_workers``. A single string or ``None`` is
    always valid.

    Parameters
    ----------
    scheduler : str, list of str, or None
        Dask scheduler address(es).
    n_dp_workers : int
        Number of data-parallel workers.

    Raises
    ------
    ValueError
        If scheduler is a list with length not matching
        ``n_dp_workers``.
    '''
    if isinstance(scheduler, (list, ListConfig)):
        if len(scheduler) != n_dp_workers:
            raise ValueError(
                f"Scheduler list length "
                f"({len(scheduler)}) does not match "
                f"n_dp_workers ({n_dp_workers})"
            )

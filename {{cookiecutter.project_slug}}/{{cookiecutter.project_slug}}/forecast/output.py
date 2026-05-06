#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: {{cookiecutter.author_name}}, {{cookiecutter.author_email}}
#
#    Copyright (C) {{cookiecutter.project_year}}  {{cookiecutter.author_name}}

r'''Lightweight pure-zarr forecast output writer.

Writes forecast predictions directly to zarr arrays using
region-based indexing without any xarray in the write path.
'''

# System modules
import logging
from typing import Dict, Iterable, List, Tuple

# External modules
import dask
import zarr
import xarray as xr
import pandas as pd
import numpy as np
from dask.delayed import Delayed
from numpy.typing import ArrayLike

# Internal modules


main_logger = logging.getLogger(__name__)


__all__ = [
    "OutputWriter",
    "_zarr_region_write",
]


def _zarr_region_write(
        zarr_array: zarr.Array,
        data: np.ndarray,
        region: tuple,
) -> None:
    r'''Write data into a zarr array at the given region.

    Parameters
    ----------
    zarr_array : zarr.Array
        Target zarr array.
    data : np.ndarray
        Data to write.
    region : tuple
        Index tuple specifying the write region.
    '''
    zarr_array.set_basic_selection(region, data)


class OutputWriter(object):
    r'''Write forecast predictions to a zarr output store.

    Opens a pre-created zarr store and writes predictions
    directly via zarr region indexing. No xarray objects are
    retained after initialisation.

    Parameters
    ----------
    data_path : str
        Path to the reference zarr store for coordinate
        and dimension metadata.
    state_variables : Iterable[str]
        List of state variable names to write.
    store_path : str
        Path to the pre-created output zarr store.
    init_times : pd.DatetimeIndex
        All possible initialization times.
    lead_times : pd.TimedeltaIndex
        All possible lead times.
    ens_mems : ArrayLike or int, optional
        Ensemble member indices or count. Default is 1.

    Attributes
    ----------
    data_path : str
        Path to the reference zarr store.
    state_variables : list of str
        State variable names.
    store_path : str
        Path to the output zarr store.
    init_times : pd.DatetimeIndex
        Initialization times.
    lead_times : pd.TimedeltaIndex
        Lead times.
    ens_mems : np.ndarray
        Ensemble member indices.
    '''
    def __init__(
            self,
            data_path: str,
            state_variables: Iterable[str],
            store_path: str,
            init_times: pd.DatetimeIndex,
            lead_times: pd.TimedeltaIndex,
            ens_mems: ArrayLike | int = 1,
    ) -> None:
        self.data_path = data_path
        self.state_variables = list(state_variables)
        self.store_path = store_path
        self.init_times = init_times
        self.lead_times = lead_times

        ens_array = np.asarray(ens_mems)
        if ens_array.ndim == 0:
            self.ens_mems = np.arange(
                int(ens_array.item())
            )
        else:
            self.ens_mems = np.unique(ens_array)

        self._spatial_meta = self._load_spatial_meta()
        self._store = zarr.open(store_path, mode='r+')

    def _load_spatial_meta(
            self,
    ) -> Dict[str, Tuple[str, ...]]:
        r'''Extract spatial dim names from reference store.

        Opens the reference zarr with xarray, extracts
        spatial dimension names per variable, then closes
        the dataset. No xarray objects are retained.

        Returns
        -------
        dict of str to tuple of str
            Maps variable name to its spatial dim names.
        '''
        ref_ds = xr.open_zarr(
            self.data_path, consolidated=False,
        )
        spatial_dims = [
            dim for dim in ref_ds.dims
            if dim not in ['time', 'ensemble']
        ]
        meta: Dict[str, Tuple[str, ...]] = {}
        for var in self.state_variables:
            ref_var = ref_ds[var]
            dims = tuple(
                d for d in ref_var.dims
                if d in spatial_dims
            )
            meta[var] = dims
        ref_ds.close()
        return meta

    def _write_batch_element(
            self,
            predictions: Dict[str, np.ndarray],
            batch_idx: int,
            init_idx: int,
            ens_idx: int,
            lead_start: int,
            lead_end: int,
    ) -> List[Delayed]:
        r'''Build delayed writes for one batch element.

        Parameters
        ----------
        predictions : dict of str to np.ndarray
            Variable names to prediction arrays.
        batch_idx : int
            Batch index to extract.
        init_idx : int
            Init time index in the output store.
        ens_idx : int
            Ensemble index in the output store.
        lead_start : int
            Start index for lead time slice.
        lead_end : int
            End index for lead time slice.

        Returns
        -------
        list of Delayed
            One delayed write per variable found in
            predictions.
        '''
        delayed_list: List[Delayed] = []
        for var in self.state_variables:
            if var not in predictions:
                continue
            data = predictions[var][batch_idx]
            n_spatial = len(self._spatial_meta[var])
            region = (
                init_idx,
                slice(lead_start, lead_end),
                ens_idx,
                *(slice(None),) * n_spatial,
            )
            zarr_array = self._store[var]
            delayed = dask.delayed(_zarr_region_write)(
                zarr_array, data, region,
            )
            delayed_list.append(delayed)
        return delayed_list

    def write(
            self,
            predictions: Dict[str, np.ndarray],
            init_times: pd.DatetimeIndex,
            ens_mems: ArrayLike,
            lead_times: pd.TimedeltaIndex,
    ) -> List[Delayed]:
        r'''Write forecast trajectories as lazy writes.

        Returns dask Delayed objects that write directly
        to zarr arrays when computed.

        Parameters
        ----------
        predictions : dict of str to np.ndarray
            Variable names to prediction arrays with shape
            ``(batch, lead_time, ...)``.
        init_times : pd.DatetimeIndex
            Initialization times for each batch element.
        ens_mems : ArrayLike
            Ensemble member indices per batch element.
        lead_times : pd.TimedeltaIndex
            Lead times for the forecast trajectory.

        Returns
        -------
        list of Delayed
            Delayed objects for each variable write.

        Raises
        ------
        ValueError
            If batch or lead-time sizes are inconsistent.
        '''
        init_times = pd.DatetimeIndex(init_times)
        ens_mems = np.asarray(ens_mems)
        lead_times = pd.TimedeltaIndex(lead_times)

        init_indices = np.array([
            np.where(self.init_times == t)[0][0]
            for t in init_times
        ])
        lead_indices = np.array([
            np.where(self.lead_times == t)[0][0]
            for t in lead_times
        ])
        ens_indices = np.array([
            np.where(self.ens_mems == e)[0][0]
            for e in ens_mems
        ])

        for var in self.state_variables:
            if var not in predictions:
                main_logger.warning(
                    f"Variable '{var}' not found in "
                    f"predictions, skipping"
                )
                continue

            pred_data = predictions[var]
            batch_size = pred_data.shape[0]
            n_lead = pred_data.shape[1]

            if batch_size != len(init_times):
                raise ValueError(
                    f"Batch size mismatch for "
                    f"'{var}': predictions has "
                    f"batch={batch_size} but "
                    f"init_times has length "
                    f"{len(init_times)}"
                )
            if n_lead != len(lead_times):
                raise ValueError(
                    f"Lead time mismatch for "
                    f"'{var}': predictions has "
                    f"lead_time={n_lead} but "
                    f"lead_times has length "
                    f"{len(lead_times)}"
                )

        lead_start = int(lead_indices[0])
        lead_end = int(lead_indices[-1]) + 1

        delayed_list: List[Delayed] = []
        batch_size = len(init_times)

        for batch_idx in range(batch_size):
            batch_delayed = self._write_batch_element(
                predictions,
                batch_idx,
                int(init_indices[batch_idx]),
                int(ens_indices[batch_idx]),
                lead_start,
                lead_end,
            )
            delayed_list.extend(batch_delayed)

        return delayed_list

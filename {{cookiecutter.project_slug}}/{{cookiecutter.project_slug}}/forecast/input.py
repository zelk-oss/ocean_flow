#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: {{cookiecutter.author_name}}, {{cookiecutter.author_email}}
#
#    Copyright (C) {{cookiecutter.project_year}}  {{cookiecutter.author_name}}

# System modules
import logging
from typing import Dict, Iterable, Optional

# External modules
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

# Internal modules


main_logger = logging.getLogger(__name__)


__all__ = [
    "InputReader",
    "dataset_to_numpy_dict",
]


def dataset_to_numpy_dict(
        ds: xr.Dataset,
) -> Dict[str, np.ndarray]:
    r'''
    Convert an in-memory xr.Dataset to a dict of arrays.

    Parameters
    ----------
    ds : xr.Dataset
        In-memory Dataset (already computed).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping variable names to numpy arrays.
    '''
    return {
        var: ds[var].values for var in ds.data_vars
    }


class InputReader(object):
    r'''
    Read initial conditions, forcings, and auxiliary data.

    Opens the state dataset (and optionally separate
    forcing/auxiliary datasets) at construction time and
    provides methods for loading batches of data indexed by
    initialization time and ensemble member.

    Parameters
    ----------
    data_path : str
        Path to the primary zarr store containing state
        variables.
    state_variables : Iterable[str]
        Names of the state variables to load.
    forcing_path : str or None, optional
        Path to a separate zarr store for forcings. If
        ``None``, forcings are loaded from the primary
        store. Default is ``None``.
    forcing_variables : Iterable[str] or None, optional
        Names of the forcing variables. Set to ``None`` to
        disable forcing loading. Default is ``None``.
    auxiliary_path : str or None, optional
        Path to a netCDF file for auxiliary data. Auxiliary
        loading is disabled when ``None``.
        Default is ``None``.
    auxiliary_variables : Iterable[str] or None, optional
        Names of the auxiliary variables. Set to ``None``
        to disable auxiliary loading. Default is ``None``.
    n_in_steps : int, optional
        Number of input time steps to load for initial
        conditions and forcing history. Default is ``1``.
    step_freq : str or None, optional
        Frequency string (e.g. ``"6h"``) for spacing
        between input steps. Required when
        ``n_in_steps > 1``. Default is ``None``.
    '''
    def __init__(
            self,
            data_path: str,
            state_variables: Iterable[str],
            forcing_path: Optional[str] = None,
            forcing_variables: Optional[
                Iterable[str]
            ] = None,
            auxiliary_path: Optional[str] = None,
            auxiliary_variables: Optional[
                Iterable[str]
            ] = None,
            n_in_steps: int = 1,
            step_freq: Optional[str] = None,
    ) -> None:
        self._state_dataset: xr.Dataset
        self._forcing_dataset: Optional[
            xr.Dataset
        ] = None
        self._auxiliary_dataset: Optional[
            xr.Dataset
        ] = None
        self.data_path = data_path
        self.state_variables = state_variables
        self.forcing_path = forcing_path
        self.forcing_variables = forcing_variables
        self.auxiliary_path = auxiliary_path
        self.auxiliary_variables = auxiliary_variables
        self.n_in_steps = n_in_steps
        self.step_freq = step_freq
        self._open_datasets()

    @property
    def use_auxiliary(self) -> bool:
        r'''Return True if auxiliary data is configured.'''
        return self._auxiliary_dataset is not None

    @property
    def use_forcings(self) -> bool:
        r'''Return True if forcing data is configured.'''
        return self._forcing_dataset is not None

    def _open_optional_dataset(
            self,
            variables: Optional[list],
            path: Optional[str],
            default_dataset: Optional[xr.Dataset]
    ) -> Optional[xr.Dataset]:
        r'''
        Open a dataset for optional variables.

        Parameters
        ----------
        variables : list of str or None
            Variable names to select. Returns ``None``
            when empty or ``None``.
        path : str or None
            Path to a zarr store to open. When ``None``,
            uses *default_dataset* instead.
        default_dataset : xr.Dataset or None
            Fallback dataset used when *path* is ``None``.

        Returns
        -------
        xr.Dataset or None
            Resulting dataset, or ``None`` if *variables*
            is empty.
        '''
        if not variables:
            return None
        if path is not None:
            ds = xr.open_zarr(path, chunks="auto")
        else:
            ds = default_dataset
        ds = ds[list(variables)]
        if "ensemble" not in ds.dims:
            ds = ds.expand_dims(ensemble=[0])
        return ds

    def _open_auxiliary_dataset(
            self,
    ) -> Optional[xr.Dataset]:
        r'''
        Open the auxiliary dataset when fully configured.

        Auxiliary loading is enabled only when both
        ``self.auxiliary_variables`` and
        ``self.auxiliary_path`` are set. The auxiliary
        source is expected to be a netCDF file.

        Returns
        -------
        xr.Dataset or None
            Loaded auxiliary dataset, or ``None`` when
            auxiliary loading is disabled.

        Raises
        ------
        ValueError
            If one of the requested auxiliary variables
            is missing.
        '''
        if not self.auxiliary_variables:
            return None
        if self.auxiliary_path is None:
            main_logger.warning(
                "Auxiliary variables specified but no "
                "auxiliary path provided. Auxiliary "
                "loading is disabled."
            )
            return None

        dataset = xr.open_dataset(
            self.auxiliary_path, chunks="auto",
        )
        dataset = dataset[list(self.auxiliary_variables)]
        if "ensemble" not in dataset.dims:
            dataset = dataset.expand_dims(ensemble=[0])
        return dataset

    def _open_datasets(self) -> None:
        r'''Open all configured datasets with chunking.'''
        dataset = forcing_dataset = xr.open_zarr(
            self.data_path, chunks="auto",
        )

        self._state_dataset = dataset[
            list(self.state_variables)
        ]
        if "ensemble" not in self._state_dataset.dims:
            self._state_dataset = (
                self._state_dataset.expand_dims(
                    ensemble=[0]
                )
            )

        self._forcing_dataset = (
            self._open_optional_dataset(
                variables=self.forcing_variables,
                path=self.forcing_path,
                default_dataset=forcing_dataset,
            )
        )

        self._auxiliary_dataset = (
            self._open_auxiliary_dataset()
        )

    def _build_time_indexer(
            self,
            init_times: pd.DatetimeIndex,
            lead_times: Optional[
                pd.TimedeltaIndex
            ] = None,
    ) -> xr.DataArray:
        r'''
        Build a batch-first time indexer for selection.

        Parameters
        ----------
        init_times : pd.DatetimeIndex
            Initial times defining the end of each input
            window.
        lead_times : pd.TimedeltaIndex or None, optional
            Lead times to append after the historical
            input window. Default is ``None``.

        Returns
        -------
        xr.DataArray
            Two-dimensional time indexer with dimensions
            ``("batch", "time_step")``.

        Raises
        ------
        ValueError
            If ``n_in_steps > 1`` and ``step_freq`` is
            not configured.
        '''
        n_lead_times = (
            0 if lead_times is None else len(lead_times)
        )
        total_steps = self.n_in_steps + n_lead_times
        time_array = np.empty(
            (len(init_times), total_steps),
            dtype="datetime64[ns]",
        )

        step_delta = pd.Timedelta(0)
        if self.n_in_steps > 1:
            if self.step_freq is None:
                raise ValueError(
                    "step_freq must be set when "
                    "n_in_steps > 1"
                )
            step_delta = pd.Timedelta(self.step_freq)

        for step_index in range(self.n_in_steps):
            offset = (
                (self.n_in_steps - 1 - step_index)
                * step_delta
            )
            time_array[:, step_index] = (
                init_times - offset
            ).values

        if lead_times is not None:
            for lead_index, lead_time in enumerate(
                lead_times
            ):
                time_array[
                    :, self.n_in_steps + lead_index
                ] = (init_times + lead_time).values

        return xr.DataArray(
            time_array,
            dims=["batch", "time_step"],
        )

    def _build_ensemble_indexer(
            self,
            dataset: xr.Dataset,
            ens_mems: ArrayLike,
    ) -> xr.DataArray:
        r'''
        Build an ensemble indexer for vectorised selection.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset with an ``ensemble`` dimension.
        ens_mems : ArrayLike
            Ensemble member indices to select.

        Returns
        -------
        xr.DataArray
            One-dimensional indexer with dim ``"batch"``.
        '''
        n_ens = len(dataset["ensemble"])
        ens_mems = np.asarray(ens_mems) % n_ens
        return xr.DataArray(ens_mems, dims="batch")

    @staticmethod
    def _drop_time_coord(
            ds: xr.Dataset,
    ) -> xr.Dataset:
        r'''
        Drop the ``time`` coordinate if present.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset potentially containing a ``time``
            coordinate.

        Returns
        -------
        xr.Dataset
            Dataset without a ``time`` coordinate.
        '''
        if "time" in ds.coords:  # pragma: no branch
            ds = ds.drop_vars("time")
        return ds

    def load_states(
            self,
            init_times: pd.DatetimeIndex,
            ens_mems: ArrayLike,
    ) -> xr.Dataset:
        r'''
        Load lazy initial conditions as xr.Dataset.

        Parameters
        ----------
        init_times : pd.DatetimeIndex
            The initial times for which to load data.
        ens_mems : ArrayLike
            The ensemble member indices to load.

        Returns
        -------
        xr.Dataset
            Lazy dask-backed Dataset with dims
            ``(batch, time_step, ...)``.
        '''
        time_indexer = self._build_time_indexer(
            init_times,
        )
        ens_indexer = self._build_ensemble_indexer(
            self._state_dataset, ens_mems,
        )

        ds = self._state_dataset.sel(
            time=time_indexer, method="nearest",
        ).isel(
            ensemble=ens_indexer,
        ).transpose(
            "batch", "time_step", ...,
        )
        return self._drop_time_coord(ds)

    def load_auxiliary(
            self,
            ens_mems: ArrayLike,
    ) -> xr.Dataset:
        r'''
        Load lazy auxiliary data as xr.Dataset.

        Parameters
        ----------
        ens_mems : ArrayLike
            The ensemble member indices to load.

        Returns
        -------
        xr.Dataset
            Lazy dask-backed Dataset with dims
            ``(batch, ...)``.
        '''
        if (
            self._auxiliary_dataset is None
            or self.auxiliary_variables is None
        ):
            raise ValueError(
                "No auxiliary dataset configured "
                "for loading"
            )

        ens_indexer = self._build_ensemble_indexer(
            self._auxiliary_dataset, ens_mems,
        )

        ds = self._auxiliary_dataset.isel(
            ensemble=ens_indexer,
        ).transpose(
            "batch", ...,
        )
        return ds

    def load_forcings(
            self,
            init_times: pd.DatetimeIndex,
            ens_mems: ArrayLike,
            lead_times: pd.TimedeltaIndex,
    ) -> xr.Dataset:
        r'''
        Load lazy forcings as xr.Dataset.

        Parameters
        ----------
        init_times : pd.DatetimeIndex
            The initial times for which to load data.
        ens_mems : ArrayLike
            The ensemble member indices to load.
        lead_times : pd.TimedeltaIndex
            The lead times for which to load data.

        Returns
        -------
        xr.Dataset
            Lazy dask-backed Dataset with dims
            ``(batch, time_step, ...)``.
        '''
        if (
            self._forcing_dataset is None
            or self.forcing_variables is None
        ):
            raise ValueError(
                "No forcing dataset configured "
                "for loading"
            )

        time_indexer = self._build_time_indexer(
            init_times=init_times,
            lead_times=lead_times,
        )
        ens_indexer = self._build_ensemble_indexer(
            self._forcing_dataset, ens_mems,
        )

        ds = self._forcing_dataset.sel(
            time=time_indexer, method="nearest",
        ).isel(
            ensemble=ens_indexer,
        ).transpose(
            "batch", "time_step", ...,
        )
        return self._drop_time_coord(ds)

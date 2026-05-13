#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Chiara Zelco, chiara.zelco@ens.psl.eu
#
#    Copyright (C) 2026  Chiara Zelco

r'''Zarr-backed PyTorch dataset for training surrogate models.'''

# System modules
import logging
from typing import Dict, Iterable, List, Optional, Tuple, Union

# External modules
import numpy as np
import tensorstore as ts
import xarray as xr
from torch.utils.data import Dataset

# Internal modules

main_logger = logging.getLogger(__name__)


__all__ = [
    "TrainDataset",
]


def _atleast_3d(array: np.ndarray) -> np.ndarray:
    r'''
    Ensure that the input array has at least 3 dimensions by prepending
    singleton dimensions.

    This is used to ensure that spatial variables have a consistent number of
    dimensions,
    even when they are 2D (e.g. latitude x longitude) rather than 3D
    (e.g. level x latitude x longitude).

    Parameters
    ----------
    array : np.ndarray
        Input array to reshape.

    Returns
    -------
    np.ndarray
        Reshaped array with at least 3 dimensions.
    '''
    if array.ndim >= 3:
        return array
    new_shape = (1,) * (3 - array.ndim) + array.shape
    return array.reshape(new_shape)


class TrainDataset(Dataset):
    r'''
    Dataset class for training using zarr datasets.

    Samples include both input and output time steps for state and
    forcing variables. The auxiliary data (e.g. mesh or land-sea
    mask) can be loaded from an auxiliary netCDF dataset and added
    to the batch.

    Attributes
    ----------
    n_times : int
        Length of the ``time`` dimension in the zarr dataset.
    time_array : np.ndarray
        Raw (CF-encoded) time values, cast to ``float32``.
    n_ensemble : int
        Number of ensemble members. ``1`` when no ensemble
        dimension is present in the dataset.
    use_ensemble : bool
        ``True`` when an ``ensemble`` dimension was detected.
    '''

    n_times: int
    time_array: np.ndarray
    n_ensemble: int
    use_ensemble: bool

    def __init__(
            self,
            data_path: str,
            state_variables: Iterable[str],
            forcing_variables: Optional[Iterable[str]] = None,
            auxiliary_path: Optional[str] = None,
            auxiliary_variables: Optional[Iterable[str]] = None,
            n_steps: int = 2,
            n_step_size: int = 1,
            threads_limit: Union[str, int] = "shared"
    ) -> None:
        r'''
        Initialize the TrainDataset.

        Parameters
        ----------
        data_path : str
            Path to the zarr dataset.
        state_variables : Iterable[str]
            List of state variable names to include in the samples.
        forcing_variables : Iterable[str], optional
            List of forcing variable names to include in the samples.
            If ``None``, no forcing variables are included.
        auxiliary_path : str, optional
            Path to an auxiliary netCDF dataset containing static variables
            (e.g. mesh, land-sea mask). If ``None``, no auxiliary data is
            included.
        auxiliary_variables : Iterable[str], optional
            List of variable names to retrieve from the auxiliary dataset.
            Only used if ``auxiliary_path`` is not ``None``.
        n_steps : int, optional
            Number of time steps to include in each sample (default is 2).
        n_step_size : int, optional
            Step size between time steps (default is 1).

        Raises
        ------
        ValueError
            If any of the specified state or forcing variables are not found in
            the dataset, if any of the specified auxiliary variables are not
            found in the auxiliary dataset, or if an auxiliary dataset's
            ensemble size does not match the main dataset's ensemble size.
        '''
        self._datasets: Optional[Dict[str, ts.TensorStore]] = None
        self._auxiliary_arrays: Dict[str, np.ndarray] = {}

        self.data_path = data_path
        self.state_variables = list(state_variables)
        self.forcing_variables = (
            list(forcing_variables)
            if forcing_variables is not None
            else []
        )
        self.auxiliary_path = auxiliary_path
        self.auxiliary_variables = (
            list(auxiliary_variables)
            if auxiliary_variables is not None
            else []
        )
        self.n_steps = n_steps
        self.n_step_size = n_step_size
        self._step_shift = (self.n_steps - 1) * self.n_step_size

        self.threads_limit = threads_limit

        self._check_metadata()
        self._load_auxiliary_dataset()

    @property
    def datasets(self) -> Dict[str, ts.TensorStore]:
        r'''
        Lazily-loaded tensorstore datasets, keyed by variable name.

        The underlying tensorstore arrays are opened on first
        access so that DataLoader worker processes each open
        their own handles after forking, avoiding shared-state
        issues with zarr.

        Returns
        -------
        Dict[str, ts.TensorStore]
            Mapping from variable name to the opened tensorstore
            array.
        '''
        if self._datasets is None:
            self._datasets = self._setup_datasets()
        return self._datasets

    def _setup_datasets(self) -> Dict[str, ts.TensorStore]:
        r'''
        Load the tensorstore datasets per variable.

        Opens one tensorstore array per variable (state, forcing
        and ``time``) under ``data_path``, sharing a single
        :class:`tensorstore.Context` so concurrency limits are
        pooled across variables. Opens are dispatched together
        and resolved at the end to amortise metadata reads.

        Called on each worker process after forking, for more
        efficient memory usage and to avoid issues with
        multiprocessing and zarr datasets.

        Returns
        -------
        Dict[str, Any]
            Mapping from variable name to the opened tensorstore
            array.
        '''
        context = ts.Context({
            'file_io_concurrency': {'limit': self.threads_limit},
            'data_copy_concurrency': {'limit': self.threads_limit},
        })
        futures = {
            name: ts.open(
                {
                    'driver': 'zarr',
                    'metadata_key': '.zarray',
                    'kvstore': {
                        'driver': 'file',
                        'path': f'{self.data_path}/{name}',
                    },
                },
                context=context,
            )
            for name in self.variables
        }
        return {
            name: future.result()
            for name, future in futures.items()
        }

    def _check_metadata(self) -> None:
        r'''
        Populate metadata attributes from the zarr dataset.

        Uses xarray for lightweight metadata inspection and
        assigns ``n_times``, ``time_array``, ``n_ensemble``,
        and ``use_ensemble`` on ``self``. Also validates that
        every requested variable exists with the expected
        leading dimensions.

        Raises
        ------
        KeyError
            If the ``time`` dimension or any requested variable
            is missing from the dataset.
        ValueError
            If the dataset is too short for the configured
            ``n_steps`` / ``n_step_size`` or if a variable has
            unexpected leading dimensions.
        '''
        with xr.open_zarr(self.data_path) as ds:
            self.n_times, self.time_array = self._infer_time_len(ds)
            self.n_ensemble, self.use_ensemble = self._infer_ensemble(ds)
            self._check_variables_in_dataset(
                ds,
                self.variables,
                ["time", "ensemble"] if self.use_ensemble
                else ["time"]
            )

    def _infer_time_len(
            self, ds: xr.Dataset,
    ) -> Tuple[int, np.ndarray]:
        r'''
        Infer the time-axis length and extract raw time values.

        Parameters
        ----------
        ds : xr.Dataset
            Open xarray view onto the zarr dataset.

        Returns
        -------
        Tuple[int, np.ndarray]
            Length of the ``time`` dimension and the raw time
            values cast to ``float32``.

        Raises
        ------
        KeyError
            If the dataset has no ``time`` dimension.
        ValueError
            If the dataset has fewer time steps than required to
            construct a single sample.
        '''
        try:
            n_times = len(ds["time"])
        except KeyError:
            raise KeyError(
                "Necessary time dimension is not included in dataset"
            )
        if n_times < self._step_shift + 1:
            raise ValueError(
                f"Dataset has only {n_times} time steps, but "
                f"n_steps={self.n_steps} and "
                f"n_step_size={self.n_step_size} require at least "
                f"{self._step_shift + 1} steps to construct a sample"
            )
        time_array = np.asarray(ds["time"].values, dtype=np.float32)
        main_logger.debug(
            "Inferred time dimension of length %d", n_times,
        )
        return n_times, time_array

    def _infer_ensemble(
            self, ds: xr.Dataset,
    ) -> Tuple[int, bool]:
        r'''
        Detect an optional ``ensemble`` dimension.

        Parameters
        ----------
        ds : xr.Dataset
            Open xarray view onto the zarr dataset.

        Returns
        -------
        Tuple[int, bool]
            Number of ensemble members and a flag indicating
            whether the ensemble dimension was present. Returns
            ``(1, False)`` when absent.
        '''
        if "ensemble" in ds.dims:
            n_ensemble = len(ds["ensemble"])
            main_logger.debug(
                "Inferred ensemble dimension of length %d", n_ensemble,
            )
            return n_ensemble, True
        main_logger.debug("Deactivated ensemble dimension")
        return 1, False

    def _load_auxiliary_dataset(self) -> None:
        r'''
        Load auxiliary/static variables from a netCDF dataset if provided.

        The auxiliary dataset is subset to ``auxiliary_variables`` and loaded
        into memory.
        '''
        if self.auxiliary_path is not None and self.auxiliary_variables:
            with xr.open_dataset(self.auxiliary_path, engine="netcdf4") as ds:
                for var_name in self.auxiliary_variables:
                    var_data = ds[var_name].values
                    var_data = np.asarray(var_data, dtype=np.float32)
                    var_data = _atleast_3d(var_data)
                    var_data = np.nan_to_num(var_data, nan=0.0)
                    self._auxiliary_arrays[var_name] = var_data
        elif self.auxiliary_path is not None:
            main_logger.warning(
                "Auxiliary path provided but no auxiliary variables "
                "specified. No auxiliary data will be included in "
                "the samples."
            )
        elif self.auxiliary_variables:
            main_logger.warning(
                "Auxiliary variables specified but no auxiliary path "
                "provided. No auxiliary data will be included in "
                "the samples."
            )
        else:
            main_logger.info(
                "No auxiliary data will be included in the samples."
            )

    def _check_variables_in_dataset(
            self,
            ds: xr.Dataset,
            var_names: List[str],
            dimensions: List[str],
    ) -> None:
        r'''
        Verify that every variable exists with the expected leading dims.

        Parameters
        ----------
        ds : xr.Dataset
            Open xarray view onto the zarr dataset.
        var_names : List[str]
            Names of variables that must be present.
        dimensions : List[str]
            Expected leading dimension names, in order.

        Raises
        ------
        KeyError
            If any variable is missing from the dataset.
        ValueError
            If any variable's leading dimensions do not match
            ``dimensions``.
        '''
        n_dims = len(dimensions)
        for var_name in var_names:
            try:
                var_dims = ds[var_name].dims
            except KeyError:
                raise KeyError(
                    f"Variable {var_name:s} is missing in the dataset."
                )
            if not list(var_dims[:n_dims]) == dimensions:
                raise ValueError(
                    f"The leading dimensions are not correct for "
                    f"{var_name:s}. Expected: {dimensions}, "
                    f"got {var_dims[:n_dims]}"
                )
        main_logger.debug(
            "All variables are included and have correct leading dimensions."
        )

    @property
    def variables(self) -> List[str]:
        r'''
        List of variable names included in the samples.

        Returns
        -------
        List[str]
            List of variable names.
        '''
        return self.state_variables + self.forcing_variables

    def __len__(self) -> int:
        r'''
        Total number of samples, determined by the time axis length,
        the ``n_steps`` and ``n_step_size`` constructor parameters, and the
        number of ensemble members when an ensemble dimension is present.

        Returns
        -------
        int
            Total number of samples in the dataset.
        '''
        curr_len = self.n_times - self._step_shift
        curr_len = curr_len * self.n_ensemble
        return curr_len

    def _get_var(
            self,
            var_name: str,
            time_slices: slice,
            ens_idx: int = 0,
    ) -> np.ndarray:
        r'''
        Retrieve samples for a specific variable at given time indices.

        Parameters
        ----------
        var_name : str
            Name of the variable to retrieve.
        time_slices : slice
            Time indices to slice from the dataset.
        ens_idx : int, optional
            Index of the ensemble member to retrieve (default is 0).
            Slicing by ensemble is only applied when an ensemble dimension was
            detected in the dataset.

        Returns
        -------
        np.ndarray
            Retrieved variable samples with shape ``(n_steps, *spatial_dims)``.
        '''
        var_data = self.datasets[var_name]
        if self.use_ensemble:
            var_data = var_data[time_slices, ens_idx]
        else:
            var_data = var_data[time_slices]
        var_data = var_data.read().result()
        var_data = np.asarray(var_data, dtype=np.float32)
        var_data = np.nan_to_num(var_data, nan=0.0)
        return var_data

    def _get_data_sample(
            self,
            time_slices: slice,
            ens_idx: int = 0,
    ) -> Dict[str, np.ndarray]:
        r'''
        Retrieve a mapping of variable names to their sampled arrays for a
        given starting time index and ensemble index.

        Parameters
        ----------
        time_slices : slice
            Time indices to slice from the datasets.
        ens_idx : int, optional
            Ensemble member index to retrieve (default is 0).

        Returns
        -------
        Dict[str, np.ndarray]
            Mapping from variable name to ndarray shaped ``(n_steps, ...)``.
        '''
        sample = {
            var_name: self._get_var(var_name, time_slices, ens_idx)
            for var_name in self.variables
        }
        return sample

    def _assemble_training_sample(
            self,
            sample: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        r'''
        Convert raw state samples to training inputs and residual targets.

        The training module expects each batch to contain ``input`` and
        ``residual`` keys. ``input`` is the state at the first time step,
        and ``residual`` is the change to the last time step.
        '''
        if self.state_variables:
            input_parts = []
            residual_parts = []
            for state_var in self.state_variables:
                state_series = sample[state_var]
                state_t = state_series[0]
                state_tdt = state_series[-1]
                if state_t.ndim == 2:
                    state_t = state_t[None, ...]
                    state_tdt = state_tdt[None, ...]
                input_parts.append(state_t)
                residual_parts.append(state_tdt - state_t)
            sample["input"] = np.concatenate(input_parts, axis=0)
            sample["residual"] = np.concatenate(residual_parts, axis=0)
        return sample

    def __getitem__(
            self, idx: int
    ) -> Dict[str, np.ndarray]:
        r'''
        Retrieve a sample consisting of state and forcing variables.

        The returned dictionary also contains a ``"time"`` key with the
        raw (CF-encoded) time value, sliced across all included time steps.

        When an ensemble dimension is present, ``idx`` is decomposed as:
        ``time_idx = idx // n_ensemble`` and
        ``ensemble_idx = idx % n_ensemble``.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with one entry per variable plus ``"time"``.
        '''
        # Determine selected time and ensemble member
        time_idx = idx // self.n_ensemble
        time_slice = slice(
            time_idx,
            time_idx + self.n_steps * self.n_step_size,
            self.n_step_size,
        )
        ens_idx = idx % self.n_ensemble

        # Load sample for selected time and ensemble member
        sample = self._get_data_sample(time_slice, ens_idx)
        sample.update(self._auxiliary_arrays)
        sample["time"] = self.time_array[time_slice]
        sample = self._assemble_training_sample(sample)
        return sample

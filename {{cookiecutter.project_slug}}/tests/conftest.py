#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: {{cookiecutter.author_name}}, {{cookiecutter.author_email}}
#
#    Copyright (C) {{cookiecutter.project_year}}  {{cookiecutter.author_name}}
r"""Shared test fixtures for the {{cookiecutter.project_name}} test suite."""

# System modules
import logging
import pathlib
from typing import (
    Any, Callable, Dict, Generator, List, Optional,
)

# External modules
import lightning.fabric
import pandas as pd
import torch
import numpy as np
import pytest
import xarray as xr
import zarr

# Internal modules
from {{cookiecutter.project_slug}}.forecast.output import OutputWriter
from {{cookiecutter.project_slug}}.forecast.validation import (
    create_output_store,
)
from {{cookiecutter.project_slug}}.pipelines import (
    PrePipeline, PostPipeline,
)


main_logger = logging.getLogger(__name__)


def _create_test_zarr(
        path: str,
        n_times: int = 20,
        n_ens: Optional[int] = None,
        n_lat: int = 4,
        n_lon: int = 8,
        n_surf_vars: int = 2,
        n_lev_vars: int = 2,
        n_levels: int = 3,
) -> zarr.Group:
    r'''
    Create a test zarr store with synthetic data.

    Generates a zarr store with ``states_surface`` and
    ``states_levels`` variables. When ``n_ens`` is provided,
    an ensemble dimension is included.

    Parameters
    ----------
    path : str
        Path to the zarr store to create.
    n_times : int, optional, default = 20
        Number of time steps.
    n_ens : int or None, optional, default = None
        Number of ensemble members. If ``None``, no ensemble
        dimension is added.
    n_lat : int, optional, default = 4
        Number of latitude grid points.
    n_lon : int, optional, default = 8
        Number of longitude grid points.
    n_surf_vars : int, optional, default = 2
        Number of surface variables.
    n_lev_vars : int, optional, default = 2
        Number of level variables.
    n_levels : int, optional, default = 3
        Number of vertical levels.

    Returns
    -------
    zarr.Group
        The opened zarr group at the specified path.
    '''
    times = np.array([
        np.datetime64('2020-01-01')
        + np.timedelta64(6 * i, 'h')
        for i in range(n_times)
    ])
    rng = np.random.default_rng(seed=20260225)
    coords = {
        "time": times,
        "latitude": np.linspace(-90, 90, n_lat),
        "longitude": np.linspace(0, 360, n_lon),
    }
    if n_ens is not None:
        surf_shape = (
            n_times, n_ens, n_surf_vars, n_lat, n_lon
        )
        lev_shape = (
            n_times, n_ens, n_lev_vars, n_levels,
            n_lat, n_lon,
        )
        surf_dims = (
            "time", "ensemble", "variable",
            "latitude", "longitude",
        )
        lev_dims = (
            "time", "ensemble", "variable", "level",
            "latitude", "longitude",
        )
        coords["ensemble"] = np.arange(n_ens)
    else:
        surf_shape = (
            n_times, n_surf_vars, n_lat, n_lon
        )
        lev_shape = (
            n_times, n_lev_vars, n_levels, n_lat, n_lon
        )
        surf_dims = (
            "time", "variable", "latitude", "longitude",
        )
        lev_dims = (
            "time", "variable", "level",
            "latitude", "longitude",
        )
    surf_data = rng.normal(size=surf_shape)
    lev_data = rng.normal(size=lev_shape)
    if n_ens is not None:
        surf_data = surf_data.astype(np.float32)
        lev_data = lev_data.astype(np.float32)
    ds = xr.Dataset(
        {
            "states_surface": (surf_dims, surf_data),
            "states_levels": (lev_dims, lev_data),
        },
        coords=coords,
    )
    ds.to_zarr(path, mode="w", consolidated=True)
    return zarr.open_group(path, mode='r')


def _create_test_auxiliary_netcdf(
        path: str,
        n_ens: Optional[int] = None,
        n_lat: int = 4,
        n_lon: int = 8,
) -> xr.Dataset:
    r'''
    Create a test auxiliary netCDF file with synthetic static
    data.

    Generates a netCDF file with ``mesh`` and ``mask``
    variables. When ``n_ens`` is provided, an ensemble
    dimension is included.

    Parameters
    ----------
    path : str
        Path to the netCDF file to create.
    n_ens : int or None, optional, default = None
        Number of ensemble members. If ``None``, no ensemble
        dimension is added.
    n_lat : int, optional, default = 4
        Number of latitude grid points.
    n_lon : int, optional, default = 8
        Number of longitude grid points.

    Returns
    -------
    xr.Dataset
        The created dataset, also saved to the netCDF path.
    '''
    rng = np.random.default_rng(seed=20260225)
    if n_ens is not None:
        mesh = rng.random(
            (n_ens, 2, n_lat, n_lon)
        ).astype(np.float32)
        mask = rng.integers(
            0, 2, size=(n_ens, n_lat, n_lon),
            dtype=np.int8,
        )
        ds = xr.Dataset({
            "mesh": (
                (
                    "ensemble", "channel",
                    "latitude", "longitude",
                ),
                mesh,
            ),
            "mask": (
                ("ensemble", "latitude", "longitude"),
                mask,
            ),
        })
    else:
        mesh = rng.random(
            (2, n_lat, n_lon)
        ).astype(np.float32)
        mask = rng.integers(
            0, 2, size=(n_lat, n_lon), dtype=np.int8,
        )
        ds = xr.Dataset({
            "mesh": (
                ("channel", "latitude", "longitude"),
                mesh,
            ),
            "mask": (
                ("latitude", "longitude"),
                mask,
            ),
        })
    ds.to_netcdf(path)
    return ds



def make_fabric(
        accelerator: str = "cpu",
        devices: int = 1,
        precision: str = "32-true",
) -> lightning.fabric.Fabric:
    r'''Create a Fabric instance for tests.

    Plain module-level factory (not a pytest fixture) that
    mirrors ``_create_test_zarr`` in purpose: shared setup
    logic callable from any test file without fixture
    injection.

    Parameters
    ----------
    accelerator : str, optional
        Fabric accelerator string, e.g. ``"cpu"`` or
        ``"gpu"``. Default is ``"cpu"``.
    devices : int, optional
        Number of devices to use. Default is ``1``.
    precision : str, optional
        Precision mode string, e.g. ``"32-true"`` or
        ``"bf16-mixed"``. Default is ``"32-true"``.

    Returns
    -------
    lightning.fabric.Fabric
        Configured ``Fabric`` instance.
    '''
    return lightning.fabric.Fabric(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
    )


@pytest.fixture()
def tmp_zarr_path(tmp_path: pathlib.Path) -> str:
    r'''
    Return a temporary file path for zarr stores.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory.

    Returns
    -------
    str
        A string path to a temporary zarr file.
    '''
    return str(tmp_path / "test.zarr")


@pytest.fixture()
def tmp_auxiliary_netcdf_path(
        tmp_path: pathlib.Path,
) -> str:
    r'''
    Return a temporary file path for auxiliary netCDF files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory.

    Returns
    -------
    str
        A string path to a temporary netCDF file.
    '''
    return str(tmp_path / "auxiliary.nc")


@pytest.fixture()
def zarr_store(tmp_path: pathlib.Path) -> zarr.Group:
    r'''
    Create a test zarr store with synthetic data.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory.

    Returns
    -------
    zarr.Group
        The opened zarr group at the created store path.
    '''
    path = str(tmp_path / "store.zarr")
    return _create_test_zarr(path)


@pytest.fixture()
def ens_zarr_store(
        tmp_path: pathlib.Path,
) -> zarr.Group:
    r'''
    Create a test zarr store with synthetic ensemble data.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory.

    Returns
    -------
    zarr.Group
        The opened zarr group at the created ensemble store path.
    '''
    path = str(tmp_path / "ens_store.zarr")
    return _create_test_zarr(path, n_ens=4)


@pytest.fixture()
def auxiliary_netcdf(
        tmp_path: pathlib.Path,
) -> xr.Dataset:
    r'''
    Create a test auxiliary netCDF file with static data.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory.

    Returns
    -------
    xr.Dataset
        The created dataset, also saved to netCDF.
    '''
    path = str(tmp_path / "auxiliary.nc")
    ds = _create_test_auxiliary_netcdf(path)
    return ds


@pytest.fixture()
def auxiliary_ens_netcdf(
        tmp_path: pathlib.Path,
) -> xr.Dataset:
    r'''
    Create a test auxiliary netCDF file with ensemble data.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory.

    Returns
    -------
    xr.Dataset
        The created ensemble dataset, also saved to netCDF.
    '''
    path = str(tmp_path / "auxiliary.nc")
    ds = _create_test_auxiliary_netcdf(path, n_ens=4)
    return ds


@pytest.fixture()
def mocked_auxiliary_netcdf(
        tmp_path: pathlib.Path,
) -> Callable:
    r'''
    Factory fixture that creates auxiliary netCDF files.

    Returns a callable that generates auxiliary netCDF files
    with configurable dimensions and returns the file path.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory.

    Returns
    -------
    Callable
        A function accepting ``n_ens``, ``n_lat``, ``n_lon``
        and returning the path to the created netCDF file.
    '''
    counter = 0

    def _factory(
            n_ens: Optional[int] = None,
            n_lat: int = 4,
            n_lon: int = 8,
    ) -> str:
        nonlocal counter
        counter += 1
        path = str(
            tmp_path / f"aux_{counter}.nc"
        )
        _create_test_auxiliary_netcdf(
            path, n_ens=n_ens,
            n_lat=n_lat, n_lon=n_lon,
        )
        return path

    return _factory


@pytest.fixture()
def data_dir(tmp_path: pathlib.Path) -> str:
    r'''
    Create a temporary data directory with zarr stores.

    Creates train, val, and test zarr stores plus an
    auxiliary netCDF file in a temporary directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory.

    Returns
    -------
    str
        Path to the temporary data directory.
    '''
    data_path = tmp_path / "data"
    data_path.mkdir()
    _create_test_zarr(str(data_path / "train.zarr"))
    _create_test_zarr(str(data_path / "val.zarr"))
    _create_test_zarr(str(data_path / "test.zarr"))
    _create_test_auxiliary_netcdf(
        str(data_path / "auxiliary.nc")
    )
    return str(data_path)


class IdentityPreModule(torch.nn.Module):
    r'''
    Identity pre-processing module.

    Passes data through unchanged for testing purposes.
    '''
    def forward(
            self,
            in_tensor: torch.Tensor,
            *args: object,
            **kwargs: object,
    ) -> torch.Tensor:
        r'''
        Return input tensor unchanged.

        Parameters
        ----------
        in_tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            The same input tensor, unchanged.
        '''
        return in_tensor


class IdentityPostModule(torch.nn.Module):
    r'''
    Identity post-processing module.

    Passes predictions through unchanged for testing.
    '''
    def forward(
            self,
            prediction: torch.Tensor,
            initial: torch.Tensor,
            *args: object,
            **kwargs: object,
    ) -> torch.Tensor:
        r'''
        Return prediction unchanged.

        Parameters
        ----------
        prediction : torch.Tensor
            Predicted tensor.
        initial : torch.Tensor
            Initial state tensor (unused).

        Returns
        -------
        torch.Tensor
            The prediction tensor, unchanged.
        '''
        return prediction

    def to_latent(
            self,
            target: torch.Tensor,
            initial: torch.Tensor,
            *args: object,
            **kwargs: object,
    ) -> torch.Tensor:
        r'''
        Return target unchanged.

        Parameters
        ----------
        target : torch.Tensor
            Target tensor.
        initial : torch.Tensor
            Initial state tensor (unused).

        Returns
        -------
        torch.Tensor
            The target tensor, unchanged.
        '''
        return target


@pytest.fixture()
def pre_pipeline() -> PrePipeline:
    r'''
    Return an identity pre-processing pipeline.

    Returns
    -------
    PrePipeline
        Pipeline with identity modules for surface and
        level variables.
    '''
    return PrePipeline(
        states_surface=IdentityPreModule(),
        states_levels=IdentityPreModule(),
    )


@pytest.fixture()
def post_pipeline() -> PostPipeline:
    r'''
    Return an identity post-processing pipeline.

    Returns
    -------
    PostPipeline
        Pipeline with identity modules for surface and
        level variables.
    '''
    return PostPipeline(
        states_surface=IdentityPostModule(),
        states_levels=IdentityPostModule(),
    )


class DummyNetwork(torch.nn.Module):
    r'''
    Dummy network that outputs zeros with correct shape.

    The output shape matches the test zarr store dimensions
    for surface and level variables.

    Attributes
    ----------
    n_surf_vars : int
        Number of surface variables.
    n_lev_vars : int
        Number of level variables.
    n_levels : int
        Number of vertical levels.
    '''
    def __init__(self) -> None:
        super().__init__()
        self.n_surf_vars = 2
        self.n_lev_vars = 2
        self.n_levels = 3
        # A parameter so the module is not empty
        self.linear = torch.nn.Linear(1, 1)

    def forward(
            self,
            x: torch.Tensor,
            **kwargs: object,
    ) -> torch.Tensor:
        r'''
        Return zeros with the expected output shape.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape
            ``(batch, channels, lat, lon)``.

        Returns
        -------
        torch.Tensor
            Zero tensor with shape
            ``(batch, out_channels, lat, lon)``.
        '''
        batch = x.size(0)
        n_lat, n_lon = x.shape[-2], x.shape[-1]
        out_channels = (
            self.n_surf_vars
            + self.n_lev_vars * self.n_levels
        )
        return torch.zeros(
            batch, out_channels, n_lat, n_lon,
        )


@pytest.fixture()
def dummy_network() -> DummyNetwork:
    r'''
    Return an instance of the DummyNetwork.

    Returns
    -------
    DummyNetwork
        A dummy network for testing.
    '''
    return DummyNetwork()


# ------------------------------------------------------------------
# Forecast test helper constants
# ------------------------------------------------------------------

_FORECAST_VARS = ["states_surface", "states_levels"]
_FORECAST_INIT_TIMES = pd.date_range(
    "2020-01-01", periods=3, freq="6h",
)
_FORECAST_LEAD_TIMES = pd.timedelta_range(
    start="0h", periods=4, freq="6h",
)


# ------------------------------------------------------------------
# Forecast test helpers
# ------------------------------------------------------------------

def _pre_create_output_store(
    data_path: str,
    store_path: str,
    state_variables: Optional[List[str]] = None,
    init_times: Optional[pd.DatetimeIndex] = None,
    lead_times: Optional[pd.TimedeltaIndex] = None,
    ens_mems: Optional[np.ndarray] = None,
    n_store_freq: Optional[int] = None,
) -> None:
    r'''Pre-create output zarr store via create_output_store.

    Uses ``create_output_store`` from ``forecast.validation``
    to set up the output zarr store before constructing an
    ``OutputWriter``.

    Parameters
    ----------
    data_path : str
        Path to the reference zarr store.
    store_path : str
        Path for the output zarr store.
    state_variables : list of str, optional
        Variables to include.
        Defaults to ``_FORECAST_VARS``.
    init_times : pd.DatetimeIndex, optional
        Initialization times.
        Defaults to ``_FORECAST_INIT_TIMES``.
    lead_times : pd.TimedeltaIndex, optional
        Lead times.
        Defaults to ``_FORECAST_LEAD_TIMES``.
    ens_mems : np.ndarray, optional
        Ensemble members.
        Defaults to ``np.arange(2)``.
    n_store_freq : int, optional
        Lead-time chunk size.
        Defaults to ``len(lead_times)``.
    '''
    variables = state_variables or _FORECAST_VARS
    itimes = (
        init_times
        if init_times is not None
        else _FORECAST_INIT_TIMES
    )
    ltimes = (
        lead_times
        if lead_times is not None
        else _FORECAST_LEAD_TIMES
    )
    ens = (
        ens_mems
        if ens_mems is not None
        else np.arange(2)
    )
    # Ensure unique ensemble members for store creation
    ens_array = np.asarray(ens)
    if ens_array.ndim > 0:
        ens = np.unique(ens_array)
    freq = (
        n_store_freq
        if n_store_freq is not None
        else len(ltimes)
    )
    create_output_store(
        data_path=data_path,
        state_variables=variables,
        store_path=store_path,
        init_times=itimes,
        lead_times=ltimes,
        ens_mems=ens,
        n_store_freq=freq,
    )


def _make_writer(
    tmp_path: Any,
    data_path: str,
    store_name: str = "output.zarr",
    state_variables: Optional[List[str]] = None,
    init_times: Optional[pd.DatetimeIndex] = None,
    lead_times: Optional[pd.TimedeltaIndex] = None,
    ens_mems: Any = 2,
    n_store_freq: Optional[int] = None,
) -> OutputWriter:
    r'''Construct an OutputWriter with sensible defaults.

    Pre-creates the output zarr store, then constructs the
    ``OutputWriter``.

    Parameters
    ----------
    tmp_path : pathlib.Path or Any
        Temporary directory for output.
    data_path : str
        Path to the reference zarr store.
    store_name : str, optional
        Name of the output zarr file.
    state_variables : list of str, optional
        Variables to write.
    init_times : pd.DatetimeIndex, optional
        Initialization times.
    lead_times : pd.TimedeltaIndex, optional
        Lead times.
    ens_mems : int or array-like, optional
        Ensemble members.
    n_store_freq : int, optional
        Lead-time chunk size.

    Returns
    -------
    OutputWriter
        Configured writer instance.
    '''
    variables = state_variables or _FORECAST_VARS
    itimes = (
        init_times
        if init_times is not None
        else _FORECAST_INIT_TIMES
    )
    ltimes = (
        lead_times
        if lead_times is not None
        else _FORECAST_LEAD_TIMES
    )
    store_path = str(tmp_path / store_name)
    freq = (
        n_store_freq
        if n_store_freq is not None
        else len(ltimes)
    )

    # Pre-create output store before constructing writer
    _pre_create_output_store(
        data_path=data_path,
        store_path=store_path,
        state_variables=variables,
        init_times=itimes,
        lead_times=ltimes,
        ens_mems=ens_mems,
        n_store_freq=freq,
    )

    return OutputWriter(
        data_path=data_path,
        state_variables=variables,
        store_path=store_path,
        init_times=itimes,
        lead_times=ltimes,
        ens_mems=ens_mems,
    )


def _make_predictions(
    batch: int = 2,
    n_lead: int = 4,
    n_surf_var: int = 2,
    n_lev_var: int = 2,
    n_levels: int = 3,
    n_lat: int = 4,
    n_lon: int = 8,
) -> Dict[str, np.ndarray]:
    r'''Create dummy prediction dict matching test zarr.

    Parameters
    ----------
    batch : int, optional
        Batch size.
    n_lead : int, optional
        Number of lead time steps.
    n_surf_var : int, optional
        Number of surface variables.
    n_lev_var : int, optional
        Number of level variables.
    n_levels : int, optional
        Number of vertical levels.
    n_lat : int, optional
        Number of latitude grid points.
    n_lon : int, optional
        Number of longitude grid points.

    Returns
    -------
    dict of str to np.ndarray
        Prediction arrays keyed by variable name.
    '''
    np.random.seed(0)
    return {
        "states_surface": np.random.randn(
            batch, n_lead, n_surf_var, n_lat, n_lon
        ).astype(np.float32),
        "states_levels": np.random.randn(
            batch, n_lead, n_lev_var, n_levels,
            n_lat, n_lon
        ).astype(np.float32),
    }


class _ConfigDouble:
    r'''Minimal config-like double with leadtime iterator.

    Attributes
    ----------
    init_times : pd.DatetimeIndex
        Initialization times for the config.
    ens_mems : Any
        Ensemble member indices.
    '''

    def __init__(
            self,
            init_times: pd.DatetimeIndex,
            ens_mems: Any,
            chunks: List[pd.TimedeltaIndex],
    ) -> None:
        self.init_times = init_times
        self.ens_mems = ens_mems
        self._chunks = chunks

    def get_leadtime_iterator(
            self,
    ) -> Generator:
        r'''Yield configured lead-time chunks.

        Yields
        ------
        pd.TimedeltaIndex
            A chunk of lead times.
        '''
        for chunk in self._chunks:
            yield chunk


def _make_config_double(
        n_chunks: int = 2,
        steps_per_chunk: int = 2,
        step_hours: int = 6,
) -> _ConfigDouble:
    r'''Build a _ConfigDouble with evenly spaced chunks.

    Parameters
    ----------
    n_chunks : int
        Number of lead-time chunks.
    steps_per_chunk : int
        Number of lead-time steps per chunk.
    step_hours : int
        Hours between each step.

    Returns
    -------
    _ConfigDouble
        A lightweight forecast config double.
    '''
    init_times = pd.DatetimeIndex(["2020-01-01"])
    ens_mems = np.array([0])
    chunks = []
    for i in range(n_chunks):
        start = (
            (i * steps_per_chunk + 1) * step_hours
        )
        end = (i + 1) * steps_per_chunk * step_hours
        chunk = pd.timedelta_range(
            start=f"{start}h",
            end=f"{end}h",
            freq=f"{step_hours}h",
        )
        chunks.append(chunk)
    return _ConfigDouble(
        init_times=init_times,
        ens_mems=ens_mems,
        chunks=chunks,
    )

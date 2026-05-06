# -*- coding: utf-8 -*-
r"""Tests for forecast/validation.py -- global validators.

Tests are organized into three classes:

- **TestValidationFunctional**: end-to-end happy-path tests
  for all seven public validators and store creation.
- **TestValidationErrors**: expected FileNotFoundError and
  ValueError conditions.
- **TestCreateOutputStoreUnittest**: unit tests for the
  create_output_store helper covering dims, chunks,
  overwrite, NaN fill, and skip-when-exists.
"""

# System modules
from pathlib import Path
from typing import List

# External modules
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

# Internal modules
from ocean_flow.forecast.validation import (
    validate_initial_conditions,
    validate_auxiliary,
    validate_forcing,
    validate_checkpoint,
    validate_output_store,
    create_output_store,
    validate_dask_addresses,
    validate_restart_config,
)
from tests.conftest import (
    _create_test_zarr,
    _create_test_auxiliary_netcdf,
)


# -----------------------------------------------------------
# Module-level constants
# -----------------------------------------------------------

_STATE_VARS: List[str] = [
    "states_surface",
    "states_levels",
]
_INIT_TIMES = pd.date_range(
    "2020-01-01", periods=3, freq="6h",
)
_LEAD_TIMES = pd.timedelta_range(
    start="0h", periods=4, freq="6h",
)
_ENS_MEMS = np.arange(2)


# -----------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------

@pytest.fixture()
def data_zarr_path(tmp_path: Path) -> str:
    r"""Create a reference zarr store and return its path."""
    path = str(tmp_path / "ref.zarr")
    _create_test_zarr(path)
    return path


@pytest.fixture()
def aux_netcdf_path(tmp_path: Path) -> str:
    r"""Create an auxiliary netCDF file and return path."""
    path = str(tmp_path / "auxiliary.nc")
    _create_test_auxiliary_netcdf(path)
    return path


@pytest.fixture()
def output_store_path(tmp_path: Path) -> str:
    r"""Return a path for the output zarr store."""
    return str(tmp_path / "output.zarr")


@pytest.fixture()
def valid_output_store(
    data_zarr_path: str,
    output_store_path: str,
) -> str:
    r"""Create a valid output store and return its path."""
    create_output_store(
        data_path=data_zarr_path,
        state_variables=_STATE_VARS,
        store_path=output_store_path,
        init_times=_INIT_TIMES,
        lead_times=_LEAD_TIMES,
        ens_mems=_ENS_MEMS,
        n_store_freq=len(_LEAD_TIMES),
        recreate=True,
    )
    return output_store_path


@pytest.fixture()
def ckpt_file(tmp_path: Path) -> str:
    r"""Create a dummy checkpoint file and return path."""
    path = tmp_path / "model.ckpt"
    path.write_text("dummy")
    return str(path)


# -----------------------------------------------------------
# Functional tests
# -----------------------------------------------------------

class TestValidationFunctional:
    r"""End-to-end happy-path tests for validators."""

    def test_validate_initial_conditions_passes_valid_store(
        self,
        data_zarr_path: str,
    ) -> None:
        r"""Valid zarr with matching variables passes."""
        # Arrange
        variables = _STATE_VARS

        # Act / Assert -- no exception raised
        validate_initial_conditions(
            data_path=data_zarr_path,
            state_variables=variables,
        )

    def test_validate_auxiliary_passes_valid_netcdf(
        self,
        aux_netcdf_path: str,
    ) -> None:
        r"""Valid netCDF with matching variables passes."""
        # Arrange
        variables = ["mesh", "mask"]

        # Act / Assert -- no exception raised
        validate_auxiliary(
            auxiliary_path=aux_netcdf_path,
            auxiliary_variables=variables,
        )

    def test_validate_forcing_passes_valid_store(
        self,
        data_zarr_path: str,
    ) -> None:
        r"""Valid zarr with matching forcing variables passes."""
        # Act / Assert -- no exception raised
        validate_forcing(
            forcing_path=None,
            forcing_variables=["states_surface"],
            data_path=data_zarr_path,
        )

    def test_validate_checkpoint_passes_existing_file(
        self,
        ckpt_file: str,
    ) -> None:
        r"""Existing checkpoint file passes validation."""
        # Act / Assert -- no exception raised
        validate_checkpoint(ckpt_path=ckpt_file)

    def test_validate_output_store_passes_correct_shape(
        self,
        valid_output_store: str,
    ) -> None:
        r"""Output store with correct shape passes."""
        # Act / Assert -- no exception raised
        validate_output_store(
            store_path=valid_output_store,
            state_variables=_STATE_VARS,
            init_times=_INIT_TIMES,
            lead_times=_LEAD_TIMES,
            ens_mems=_ENS_MEMS,
        )

    def test_create_output_store_creates_zarr(
        self,
        data_zarr_path: str,
        output_store_path: str,
    ) -> None:
        r"""create_output_store writes a zarr directory."""
        # Arrange
        assert not Path(output_store_path).exists()

        # Act
        create_output_store(
            data_path=data_zarr_path,
            state_variables=_STATE_VARS,
            store_path=output_store_path,
            init_times=_INIT_TIMES,
            lead_times=_LEAD_TIMES,
            ens_mems=_ENS_MEMS,
            n_store_freq=len(_LEAD_TIMES),
            recreate=True,
        )

        # Assert
        assert Path(output_store_path).exists()

    def test_validate_dask_addresses_single_address(
        self,
    ) -> None:
        r"""Single string scheduler passes validation."""
        # Act / Assert -- no exception raised
        validate_dask_addresses(
            scheduler="tcp://localhost:8786",
            n_dp_workers=4,
        )

    def test_validate_dask_addresses_list_matching(
        self,
    ) -> None:
        r"""List of addresses matching n_dp_workers passes."""
        # Arrange
        addrs = [
            "tcp://host1:8786",
            "tcp://host2:8786",
        ]

        # Act / Assert -- no exception raised
        validate_dask_addresses(
            scheduler=addrs,
            n_dp_workers=2,
        )

    def test_validate_auxiliary_both_none_returns(
        self,
    ) -> None:
        r"""Both args None returns without exception."""
        # Act / Assert -- no exception raised
        validate_auxiliary(
            auxiliary_path=None,
            auxiliary_variables=None,
        )

    def test_validate_checkpoint_none_returns(
        self,
    ) -> None:
        r"""None checkpoint returns without exception."""
        # Act / Assert -- no exception raised
        validate_checkpoint(ckpt_path=None)

    def test_validate_forcing_both_none_returns(
        self,
        data_zarr_path: str,
    ) -> None:
        r"""Both forcing args None returns without exception."""
        # Act / Assert -- no exception raised
        validate_forcing(
            forcing_path=None,
            forcing_variables=None,
            data_path=data_zarr_path,
        )


# -----------------------------------------------------------
# Error tests
# -----------------------------------------------------------

class TestValidationErrors:
    r"""Tests for expected error conditions."""

    def test_validate_initial_conditions_missing_store(
        self,
        tmp_path: Path,
    ) -> None:
        r"""Missing zarr store raises FileNotFoundError."""
        # Arrange
        bad_path = str(tmp_path / "nonexistent.zarr")

        # Act / Assert
        with pytest.raises(FileNotFoundError):
            validate_initial_conditions(
                data_path=bad_path,
                state_variables=_STATE_VARS,
            )

    def test_validate_initial_conditions_missing_variable(
        self,
        data_zarr_path: str,
    ) -> None:
        r"""Missing variable in zarr raises ValueError."""
        # Act / Assert
        with pytest.raises(ValueError):
            validate_initial_conditions(
                data_path=data_zarr_path,
                state_variables=["nonexistent_var"],
            )

    def test_validate_auxiliary_missing_netcdf(
        self,
        tmp_path: Path,
    ) -> None:
        r"""Missing netCDF file raises FileNotFoundError."""
        # Arrange
        bad_path = str(tmp_path / "missing.nc")

        # Act / Assert
        with pytest.raises(FileNotFoundError):
            validate_auxiliary(
                auxiliary_path=bad_path,
                auxiliary_variables=["mesh"],
            )

    def test_validate_auxiliary_missing_variable(
        self,
        aux_netcdf_path: str,
    ) -> None:
        r"""Missing variable in netCDF raises ValueError."""
        # Act / Assert
        with pytest.raises(ValueError):
            validate_auxiliary(
                auxiliary_path=aux_netcdf_path,
                auxiliary_variables=["nonexistent"],
            )

    def test_validate_auxiliary_path_without_variables(
        self,
        aux_netcdf_path: str,
    ) -> None:
        r"""Path set but no variables raises ValueError."""
        # Act / Assert
        with pytest.raises(ValueError):
            validate_auxiliary(
                auxiliary_path=aux_netcdf_path,
                auxiliary_variables=None,
            )

    def test_validate_auxiliary_variables_without_path(
        self,
    ) -> None:
        r"""Variables set but no path raises ValueError."""
        # Act / Assert
        with pytest.raises(ValueError):
            validate_auxiliary(
                auxiliary_path=None,
                auxiliary_variables=["mesh"],
            )

    def test_validate_forcing_missing_variable(
        self,
        data_zarr_path: str,
    ) -> None:
        r"""Missing forcing variable raises ValueError."""
        # Act / Assert
        with pytest.raises(ValueError):
            validate_forcing(
                forcing_path=None,
                forcing_variables=["nonexistent_var"],
                data_path=data_zarr_path,
            )

    def test_validate_checkpoint_missing_file(
        self,
        tmp_path: Path,
    ) -> None:
        r"""Missing checkpoint file raises FileNotFoundError."""
        # Arrange
        bad_path = str(tmp_path / "missing.ckpt")

        # Act / Assert
        with pytest.raises(FileNotFoundError):
            validate_checkpoint(ckpt_path=bad_path)

    def test_validate_output_store_missing_variable(
        self,
        valid_output_store: str,
    ) -> None:
        r"""Missing variable in output store raises ValueError."""
        # Act / Assert
        with pytest.raises(ValueError, match="not found"):
            validate_output_store(
                store_path=valid_output_store,
                state_variables=["nonexistent_var"],
                init_times=_INIT_TIMES,
                lead_times=_LEAD_TIMES,
                ens_mems=_ENS_MEMS,
            )

    def test_validate_output_store_checks_all_state_variables(
        self,
        data_zarr_path: str,
        tmp_path: Path,
    ) -> None:
        r"""Missing later state variable in output store raises ValueError."""
        # Arrange
        store_path = str(tmp_path / "missing_levels.zarr")
        create_output_store(
            data_path=data_zarr_path,
            state_variables=["states_surface"],
            store_path=store_path,
            init_times=_INIT_TIMES,
            lead_times=_LEAD_TIMES,
            ens_mems=_ENS_MEMS,
            n_store_freq=len(_LEAD_TIMES),
            recreate=True,
        )

        # Act / Assert
        with pytest.raises(ValueError, match=r"states_levels|not found"):
            validate_output_store(
                store_path=store_path,
                state_variables=["states_surface", "states_levels"],
                init_times=_INIT_TIMES,
                lead_times=_LEAD_TIMES,
                ens_mems=_ENS_MEMS,
            )

    def test_validate_output_store_missing_dimension(
        self,
        data_zarr_path: str,
        tmp_path: Path,
    ) -> None:
        r"""Variable without expected dims raises ValueError."""
        # Arrange -- create a store with a variable that
        # lacks init_time/lead_time/ensemble dimensions
        store_path = str(tmp_path / "bad_dims.zarr")
        ds = xr.Dataset({
            "states_surface": xr.DataArray(
                np.zeros((4, 8), dtype=np.float32),
                dims=("lat", "lon"),
            ),
        })
        ds.to_zarr(store_path, mode="w")

        # Act / Assert
        with pytest.raises(ValueError, match="not found"):
            validate_output_store(
                store_path=store_path,
                state_variables=["states_surface"],
                init_times=_INIT_TIMES,
                lead_times=_LEAD_TIMES,
                ens_mems=_ENS_MEMS,
            )

    def test_validate_output_store_wrong_shape(
        self,
        valid_output_store: str,
    ) -> None:
        r"""Wrong ensemble size raises ValueError."""
        # Act / Assert
        with pytest.raises(ValueError):
            validate_output_store(
                store_path=valid_output_store,
                state_variables=_STATE_VARS,
                init_times=_INIT_TIMES,
                lead_times=_LEAD_TIMES,
                ens_mems=np.arange(5),
            )

    def test_validate_dask_addresses_list_wrong_length(
        self,
    ) -> None:
        r"""List length != n_dp_workers raises ValueError."""
        # Arrange
        addrs = [
            "tcp://host1:8786",
            "tcp://host2:8786",
        ]

        # Act / Assert
        with pytest.raises(ValueError):
            validate_dask_addresses(
                scheduler=addrs,
                n_dp_workers=4,
            )


# -----------------------------------------------------------
# Unit tests for create_output_store
# -----------------------------------------------------------

class TestCreateOutputStoreUnittest:
    r"""Unit tests for create_output_store internals."""

    def test_creates_zarr_with_correct_dims(
        self,
        data_zarr_path: str,
        output_store_path: str,
    ) -> None:
        r"""Created store has init_time, lead_time, ensemble."""
        # Act
        create_output_store(
            data_path=data_zarr_path,
            state_variables=_STATE_VARS,
            store_path=output_store_path,
            init_times=_INIT_TIMES,
            lead_times=_LEAD_TIMES,
            ens_mems=_ENS_MEMS,
            n_store_freq=len(_LEAD_TIMES),
            recreate=True,
        )

        # Assert
        ds = xr.open_zarr(
            output_store_path, consolidated=False,
        )
        for dim in ("init_time", "lead_time", "ensemble"):
            assert dim in ds.dims, (
                f"Missing dimension '{dim}'"
            )
        assert ds.sizes["init_time"] == len(_INIT_TIMES)
        assert ds.sizes["lead_time"] == len(_LEAD_TIMES)
        assert ds.sizes["ensemble"] == len(_ENS_MEMS)

    def test_creates_zarr_with_correct_chunks(
        self,
        data_zarr_path: str,
        output_store_path: str,
    ) -> None:
        r"""Chunks follow (1, n_store_freq, 1, *spatial)."""
        # Arrange
        n_store_freq = 2

        # Act
        create_output_store(
            data_path=data_zarr_path,
            state_variables=_STATE_VARS,
            store_path=output_store_path,
            init_times=_INIT_TIMES,
            lead_times=_LEAD_TIMES,
            ens_mems=_ENS_MEMS,
            n_store_freq=n_store_freq,
            recreate=True,
        )

        # Assert
        store = zarr.open(output_store_path, mode="r")
        for var in _STATE_VARS:
            chunks = store[var].chunks
            assert chunks[0] == 1, (
                f"init_time chunk should be 1, "
                f"got {chunks[0]}"
            )
            assert chunks[1] == n_store_freq, (
                f"lead_time chunk should be "
                f"{n_store_freq}, got {chunks[1]}"
            )
            assert chunks[2] == 1, (
                f"ensemble chunk should be 1, "
                f"got {chunks[2]}"
            )

    def test_recreate_overwrites_existing(
        self,
        data_zarr_path: str,
        output_store_path: str,
    ) -> None:
        r"""recreate=True overwrites an existing store."""
        # Arrange -- create once with 2 ensemble members
        create_output_store(
            data_path=data_zarr_path,
            state_variables=_STATE_VARS,
            store_path=output_store_path,
            init_times=_INIT_TIMES,
            lead_times=_LEAD_TIMES,
            ens_mems=np.arange(2),
            n_store_freq=len(_LEAD_TIMES),
            recreate=True,
        )

        # Act -- recreate with 5 ensemble members
        create_output_store(
            data_path=data_zarr_path,
            state_variables=_STATE_VARS,
            store_path=output_store_path,
            init_times=_INIT_TIMES,
            lead_times=_LEAD_TIMES,
            ens_mems=np.arange(5),
            n_store_freq=len(_LEAD_TIMES),
            recreate=True,
        )

        # Assert -- store has 5 ensemble members
        ds = xr.open_zarr(
            output_store_path, consolidated=False,
        )
        assert ds.sizes["ensemble"] == 5

    def test_nan_fill(
        self,
        data_zarr_path: str,
        output_store_path: str,
    ) -> None:
        r"""Newly created store is NaN-filled."""
        # Act
        create_output_store(
            data_path=data_zarr_path,
            state_variables=_STATE_VARS,
            store_path=output_store_path,
            init_times=_INIT_TIMES,
            lead_times=_LEAD_TIMES,
            ens_mems=_ENS_MEMS,
            n_store_freq=len(_LEAD_TIMES),
            recreate=True,
        )

        # Assert
        ds = xr.open_zarr(
            output_store_path, consolidated=False,
        )
        for var in _STATE_VARS:
            values = ds[var].values
            assert np.all(np.isnan(values)), (
                f"Variable '{var}' is not all NaN"
            )

    def test_create_output_store_uses_zarr_fill_value_without_materializing_full_array(
        self,
        data_zarr_path: str,
        output_store_path: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        r"""create_output_store succeeds without calling np.full."""
        # Arrange
        def _raise_on_full(
            *args: object, **kwargs: object,
        ) -> None:
            raise AssertionError(
                "np.full should not be called"
            )

        monkeypatch.setattr(np, "full", _raise_on_full)

        # Act
        create_output_store(
            data_path=data_zarr_path,
            state_variables=_STATE_VARS,
            store_path=output_store_path,
            init_times=_INIT_TIMES,
            lead_times=_LEAD_TIMES,
            ens_mems=_ENS_MEMS,
            n_store_freq=len(_LEAD_TIMES),
            recreate=True,
        )

        # Assert
        ds = xr.open_zarr(
            output_store_path, consolidated=False,
        )
        for var in _STATE_VARS:
            values = ds[var].values
            assert np.all(np.isnan(values)), (
                f"Variable '{var}' is not all NaN"
            )

    def test_skips_creation_when_not_recreate_and_exists(
        self,
        data_zarr_path: str,
        output_store_path: str,
    ) -> None:
        r"""recreate=False with existing store skips creation."""
        # Arrange -- create store with 2 ensemble members
        create_output_store(
            data_path=data_zarr_path,
            state_variables=_STATE_VARS,
            store_path=output_store_path,
            init_times=_INIT_TIMES,
            lead_times=_LEAD_TIMES,
            ens_mems=np.arange(2),
            n_store_freq=len(_LEAD_TIMES),
            recreate=True,
        )

        # Act -- call with recreate=False and different
        # ensemble count; store should NOT be overwritten
        create_output_store(
            data_path=data_zarr_path,
            state_variables=_STATE_VARS,
            store_path=output_store_path,
            init_times=_INIT_TIMES,
            lead_times=_LEAD_TIMES,
            ens_mems=np.arange(5),
            n_store_freq=len(_LEAD_TIMES),
            recreate=False,
        )

        # Assert -- store still has 2 ensemble members
        ds = xr.open_zarr(
            output_store_path, consolidated=False,
        )
        assert ds.sizes["ensemble"] == 2


# -----------------------------------------------------------
# Unit tests for validate_restart_config
# -----------------------------------------------------------

class TestValidateRestartConfig:
    r'''Direct tests for validate_restart_config.'''

    def test_restart_true_recreate_true_raises(
        self,
    ) -> None:
        r'''Both True raises ValueError.'''
        # Act / Assert
        with pytest.raises(ValueError):
            validate_restart_config(
                restart=True,
                recreate_store=True,
            )

    def test_restart_true_recreate_false_passes(
        self,
    ) -> None:
        r'''restart=True, recreate_store=False passes.'''
        # Act / Assert -- no exception raised
        validate_restart_config(
            restart=True,
            recreate_store=False,
        )

    def test_restart_false_recreate_true_passes(
        self,
    ) -> None:
        r'''restart=False, recreate_store=True passes.'''
        # Act / Assert -- no exception raised
        validate_restart_config(
            restart=False,
            recreate_store=True,
        )

    def test_restart_false_recreate_false_passes(
        self,
    ) -> None:
        r'''Both False passes without error.'''
        # Act / Assert -- no exception raised
        validate_restart_config(
            restart=False,
            recreate_store=False,
        )

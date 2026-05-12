# -*- coding: utf-8 -*-
r'''Tests for src/forecast/input.py -- InputReader.

Tests are organized into four classes:

- **Functional**: end-to-end load_states / load_auxiliary /
  load_forcings round-trips.
- **Unittest**: property flags, dimension handling, and
  private helper _open_optional_dataset.
- **Errors**: expected ValueErrors and KeyErrors.
- **EdgeCases**: ensemble wrapping and modulo indexing.
'''

# External modules
import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Internal modules
from {{cookiecutter.project_slug}}.forecast.input import InputReader
from tests.conftest import _create_test_zarr


# -----------------------------------------------------------
# Module-level constants
# -----------------------------------------------------------

_INIT_TIMES = pd.DatetimeIndex([
    pd.Timestamp("2020-01-01"),
    pd.Timestamp("2020-01-01 06:00"),
])
_LEAD_TIMES = pd.timedelta_range(
    start="6h", periods=3, freq="6h",
)


# -----------------------------------------------------------
# Module-level fixtures
# -----------------------------------------------------------

@pytest.fixture()
def zarr_path_no_ens(tmp_path) -> str:
    r'''Return path to a zarr store without ensemble dim.'''
    path = str(tmp_path / "no_ens.zarr")
    _create_test_zarr(path, n_times=20, n_ens=None)
    return path


@pytest.fixture()
def zarr_path_with_ens(tmp_path) -> str:
    r'''Return path to a zarr store with ensemble dim.'''
    path = str(tmp_path / "ens.zarr")
    _create_test_zarr(path, n_times=20, n_ens=2)
    return path


@pytest.fixture()
def reader_no_ens(zarr_path_no_ens) -> InputReader:
    r'''Return InputReader for a store without ensemble.'''
    return InputReader(
        data_path=zarr_path_no_ens,
        state_variables=[
            "states_surface", "states_levels",
        ],
    )


@pytest.fixture()
def reader_with_ens(zarr_path_with_ens) -> InputReader:
    r'''Return InputReader for a store with ensemble.'''
    return InputReader(
        data_path=zarr_path_with_ens,
        state_variables=[
            "states_surface", "states_levels",
        ],
    )


# -----------------------------------------------------------
# Functional tests
# -----------------------------------------------------------

class TestForecastInputFunctional:
    r'''End-to-end tests for load_states, load_auxiliary,
    and load_forcings.
    '''

    def test_returns_dict_with_state_variables(
        self, reader_no_ens,
    ) -> None:
        r'''load_states returns a dict keyed by state
        variable names.
        '''
        states = reader_no_ens.load_states(
            _INIT_TIMES[:1], np.array([0]),
        )
        assert "states_surface" in states
        assert "states_levels" in states

    def test_shape_has_batch_as_first_dim(
        self, reader_no_ens,
    ) -> None:
        r'''Loading 2 init_times yields shape[0] == 2.'''
        states = reader_no_ens.load_states(
            _INIT_TIMES, np.array([0, 0]),
        )
        assert states["states_surface"].shape[0] == 2

    def test_configured_auxiliary_loads_from_netcdf_path(
        self,
        zarr_path_no_ens,
        mocked_auxiliary_netcdf,
    ) -> None:
        r'''Configured auxiliary data loads from a netCDF
        file path.
        '''
        aux_path = mocked_auxiliary_netcdf(
            n_ens=None, n_lat=2, n_lon=3,
        )
        reader = InputReader(
            data_path=zarr_path_no_ens,
            state_variables=["states_surface"],
            auxiliary_path=aux_path,
            auxiliary_variables=["mesh", "mask"],
        )
        aux = reader.load_auxiliary(np.array([0]))
        assert sorted(aux.keys()) == ["mask", "mesh"]

        with xr.open_dataset(aux_path) as expected:
            np.testing.assert_array_equal(
                aux["mesh"],
                expected["mesh"].values[None, ...],
            )
            np.testing.assert_array_equal(
                aux["mask"],
                expected["mask"].values[None, ...],
            )

    def test_returns_dict_with_forcing_variables(
        self, zarr_path_no_ens,
    ) -> None:
        r'''load_forcings returns the requested forcing
        variables.
        '''
        reader = InputReader(
            data_path=zarr_path_no_ens,
            state_variables=["states_surface"],
            forcing_variables=["states_surface"],
        )
        forcings = reader.load_forcings(
            _INIT_TIMES[:1], np.array([0]), _LEAD_TIMES,
        )
        assert "states_surface" in forcings

    def test_shape_has_batch_and_lead_time_dims(
        self, zarr_path_no_ens,
    ) -> None:
        r'''Loading 2 init_times x 3 lead_times gives
        shape (2, 4, ...) with n_in_steps=1.
        '''
        reader = InputReader(
            data_path=zarr_path_no_ens,
            state_variables=["states_surface"],
            forcing_variables=["states_surface"],
            n_in_steps=1,
            step_freq="6h",
        )
        forcings = reader.load_forcings(
            _INIT_TIMES, np.array([0, 0]), _LEAD_TIMES,
        )
        assert forcings["states_surface"].shape[0] == 2
        assert forcings["states_surface"].shape[1] == 4

    def test_forcing_path_uses_separate_zarr_store(
        self,
        zarr_path_no_ens,
        zarr_path_with_ens,
    ) -> None:
        r'''A dedicated forcing path is opened and used for
        forcings.
        '''
        reader = InputReader(
            data_path=zarr_path_no_ens,
            state_variables=["states_surface"],
            forcing_path=zarr_path_with_ens,
            forcing_variables=["states_surface"],
        )
        forcings = reader.load_forcings(
            _INIT_TIMES[:1],
            np.array([1]),
            _LEAD_TIMES[:1],
        )
        assert forcings["states_surface"].shape[0] == 1

    def test_load_states_default_n_in_has_time_axis(
        self, reader_no_ens,
    ) -> None:
        r'''With default n_in_steps=1, load_states returns
        shape (batch, 1, ...).
        '''
        states = reader_no_ens.load_states(
            _INIT_TIMES[:1], np.array([0]),
        )
        assert states["states_surface"].shape[1] == 1

    def test_load_states_n_in_steps_2_returns_two_timesteps(
        self, zarr_path_no_ens,
    ) -> None:
        r'''With n_in_steps=2, load_states returns
        shape[1] == 2.
        '''
        reader = InputReader(
            data_path=zarr_path_no_ens,
            state_variables=["states_surface", "states_levels"],
            n_in_steps=2,
            step_freq="6h",
        )
        states = reader.load_states(
            _INIT_TIMES[:1], np.array([0]),
        )
        assert states["states_surface"].shape[1] == 2
        assert states["states_levels"].shape[1] == 2

    def test_load_states_time_ordering_with_n_in_steps_2(
        self, zarr_path_no_ens,
    ) -> None:
        r'''With n_in_steps=2, init_time T loads steps T-6h
        and T in order.
        '''
        reader_2step = InputReader(
            data_path=zarr_path_no_ens,
            state_variables=["states_surface"],
            n_in_steps=2,
            step_freq="6h",
        )
        reader_single = InputReader(
            data_path=zarr_path_no_ens,
            state_variables=["states_surface"],
        )
        # Use _INIT_TIMES[1:2] = 2020-01-01 06:00
        # Should load 2020-01-01 00:00 (T-6h) and 2020-01-01 06:00 (T)

        # Act: 2-step load at T; reference single-step load at T
        states = reader_2step.load_states(
            _INIT_TIMES[1:2], np.array([0]),
        )
        state_t = reader_single.load_states(
            _INIT_TIMES[1:2], np.array([0]),
        )

        # Verify index 0 and index 1 differ (T-6h != T)
        assert not np.array_equal(
            states["states_surface"][0, 0],
            states["states_surface"][0, 1],
        )
        # Verify second time step (index 1) equals the reference T
        np.testing.assert_array_equal(
            states["states_surface"][0, 1],
            state_t["states_surface"][0, 0],
        )

    def test_load_forcings_n_in_steps_2_shape(
        self, zarr_path_no_ens,
    ) -> None:
        r'''With n_in_steps=2 and 3 lead_times, forcings
        have shape[1] == 5.
        '''
        reader = InputReader(
            data_path=zarr_path_no_ens,
            state_variables=["states_surface"],
            forcing_variables=["states_surface"],
            n_in_steps=2,
            step_freq="6h",
        )
        forcings = reader.load_forcings(
            _INIT_TIMES[:1], np.array([0]), _LEAD_TIMES,
        )
        assert forcings["states_surface"].shape[1] == 5


# -----------------------------------------------------------
# Unittest tests
# -----------------------------------------------------------

class TestForecastInputUnittest:
    r'''Unit tests for properties, dimension handling,
    and _open_optional_dataset.
    '''

    # -- fixtures for _open_optional_dataset tests ----------

    @pytest.fixture()
    def opt_zarr_path(self, tmp_path) -> str:
        r'''Path to a zarr store for optional-dataset
        tests (n_times=10, no ensemble).
        '''
        path = str(tmp_path / "data.zarr")
        _create_test_zarr(
            path, n_times=10, n_ens=None,
        )
        return path

    @pytest.fixture()
    def second_zarr_path(self, tmp_path) -> str:
        r'''Path to a second zarr store for
        separate-path tests.
        '''
        path = str(tmp_path / "extra.zarr")
        _create_test_zarr(
            path, n_times=10, n_ens=None,
        )
        return path

    @pytest.fixture()
    def opt_reader(
        self, opt_zarr_path,
    ) -> InputReader:
        r'''InputReader built on the opt_zarr_path store.'''
        return InputReader(
            data_path=opt_zarr_path,
            state_variables=[
                "states_surface", "states_levels",
            ],
        )

    @pytest.fixture()
    def default_ds(
        self, opt_zarr_path,
    ) -> xr.Dataset:
        r'''Default xr.Dataset opened from opt_zarr_path.'''
        return xr.open_zarr(opt_zarr_path)

    # -- property tests -------------------------------------

    def test_use_auxiliary_false_by_default(
        self, reader_no_ens,
    ) -> None:
        r'''use_auxiliary is False when no auxiliary
        variables are given.
        '''
        assert reader_no_ens.use_auxiliary is False

    def test_use_forcings_false_by_default(
        self, reader_no_ens,
    ) -> None:
        r'''use_forcings is False when no forcing variables
        are given.
        '''
        assert reader_no_ens.use_forcings is False

    def test_use_auxiliary_false_without_auxiliary_path(
        self, zarr_path_no_ens,
    ) -> None:
        r'''use_auxiliary is False when only auxiliary vars
        are provided.
        '''
        reader = InputReader(
            data_path=zarr_path_no_ens,
            state_variables=["states_surface"],
            auxiliary_variables=["states_surface"],
        )
        assert reader.use_auxiliary is False

    def test_use_auxiliary_true_when_path_and_variables_set(
        self,
        zarr_path_no_ens,
        mocked_auxiliary_netcdf,
    ) -> None:
        r'''use_auxiliary is True when auxiliary path and
        vars are set.
        '''
        aux_path = mocked_auxiliary_netcdf(
            n_ens=None, n_lat=2, n_lon=3,
        )
        reader = InputReader(
            data_path=zarr_path_no_ens,
            state_variables=["states_surface"],
            auxiliary_path=aux_path,
            auxiliary_variables=["mesh"],
        )
        assert reader.use_auxiliary is True

    def test_use_forcings_true_when_variables_set(
        self, zarr_path_no_ens,
    ) -> None:
        r'''use_forcings is True when forcing_variables
        are provided.
        '''
        reader = InputReader(
            data_path=zarr_path_no_ens,
            state_variables=["states_surface"],
            forcing_variables=["states_surface"],
        )
        assert reader.use_forcings is True

    def test_ensemble_dim_added_when_missing(
        self, zarr_path_no_ens,
    ) -> None:
        r'''Reader expands an ensemble dim when the store
        has none.
        '''
        reader = InputReader(
            data_path=zarr_path_no_ens,
            state_variables=["states_surface"],
        )
        assert "ensemble" in reader._state_dataset.dims

    def test_load_states_does_not_return_time_key(
        self, reader_no_ens,
    ) -> None:
        r'''load_states returns only state vars -- no time
        key injected.
        '''
        states = reader_no_ens.load_states(
            _INIT_TIMES[:1], np.array([0]),
        )
        assert "time" not in states

    def test_n_in_steps_stored_on_reader(
        self, zarr_path_no_ens,
    ) -> None:
        r'''n_in_steps and step_freq are stored as instance
        attributes.
        '''
        reader = InputReader(
            data_path=zarr_path_no_ens,
            state_variables=["states_surface"],
            n_in_steps=3,
            step_freq="6h",
        )
        assert reader.n_in_steps == 3
        assert reader.step_freq == "6h"

    # -- _open_optional_dataset tests -----------------------

    def test_open_optional_dataset_uses_default_when_path_is_none(
        self, opt_reader, default_ds,
    ) -> None:
        r'''When path=None, _open_optional_dataset uses
        default_dataset.
        '''
        result = opt_reader._open_optional_dataset(
            variables=["states_surface"],
            path=None,
            default_dataset=default_ds,
        )
        assert "states_surface" in result.data_vars
        assert set(result.data_vars) == {
            "states_surface",
        }

    def test_open_optional_dataset_opens_separate_zarr_when_path_given(
        self,
        opt_reader,
        default_ds,
        second_zarr_path,
    ) -> None:
        r'''When a real path is given, the dataset is opened
        from that path.
        '''
        result = opt_reader._open_optional_dataset(
            variables=["states_surface"],
            path=second_zarr_path,
            default_dataset=default_ds,
        )
        assert "states_surface" in result.data_vars

    def test_open_optional_dataset_adds_ensemble_dim_when_missing(
        self, opt_reader, default_ds,
    ) -> None:
        r'''Result always has an ensemble dimension.'''
        result = opt_reader._open_optional_dataset(
            variables=["states_surface"],
            path=None,
            default_dataset=default_ds,
        )
        assert "ensemble" in result.dims


# -----------------------------------------------------------
# Error tests
# -----------------------------------------------------------

class TestForecastInputErrors:
    r'''Tests for expected ValueErrors and KeyErrors.'''

    def test_raises_valueerror_when_auxiliary_not_configured(
        self, reader_no_ens,
    ) -> None:
        r'''load_auxiliary raises ValueError when no
        auxiliary vars given.
        '''
        with pytest.raises(ValueError):
            reader_no_ens.load_auxiliary(np.array([0]))

    def test_raises_valueerror_when_forcings_not_configured(
        self, reader_no_ens,
    ) -> None:
        r'''load_forcings raises ValueError when no forcing
        vars given.
        '''
        with pytest.raises(ValueError):
            reader_no_ens.load_forcings(
                _INIT_TIMES[:1],
                np.array([0]),
                _LEAD_TIMES,
            )

    def test_load_states_raises_without_step_freq_for_multi_input(
        self, zarr_path_no_ens,
    ) -> None:
        r'''load_states raises ValueError for n_in_steps>1
        without step_freq.
        '''
        reader = InputReader(
            data_path=zarr_path_no_ens,
            state_variables=["states_surface"],
            n_in_steps=2,
        )

        with pytest.raises(ValueError, match="step_freq"):
            reader.load_states(_INIT_TIMES[:1], np.array([0]))

    def test_load_forcings_raises_without_step_freq_for_multi_input(
        self, zarr_path_no_ens,
    ) -> None:
        r'''load_forcings raises ValueError for
        n_in_steps>1 without step_freq.
        '''
        reader = InputReader(
            data_path=zarr_path_no_ens,
            state_variables=["states_surface"],
            forcing_variables=["states_surface"],
            n_in_steps=2,
        )

        with pytest.raises(ValueError, match="step_freq"):
            reader.load_forcings(
                _INIT_TIMES[:1],
                np.array([0]),
                _LEAD_TIMES,
            )

    def test_missing_auxiliary_variable_raises_value_error(
        self,
        zarr_path_no_ens,
        mocked_auxiliary_netcdf,
    ) -> None:
        r'''Missing auxiliary variables raise a clear
        KeyError.
        '''
        aux_path = mocked_auxiliary_netcdf(
            n_ens=None, n_lat=2, n_lon=3,
        )
        with pytest.raises(KeyError):
            InputReader(
                data_path=zarr_path_no_ens,
                state_variables=["states_surface"],
                auxiliary_path=aux_path,
                auxiliary_variables=["missing"],
            )


# -----------------------------------------------------------
# Edge-case tests
# -----------------------------------------------------------

class TestForecastInputEdgeCases:
    r'''Tests for ensemble wrapping and modulo indexing.'''

    def test_ensemble_wrapping(
        self, zarr_path_with_ens,
    ) -> None:
        r'''Requesting ens_mem=3 with 2 members uses
        index 3 % 2 = 1.
        '''
        reader = InputReader(
            data_path=zarr_path_with_ens,
            state_variables=["states_surface"],
        )
        states_wrapped = reader.load_states(
            _INIT_TIMES[:1], np.array([3]),
        )
        states_direct = reader.load_states(
            _INIT_TIMES[:1], np.array([1]),
        )
        np.testing.assert_array_equal(
            states_wrapped["states_surface"],
            states_direct["states_surface"],
        )

    def test_no_ensemble_auxiliary_expands_to_batch_dimension(
        self,
        zarr_path_no_ens,
        mocked_auxiliary_netcdf,
    ) -> None:
        r'''No-ensemble auxiliary data is expanded to
        batch-first output.
        '''
        aux_path = mocked_auxiliary_netcdf(
            n_ens=None, n_lat=2, n_lon=3,
        )
        reader = InputReader(
            data_path=zarr_path_no_ens,
            state_variables=["states_surface"],
            auxiliary_path=aux_path,
            auxiliary_variables=["mesh", "mask"],
        )
        aux = reader.load_auxiliary(np.array([0, 0]))
        assert aux["mesh"].shape == (2, 2, 2, 3)
        assert aux["mask"].shape == (2, 2, 3)
        np.testing.assert_array_equal(
            aux["mesh"][0], aux["mesh"][1],
        )
        np.testing.assert_array_equal(
            aux["mask"][0], aux["mask"][1],
        )

    def test_ensemble_auxiliary_uses_modulo_indexing(
        self,
        zarr_path_no_ens,
        mocked_auxiliary_netcdf,
    ) -> None:
        r'''Auxiliary ensemble indexing wraps with
        ens_mems % n_ens.
        '''
        aux_path = mocked_auxiliary_netcdf(
            n_ens=2, n_lat=2, n_lon=3,
        )
        ens_mems = np.array([0, 1, 2, 3, 4])
        expected_indices = ens_mems % 2

        reader = InputReader(
            data_path=zarr_path_no_ens,
            state_variables=["states_surface"],
            auxiliary_path=aux_path,
            auxiliary_variables=["mesh", "mask"],
        )
        aux = reader.load_auxiliary(ens_mems)

        with xr.open_dataset(aux_path) as expected:
            np.testing.assert_array_equal(
                aux["mesh"],
                expected["mesh"].values[
                    expected_indices
                ],
            )
            np.testing.assert_array_equal(
                aux["mask"],
                expected["mask"].values[
                    expected_indices
                ],
            )

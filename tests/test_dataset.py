# -*- coding: utf-8 -*-
r'''Tests for the TrainDataset class in src/data/dataset.py.

The tests aim for 100% coverage of dataset.py.
'''

# System modules
import logging

# External modules
import numpy as np
import xarray as xr
import zarr
import pytest
import torch.utils.data

# Internal modules
from ocean_flow.data.dataset import TrainDataset
from tests.conftest import _create_test_zarr


# -----------------------------------------------------------
# Private helper
# -----------------------------------------------------------

def _make_ds(path: str, **kwargs: object) -> TrainDataset:
    r'''
    Construct a ``TrainDataset`` and work around bug B-2.

    ``TrainDataset.__init__`` declares ``_datasets`` as a
    type annotation only; no attribute value is ever assigned.
    The ``datasets`` lazy-load property checks
    ``self._datasets is None``, which raises ``AttributeError``
    on first access.  Setting the attribute to ``None`` here
    restores the intended lazy-loading behaviour.

    Note: once B-1 is fixed and the constructor works, this
    helper also handles B-2 automatically.

    Parameters
    ----------
    path : str
        Zarr store path forwarded to ``TrainDataset``.
    **kwargs : object
        Additional keyword arguments forwarded to
        ``TrainDataset``.

    Returns
    -------
    TrainDataset
        Dataset instance with ``_datasets`` initialised to
        ``None`` so that lazy loading works correctly.
    '''
    ds = TrainDataset(path, **kwargs)
    ds._datasets = None  # work around B-2
    return ds


# -----------------------------------------------------------
# Functional tests
# -----------------------------------------------------------

class TestDatasetFunctional:
    r'''End-to-end functional tests for TrainDataset.'''

    def test_variables_property(
            self,
            zarr_store: zarr.Group,
    ) -> None:
        r'''variables returns state names then forcing names.'''
        ds = TrainDataset(
            str(zarr_store.store.path),
            state_variables=["states_surface"],
            forcing_variables=["states_levels"],
        )
        assert ds.variables == [
            "states_surface",
            "states_levels",
        ]

    def test_getitem_returns_expected_keys(
            self,
            tmp_path: object,
    ) -> None:
        r'''__getitem__ includes state variables and time.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=5)
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            forcing_variables=["states_levels"],
            n_steps=2,
            n_step_size=1,
        )
        sample = ds[1]
        assert "states_surface" in sample
        assert "states_levels" in sample
        assert "time" in sample

    def test_getitem_surface_shape(
            self,
            tmp_path: object,
    ) -> None:
        r'''states_surface shape is (n_steps, var, lat, lon).'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=5)
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            n_steps=2,
            n_step_size=1,
        )
        assert ds[0]["states_surface"].shape == (2, 2, 4, 8)

    def test_getitem_shape_with_ensemble(
            self,
            tmp_path: object,
    ) -> None:
        r'''Ensemble axis absent from returned variable arrays.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=5, n_ens=3)
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            n_steps=2,
            n_step_size=1,
        )
        assert ds[0]["states_surface"].ndim == 4

    def test_different_ensemble_members_differ(
            self,
            tmp_path: object,
    ) -> None:
        r'''Different ensemble indices return different data.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4, n_ens=3)
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            n_steps=1,
        )
        s0 = ds[0]   # ens=0
        s1 = ds[1]   # ens=1, same time
        assert not np.allclose(
            s0["states_surface"],
            s1["states_surface"],
        )

    def test_auxiliary_vars_in_sample(
            self,
            tmp_path: object,
            mocked_auxiliary_netcdf: object,
    ) -> None:
        r'''Auxiliary variables appear in every sample.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4)
        aux_path = mocked_auxiliary_netcdf(
            n_ens=None, n_lat=4, n_lon=8,
        )
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            auxiliary_path=aux_path,
            auxiliary_variables=["mask", "mesh"],
        )
        samp = ds[0]
        assert "mask" in samp
        assert "mesh" in samp

    def test_auxiliary_shapes_no_ensemble(
            self,
            tmp_path: object,
            mocked_auxiliary_netcdf: object,
    ) -> None:
        r'''Auxiliary shapes are at-least-3D after processing.

        ``_process_var_data`` calls ``_atleast_3d``, which
        promotes 2-D arrays (e.g. ``mask`` with shape
        ``(lat, lon)``) to 3-D by prepending a singleton
        dimension.  3-D arrays (e.g. ``mesh``) are unchanged.
        '''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4)
        aux_path = mocked_auxiliary_netcdf(
            n_ens=None, n_lat=4, n_lon=8,
        )
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            auxiliary_path=aux_path,
            auxiliary_variables=["mask", "mesh"],
        )
        samp = ds[0]
        # mask is (lat, lon) in netCDF → (1, lat, lon) after
        # _atleast_3d
        assert samp["mask"].shape == (1, 4, 8)
        # mesh is (channel, lat, lon) in netCDF → unchanged
        assert samp["mesh"].shape == (2, 4, 8)

    def test_auxiliary_not_sliced_by_ensemble(
            self,
            tmp_path: object,
            mocked_auxiliary_netcdf: object,
    ) -> None:
        r'''Auxiliary arrays returned unchanged for all ens idx.

        The source inserts the full auxiliary array into every
        sample via ``sample.update(self._auxiliary_arrays)``
        without slicing by ensemble index.  Both ensemble-0
        and ensemble-1 samples therefore receive the identical
        full array, retaining the ensemble dimension.
        '''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4, n_ens=3)
        aux_path = mocked_auxiliary_netcdf(
            n_ens=3, n_lat=4, n_lon=8,
        )
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            auxiliary_path=aux_path,
            auxiliary_variables=["mask"],
            n_steps=1,
        )
        s0 = ds[0]   # ens=0
        s1 = ds[1]   # ens=1, same time
        # Full ensemble dim is retained (not sliced)
        assert s0["mask"].shape == (3, 4, 8)
        assert np.array_equal(s0["mask"], s1["mask"])

    def test_full_iteration_count(
            self,
            tmp_path: object,
    ) -> None:
        r'''Iterating yields exactly len(ds) samples.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=50)
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            forcing_variables=["states_levels"],
            n_steps=5,
            n_step_size=1,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=None)

        assert len(ds) == 46
        assert sum(1 for _ in loader) == 46

    def test_full_iteration_with_ensemble(
            self,
            tmp_path: object,
    ) -> None:
        r'''Iteration count correct with ensemble members.'''
        n_times, n_ens = 5, 2
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(
            path, n_times=n_times, n_ens=n_ens,
        )
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            n_steps=2,
            n_step_size=1,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=None)

        assert sum(1 for _ in loader) == (n_times - 1) * n_ens


# -----------------------------------------------------------
# Unit tests
# -----------------------------------------------------------

class TestDatasetUnittest:
    r'''Isolated unit tests for TrainDataset helpers.'''

    def test_n_times_set_correctly(
            self,
            tmp_path: object,
    ) -> None:
        r'''n_times equals the time-axis length of the zarr.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=7)
        ds = TrainDataset(
            path,
            state_variables=["states_surface"],
        )
        assert ds.n_times == 7

    def test_step_shift(
            self,
            tmp_path: object,
    ) -> None:
        r'''_step_shift = (n_steps - 1) * n_step_size.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=10)
        ds = TrainDataset(
            path,
            state_variables=["states_surface"],
            n_steps=3,
            n_step_size=2,
        )
        assert ds._step_shift == (3 - 1) * 2

    def test_len_no_ensemble(
            self,
            tmp_path: object,
    ) -> None:
        r'''len = n_times - _step_shift without ensemble.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=6)
        ds = TrainDataset(
            path,
            state_variables=["states_surface"],
            n_steps=2,
            n_step_size=1,
        )
        assert len(ds) == 5

    def test_len_with_ensemble(
            self,
            tmp_path: object,
    ) -> None:
        r'''len = (n_times - _step_shift) * n_ensemble.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=6, n_ens=3)
        ds = TrainDataset(
            path,
            state_variables=["states_surface"],
            n_steps=2,
            n_step_size=1,
        )
        assert len(ds) == 5 * 3

    def test_n_steps_one_len(
            self,
            tmp_path: object,
    ) -> None:
        r'''n_steps=1 gives one sample per time step.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=3)
        ds = TrainDataset(
            path,
            state_variables=["states_surface"],
            n_steps=1,
            n_step_size=1,
        )
        assert len(ds) == 3

    def test_step_size_two_len(
            self,
            tmp_path: object,
    ) -> None:
        r'''n_step_size=2 doubles spacing, reduces sample count.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=5)
        ds = TrainDataset(
            path,
            state_variables=["states_surface"],
            n_steps=2,
            n_step_size=2,
        )
        # 5 - (2-1)*2 = 3
        assert len(ds) == 3

    def test_n_ensemble_default_one(
            self,
            tmp_path: object,
    ) -> None:
        r'''n_ensemble=1 and use_ensemble=False by default.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4)
        ds = TrainDataset(
            path,
            state_variables=["states_surface"],
            n_steps=1,
        )
        assert ds.n_ensemble == 1
        assert ds.use_ensemble is False

    def test_n_ensemble_set_correctly(
            self,
            tmp_path: object,
    ) -> None:
        r'''n_ensemble and use_ensemble=True when ens dim present.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4, n_ens=5)
        ds = TrainDataset(
            path,
            state_variables=["states_surface"],
            n_steps=1,
        )
        assert ds.n_ensemble == 5
        assert ds.use_ensemble is True

    def test_time_array_dtype_and_shape(
            self,
            tmp_path: object,
    ) -> None:
        r'''time_array is float32 with length n_times.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=5)
        ds = TrainDataset(
            path,
            state_variables=["states_surface"],
        )
        assert ds.time_array.dtype == np.float32
        assert ds.time_array.shape == (5,)

    def test_get_var_shape_no_ensemble(
            self,
            tmp_path: object,
    ) -> None:
        r'''_get_var returns (n_steps, var, lat, lon) shape.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(
            path, n_times=10, n_lat=4, n_lon=8,
        )
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            n_steps=3,
            n_step_size=2,
        )
        time_slice = slice(0, 3 * 2, 2)
        arr = ds._get_var("states_surface", time_slice)
        assert arr.shape == (3, 2, 4, 8)

    def test_get_var_selects_ensemble_member(
            self,
            tmp_path: object,
    ) -> None:
        r'''_get_var with different ens_idx returns diff data.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4, n_ens=3)
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            n_steps=1,
        )
        time_slice = slice(0, 1, 1)
        arr0 = ds._get_var(
            "states_surface", time_slice, ens_idx=0,
        )
        arr1 = ds._get_var(
            "states_surface", time_slice, ens_idx=1,
        )
        assert not np.allclose(arr0, arr1)

    def test_get_var_default_ens_idx_zero(
            self,
            tmp_path: object,
    ) -> None:
        r'''_get_var defaults to ens_idx=0.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4, n_ens=3)
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            n_steps=1,
        )
        time_slice = slice(0, 1, 1)
        arr_default = ds._get_var(
            "states_surface", time_slice,
        )
        arr_ens0 = ds._get_var(
            "states_surface", time_slice, ens_idx=0,
        )
        assert np.array_equal(arr_default, arr_ens0)

    def test_get_var_returns_float32(
            self,
            tmp_path: object,
    ) -> None:
        r'''_get_var output is always float32.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4)
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            n_steps=1,
        )
        arr = ds._get_var(
            "states_surface", slice(0, 1, 1),
        )
        assert arr.dtype == np.float32

    def test_getitem_casts_float64_zarr_to_float32(
            self,
            tmp_path: object,
    ) -> None:
        r'''__getitem__ returns float32 when zarr is float64.'''
        path = str(tmp_path / "t.zarr")
        rng = np.random.default_rng(seed=19921225)
        n_times, n_surf, n_lev, n_levels = 5, 2, 2, 3
        n_lat, n_lon = 4, 8
        surf = rng.normal(
            size=(n_times, n_surf, n_lat, n_lon),
        ).astype(np.float64)
        lev = rng.normal(
            size=(n_times, n_lev, n_levels, n_lat, n_lon),
        ).astype(np.float64)
        times = np.array([
            np.datetime64('2020-01-01')
            + np.timedelta64(6 * i, 'h')
            for i in range(n_times)
        ])
        ds_xr = xr.Dataset(
            {
                "states_surface": (
                    ("time", "variable",
                     "latitude", "longitude"),
                    surf,
                ),
                "states_levels": (
                    ("time", "variable", "level",
                     "latitude", "longitude"),
                    lev,
                ),
            },
            coords={
                "time": times,
                "latitude": np.linspace(-90, 90, n_lat),
                "longitude": np.linspace(0, 360, n_lon),
            },
        )
        ds_xr.to_zarr(path, mode="w", consolidated=True)

        assert (
            ds_xr["states_surface"].dtype == np.float64
        )
        assert (
            ds_xr["states_levels"].dtype == np.float64
        )

        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            forcing_variables=["states_levels"],
            n_steps=2,
            n_step_size=1,
        )
        sample = ds[0]

        assert sample["states_surface"].dtype == np.float32
        assert sample["states_levels"].dtype == np.float32

    def test_get_data_sample_all_vars(
            self,
            tmp_path: object,
    ) -> None:
        r'''_get_data_sample returns state and forcing arrays.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=5)
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            forcing_variables=["states_levels"],
            n_steps=2,
        )
        time_slice = slice(0, 2, 1)
        samp = ds._get_data_sample(time_slice)
        assert "states_surface" in samp
        assert "states_levels" in samp
        assert samp["states_surface"].shape == (2, 2, 4, 8)

    def test_getitem_time_is_float32_array(
            self,
            tmp_path: object,
    ) -> None:
        r'''time in sample is 1-D float32 array of len n_steps.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=5)
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            n_steps=2,
            n_step_size=1,
        )
        samp = ds[0]
        assert isinstance(samp["time"], np.ndarray)
        assert samp["time"].shape == (2,)
        assert samp["time"].dtype == np.float32

    def test_getitem_time_advances(
            self,
            tmp_path: object,
    ) -> None:
        r'''Successive samples have advancing time values.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=5)
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            n_steps=1,
            n_step_size=1,
        )
        t0 = ds[0]["time"]
        t1 = ds[1]["time"]
        # Arrays of length 1; compare first element
        assert t1[0] > t0[0]

    def test_idx_decomposition_same_time(
            self,
            tmp_path: object,
    ) -> None:
        r'''Consecutive ensemble indices share the same time.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4, n_ens=3)
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            n_steps=1,
        )
        s0 = ds[0]   # time=0, ens=0
        s1 = ds[1]   # time=0, ens=1
        s3 = ds[3]   # time=1, ens=0
        assert np.array_equal(s0["time"], s1["time"])
        assert not np.array_equal(s3["time"], s0["time"])

    def test_idx_ensemble_decomposition_exact(
            self,
            tmp_path: object,
    ) -> None:
        r'''ensemble_idx = idx % n_ensemble selects right member.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4, n_ens=3)
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            n_steps=1,
        )
        # idx=5 → time_idx=1, ensemble_idx=2
        sample = ds[5]
        expected = ds._get_var(
            "states_surface",
            slice(1, 2, 1),
            ens_idx=2,
        )
        assert np.array_equal(
            sample["states_surface"], expected,
        )

    def test_check_variables_valid_passes(
            self,
            tmp_path: object,
    ) -> None:
        r'''_check_variables_in_dataset ok for present vars.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4)
        ds = TrainDataset(
            path,
            state_variables=["states_surface"],
        )
        with xr.open_zarr(
            path,
            consolidated=True,
            decode_times=False,
            decode_cf=False,
            decode_coords=False,
        ) as meta:
            ds._check_variables_in_dataset(
                meta,
                ["states_surface"],
                ["time"],
            )

    def test_auxiliary_arrays_keys(
            self,
            tmp_path: object,
            mocked_auxiliary_netcdf: object,
    ) -> None:
        r'''_auxiliary_arrays contains the requested keys.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4)
        aux_path = mocked_auxiliary_netcdf(
            n_ens=None, n_lat=4, n_lon=8,
        )
        ds = TrainDataset(
            path,
            state_variables=["states_surface"],
            auxiliary_path=aux_path,
            auxiliary_variables=["mask", "mesh"],
        )
        assert "mask" in ds._auxiliary_arrays
        assert "mesh" in ds._auxiliary_arrays

    def test_auxiliary_arrays_dtype(
            self,
            tmp_path: object,
            mocked_auxiliary_netcdf: object,
    ) -> None:
        r'''_auxiliary_arrays values are float32 ndarrays.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4)
        aux_path = mocked_auxiliary_netcdf(
            n_ens=None, n_lat=4, n_lon=8,
        )
        ds = TrainDataset(
            path,
            state_variables=["states_surface"],
            auxiliary_path=aux_path,
            auxiliary_variables=["mask"],
        )
        assert isinstance(
            ds._auxiliary_arrays["mask"], np.ndarray,
        )
        assert (
            ds._auxiliary_arrays["mask"].dtype == np.float32
        )

    def test_auxiliary_arrays_empty_when_no_aux(
            self,
            tmp_path: object,
    ) -> None:
        r'''_auxiliary_arrays is empty dict when no aux given.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4)
        ds = TrainDataset(
            path,
            state_variables=["states_surface"],
        )
        assert ds._auxiliary_arrays == {}

    def test_warns_aux_path_no_vars(
            self,
            tmp_path: object,
            caplog: pytest.LogCaptureFixture,
    ) -> None:
        r'''Warning when aux path given but no variables.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4)
        caplog.set_level(logging.WARNING)
        TrainDataset(
            path,
            state_variables=["states_surface"],
            auxiliary_path="/nonexistent/path.nc",
            auxiliary_variables=None,
        )
        assert "Auxiliary path provided" in caplog.text

    def test_warns_aux_vars_no_path(
            self,
            tmp_path: object,
            caplog: pytest.LogCaptureFixture,
    ) -> None:
        r'''Warning when aux vars given but no path.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4)
        caplog.set_level(logging.WARNING)
        TrainDataset(
            path,
            state_variables=["states_surface"],
            auxiliary_path=None,
            auxiliary_variables=["mask"],
        )
        assert "Auxiliary variables specified" in caplog.text

    def test_no_aux_info_log(
            self,
            tmp_path: object,
            caplog: pytest.LogCaptureFixture,
    ) -> None:
        r'''Info message logged when no auxiliary data at all.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4)
        caplog.set_level(logging.INFO)
        ds = TrainDataset(
            path,
            state_variables=["states_surface"],
        )
        assert ds._auxiliary_arrays == {}
        assert (
            "No auxiliary data will be included"
            in caplog.text
        )

    def test_no_forcing_variables_empty_list(
            self,
            tmp_path: object,
    ) -> None:
        r'''forcing_variables=None sets an empty list.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4)
        ds = TrainDataset(
            path,
            state_variables=["states_surface"],
            forcing_variables=None,
        )
        assert ds.forcing_variables == []
        assert ds.variables == ["states_surface"]


# -----------------------------------------------------------
# Error tests
# -----------------------------------------------------------

class TestDatasetErrors:
    r'''Error condition tests for TrainDataset.'''

    def test_missing_state_variable_raises(
            self,
            tmp_path: object,
    ) -> None:
        r'''Missing state variable raises KeyError (bug B-3).

        The docstring documents ``ValueError``; source raises
        ``KeyError`` via ``_check_variables_in_dataset``.
        '''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=3)
        with pytest.raises(KeyError):
            TrainDataset(
                path,
                state_variables=["nonexistent"],
            )

    def test_missing_forcing_variable_raises(
            self,
            tmp_path: object,
    ) -> None:
        r'''Missing forcing variable raises KeyError (bug B-3).'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=3)
        with pytest.raises(KeyError):
            TrainDataset(
                path,
                state_variables=["states_surface"],
                forcing_variables=["nonexistent"],
            )

    def test_missing_aux_variable_raises(
            self,
            tmp_path: object,
            mocked_auxiliary_netcdf: object,
    ) -> None:
        r'''Missing auxiliary variable raises KeyError.

        Source accesses ``ds[var_name]`` inside a dict
        comprehension; xarray raises ``KeyError`` for unknown
        variable names.
        '''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=3)
        aux_path = mocked_auxiliary_netcdf(
            n_ens=None, n_lat=4, n_lon=8,
        )
        with pytest.raises(KeyError):
            TrainDataset(
                path,
                state_variables=["states_surface"],
                auxiliary_path=aux_path,
                auxiliary_variables=["nonexistent_var"],
            )

    def test_too_few_time_steps_raises(
            self,
            tmp_path: object,
    ) -> None:
        r'''ValueError when dataset has too few time steps.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=2)
        with pytest.raises(
            ValueError, match="require at least",
        ):
            TrainDataset(
                path,
                state_variables=["states_surface"],
                n_steps=3,
                n_step_size=2,
            )

    def test_missing_time_dimension_raises(
            self,
            tmp_path: object,
    ) -> None:
        r'''KeyError when zarr has no time dimension.'''
        path = str(tmp_path / "no_time.zarr")
        xr.Dataset({
            "states_surface": (
                ("lat", "lon"),
                np.zeros((4, 8)),
            ),
        }).to_zarr(path, mode="w", consolidated=True)
        with pytest.raises(KeyError):
            TrainDataset(
                path,
                state_variables=["states_surface"],
            )

    def test_check_variables_missing_raises_keyerror(
            self,
            tmp_path: object,
    ) -> None:
        r'''_check_variables_in_dataset raises KeyError (bug B-3).

        Documented interface says ``ValueError``; source uses
        ``ds[var_name]`` which raises ``KeyError``.
        '''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4)
        ds = TrainDataset(
            path,
            state_variables=["states_surface"],
        )
        with xr.open_zarr(
            path,
            consolidated=True,
            decode_times=False,
            decode_cf=False,
            decode_coords=False,
        ) as meta:
            with pytest.raises(KeyError):
                ds._check_variables_in_dataset(
                    meta,
                    ["no_such_var"],
                    ["time"],
                )

    def test_check_variables_wrong_dims_raises(
            self,
            tmp_path: object,
    ) -> None:
        r'''_check_variables raises ValueError for wrong dims.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=4)
        ds = TrainDataset(
            path,
            state_variables=["states_surface"],
        )
        with xr.open_zarr(
            path,
            consolidated=True,
            decode_times=False,
            decode_cf=False,
            decode_coords=False,
        ) as meta:
            with pytest.raises(
                ValueError,
                match="not correct",
            ):
                ds._check_variables_in_dataset(
                    meta,
                    ["states_surface"],
                    ["time", "ensemble"],
                )


# -----------------------------------------------------------
# Edge cases
# -----------------------------------------------------------

class TestDatasetEdgeCases:
    r'''Boundary condition tests for TrainDataset.'''

    def test_minimum_one_sample(
            self,
            tmp_path: object,
    ) -> None:
        r'''Dataset with minimum valid length has 1 sample.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=2)
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            n_steps=2,
            n_step_size=1,
        )
        assert len(ds) == 1
        _ = ds[0]

    def test_fixture_zarr_store(
            self,
            zarr_store: zarr.Group,
    ) -> None:
        r'''TrainDataset works with the zarr_store fixture.'''
        store_path = str(zarr_store.store.path)
        ds = TrainDataset(
            store_path,
            state_variables=["states_surface"],
        )
        assert ds.n_times == zarr_store["time"].shape[0]

    def test_n_steps_one_getitem_shape(
            self,
            tmp_path: object,
    ) -> None:
        r'''n_steps=1 yields a length-1 time axis per sample.'''
        path = str(tmp_path / "t.zarr")
        _create_test_zarr(path, n_times=3)
        ds = _make_ds(
            path,
            state_variables=["states_surface"],
            n_steps=1,
            n_step_size=1,
        )
        samp = ds[2]
        assert samp["states_surface"].shape == (1, 2, 4, 8)

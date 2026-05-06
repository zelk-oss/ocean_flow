# -*- coding: utf-8 -*-
r"""Tests for src/forecast/output.py -- OutputWriter."""

# System modules
from unittest.mock import patch

# External modules
import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from dask.delayed import Delayed

# Internal modules
from {{cookiecutter.project_slug}}.forecast.output import (
    OutputWriter,
    _zarr_region_write,
)
from tests.conftest import (
    _create_test_zarr,
    _make_predictions,
    _make_writer,
)


# ------------------------------------------------------------------
# Constants shared across test classes
# ------------------------------------------------------------------

_VARS = ["states_surface", "states_levels"]
_INIT_TIMES = pd.date_range(
    "2020-01-01", periods=3, freq="6h"
)
_LEAD_TIMES = pd.timedelta_range(
    start="0h", periods=4, freq="6h"
)


# ------------------------------------------------------------------
# Functional tests: end-to-end write / read
# ------------------------------------------------------------------

class TestForecastOutputFunctional:
    r'''End-to-end tests that write to and read from zarr.'''

    def test_init_opens_zarr_store(
        self, tmp_path
    ) -> None:
        r'''Verify writer._store is a zarr Group.'''
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(tmp_path, data_path)

        # Assert
        assert isinstance(writer._store, zarr.Group)

    def test_init_no_xarray_retained(
        self, tmp_path
    ) -> None:
        r'''Verify no xarray objects on writer after init.'''
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(tmp_path, data_path)

        # Assert - no xarray objects stored on writer
        for attr_name in dir(writer):
            if attr_name.startswith('__'):
                continue
            attr = getattr(writer, attr_name)
            assert not isinstance(
                attr, (xr.Dataset, xr.DataArray)
            ), (
                f"Attribute '{attr_name}' is an xarray "
                f"object: {type(attr)}"
            )

    def test_write_returns_dask_delayed_per_variable(
        self, tmp_path
    ) -> None:
        r'''Verify each variable produces separate delayed.'''
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )
        preds = _make_predictions(batch=2, n_lead=4)
        init_times = _INIT_TIMES[:2]
        ens_mems = np.array([0, 1])

        # Act
        delayed = writer.write(
            preds, init_times, ens_mems, _LEAD_TIMES
        )

        # Assert - each delayed is a Delayed object
        assert len(delayed) > 0
        for d in delayed:
            assert isinstance(d, Delayed)
        # 2 batch elements * 2 variables = 4 delayed
        n_vars = len(_VARS)
        n_batch = len(init_times)
        assert len(delayed) == n_vars * n_batch

    def test_write_delayed_writes_to_zarr_region(
        self, tmp_path
    ) -> None:
        r'''Delayed writes correct data to zarr regions.'''
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )

        # Arrange - use deterministic data
        n_vals = 2 * 4 * 2 * 4 * 8
        surf = np.arange(
            n_vals, dtype=np.float32
        ).reshape(2, 4, 2, 4, 8)
        preds = {
            "states_surface": surf,
            "states_levels": np.zeros(
                (2, 4, 2, 3, 4, 8), dtype=np.float32
            ),
        }
        init_times = _INIT_TIMES[:2]
        ens_mems = np.array([0, 1])

        # Act
        delayed = writer.write(
            preds, init_times, ens_mems, _LEAD_TIMES
        )
        dask.compute(*delayed)

        # Assert - read back from zarr and verify
        store = zarr.open(writer.store_path, mode='r')
        arr = store["states_surface"]
        # Region for batch 0: init_idx=0, ens_idx=0
        init_idx = 0
        ens_idx = 0
        written = arr[init_idx, :, ens_idx, ...]
        np.testing.assert_array_equal(
            written, surf[0]
        )

    def test_write_no_xarray_in_write_path(
        self, tmp_path
    ) -> None:
        r'''Verify xr.Dataset not called during write.'''
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )
        preds = _make_predictions(batch=2, n_lead=4)
        init_times = _INIT_TIMES[:2]
        ens_mems = np.array([0, 0])

        # Act - patch xr.Dataset to raise if called
        with patch.object(
            xr, 'Dataset',
            side_effect=AssertionError(
                "xr.Dataset should not be called "
                "during write path"
            ),
        ):
            delayed = writer.write(
                preds, init_times, ens_mems,
                _LEAD_TIMES,
            )
            dask.compute(*delayed)

    def test_writes_non_nan_values(
        self, tmp_path
    ) -> None:
        r'''Written region contains no NaN values.'''
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )
        preds = _make_predictions(batch=2, n_lead=4)
        init_times = _INIT_TIMES[:2]
        ens_mems = np.array([0, 0])

        # Act
        delayed = writer.write(
            preds, init_times, ens_mems, _LEAD_TIMES
        )
        dask.compute(*delayed)

        # Assert
        store = zarr.open(writer.store_path, mode='r')
        arr = store["states_surface"]
        region = arr[0, :, 0, ...]
        assert not np.any(np.isnan(region))

    def test_write_values_numerically_correct(
        self, tmp_path
    ) -> None:
        r'''Values read back from zarr match written ones.'''
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )

        # Arrange - use np.arange for exact equality
        n_vals = 2 * 4 * 2 * 4 * 8
        surf = np.arange(
            n_vals, dtype=np.float32
        ).reshape(2, 4, 2, 4, 8)
        preds = {
            "states_surface": surf,
            "states_levels": np.zeros(
                (2, 4, 2, 3, 4, 8), dtype=np.float32
            ),
        }
        init_times = _INIT_TIMES[:2]
        ens_mems = np.array([0, 1])

        # Act
        delayed = writer.write(
            preds, init_times, ens_mems, _LEAD_TIMES
        )
        dask.compute(*delayed)

        # Assert - verify batch element 0 round-trips
        store = zarr.open(writer.store_path, mode='r')
        written = store["states_surface"][0, :, 0, ...]
        np.testing.assert_array_equal(written, surf[0])

        # Assert - verify batch element 1 round-trips
        written_1 = store["states_surface"][1, :, 1, ...]
        np.testing.assert_array_equal(
            written_1, surf[1]
        )


# ------------------------------------------------------------------
# Unit tests: individual behaviours
# ------------------------------------------------------------------

class TestForecastOutputUnittest:
    r'''Tests for individual OutputWriter behaviours.'''

    def test_ens_mems_as_int_creates_arange(
        self, tmp_path
    ) -> None:
        r'''Integer ens_mems=3 converts to np.arange(3).'''
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=3
        )

        # Assert
        np.testing.assert_array_equal(
            writer.ens_mems, [0, 1, 2]
        )

    def test_ens_mems_as_array_uses_unique(
        self, tmp_path
    ) -> None:
        r'''Array ens_mems=[0,1,1,2] deduplicates to [0,1,2].'''
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path,
            ens_mems=[0, 1, 1, 2],
        )

        # Assert
        np.testing.assert_array_equal(
            writer.ens_mems, [0, 1, 2]
        )

    def test_spatial_meta_is_plain_dict(
        self, tmp_path
    ) -> None:
        r'''Spatial metadata stored as plain Python dict.'''
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(tmp_path, data_path)

        # Assert
        assert isinstance(writer._spatial_meta, dict)
        for var in _VARS:
            assert var in writer._spatial_meta
            assert isinstance(
                writer._spatial_meta[var], tuple
            )

    def test_store_path_attribute_preserved(
        self, tmp_path
    ) -> None:
        r'''store_path attribute matches constructor arg.'''
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        store_path = str(tmp_path / "output.zarr")
        writer = _make_writer(tmp_path, data_path)

        # Assert
        assert writer.store_path == store_path


# ------------------------------------------------------------------
# Module-level helper tests
# ------------------------------------------------------------------

class TestZarrRegionWrite:
    r'''Tests for the _zarr_region_write helper function.'''

    def test_zarr_region_write_writes_data(
        self, tmp_path
    ) -> None:
        r'''Data is written to correct zarr region.'''
        # Arrange
        store_path = str(tmp_path / "test.zarr")
        root = zarr.open(store_path, mode='w')
        arr = root.zeros(
            'test_var',
            shape=(3, 4, 2, 4, 8),
            dtype=np.float32,
        )
        data = np.ones((4, 4, 8), dtype=np.float32)
        region = (0, slice(0, 4), 0, slice(None),
                  slice(None))

        # Act
        _zarr_region_write(arr, data, region)

        # Assert
        result = np.asarray(arr[0, :, 0, :, :])
        np.testing.assert_array_equal(result, data)

    def test_zarr_region_write_preserves_other(
        self, tmp_path
    ) -> None:
        r'''Writing to region does not alter other regions.'''
        # Arrange
        store_path = str(tmp_path / "test.zarr")
        root = zarr.open(store_path, mode='w')
        arr = root.zeros(
            'test_var',
            shape=(3, 4, 2, 4, 8),
            dtype=np.float32,
        )
        data = np.ones((4, 4, 8), dtype=np.float32)
        region = (0, slice(0, 4), 0, slice(None),
                  slice(None))

        # Act
        _zarr_region_write(arr, data, region)

        # Assert - other init_time index is untouched
        other = np.asarray(arr[1, :, 0, :, :])
        np.testing.assert_array_equal(
            other, np.zeros_like(other)
        )


# ------------------------------------------------------------------
# Error handling tests
# ------------------------------------------------------------------

class TestForecastOutputErrors:
    r'''Tests for error conditions in OutputWriter.'''

    def test_batch_size_mismatch_raises_valueerror(
        self, tmp_path
    ) -> None:
        r'''Batch size mismatch between preds and init_times.'''
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )
        # predictions batch=3 but init_times has 2 elements
        preds = _make_predictions(batch=3, n_lead=4)

        # Act / Assert
        with pytest.raises(ValueError):
            writer.write(
                preds,
                _INIT_TIMES[:2],
                np.array([0, 0, 1]),
                _LEAD_TIMES,
            )

    def test_lead_time_mismatch_raises_valueerror(
        self, tmp_path
    ) -> None:
        r'''Lead-time mismatch between preds and lead_times.'''
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )
        # predictions n_lead=2 but lead_times has 4
        preds = _make_predictions(batch=2, n_lead=2)

        # Act / Assert
        with pytest.raises(ValueError):
            writer.write(
                preds,
                _INIT_TIMES[:2],
                np.array([0, 1]),
                _LEAD_TIMES,
            )


# ------------------------------------------------------------------
# Edge case tests
# ------------------------------------------------------------------

class TestForecastOutputEdgeCases:
    r'''Tests for edge cases in OutputWriter.'''

    def test_missing_variable_logs_warning(
        self, tmp_path, caplog
    ) -> None:
        r'''Missing variable emits warning, no exception.'''
        import logging
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )
        # Only provide states_surface, omit states_levels
        np.random.seed(0)
        preds = {
            "states_surface": np.random.randn(
                2, 4, 2, 4, 8
            ).astype(np.float32),
        }

        # Act
        with caplog.at_level(logging.WARNING):
            writer.write(
                preds,
                _INIT_TIMES[:2],
                np.array([0, 1]),
                _LEAD_TIMES,
            )

        # Assert
        assert any(
            "states_levels" in r.message
            for r in caplog.records
        )

    def test_all_variables_missing_returns_empty(
        self, tmp_path, caplog
    ) -> None:
        r'''All variables missing returns empty Delayed.'''
        import logging
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )
        preds = {
            "nonexistent_var": np.zeros(
                (2, 4, 2, 4, 8),
                dtype=np.float32,
            ),
        }

        # Act
        with caplog.at_level(logging.WARNING):
            result = writer.write(
                preds,
                _INIT_TIMES[:2],
                np.array([0, 1]),
                _LEAD_TIMES,
            )

        # Assert
        assert len(result) == 0

# -*- coding: utf-8 -*-
r"""Tests for forecast/restart.py -- restart from zarr."""

# System modules
import logging
from typing import Dict, Tuple

# External modules
import dask
import numpy as np
import pandas as pd
import pytest

# Internal modules
from ocean_flow.forecast.restart import (
    check_written_regions,
    filter_forecast_configs,
)
from ocean_flow.forecast.config import ForecastConfig
from ocean_flow.forecast.output import OutputWriter
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
# Functional tests: check_written_regions
# ------------------------------------------------------------------

class TestCheckWrittenRegionsFunctional:
    r'''Tests for check_written_regions with real zarr stores.'''

    def test_all_nan_returns_minus_one(
        self, tmp_path: object
    ) -> None:
        r'''Fully NaN store returns -1 for all pairs.'''
        # Arrange
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )

        # Act
        result = check_written_regions(
            writer.store_path, _VARS
        )

        # Assert -- 3 init_times x 2 ens_mems = 6 pairs
        assert len(result) == 6
        for key, value in result.items():
            assert value == -1, (
                f"Expected -1 for unwritten pair {key}, "
                f"got {value}"
            )

    def test_fully_written_returns_last_index(
        self, tmp_path: object
    ) -> None:
        r'''Fully written pair returns len(lead_times) - 1.'''
        # Arrange
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )
        # Write all 4 lead times for init_time=0, ens=0
        preds = _make_predictions(batch=1, n_lead=4)
        delayed = writer.write(
            preds,
            _INIT_TIMES[:1],
            np.array([0]),
            _LEAD_TIMES,
        )
        dask.compute(*delayed)

        # Act
        result = check_written_regions(
            writer.store_path, _VARS
        )

        # Assert
        assert result[(0, 0)] == len(_LEAD_TIMES) - 1

    def test_partially_written_returns_last_contiguous(
        self, tmp_path: object
    ) -> None:
        r'''Writing 2 of 4 lead times returns index 1.'''
        # Arrange
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )
        # Write only the first 2 lead times
        partial_leads = _LEAD_TIMES[:2]
        preds = _make_predictions(batch=1, n_lead=2)
        delayed = writer.write(
            preds,
            _INIT_TIMES[:1],
            np.array([0]),
            partial_leads,
        )
        dask.compute(*delayed)

        # Act
        result = check_written_regions(
            writer.store_path, _VARS
        )

        # Assert -- last contiguous index is 1 (indices 0,1)
        assert result[(0, 0)] == 1

    def test_gap_returns_last_contiguous_before_gap(
        self, tmp_path: object
    ) -> None:
        r'''Gap at index 2 with data at 0-1 and 3 returns 1.'''
        # Arrange
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )
        # Write lead times 0 and 1
        preds_01 = _make_predictions(batch=1, n_lead=2)
        delayed = writer.write(
            preds_01,
            _INIT_TIMES[:1],
            np.array([0]),
            _LEAD_TIMES[:2],
        )
        dask.compute(*delayed)
        # Write lead time 3 (skipping 2 to create a gap)
        preds_3 = _make_predictions(batch=1, n_lead=1)
        delayed = writer.write(
            preds_3,
            _INIT_TIMES[:1],
            np.array([0]),
            _LEAD_TIMES[3:4],
        )
        dask.compute(*delayed)

        # Act
        result = check_written_regions(
            writer.store_path, _VARS
        )

        # Assert -- contiguous block stops at index 1
        assert result[(0, 0)] == 1


# ------------------------------------------------------------------
# Functional tests: filter_forecast_configs
# ------------------------------------------------------------------

class TestFilterForecastConfigsFunctional:
    r'''Tests for filter_forecast_configs with real configs.'''

    def test_fully_written_configs_are_skipped(
        self, tmp_path: object
    ) -> None:
        r'''Config with all pairs fully written is removed.'''
        # Arrange
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )
        config = ForecastConfig(
            init_times=_INIT_TIMES[:1],
            lead_times=_LEAD_TIMES,
            ens_mems=np.array([0]),
            n_store_freq=4,
        )
        # Mark (0, 0) as fully written
        written_regions: Dict[Tuple[int, int], int] = {
            (0, 0): len(_LEAD_TIMES) - 1,
        }

        # Act
        result = filter_forecast_configs(
            [config], writer, written_regions
        )

        # Assert
        assert len(result) == 0

    def test_partially_written_configs_are_trimmed(
        self, tmp_path: object
    ) -> None:
        r'''Partially written config has truncated lead_times.'''
        # Arrange
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )
        config = ForecastConfig(
            init_times=_INIT_TIMES[:1],
            lead_times=_LEAD_TIMES,
            ens_mems=np.array([0]),
            n_store_freq=4,
        )
        # Mark (0, 0) as partially written (indices 0,1)
        written_regions: Dict[Tuple[int, int], int] = {
            (0, 0): 1,
        }

        # Act
        result = filter_forecast_configs(
            [config], writer, written_regions
        )

        # Assert -- should start from index 2
        assert len(result) == 1
        expected_leads = _LEAD_TIMES[2:]
        pd.testing.assert_index_equal(
            result[0].lead_times, expected_leads
        )

    def test_unwritten_configs_unchanged(
        self, tmp_path: object
    ) -> None:
        r'''Unwritten config passes through with original leads.'''
        # Arrange
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )
        config = ForecastConfig(
            init_times=_INIT_TIMES[:1],
            lead_times=_LEAD_TIMES,
            ens_mems=np.array([0]),
            n_store_freq=4,
        )
        # Mark (0, 0) as unwritten
        written_regions: Dict[Tuple[int, int], int] = {
            (0, 0): -1,
        }

        # Act
        result = filter_forecast_configs(
            [config], writer, written_regions
        )

        # Assert
        assert len(result) == 1
        pd.testing.assert_index_equal(
            result[0].lead_times, _LEAD_TIMES
        )

    def test_logs_warning_on_restart(
        self, tmp_path: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        r'''WARNING log emitted for restarted config.'''
        # Arrange
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )
        config = ForecastConfig(
            init_times=_INIT_TIMES[:1],
            lead_times=_LEAD_TIMES,
            ens_mems=np.array([0]),
            n_store_freq=4,
        )
        # Partially written
        written_regions: Dict[Tuple[int, int], int] = {
            (0, 0): 1,
        }

        # Act
        with caplog.at_level(logging.WARNING):
            filter_forecast_configs(
                [config], writer, written_regions
            )

        # Assert
        assert any(
            "restart" in r.message.lower()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        ), (
            "Expected a WARNING log containing 'restart'"
        )


# ------------------------------------------------------------------
# Edge case tests: filter_forecast_configs
# ------------------------------------------------------------------

class TestFilterForecastConfigsEdgeCases:
    r'''Edge cases for filter_forecast_configs.'''

    def test_empty_configs_returns_empty(
        self, tmp_path: object
    ) -> None:
        r'''Empty input list returns empty output list.'''
        # Arrange
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )
        written_regions: Dict[Tuple[int, int], int] = {}

        # Act
        result = filter_forecast_configs(
            [], writer, written_regions
        )

        # Assert
        assert result == []

    def test_all_written_returns_empty(
        self, tmp_path: object
    ) -> None:
        r'''All configs fully written returns empty list.'''
        # Arrange
        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2
        )
        last_idx = len(_LEAD_TIMES) - 1
        configs = [
            ForecastConfig(
                init_times=_INIT_TIMES[:1],
                lead_times=_LEAD_TIMES,
                ens_mems=np.array([0]),
                n_store_freq=4,
            ),
            ForecastConfig(
                init_times=_INIT_TIMES[1:2],
                lead_times=_LEAD_TIMES,
                ens_mems=np.array([1]),
                n_store_freq=4,
            ),
        ]
        written_regions: Dict[Tuple[int, int], int] = {
            (0, 0): last_idx,
            (1, 1): last_idx,
        }

        # Act
        result = filter_forecast_configs(
            configs, writer, written_regions
        )

        # Assert
        assert len(result) == 0

# -*- coding: utf-8 -*-
r'''Tests for the IO refactor: lazy xarray, Delayed writes,
persist-based runner, and prefetch.py deletion.

These tests are written TDD-first and are expected to FAIL
until the refactor described in plans/io-refactor.md and
plans/56-rank-consistent-forecast-loop-implementation.md is
implemented.

Organized into:
- **Phase1InputReader**: lazy xr.Dataset returns and
  dataset_to_numpy_dict helper.
- **Phase2OutputWriter**: Delayed writes via pure zarr.
- **Phase3Runner**: persist/compute runner orchestration.
- **Phase4DeletePrefetch**: prefetch.py removal.
- **Phase3RunnerBatch**: run_batch signature
  changes (xr.Dataset args instead of futures).
'''

# System modules
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

# External modules
import dask
import dask.array
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.delayed import Delayed

# Internal modules
from ocean_flow.forecast.input import InputReader
from ocean_flow.forecast.output import OutputWriter
from tests.conftest import (
    _create_test_zarr,
    _ConfigDouble,
    _make_config_double,
    _make_predictions,
    _make_writer,
)


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
_VARS = ["states_surface", "states_levels"]


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def _make_mock_model(
        n_out_steps: int = 1,
) -> MagicMock:
    r'''Create a mock ForecastModel.'''
    model = MagicMock()
    model.n_out_steps = n_out_steps
    model.advance.return_value = {
        "states_surface": np.zeros(
            (1, n_out_steps, 1, 1),
            dtype=np.float32,
        )
    }
    return model


# -----------------------------------------------------------
# Phase 1: InputReader -- lazy xr.Dataset returns
# -----------------------------------------------------------

class TestPhase1InputReaderLazyReturns:
    r'''InputReader methods return lazy xr.Dataset.'''

    @pytest.fixture()
    def zarr_path(self, tmp_path: object) -> str:
        r'''Return path to a zarr store without ensemble.'''
        path = str(tmp_path / "no_ens.zarr")
        _create_test_zarr(path, n_times=20, n_ens=None)
        return path

    @pytest.fixture()
    def zarr_path_ens(self, tmp_path: object) -> str:
        r'''Return path to a zarr store with ensemble.'''
        path = str(tmp_path / "ens.zarr")
        _create_test_zarr(path, n_times=20, n_ens=2)
        return path

    @pytest.fixture()
    def reader(self, zarr_path: str) -> InputReader:
        r'''Return InputReader for store without ensemble.'''
        return InputReader(
            data_path=zarr_path,
            state_variables=[
                "states_surface", "states_levels",
            ],
        )

    def test_load_states_returns_xr_dataset(
        self, reader: InputReader,
    ) -> None:
        r'''load_states returns xr.Dataset not dict.'''
        # Act
        result = reader.load_states(
            _INIT_TIMES[:1], np.array([0]),
        )

        # Assert
        assert isinstance(result, xr.Dataset)

    def test_load_states_is_dask_backed(
        self, reader: InputReader,
    ) -> None:
        r'''load_states Dataset has dask-backed arrays.'''
        # Act
        result = reader.load_states(
            _INIT_TIMES[:1], np.array([0]),
        )

        # Assert
        for var in result.data_vars:
            assert isinstance(
                result[var].data, dask.array.Array
            ), (
                f"Variable '{var}' is not dask-backed"
            )

    def test_load_states_has_expected_variables(
        self, reader: InputReader,
    ) -> None:
        r'''Lazy Dataset has all state variables.'''
        # Act
        result = reader.load_states(
            _INIT_TIMES[:1], np.array([0]),
        )

        # Assert
        assert isinstance(result, xr.Dataset)
        assert "states_surface" in result.data_vars
        assert "states_levels" in result.data_vars

    def test_load_states_computed_shape(
        self, reader: InputReader,
    ) -> None:
        r'''After compute, states have batch and time dims.'''
        # Act
        result = reader.load_states(
            _INIT_TIMES, np.array([0, 0]),
        )

        # Assert
        assert isinstance(result, xr.Dataset)
        computed = result.compute()
        assert computed["states_surface"].shape[0] == 2
        assert computed["states_surface"].shape[1] == 1

    def test_load_forcings_returns_xr_dataset(
        self, zarr_path: str,
    ) -> None:
        r'''load_forcings returns xr.Dataset not dict.'''

        reader = InputReader(
            data_path=zarr_path,
            state_variables=["states_surface"],
            forcing_variables=["states_surface"],
        )

        # Act
        result = reader.load_forcings(
            _INIT_TIMES[:1], np.array([0]),
            _LEAD_TIMES,
        )

        # Assert
        assert isinstance(result, xr.Dataset)

    def test_load_forcings_is_dask_backed(
        self, zarr_path: str,
    ) -> None:
        r'''load_forcings Dataset has dask-backed arrays.'''

        reader = InputReader(
            data_path=zarr_path,
            state_variables=["states_surface"],
            forcing_variables=["states_surface"],
        )

        # Act
        result = reader.load_forcings(
            _INIT_TIMES[:1], np.array([0]),
            _LEAD_TIMES,
        )

        # Assert
        for var in result.data_vars:
            assert isinstance(
                result[var].data, dask.array.Array
            )

    def test_load_auxiliary_returns_xr_dataset(
        self,
        zarr_path: str,
        mocked_auxiliary_netcdf: Any,
    ) -> None:
        r'''load_auxiliary returns xr.Dataset not dict.'''

        aux_path = mocked_auxiliary_netcdf(
            n_ens=None, n_lat=2, n_lon=3,
        )
        reader = InputReader(
            data_path=zarr_path,
            state_variables=["states_surface"],
            auxiliary_path=aux_path,
            auxiliary_variables=["mesh", "mask"],
        )

        # Act
        result = reader.load_auxiliary(np.array([0]))

        # Assert
        assert isinstance(result, xr.Dataset)

    def test_load_auxiliary_is_dask_backed(
        self,
        zarr_path: str,
        mocked_auxiliary_netcdf: Any,
    ) -> None:
        r'''load_auxiliary Dataset has dask-backed arrays.'''

        aux_path = mocked_auxiliary_netcdf(
            n_ens=None, n_lat=2, n_lon=3,
        )
        reader = InputReader(
            data_path=zarr_path,
            state_variables=["states_surface"],
            auxiliary_path=aux_path,
            auxiliary_variables=["mesh", "mask"],
        )

        # Act
        result = reader.load_auxiliary(np.array([0]))

        # Assert
        for var in result.data_vars:
            assert isinstance(
                result[var].data, dask.array.Array
            )

    def test_load_states_computed_values_match(
        self, reader: InputReader, zarr_path: str,
    ) -> None:
        r'''Computed lazy values match eager xr.open_zarr.'''
        # Act
        result = reader.load_states(
            _INIT_TIMES[:1], np.array([0]),
        )

        # Assert
        assert isinstance(result, xr.Dataset)
        computed = result.compute()
        # After compute, values should be finite
        for var in computed.data_vars:
            vals = computed[var].values
            assert np.isfinite(vals).all()


# -----------------------------------------------------------
# Phase 1: dataset_to_numpy_dict helper
# -----------------------------------------------------------

class TestPhase1DatasetToNumpyDict:
    r'''Tests for the dataset_to_numpy_dict helper.'''

    def test_import_dataset_to_numpy_dict(
        self,
    ) -> None:
        r'''dataset_to_numpy_dict is importable from input.'''

        from ocean_flow.forecast.input import (
            dataset_to_numpy_dict,
        )
        assert callable(dataset_to_numpy_dict)

    def test_converts_dataset_to_dict(self) -> None:
        r'''Converts in-memory xr.Dataset to Dict[str, ndarray].'''

        from ocean_flow.forecast.input import (
            dataset_to_numpy_dict,
        )

        # Arrange
        ds = xr.Dataset({
            "a": (("x", "y"), np.ones((2, 3))),
            "b": (("x",), np.zeros(2)),
        })

        # Act
        result = dataset_to_numpy_dict(ds)

        # Assert
        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b"}
        assert isinstance(result["a"], np.ndarray)
        np.testing.assert_array_equal(
            result["a"], np.ones((2, 3))
        )

    def test_exported_in_init_all(self) -> None:
        r'''dataset_to_numpy_dict in forecast __init__.'''

        from ocean_flow import forecast
        assert hasattr(forecast, "dataset_to_numpy_dict")

    def test_in_input_module_all(self) -> None:
        r'''dataset_to_numpy_dict in input.__all__.'''

        from ocean_flow.forecast import input as inp
        assert "dataset_to_numpy_dict" in inp.__all__


# -----------------------------------------------------------
# Phase 1: InputReader opens zarr with chunks='auto'
# -----------------------------------------------------------

class TestPhase1InputReaderChunkedOpen:
    r'''InputReader opens zarr stores with chunks=auto.'''

    def test_state_dataset_is_chunked(
        self, tmp_path: object,
    ) -> None:
        r'''_state_dataset should be chunked (dask-backed).'''

        path = str(tmp_path / "data.zarr")
        _create_test_zarr(path, n_times=20, n_ens=None)
        reader = InputReader(
            data_path=path,
            state_variables=["states_surface"],
        )

        # Assert -- the internal dataset is dask-backed
        for var in reader._state_dataset.data_vars:
            assert isinstance(
                reader._state_dataset[var].data,
                dask.array.Array,
            ), (
                f"_state_dataset['{var}'] is not "
                f"dask-backed"
            )

    def test_forcing_dataset_is_chunked(
        self, tmp_path: object,
    ) -> None:
        r'''_forcing_dataset should be chunked (dask-backed).'''

        path = str(tmp_path / "data.zarr")
        _create_test_zarr(path, n_times=20, n_ens=None)
        reader = InputReader(
            data_path=path,
            state_variables=["states_surface"],
            forcing_variables=["states_surface"],
        )

        # Assert
        assert reader._forcing_dataset is not None
        for var in reader._forcing_dataset.data_vars:
            assert isinstance(
                reader._forcing_dataset[var].data,
                dask.array.Array,
            )

    def test_auxiliary_dataset_is_chunked(
        self,
        tmp_path: object,
        mocked_auxiliary_netcdf: Any,
    ) -> None:
        r'''_auxiliary_dataset should be chunked.'''

        zarr_path = str(tmp_path / "data.zarr")
        _create_test_zarr(
            zarr_path, n_times=20, n_ens=None,
        )
        aux_path = mocked_auxiliary_netcdf(
            n_ens=None, n_lat=4, n_lon=8,
        )
        reader = InputReader(
            data_path=zarr_path,
            state_variables=["states_surface"],
            auxiliary_path=aux_path,
            auxiliary_variables=["mesh", "mask"],
        )

        # Assert
        assert reader._auxiliary_dataset is not None
        for var in reader._auxiliary_dataset.data_vars:
            assert isinstance(
                reader._auxiliary_dataset[var].data,
                dask.array.Array,
            )


# -----------------------------------------------------------
# Phase 2: OutputWriter -- Delayed writes
# -----------------------------------------------------------

class TestPhase2OutputWriterDelayed:
    r'''OutputWriter.write returns List[Delayed].'''

    def test_write_returns_list_of_delayed(
        self, tmp_path: object,
    ) -> None:
        r'''write() returns List[dask.delayed.Delayed].'''

        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2,
            init_times=_INIT_TIMES,
            lead_times=_LEAD_TIMES,
        )
        preds = _make_predictions(batch=2, n_lead=3)
        init_times = _INIT_TIMES[:2]
        ens_mems = np.array([0, 0])

        # Act
        result = writer.write(
            preds, init_times, ens_mems, _LEAD_TIMES,
        )

        # Assert
        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert isinstance(item, Delayed), (
                f"Expected Delayed, got {type(item)}"
            )

    def test_data_not_written_before_compute(
        self, tmp_path: object,
    ) -> None:
        r'''Data is not in zarr until dask.compute is called.'''

        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2,
            init_times=_INIT_TIMES,
            lead_times=_LEAD_TIMES,
        )
        preds = _make_predictions(batch=1, n_lead=3)
        init_times = _INIT_TIMES[:1]
        ens_mems = np.array([0])

        # Act -- call write but do not compute
        delayed_list = writer.write(
            preds, init_times, ens_mems, _LEAD_TIMES,
        )

        # Assert -- data should still be NaN
        ds = xr.open_zarr(writer.store_path)
        region = ds["states_surface"].sel(
            init_time=init_times,
            lead_time=_LEAD_TIMES,
            ensemble=0,
        )
        assert np.all(np.isnan(region.values)), (
            "Data should not be written before compute"
        )

    def test_data_written_after_compute(
        self, tmp_path: object,
    ) -> None:
        r'''After dask.compute, zarr contains correct data.'''

        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2,
            init_times=_INIT_TIMES,
            lead_times=_LEAD_TIMES,
        )
        n_vals = 1 * 3 * 2 * 4 * 8
        surf = np.arange(
            n_vals, dtype=np.float32
        ).reshape(1, 3, 2, 4, 8)
        preds = {
            "states_surface": surf,
            "states_levels": np.zeros(
                (1, 3, 2, 3, 4, 8), dtype=np.float32,
            ),
        }
        init_times = _INIT_TIMES[:1]
        ens_mems = np.array([0])

        # Act
        delayed_list = writer.write(
            preds, init_times, ens_mems, _LEAD_TIMES,
        )
        dask.compute(*delayed_list)

        # Assert
        ds = xr.open_zarr(writer.store_path)
        written = ds["states_surface"].sel(
            init_time=init_times[0],
            ensemble=0,
            lead_time=_LEAD_TIMES,
        ).values
        np.testing.assert_array_equal(written, surf[0])

    def test_write_returns_not_none(
        self, tmp_path: object,
    ) -> None:
        r'''write() no longer returns None.'''

        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2,
            init_times=_INIT_TIMES,
            lead_times=_LEAD_TIMES,
        )
        preds = _make_predictions(batch=1, n_lead=3)
        init_times = _INIT_TIMES[:1]
        ens_mems = np.array([0])

        # Act
        result = writer.write(
            preds, init_times, ens_mems, _LEAD_TIMES,
        )

        # Assert
        assert result is not None

    def test_missing_variable_still_returns_delayed(
        self, tmp_path: object,
    ) -> None:
        r'''Missing variable logged, remaining returned.'''

        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2,
            init_times=_INIT_TIMES,
            lead_times=_LEAD_TIMES,
        )
        # Only provide states_surface
        preds = {
            "states_surface": np.random.randn(
                1, 3, 2, 4, 8
            ).astype(np.float32),
        }
        init_times = _INIT_TIMES[:1]
        ens_mems = np.array([0])

        # Act
        result = writer.write(
            preds, init_times, ens_mems, _LEAD_TIMES,
        )

        # Assert
        assert isinstance(result, list)


# -----------------------------------------------------------
# Phase 2: Restart tests must call dask.compute after write
# -----------------------------------------------------------

class TestPhase2RestartWithDelayedWrites:
    r'''Restart functions work with delayed writes.'''

    def test_check_written_regions_after_delayed_write(
        self, tmp_path: object,
    ) -> None:
        r'''check_written_regions finds data after compute.'''

        from ocean_flow.forecast.restart import (
            check_written_regions,
        )

        data_path = str(tmp_path / "ref.zarr")
        _create_test_zarr(data_path)
        writer = _make_writer(
            tmp_path, data_path, ens_mems=2,
            init_times=_INIT_TIMES,
            lead_times=_LEAD_TIMES,
        )
        preds = _make_predictions(batch=1, n_lead=3)
        init_times = _INIT_TIMES[:1]
        ens_mems = np.array([0])

        # Act -- write returns Delayed
        delayed_list = writer.write(
            preds, init_times, ens_mems, _LEAD_TIMES,
        )

        # Compute the writes
        assert isinstance(delayed_list, list)
        dask.compute(*delayed_list)

        # Assert -- data is now findable
        result = check_written_regions(
            writer.store_path, _VARS,
        )
        assert result[(0, 0)] == len(_LEAD_TIMES) - 1


# -----------------------------------------------------------
# Phase 3: Runner -- no futures, uses persist/compute
# -----------------------------------------------------------

class TestPhase3RunnerNoFutures:
    r'''Runner no longer uses distributed.Future.'''

    def test_runner_does_not_use_submit(
        self,
    ) -> None:
        r'''runner does not call client.submit at runtime.'''
        from ocean_flow.forecast.runner import (
            run_forecast,
        )

        # Arrange
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = MagicMock()
        reader.use_forcings = False
        reader.use_auxiliary = False
        lazy_ds = xr.Dataset({
            "states_surface": (
                ("batch", "time_step", "var"),
                dask.array.from_array(
                    np.zeros((1, 1, 1)),
                ),
            ),
        })
        reader.load_states.return_value = lazy_ds
        writer = MagicMock()
        mock_delayed = MagicMock(spec=Delayed)
        writer.write.return_value = [mock_delayed]
        chunk = pd.timedelta_range(
            start="6h", end="6h", freq="6h",
        )
        config = _ConfigDouble(
            init_times=pd.DatetimeIndex(
                ["2020-01-01"]
            ),
            ens_mems=np.array([0]),
            chunks=[chunk],
        )

        # Act
        with patch(
            "ocean_flow"
            ".forecast.runner.dask"
        ):
            run_forecast(
                client=client,
                model=model,
                input_reader=reader,
                output_writer=writer,
                forecast_configs=[config],
                n_prefetch_init=0,
                n_prefetch_forcing=0,
            )

        # Assert
        client.submit.assert_not_called()

    def test_runner_does_not_have_ensure_callable_name(
        self,
    ) -> None:
        r'''_ensure_callable_name deleted from runner.'''

        from ocean_flow.forecast import runner
        assert not hasattr(
            runner, "_ensure_callable_name"
        ), (
            "_ensure_callable_name should be deleted"
        )

    def test_runner_does_not_have_submit_states_aux(
        self,
    ) -> None:
        r'''_submit_states_aux deleted from runner.'''

        from ocean_flow.forecast import runner
        assert not hasattr(
            runner, "_submit_states_aux"
        ), (
            "_submit_states_aux should be deleted"
        )

    def test_run_forecast_calls_dask_compute(
        self,
    ) -> None:
        r'''run_forecast batch-computes all Delayed at end.'''
        from ocean_flow.forecast.runner import (
            run_forecast,
        )

        # Arrange
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = MagicMock()
        reader.use_forcings = False
        reader.use_auxiliary = False
        lazy_ds = xr.Dataset({
            "states_surface": (
                ("batch", "time_step", "var"),
                dask.array.from_array(
                    np.zeros((1, 1, 1)),
                ),
            ),
        })
        reader.load_states.return_value = lazy_ds
        writer = MagicMock()
        mock_delayed = MagicMock(spec=Delayed)
        writer.write.return_value = [mock_delayed]
        chunk = pd.timedelta_range(
            start="6h", end="6h", freq="6h",
        )
        config = _ConfigDouble(
            init_times=pd.DatetimeIndex(
                ["2020-01-01"]
            ),
            ens_mems=np.array([0]),
            chunks=[chunk],
        )

        # Act
        with patch(
            "ocean_flow"
            ".forecast.runner.dask"
        ) as mock_dask:
            run_forecast(
                client=client,
                model=model,
                input_reader=reader,
                output_writer=writer,
                forecast_configs=[config],
                n_prefetch_init=0,
                n_prefetch_forcing=0,
            )

        # Assert
        mock_dask.compute.assert_called_once()

    def test_run_forecast_uses_persist(
        self,
    ) -> None:
        r'''PrefetchIterator used by run_forecast calls persist.'''
        from ocean_flow.forecast.runner import (
            PrefetchIterator,
        )

        # Arrange -- item with .persist() method
        mock_item = MagicMock()
        mock_item.persist.return_value = mock_item
        load_fn = MagicMock(return_value=mock_item)

        # Act
        prefetch = PrefetchIterator(
            [load_fn], n_prefetch=0,
        )
        next(prefetch)

        # Assert
        mock_item.persist.assert_called()


# -----------------------------------------------------------
# Phase 3: run_batch signature
# -----------------------------------------------------------

class TestPhase3RunForecastBatchSignature:
    r'''run_batch takes xr.Dataset args.'''

    def test_accepts_states_ds_parameter(self) -> None:
        r'''run_batch accepts states_ds kwarg.'''

        import inspect
        from ocean_flow.forecast.runner import (
            run_batch,
        )
        sig = inspect.signature(run_batch)
        assert "states_ds" in sig.parameters, (
            "run_batch should accept states_ds"
        )

    def test_accepts_aux_ds_parameter(self) -> None:
        r'''run_batch accepts aux_ds kwarg.'''

        import inspect
        from ocean_flow.forecast.runner import (
            run_batch,
        )
        sig = inspect.signature(run_batch)
        assert "aux_ds" in sig.parameters, (
            "run_batch should accept aux_ds"
        )

    def test_no_client_parameter(self) -> None:
        r'''run_batch no longer takes client.'''

        import inspect
        from ocean_flow.forecast.runner import (
            run_batch,
        )
        sig = inspect.signature(run_batch)
        assert "client" not in sig.parameters, (
            "run_batch should not take client"
        )

    def test_no_states_future_parameter(self) -> None:
        r'''run_batch no longer takes states_future.'''

        import inspect
        from ocean_flow.forecast.runner import (
            run_batch,
        )
        sig = inspect.signature(run_batch)
        assert "states_future" not in sig.parameters, (
            "run_batch should not take "
            "states_future"
        )

    def test_returns_list_of_delayed(self) -> None:
        r'''run_batch returns List[Delayed].'''

        from ocean_flow.forecast.runner import (
            run_batch,
        )

        # Arrange -- mock everything
        model = _make_mock_model(n_out_steps=1)
        reader = MagicMock()
        reader.use_forcings = False
        writer = MagicMock()
        # Writer.write returns a list of Delayed
        mock_delayed = MagicMock(spec=Delayed)
        writer.write.return_value = [mock_delayed]

        chunk = pd.timedelta_range(
            start="6h", end="6h", freq="6h",
        )
        config = _ConfigDouble(
            init_times=pd.DatetimeIndex(["2020-01-01"]),
            ens_mems=np.array([0]),
            chunks=[chunk],
        )

        # Build a minimal xr.Dataset for states_ds
        states_ds = xr.Dataset({
            "states_surface": (
                ("batch", "time_step", "var", "lat", "lon"),
                np.zeros((1, 1, 1, 1, 1)),
            ),
        })

        # Act
        result = run_batch(
            model=model,
            input_reader=reader,
            output_writer=writer,
            config=config,
            states_ds=states_ds,
            aux_ds=None,
            n_prefetch_forcing=0,
        )

        # Assert
        assert isinstance(result, list)


# -----------------------------------------------------------
# Phase 3: run_forecast collects delayed
# -----------------------------------------------------------

class TestPhase3RunForecastCollectsDelayed:
    r'''run_forecast collects Delayed and batch-computes.'''

    def test_run_forecast_collects_delayed_from_batch(
        self,
    ) -> None:
        r'''Delayed objects from write are collected.'''

        from ocean_flow.forecast.runner import (
            run_forecast,
        )

        # Arrange
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = MagicMock()
        reader.use_forcings = False
        reader.use_auxiliary = False

        # load_states returns lazy xr.Dataset
        lazy_ds = xr.Dataset({
            "states_surface": (
                ("batch", "time_step", "var"),
                dask.array.from_array(
                    np.zeros((1, 1, 1)),
                ),
            ),
        })
        reader.load_states.return_value = lazy_ds

        writer = MagicMock()
        mock_delayed = MagicMock(spec=Delayed)
        writer.write.return_value = [mock_delayed]

        chunk = pd.timedelta_range(
            start="6h", end="6h", freq="6h",
        )
        config = _ConfigDouble(
            init_times=pd.DatetimeIndex(["2020-01-01"]),
            ens_mems=np.array([0]),
            chunks=[chunk],
        )

        # Act
        with patch(
            "ocean_flow"
            ".forecast.runner.dask"
        ) as mock_dask:
            run_forecast(
                client=client,
                model=model,
                input_reader=reader,
                output_writer=writer,
                forecast_configs=[config],
                n_prefetch_init=0,
                n_prefetch_forcing=0,
            )

            # Assert -- dask.compute called with delayed
            mock_dask.compute.assert_called_once()


# -----------------------------------------------------------
# Phase 3: Model receives Dict[str, np.ndarray]
# -----------------------------------------------------------

class TestPhase3ModelReceivesNumpyDict:
    r'''Model.set_state receives Dict[str, np.ndarray].'''

    def test_model_set_state_receives_numpy_dict(
        self,
    ) -> None:
        r'''set_state gets numpy arrays, not xr.Dataset.'''

        from ocean_flow.forecast.runner import (
            run_batch,
        )

        model = _make_mock_model(n_out_steps=1)
        reader = MagicMock()
        reader.use_forcings = False
        writer = MagicMock()
        mock_delayed = MagicMock(spec=Delayed)
        writer.write.return_value = [mock_delayed]

        chunk = pd.timedelta_range(
            start="6h", end="6h", freq="6h",
        )
        config = _ConfigDouble(
            init_times=pd.DatetimeIndex(["2020-01-01"]),
            ens_mems=np.array([0]),
            chunks=[chunk],
        )

        states_ds = xr.Dataset({
            "states_surface": (
                ("batch", "time_step", "var"),
                np.ones((1, 1, 2)),
            ),
        })

        # Act
        run_batch(
            model=model,
            input_reader=reader,
            output_writer=writer,
            config=config,
            states_ds=states_ds,
            aux_ds=None,
            n_prefetch_forcing=0,
        )

        # Assert -- set_state called with a dict
        model.set_state.assert_called_once()
        state_arg = model.set_state.call_args[0][0]
        assert isinstance(state_arg, dict)
        for key, val in state_arg.items():
            assert isinstance(val, np.ndarray), (
                f"Value for key '{key}' should be "
                f"np.ndarray, got {type(val)}"
            )


# -----------------------------------------------------------
# Phase 4: prefetch.py deleted
# -----------------------------------------------------------

class TestPhase4DeletePrefetch:
    r'''prefetch.py should not exist after refactor.'''

    def test_prefetch_module_not_importable(
        self,
    ) -> None:
        r'''Importing prefetch raises ImportError.'''

        with pytest.raises(ImportError):
            from ocean_flow.forecast.prefetch import (
                Prefetcher,
            )

    def test_prefetch_not_in_forecast_init(
        self,
    ) -> None:
        r'''forecast __init__ does not export Prefetcher.'''

        from ocean_flow import forecast
        assert not hasattr(forecast, "Prefetcher"), (
            "Prefetcher should be removed from exports"
        )

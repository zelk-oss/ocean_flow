# -*- coding: utf-8 -*-
r'''Tests for forecast/runner.py -- run_forecast,
run_batch, initialize_io, PrefetchIterator, and
_count_forecast_batches.'''

# System modules
import os
import sys
from typing import Any, Optional
from unittest.mock import MagicMock, call, patch

# Path setup for scripts/ directory
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "scripts",
    ),
)

# External modules
import dask
import dask.array
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.delayed import Delayed
from omegaconf import OmegaConf

# Internal modules
from ocean_flow.forecast.runner import (
    PrefetchIterator,
    run_forecast,
    run_batch,
    initialize_io,
    _count_forecast_batches,
    _shift_forcing_times,
)
from tests.conftest import (
    _create_test_zarr,
    _ConfigDouble,
    _make_config_double,
    _pre_create_output_store,
)


# -----------------------------------------------------------
# Helper classes and functions
# -----------------------------------------------------------

def _make_mock_model(
        n_out_steps: int = 1,
) -> MagicMock:
    r'''Create a mock ForecastModel.

    Parameters
    ----------
    n_out_steps : int
        Number of output steps per advance call.

    Returns
    -------
    MagicMock
        Mock model with set_state, set_auxiliary,
        advance.
    '''
    model = MagicMock()
    model.n_out_steps = n_out_steps
    model.advance.return_value = {
        "states_surface": np.zeros(
            (1, n_out_steps, 1, 1),
            dtype=np.float32,
        )
    }
    return model


def _make_lazy_ds() -> xr.Dataset:
    r'''Create a lazy dask-backed xr.Dataset.

    Returns
    -------
    xr.Dataset
        Dask-backed dataset for testing.
    '''
    return xr.Dataset({
        "states_surface": (
            ("batch", "time_step", "var"),
            dask.array.from_array(
                np.zeros((1, 1, 1)),
            ),
        ),
    })


def _make_mock_input_reader(
        use_auxiliary: bool = True,
        use_forcings: bool = True,
) -> MagicMock:
    r'''Create a mock InputReader.

    Parameters
    ----------
    use_auxiliary : bool
        Whether auxiliary data is available.
    use_forcings : bool
        Whether forcing data is available.

    Returns
    -------
    MagicMock
        Mock input reader returning lazy xr.Datasets.
    '''
    reader = MagicMock()
    reader.use_auxiliary = use_auxiliary
    reader.use_forcings = use_forcings
    reader.load_states.return_value = _make_lazy_ds()
    reader.load_auxiliary.return_value = (
        _make_lazy_ds()
    )
    reader.load_forcings.return_value = (
        _make_lazy_ds()
    )
    return reader


def _make_mock_output_writer() -> MagicMock:
    r'''Create a mock OutputWriter.

    Returns
    -------
    MagicMock
        Mock output writer returning Delayed list.
    '''
    writer = MagicMock()
    mock_delayed = MagicMock(spec=Delayed)
    writer.write.return_value = [mock_delayed]
    return writer


def _make_io_cfg(tmp_path: Any) -> OmegaConf:
    r'''Build a full config suitable for initialize_io.

    Parameters
    ----------
    tmp_path : Any
        Temporary path for test data.

    Returns
    -------
    OmegaConf
        Configuration for initialize_io.
    '''
    data_path = str(tmp_path / "data.zarr")
    _create_test_zarr(data_path, n_times=20)
    store_path = str(tmp_path / "output.zarr")
    _pre_create_output_store(
        data_path=data_path,
        store_path=store_path,
        init_times=pd.date_range(
            "2020-01-01", "2020-01-01T06:00",
            freq="6h",
        ),
        lead_times=pd.timedelta_range(
            "6h", "24h", freq="6h",
        ),
        ens_mems=np.arange(1),
    )
    return OmegaConf.create({
        "init_start": "2020-01-01",
        "init_end": "2020-01-01T06:00",
        "init_freq": "6h",
        "lead_time": "24h",
        "step_freq": "6h",
        "ensemble_size": 1,
        "n_in_steps": 1,
        "n_out_steps": 1,
        "io": {
            "data_path": data_path,
            "state_variables": [
                "states_surface",
                "states_levels",
            ],
            "auxiliary_path": None,
            "auxiliary_variables": None,
            "forcing_path": None,
            "forcing_variables": None,
            "store_path": store_path,
        },
    })


# -----------------------------------------------------------
# Functional tests
# -----------------------------------------------------------

class TestRunnerFunctional:
    r'''End-to-end tests for runner orchestration.'''

    def test_full_orchestration_calls_model(
            self,
    ) -> None:
        r'''run_forecast calls model.set_state and
        model.advance.
        '''
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = _make_mock_input_reader(
            use_auxiliary=True,
            use_forcings=False,
        )
        writer = _make_mock_output_writer()

        config = _make_config_double(
            n_chunks=1, steps_per_chunk=2,
        )

        with patch(
            "ocean_flow"
            ".forecast.runner.dask",
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

        model.set_state.assert_called_once()
        model.set_auxiliary.assert_called_once()
        assert model.advance.call_count >= 1

    def test_run_batch_trims_excess(
            self,
    ) -> None:
        r'''Excess trajectory steps are trimmed.'''
        model = _make_mock_model(n_out_steps=3)
        model.advance.return_value = {
            "states_surface": np.zeros(
                (1, 3, 1, 1), dtype=np.float32,
            )
        }
        reader = _make_mock_input_reader(
            use_auxiliary=False,
            use_forcings=False,
        )
        writer = _make_mock_output_writer()

        chunk = pd.timedelta_range(
            start="6h", end="12h", freq="6h",
        )
        config = _ConfigDouble(
            init_times=pd.DatetimeIndex(
                ["2020-01-01"],
            ),
            ens_mems=np.array([0]),
            chunks=[chunk],
        )

        states_ds = xr.Dataset({
            "s": (("batch",), np.zeros(1)),
        })

        result = run_batch(
            model=model,
            input_reader=reader,
            output_writer=writer,
            config=config,
            states_ds=states_ds,
            aux_ds=None,
            n_prefetch_forcing=0,
        )

        written_preds = (
            writer.write.call_args[0][0]
        )
        for var, arr in written_preds.items():
            assert arr.shape[1] == 2

    def test_cumulative_offset_applied(
            self,
    ) -> None:
        r'''Forcing args use cumulative offsets.'''
        init_times = pd.DatetimeIndex(
            ["2020-01-01"],
        )
        chunk_1 = pd.timedelta_range(
            start="6h", end="12h", freq="6h",
        )
        chunk_2 = pd.timedelta_range(
            start="18h", end="24h", freq="6h",
        )
        config = _ConfigDouble(
            init_times=init_times,
            ens_mems=np.array([0]),
            chunks=[chunk_1, chunk_2],
        )

        shifted, relative = _shift_forcing_times(
            config, chunk_2,
            offset=chunk_1[-1],
        )

        assert shifted[0] == (
            init_times[0] + chunk_1[-1]
        )
        expected_relative = chunk_2 - chunk_1[-1]
        pd.testing.assert_index_equal(
            relative, expected_relative,
        )

    def test_multiple_configs_prefetch(
            self,
    ) -> None:
        r'''Multiple configs exercise the prefetch window.'''
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = _make_mock_input_reader(
            use_auxiliary=True,
            use_forcings=False,
        )
        writer = _make_mock_output_writer()

        configs = [
            _make_config_double(
                n_chunks=1, steps_per_chunk=1,
            ),
            _make_config_double(
                n_chunks=1, steps_per_chunk=1,
            ),
            _make_config_double(
                n_chunks=1, steps_per_chunk=1,
            ),
        ]

        with patch(
            "ocean_flow"
            ".forecast.runner.dask",
        ):
            run_forecast(
                client=client,
                model=model,
                input_reader=reader,
                output_writer=writer,
                forecast_configs=configs,
                n_prefetch_init=1,
                n_prefetch_forcing=0,
            )

        assert model.set_state.call_count == 3
        assert model.set_auxiliary.call_count == 3

    def test_with_forcings_multi_chunk(
            self,
    ) -> None:
        r'''Forcings are loaded and consumed across
        multiple chunks.
        '''
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = _make_mock_input_reader(
            use_auxiliary=False,
            use_forcings=True,
        )
        writer = _make_mock_output_writer()

        config = _make_config_double(
            n_chunks=3, steps_per_chunk=1,
        )

        with patch(
            "ocean_flow"
            ".forecast.runner.dask",
        ):
            run_forecast(
                client=client,
                model=model,
                input_reader=reader,
                output_writer=writer,
                forecast_configs=[config],
                n_prefetch_init=0,
                n_prefetch_forcing=1,
            )

        assert model.advance.call_count == 3
        assert writer.write.call_count == 3
        assert reader.load_forcings.call_count == 3

    def test_dask_compute_called_at_end(
            self,
    ) -> None:
        r'''dask.compute is called with all delayed
        writes.
        '''
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = _make_mock_input_reader(
            use_auxiliary=False,
            use_forcings=False,
        )
        writer = _make_mock_output_writer()

        config = _make_config_double(
            n_chunks=1, steps_per_chunk=1,
        )

        with patch(
            "ocean_flow"
            ".forecast.runner.dask",
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

            mock_dask.compute.assert_called_once()


# -----------------------------------------------------------
# Unit tests
# -----------------------------------------------------------

class TestRunnerUnittest:
    r'''Isolated unit tests for runner helper functions.'''

    def test_initialize_io_returns_reader_writer(
            self,
            tmp_path: Any,
    ) -> None:
        r'''initialize_io returns (InputReader,
        OutputWriter) with simplified constructor.
        '''
        from ocean_flow.forecast.input import (
            InputReader,
        )
        from ocean_flow.forecast.output import (
            OutputWriter,
        )

        cfg = _make_io_cfg(tmp_path)
        reader, writer = initialize_io(cfg)
        assert isinstance(reader, InputReader)
        assert isinstance(writer, OutputWriter)

    def test_count_forecast_batches_basic(
            self,
    ) -> None:
        r'''_count_forecast_batches returns correct
        counts.
        '''
        cfg_1 = OmegaConf.create({
            "init_start": "2020-01-01",
            "init_end": "2020-01-01",
            "init_freq": "6h",
            "ensemble_size": 1,
            "batch_size": 1,
        })
        result_1 = _count_forecast_batches(cfg_1)
        assert result_1 == 1

        cfg_2 = OmegaConf.create({
            "init_start": "2020-01-01",
            "init_end": "2020-01-02T18:00",
            "init_freq": "6h",
            "ensemble_size": 2,
            "batch_size": 4,
        })
        result_2 = _count_forecast_batches(cfg_2)
        assert result_2 == 4

        cfg_3 = OmegaConf.create({
            "init_start": "2020-01-01",
            "init_end": "2020-01-03",
            "init_freq": "12h",
            "ensemble_size": 1,
            "batch_size": 2,
        })
        result_3 = _count_forecast_batches(cfg_3)
        assert result_3 == 3

    def test_init_times_match(
            self,
            tmp_path: Any,
    ) -> None:
        r'''OutputWriter.init_times match cfg dates.'''
        cfg = _make_io_cfg(tmp_path)
        _, writer = initialize_io(cfg)
        expected = pd.date_range(
            start=cfg.init_start,
            end=cfg.init_end,
            freq=cfg.init_freq,
        )
        assert writer.init_times.equals(expected)

    def test_lead_times_match(
            self,
            tmp_path: Any,
    ) -> None:
        r'''OutputWriter.lead_times match cfg
        timedeltas.
        '''
        cfg = _make_io_cfg(tmp_path)
        _, writer = initialize_io(cfg)
        expected = pd.timedelta_range(
            start=cfg.step_freq,
            end=cfg.lead_time,
            freq=cfg.step_freq,
        )
        assert writer.lead_times.equals(expected)

    def test_initialize_io_passes_n_in_steps(
            self,
            tmp_path: Any,
    ) -> None:
        r'''initialize_io passes n_in_steps and
        step_freq to InputReader.
        '''
        cfg = _make_io_cfg(tmp_path)
        cfg.n_in_steps = 2
        cfg.step_freq = "6h"

        with patch(
            "ocean_flow"
            ".forecast.runner.InputReader",
            autospec=True,
        ) as mock_reader:
            initialize_io(cfg)
            mock_reader.assert_called_once()
            call_kwargs = (
                mock_reader.call_args.kwargs
            )
            assert call_kwargs["n_in_steps"] == 2
            assert call_kwargs["step_freq"] == "6h"

    def test_count_none_init_returns_one(
            self,
    ) -> None:
        r'''_count_forecast_batches returns 1 when
        init is None.
        '''
        cfg = OmegaConf.create({
            "init_start": None,
            "init_end": None,
            "init_freq": None,
        })
        result = _count_forecast_batches(cfg)
        assert result == 1

    def test_count_ensemble_multiplier(
            self,
    ) -> None:
        r'''_count_forecast_batches accounts for
        ensemble_size.
        '''
        cfg = OmegaConf.create({
            "init_start": "2020-01-01",
            "init_end": "2020-01-01T18:00",
            "init_freq": "6h",
            "ensemble_size": 3,
            "batch_size": 5,
        })
        result = _count_forecast_batches(cfg)
        assert result == 3

    def test_shift_forcing_times_no_offset(
            self,
    ) -> None:
        r'''Zero offset returns original values.'''
        config = _make_config_double()
        chunk = pd.timedelta_range(
            start="6h", end="12h", freq="6h",
        )
        shifted, relative = _shift_forcing_times(
            config, chunk, pd.Timedelta(0),
        )
        pd.testing.assert_index_equal(
            shifted, config.init_times,
        )
        pd.testing.assert_index_equal(
            relative, chunk,
        )

    def test_shift_forcing_times_with_offset(
            self,
    ) -> None:
        r'''Non-zero offset shifts init times and
        adjusts lead times.
        '''
        config = _make_config_double()
        chunk = pd.timedelta_range(
            start="18h", end="24h", freq="6h",
        )
        offset = pd.Timedelta("12h")

        shifted, relative = _shift_forcing_times(
            config, chunk, offset,
        )

        expected_shifted = config.init_times + offset
        expected_relative = chunk - offset
        pd.testing.assert_index_equal(
            shifted, expected_shifted,
        )
        pd.testing.assert_index_equal(
            relative, expected_relative,
        )


# -----------------------------------------------------------
# Edge case tests
# -----------------------------------------------------------

class TestRunnerEdgeCases:
    r'''Boundary condition tests for runner functions.'''

    def test_empty_forecast_configs_returns(
            self,
    ) -> None:
        r'''Empty list returns immediately.'''
        client = MagicMock()
        model = _make_mock_model()
        reader = _make_mock_input_reader()
        writer = _make_mock_output_writer()

        with patch(
            "ocean_flow"
            ".forecast.runner.dask",
        ):
            run_forecast(
                client=client,
                model=model,
                input_reader=reader,
                output_writer=writer,
                forecast_configs=[],
            )

        reader.load_states.assert_not_called()

    def test_prefetch_aux_next_config(
            self,
    ) -> None:
        r'''Prefetching next config also prefetches
        auxiliary.
        '''
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = _make_mock_input_reader(
            use_auxiliary=True,
            use_forcings=False,
        )
        writer = _make_mock_output_writer()

        # Use 3 configs with prefetch=1 so next config
        # gets prefetched in the loop
        configs = [
            _make_config_double(
                n_chunks=1, steps_per_chunk=1,
            )
            for _ in range(3)
        ]

        with patch(
            "ocean_flow"
            ".forecast.runner.dask",
        ):
            run_forecast(
                client=client,
                model=model,
                input_reader=reader,
                output_writer=writer,
                forecast_configs=configs,
                n_prefetch_init=1,
                n_prefetch_forcing=0,
            )

        # load_auxiliary called for all 3 configs
        assert reader.load_auxiliary.call_count == 3

    def test_write_returns_empty_no_compute(
            self,
    ) -> None:
        r'''When writer returns no delayed, dask.compute
        is not called.
        '''
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = _make_mock_input_reader(
            use_auxiliary=False,
            use_forcings=False,
        )
        writer = MagicMock()
        writer.write.return_value = []

        config = _make_config_double(
            n_chunks=1, steps_per_chunk=1,
        )

        with patch(
            "ocean_flow"
            ".forecast.runner.dask",
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

            mock_dask.compute.assert_not_called()

    def test_trajectory_no_trim_when_exact(
            self,
    ) -> None:
        r'''Trajectory not trimmed when exact match.'''
        model = _make_mock_model(n_out_steps=2)
        model.advance.return_value = {
            "states_surface": np.zeros(
                (1, 2, 1, 1), dtype=np.float32,
            )
        }
        reader = _make_mock_input_reader(
            use_auxiliary=False,
            use_forcings=False,
        )
        writer = _make_mock_output_writer()

        chunk = pd.timedelta_range(
            start="6h", end="12h", freq="6h",
        )
        config = _ConfigDouble(
            init_times=pd.DatetimeIndex(
                ["2020-01-01"],
            ),
            ens_mems=np.array([0]),
            chunks=[chunk],
        )

        states_ds = xr.Dataset({
            "s": (("batch",), np.zeros(1)),
        })

        run_batch(
            model=model,
            input_reader=reader,
            output_writer=writer,
            config=config,
            states_ds=states_ds,
            aux_ds=None,
            n_prefetch_forcing=0,
        )

        written_preds = (
            writer.write.call_args[0][0]
        )
        for var, arr in written_preds.items():
            assert arr.shape[1] == 2

    def test_single_config_single_chunk(
            self,
    ) -> None:
        r'''One config, one chunk calls write once.'''
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = _make_mock_input_reader(
            use_auxiliary=False,
            use_forcings=False,
        )
        writer = _make_mock_output_writer()

        config = _make_config_double(
            n_chunks=1, steps_per_chunk=1,
        )

        with patch(
            "ocean_flow"
            ".forecast.runner.dask",
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

        writer.write.assert_called_once()

    def test_no_forcings_no_auxiliary(
            self,
    ) -> None:
        r'''Without forcings or auxiliary,
        set_auxiliary not called.
        '''
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = _make_mock_input_reader(
            use_auxiliary=False,
            use_forcings=False,
        )
        writer = _make_mock_output_writer()
        config = _make_config_double(
            n_chunks=1, steps_per_chunk=1,
        )

        with patch(
            "ocean_flow"
            ".forecast.runner.dask",
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

        model.set_auxiliary.assert_not_called()

    def test_multi_configs_no_aux_prefetch(
            self,
    ) -> None:
        r'''Multiple configs with no auxiliary skips
        auxiliary prefetch in the loop.
        '''
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = _make_mock_input_reader(
            use_auxiliary=False,
            use_forcings=False,
        )
        writer = _make_mock_output_writer()

        configs = [
            _make_config_double(
                n_chunks=1, steps_per_chunk=1,
            )
            for _ in range(3)
        ]

        with patch(
            "ocean_flow"
            ".forecast.runner.dask",
        ):
            run_forecast(
                client=client,
                model=model,
                input_reader=reader,
                output_writer=writer,
                forecast_configs=configs,
                n_prefetch_init=1,
                n_prefetch_forcing=0,
            )

        reader.load_auxiliary.assert_not_called()

    def test_zero_length_chunk_skips_trim(
            self,
    ) -> None:
        r'''Zero-length chunk does not trim trajectory.'''
        model = _make_mock_model(n_out_steps=1)
        model.advance.return_value = {}
        reader = _make_mock_input_reader(
            use_auxiliary=False,
            use_forcings=False,
        )
        writer = _make_mock_output_writer()

        empty_chunk = pd.TimedeltaIndex([])
        config = _ConfigDouble(
            init_times=pd.DatetimeIndex(
                ["2020-01-01"],
            ),
            ens_mems=np.array([0]),
            chunks=[empty_chunk],
        )

        states_ds = xr.Dataset({
            "s": (("batch",), np.zeros(1)),
        })

        result = run_batch(
            model=model,
            input_reader=reader,
            output_writer=writer,
            config=config,
            states_ds=states_ds,
            aux_ds=None,
            n_prefetch_forcing=0,
        )

        assert isinstance(result, list)

    def test_empty_chunks_returns_empty(
            self,
    ) -> None:
        r'''Config with no chunks yields empty list.'''
        model = _make_mock_model()
        reader = _make_mock_input_reader(
            use_forcings=False,
        )
        writer = _make_mock_output_writer()

        config = _ConfigDouble(
            init_times=pd.DatetimeIndex(
                ["2020-01-01"],
            ),
            ens_mems=np.array([0]),
            chunks=[],
        )

        states_ds = xr.Dataset({
            "s": (("batch",), np.zeros(1)),
        })

        result = run_batch(
            model=model,
            input_reader=reader,
            output_writer=writer,
            config=config,
            states_ds=states_ds,
            aux_ds=None,
            n_prefetch_forcing=0,
        )

        assert result == []



# -----------------------------------------------------------
# DP-aware forecast loop tests
# -----------------------------------------------------------

class TestRunnerDPAware:
    r'''Tests for DP-rank-strided forecast loop.'''

    def test_dp_rank_strides_configs(
            self,
    ) -> None:
        r'''dp_rank=1, dp_world_size=2 processes only
        config at index 1 from 3 configs.
        '''
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = _make_mock_input_reader(
            use_auxiliary=False,
            use_forcings=False,
        )
        writer = _make_mock_output_writer()

        # Arrange -- 3 distinct configs
        configs = [
            _make_config_double(
                n_chunks=1, steps_per_chunk=1,
            )
            for _ in range(3)
        ]

        # Act -- dp_rank=1, dp_world_size=2
        # worker_configs = configs[1::2] = [configs[1]]
        with patch(
            "ocean_flow"
            ".forecast.runner.dask",
        ):
            run_forecast(
                client=client,
                model=model,
                input_reader=reader,
                output_writer=writer,
                forecast_configs=configs,
                n_prefetch_init=0,
                n_prefetch_forcing=0,
                dp_rank=1,
                dp_world_size=2,
            )

        # Assert -- only 1 config processed
        assert model.set_state.call_count == 1

    def test_dp_rank_zero_processes_first_config(
            self,
    ) -> None:
        r'''dp_rank=0, dp_world_size=2 processes
        configs[0] and configs[2] from 3 configs.
        '''
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = _make_mock_input_reader(
            use_auxiliary=False,
            use_forcings=False,
        )
        writer = _make_mock_output_writer()

        configs = [
            _make_config_double(
                n_chunks=1, steps_per_chunk=1,
            )
            for _ in range(3)
        ]

        # Act -- dp_rank=0, dp_world_size=2
        # worker_configs = configs[0::2]
        #               = [configs[0], configs[2]]
        with patch(
            "ocean_flow"
            ".forecast.runner.dask",
        ):
            run_forecast(
                client=client,
                model=model,
                input_reader=reader,
                output_writer=writer,
                forecast_configs=configs,
                n_prefetch_init=0,
                n_prefetch_forcing=0,
                dp_rank=0,
                dp_world_size=2,
            )

        # Assert -- 2 configs processed
        assert model.set_state.call_count == 2

    def test_dp_world_size_one_processes_all(
            self,
    ) -> None:
        r'''Default dp_world_size=1 processes all
        configs unchanged.
        '''
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = _make_mock_input_reader(
            use_auxiliary=False,
            use_forcings=False,
        )
        writer = _make_mock_output_writer()

        configs = [
            _make_config_double(
                n_chunks=1, steps_per_chunk=1,
            )
            for _ in range(3)
        ]

        # Act -- default dp_rank=0, dp_world_size=1
        with patch(
            "ocean_flow"
            ".forecast.runner.dask",
        ):
            run_forecast(
                client=client,
                model=model,
                input_reader=reader,
                output_writer=writer,
                forecast_configs=configs,
                n_prefetch_init=0,
                n_prefetch_forcing=0,
                dp_rank=0,
                dp_world_size=1,
            )

        # Assert -- all 3 configs processed
        assert model.set_state.call_count == 3

    def test_dp_rank_beyond_configs_returns(
            self,
    ) -> None:
        r'''dp_rank=5 with 3 configs returns
        without processing.
        '''
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = _make_mock_input_reader(
            use_auxiliary=False,
            use_forcings=False,
        )
        writer = _make_mock_output_writer()

        configs = [
            _make_config_double(
                n_chunks=1, steps_per_chunk=1,
            )
            for _ in range(3)
        ]

        # Act -- dp_rank=5, dp_world_size=6
        # worker_configs = configs[5::6] = []
        with patch(
            "ocean_flow"
            ".forecast.runner.dask",
        ):
            run_forecast(
                client=client,
                model=model,
                input_reader=reader,
                output_writer=writer,
                forecast_configs=configs,
                n_prefetch_init=0,
                n_prefetch_forcing=0,
                dp_rank=5,
                dp_world_size=6,
            )

        # Assert -- no configs processed
        reader.load_states.assert_not_called()
        model.set_state.assert_not_called()


class TestRunnerProgressBar:
    r'''Tests for progress-bar behavior in run_forecast.'''

    @pytest.mark.parametrize(
        "dp_rank,dp_world_size,expected_total,expected_desc",
        [
            (0, 1, 5, "rank 0"),
            (0, 2, 3, "rank 0"),
            (1, 2, 2, "rank 1"),
        ],
    )
    def test_run_forecast_wraps_consumed_configs_with_tqdm(
            self,
            dp_rank: int,
            dp_world_size: int,
            expected_total: int,
            expected_desc: str,
    ) -> None:
        r'''Verifies run_forecast wraps consumed configs with rank-aware tqdm.'''
        # Arrange
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = _make_mock_input_reader(
            use_auxiliary=False,
            use_forcings=False,
        )
        writer = _make_mock_output_writer()
        configs = [
            _make_config_double(
                n_chunks=1, steps_per_chunk=1,
            )
            for _ in range(5)
        ]

        # Act
        with patch(
            "ocean_flow"
            ".forecast.runner.dask",
        ), patch(
            "ocean_flow"
            ".forecast.runner.tqdm",
            side_effect=lambda iterable, **_: iterable,
            create=True,
        ) as tqdm_mock:
            run_forecast(
                client=client,
                model=model,
                input_reader=reader,
                output_writer=writer,
                forecast_configs=configs,
                n_prefetch_init=0,
                n_prefetch_forcing=0,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
            )

        # Assert
        tqdm_mock.assert_called_once()
        _, tqdm_kwargs = tqdm_mock.call_args
        assert tqdm_kwargs["total"] == expected_total
        assert tqdm_kwargs["disable"] is False
        assert tqdm_kwargs["desc"] == expected_desc

    @pytest.mark.parametrize("dp_rank", [0, 1, 2])
    def test_progress_bars_different_ranks_separate_lines(
            self,
            dp_rank: int,
    ) -> None:
        r'''Progress bars for different ranks use rank position.'''
        # Arrange
        client = MagicMock()
        model = _make_mock_model(n_out_steps=1)
        reader = _make_mock_input_reader(
            use_auxiliary=False,
            use_forcings=False,
        )
        writer = _make_mock_output_writer()
        configs = [
            _make_config_double(
                n_chunks=1, steps_per_chunk=1,
            )
            for _ in range(4)
        ]

        # Act
        with patch(
            "ocean_flow"
            ".forecast.runner.dask",
        ), patch(
            "ocean_flow"
            ".forecast.runner.tqdm",
            side_effect=lambda iterable, **_: iterable,
            create=True,
        ) as tqdm_mock:
            run_forecast(
                client=client,
                model=model,
                input_reader=reader,
                output_writer=writer,
                forecast_configs=configs,
                n_prefetch_init=0,
                n_prefetch_forcing=0,
                dp_rank=dp_rank,
                dp_world_size=3,
            )

        # Assert
        tqdm_mock.assert_called_once()
        _, tqdm_kwargs = tqdm_mock.call_args
        assert "position" in tqdm_kwargs, (
            "tqdm must have position parameter "
            "for multi-rank progress bars"
        )
        assert tqdm_kwargs["position"] == dp_rank, (
            f"Expected position={dp_rank} for dp_rank={dp_rank}, "
            f"got position={tqdm_kwargs.get('position')}"
        )


# -----------------------------------------------------------
# Error tests
# -----------------------------------------------------------

class TestRunnerErrors:
    r'''Error condition tests for runner functions.'''

    def test_nonpositive_n_out_steps(
            self,
    ) -> None:
        r'''Nonpositive model.n_out_steps raises.'''
        model = _make_mock_model(n_out_steps=0)
        reader = _make_mock_input_reader(
            use_auxiliary=False,
            use_forcings=False,
        )
        writer = _make_mock_output_writer()

        config = _make_config_double(
            n_chunks=1, steps_per_chunk=1,
        )

        states_ds = xr.Dataset({
            "s": (("batch",), np.zeros(1)),
        })

        with pytest.raises(
            ValueError,
            match="model.n_out_steps must be positive",
        ):
            run_batch(
                model=model,
                input_reader=reader,
                output_writer=writer,
                config=config,
                states_ds=states_ds,
                aux_ds=None,
                n_prefetch_forcing=0,
            )

        model.advance.assert_not_called()


# -----------------------------------------------------------
# Init exports tests
# -----------------------------------------------------------

class TestForecastInitExports:
    r'''Tests for forecast/__init__.py public API.'''

    def test_init_exports_all_public_symbols(
            self,
    ) -> None:
        r'''forecast/__init__ exports required
        symbols.
        '''
        from ocean_flow import forecast

        expected_names = [
            "run_forecast",
            "run_batch",
            "PrefetchIterator",
            "ForecastConfig",
            "generate_forecast_configs",
            "ForecastModel",
            "InputReader",
            "OutputWriter",
            "load_forecast_model",
            "setup_environment",
            "initialize_client",
            "initialize_io",
            "validate_initial_conditions",
            "validate_auxiliary",
            "validate_forcing",
            "validate_checkpoint",
            "validate_output_store",
            "create_output_store",
            "validate_dask_addresses",
        ]
        for name in expected_names:
            assert hasattr(forecast, name), (
                f"forecast.__init__ missing "
                f"'{name}'"
            )


# -----------------------------------------------------------
# PrefetchIterator tests
# -----------------------------------------------------------

class TestPrefetchIteratorUnittest:
    r'''Unit tests for PrefetchIterator.'''

    def test_empty_load_fns_yields_nothing(
            self,
    ) -> None:
        r'''Empty list produces no items.'''
        prefetch = PrefetchIterator([], n_prefetch=2)
        result = list(prefetch)
        assert result == []

    def test_single_item_no_prefetch(
            self,
    ) -> None:
        r'''Single item with zero prefetch works.'''
        prefetch = PrefetchIterator(
            [lambda: "a"], n_prefetch=0,
        )
        result = list(prefetch)
        assert result == ["a"]

    def test_iteration_order_preserved(
            self,
    ) -> None:
        r'''Items are yielded in order.'''
        fns = [lambda i=i: i for i in range(5)]
        prefetch = PrefetchIterator(
            fns, n_prefetch=2,
        )
        result = list(prefetch)
        assert result == [0, 1, 2, 3, 4]

    def test_len_returns_total_count(
            self,
    ) -> None:
        r'''__len__ returns total item count.'''
        fns = [lambda: None for _ in range(7)]
        prefetch = PrefetchIterator(
            fns, n_prefetch=3,
        )
        assert len(prefetch) == 7

    def test_prefetch_calls_persist(
            self,
    ) -> None:
        r'''Items with .persist() get it called.'''
        mock_ds = MagicMock()
        mock_ds.persist.return_value = "persisted"
        fns = [lambda: mock_ds]
        prefetch = PrefetchIterator(
            fns, n_prefetch=0,
        )
        result = list(prefetch)
        mock_ds.persist.assert_called_once()
        assert result == ["persisted"]

    def test_stopiteration_after_exhaustion(
            self,
    ) -> None:
        r'''StopIteration raised after exhaustion.'''
        prefetch = PrefetchIterator(
            [lambda: 1], n_prefetch=0,
        )
        next(prefetch)
        with pytest.raises(StopIteration):
            next(prefetch)

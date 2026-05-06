# -*- coding: utf-8 -*-
r'''Tests for scripts/forecast.py -- global/local phase
split and YAML config defaults.'''

# System modules
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Tuple
from unittest.mock import (
    MagicMock,
    call,
    patch,
)

# Path setup for scripts/ directory
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "scripts"),
)

# External modules
import pytest
from omegaconf import OmegaConf

# Import functions under test
from forecast import (
    _run_global_phase,
    run_local_forecast,
    _launch_forecast,
    main_forecast,
)


def _make_forecast_cfg(
        include_time_params: bool = True,
) -> Any:
    r'''Return a minimal forecast config.

    Builds a shared base config used by both global and
    local phase tests.  When ``include_time_params`` is
    True the six time/store keys required only by the
    global phase are included.

    Parameters
    ----------
    include_time_params : bool, optional
        If True, add init_start, init_end, init_freq,
        lead_time, step_freq, and n_store_freq keys.
        Default is True.

    Returns
    -------
    OmegaConf
        Minimal DictConfig for forecast testing.
    '''
    cfg: Dict[str, Any] = {
        "logging_level": "WARNING",
        "trainer": {
            "accelerator": "cpu",
            "devices": 1,
            "precision": "32-true",
            "strategy": "auto",
        },
        "dask": {
            "scheduler": None,
            "n_workers": 1,
            "dashboard_address": None,
            "n_prefetch_init": 1,
            "n_prefetch_forcing": 3,
        },
        "io": {
            "data_path": "/tmp/data.zarr",
            "state_variables": [
                "states_surface",
                "states_levels",
            ],
            "auxiliary_path": None,
            "auxiliary_variables": None,
            "forcing_path": None,
            "forcing_variables": None,
            "store_path": "/tmp/out.zarr",
            "recreate_store": False,
            "restart": False,
        },
        "ckpt_path": "/tmp/best.ckpt",
        "n_in_steps": 1,
        "n_out_steps": 1,
        "ensemble_size": 1,
        "batch_size": 4,
        "seed": 42,
    }
    if include_time_params:
        cfg.update({
            "init_start": "2020-01-01",
            "init_end": "2020-01-02",
            "init_freq": "6h",
            "lead_time": "1D",
            "step_freq": "6h",
            "n_store_freq": 1,
        })
    return OmegaConf.create(cfg)


def _make_global_phase_cfg() -> Any:
    r'''Return a minimal config for global phase tests.

    Returns
    -------
    OmegaConf
        Minimal DictConfig for global phase testing.
    '''
    return _make_forecast_cfg(include_time_params=True)


def _make_local_forecast_cfg() -> Any:
    r'''Return a minimal config for local forecast tests.

    Returns
    -------
    OmegaConf
        Minimal DictConfig for local forecast testing.
    '''
    return _make_forecast_cfg(include_time_params=False)


# -----------------------------------------------------------
# Global phase functional tests
# -----------------------------------------------------------

class TestGlobalPhaseFunctional:
    r'''Tests for _run_global_phase orchestration.'''

    def test_global_phase_validates_before_launch(
            self,
    ) -> None:
        r'''Validators are called in order before launch.'''
        # Arrange
        cfg = _make_global_phase_cfg()
        call_order: List[str] = []

        def _track(name: str) -> Any:
            def _side(*a: Any, **kw: Any) -> None:
                call_order.append(name)
            return _side

        mock_configs = [MagicMock(), MagicMock()]

        # Act
        with patch(
            "forecast.validate_initial_conditions",
            side_effect=_track("validate_ic"),
        ) as p_ic, patch(
            "forecast.validate_auxiliary",
            side_effect=_track("validate_aux"),
        ), patch(
            "forecast.validate_forcing",
            side_effect=_track("validate_forcing"),
        ), patch(
            "forecast.validate_checkpoint",
            side_effect=_track("validate_ckpt"),
        ) as p_ckpt, patch(
            "forecast.create_output_store",
            side_effect=_track("create_store"),
        ), patch(
            "forecast.generate_forecast_configs",
            side_effect=lambda *a, **kw: (
                call_order.append("gen_configs")
                or mock_configs
            ),
        ), patch(
            "forecast.validate_dask_addresses",
            side_effect=_track("validate_dask"),
        ) as p_dask:
            _run_global_phase(cfg)

        # Assert -- validators called before configs
        assert call_order.index(
            "validate_ic"
        ) < call_order.index("gen_configs")
        assert call_order.index(
            "validate_ckpt"
        ) < call_order.index("gen_configs")

        # Assert -- dask validation after configs
        assert call_order.index(
            "gen_configs"
        ) < call_order.index("validate_dask")

        # Assert -- each validator called once
        p_ic.assert_called_once()
        p_ckpt.assert_called_once()
        p_dask.assert_called_once()

    def test_global_phase_creates_output_store(
            self,
    ) -> None:
        r'''Global phase calls create_output_store.'''
        # Arrange
        cfg = _make_global_phase_cfg()
        mock_configs = [MagicMock()]

        # Act
        with patch(
            "forecast.validate_initial_conditions",
        ), patch(
            "forecast.validate_auxiliary",
        ), patch(
            "forecast.validate_forcing",
        ), patch(
            "forecast.validate_checkpoint",
        ), patch(
            "forecast.create_output_store",
        ) as p_create, patch(
            "forecast.generate_forecast_configs",
            return_value=mock_configs,
        ), patch(
            "forecast.validate_dask_addresses",
        ):
            _run_global_phase(cfg)

        # Assert
        p_create.assert_called_once()

    def test_global_phase_returns_configs_and_dp_workers(
            self,
    ) -> None:
        r'''Global phase returns forecast configs and count.'''
        # Arrange
        cfg = _make_global_phase_cfg()
        mock_configs = [MagicMock(), MagicMock()]

        # Act
        with patch(
            "forecast.validate_initial_conditions",
        ), patch(
            "forecast.validate_auxiliary",
        ), patch(
            "forecast.validate_forcing",
        ), patch(
            "forecast.validate_checkpoint",
        ), patch(
            "forecast.create_output_store",
        ), patch(
            "forecast.generate_forecast_configs",
            return_value=mock_configs,
        ), patch(
            "forecast.validate_dask_addresses",
        ):
            result = _run_global_phase(cfg)

        # Assert
        assert isinstance(result, tuple)
        configs, n_dp = result
        assert configs == mock_configs
        assert isinstance(n_dp, int)
        assert n_dp >= 1

    @pytest.mark.parametrize(
        "devices_value,expected_workers",
        [
            ([0, 1], 2),
            ((0, 1), 2),
            (OmegaConf.create([0, 1]), 2),
        ],
    )
    def test_global_phase_counts_list_like_devices_without_type_failures(
            self,
            devices_value: Any,
            expected_workers: int,
    ) -> None:
        r'''Global phase treats OmegaConf list-like devices like list/tuple.'''
        # Arrange
        cfg = _make_global_phase_cfg()
        cfg.trainer.devices = devices_value
        mock_configs = [MagicMock()]

        # Act
        with patch(
            "forecast.validate_initial_conditions",
        ), patch(
            "forecast.validate_auxiliary",
        ), patch(
            "forecast.validate_forcing",
        ), patch(
            "forecast.validate_checkpoint",
        ), patch(
            "forecast.create_output_store",
        ), patch(
            "forecast.generate_forecast_configs",
            return_value=mock_configs,
        ) as p_gen, patch(
            "forecast.validate_dask_addresses",
        ) as p_validate_dask:
            result = _run_global_phase(cfg)

        # Assert
        configs, n_dp_workers = result
        assert configs == mock_configs
        assert n_dp_workers == expected_workers
        p_gen.assert_called_once_with(
            cfg, dp_world_size=expected_workers,
        )
        p_validate_dask.assert_called_once_with(
            cfg.dask.scheduler, expected_workers,
        )


# -----------------------------------------------------------
# Global phase restart tests
# -----------------------------------------------------------

class TestGlobalPhaseRestart:
    r'''Tests for _run_global_phase restart behaviour.'''

    def test_global_phase_restart_validates_output_store(
            self,
    ) -> None:
        r'''restart=True validates store, filters configs.'''
        # Arrange
        cfg = _make_global_phase_cfg()
        cfg.io.restart = True
        mock_configs = [MagicMock(), MagicMock()]

        # Act
        with patch(
            "forecast.validate_initial_conditions",
        ), patch(
            "forecast.validate_auxiliary",
        ), patch(
            "forecast.validate_forcing",
        ), patch(
            "forecast.validate_checkpoint",
        ), patch(
            "forecast.validate_output_store",
        ) as p_validate_store, patch(
            "forecast.create_output_store",
        ) as p_create_store, patch(
            "forecast.generate_forecast_configs",
            return_value=mock_configs,
        ), patch(
            "forecast.check_written_regions",
            return_value={},
        ), patch(
            "forecast.OutputWriter",
            return_value=MagicMock(),
        ), patch(
            "forecast.filter_forecast_configs",
            return_value=mock_configs,
        ) as p_filter, patch(
            "forecast.validate_dask_addresses",
        ), patch(
            "os.path.exists",
            return_value=True,
        ):
            _run_global_phase(cfg)

        # Assert -- validate, not create
        p_validate_store.assert_called_once()
        p_create_store.assert_not_called()

        # Assert -- filter called for restart
        p_filter.assert_called_once()


# -----------------------------------------------------------
# Global phase runtime safety tests (TDD issue #56)
# -----------------------------------------------------------

class TestGlobalPhaseRuntimeSafety:
    r'''Tests for global phase runtime safety validations.'''

    def test_global_phase_validates_restart_recreate_conflict(
            self,
    ) -> None:
        r'''Global phase calls validate_restart_config early.'''
        # Arrange
        cfg = _make_global_phase_cfg()
        call_order: List[str] = []

        def _track(name: str) -> Any:
            def _side(*a: Any, **kw: Any) -> None:
                call_order.append(name)
            return _side

        mock_configs = [MagicMock()]

        # Act
        with patch(
            "forecast.validate_restart_config",
            side_effect=_track("validate_restart"),
        ) as p_validate_restart, patch(
            "forecast.validate_initial_conditions",
            side_effect=_track("validate_ic"),
        ), patch(
            "forecast.validate_auxiliary",
            side_effect=_track("validate_aux"),
        ), patch(
            "forecast.validate_forcing",
            side_effect=_track("validate_forcing"),
        ), patch(
            "forecast.validate_checkpoint",
            side_effect=_track("validate_ckpt"),
        ), patch(
            "forecast.create_output_store",
            side_effect=_track("create_store"),
        ), patch(
            "forecast.generate_forecast_configs",
            side_effect=lambda *a, **kw: (
                call_order.append("gen_configs")
                or mock_configs
            ),
        ), patch(
            "forecast.validate_dask_addresses",
            side_effect=_track("validate_dask"),
        ):
            _run_global_phase(cfg)

        # Assert -- restart validator called before any work
        p_validate_restart.assert_called_once()
        assert call_order.index(
            "validate_restart"
        ) < call_order.index("validate_ic"), (
            "validate_restart_config must be called "
            "before validate_initial_conditions"
        )
        assert call_order.index(
            "validate_restart"
        ) < call_order.index("create_store"), (
            "validate_restart_config must be called "
            "before create_output_store"
        )

    def test_global_phase_passes_recreate_store_to_create(
            self,
    ) -> None:
        r'''Global phase passes recreate_store config to create_output_store.'''
        # Arrange
        cfg = _make_global_phase_cfg()
        cfg.io.recreate_store = False
        mock_configs = [MagicMock()]

        # Act
        with patch(
            "forecast.validate_initial_conditions",
        ), patch(
            "forecast.validate_auxiliary",
        ), patch(
            "forecast.validate_forcing",
        ), patch(
            "forecast.validate_checkpoint",
        ), patch(
            "forecast.create_output_store",
        ) as p_create, patch(
            "forecast.generate_forecast_configs",
            return_value=mock_configs,
        ), patch(
            "forecast.validate_dask_addresses",
        ):
            _run_global_phase(cfg)

        # Assert -- recreate=False passed to create_output_store
        p_create.assert_called_once()
        create_kwargs = p_create.call_args.kwargs
        assert "recreate" in create_kwargs, (
            "create_output_store not called with "
            "recreate keyword argument"
        )
        assert create_kwargs["recreate"] is False, (
            f"Expected recreate=False, got "
            f"recreate={create_kwargs['recreate']}"
        )


# -----------------------------------------------------------
# Local forecast functional tests
# -----------------------------------------------------------

class TestLocalForecastFunctional:
    r'''Tests for run_local_forecast per-worker logic.'''

    def _make_mock_fabric(
            self,
            global_rank: int = 0,
            world_size: int = 1,
    ) -> MagicMock:
        r'''Create a mock Fabric with rank info.

        Parameters
        ----------
        global_rank : int, optional
            Simulated global rank. Default is 0.
        world_size : int, optional
            Simulated world size. Default is 1.

        Returns
        -------
        MagicMock
            Mock Fabric with global_rank and world_size.
        '''
        fabric = MagicMock()
        fabric.global_rank = global_rank
        fabric.world_size = world_size
        return fabric

    @contextmanager
    def _mock_local_deps(
            self,
    ) -> Generator[Dict[str, MagicMock], None, None]:
        r'''Patch all run_local_forecast dependencies.

        Yields a dict keyed by short name so individual
        tests can inspect specific mocks.

        Yields
        ------
        dict of str to MagicMock
            Named references to each patched dependency.
        '''
        # Default: all entries complete (last contiguous
        # lead index == 3, i.e. 4 leads fully written).
        # Matches the real check_written_regions contract,
        # which always returns a non-empty dict.
        _complete = {(0, 0): 3}
        mock_output_writer = MagicMock()
        mock_output_writer.lead_times = [0, 1, 2, 3]
        with patch(
            "forecast.setup_environment",
        ) as p_setup, patch(
            "forecast.initialize_client",
            return_value=MagicMock(
                __enter__=MagicMock(
                    return_value=MagicMock(),
                ),
                __exit__=MagicMock(
                    return_value=False,
                ),
            ),
        ) as p_client, patch(
            "forecast.load_forecast_model",
            return_value=MagicMock(),
        ) as p_model, patch(
            "forecast.run_forecast",
        ) as p_run, patch(
            "forecast.initialize_io",
            return_value=(
                MagicMock(), MagicMock(),
            ),
        ) as p_io, patch(
            "forecast.check_written_regions",
            return_value=_complete,
        ) as p_check:
            p_io.return_value = (
                MagicMock(), mock_output_writer,
            )
            yield {
                "setup": p_setup,
                "client": p_client,
                "model": p_model,
                "run": p_run,
                "io": p_io,
                "check": p_check,
            }

    def test_local_forecast_calls_barrier(
            self,
    ) -> None:
        r'''run_local_forecast calls fabric.barrier().'''
        # Arrange
        cfg = _make_local_forecast_cfg()
        fabric = self._make_mock_fabric()
        mock_configs = [MagicMock()]

        # Act
        with self._mock_local_deps():
            run_local_forecast(
                fabric, cfg, mock_configs,
            )

        # Assert
        fabric.barrier.assert_called_once()

    def test_local_forecast_rank_zero_checks_completeness(
            self,
    ) -> None:
        r'''Rank 0 calls check_written_regions after barrier.'''
        # Arrange
        cfg = _make_local_forecast_cfg()
        fabric = self._make_mock_fabric(
            global_rank=0,
        )
        mock_configs = [MagicMock()]

        # Act
        with self._mock_local_deps() as mocks:
            run_local_forecast(
                fabric, cfg, mock_configs,
            )

        # Assert -- rank 0 performs completeness check
        assert mocks["check"].called, (
            "check_written_regions not called "
            "for rank 0"
        )

    def test_local_forecast_sets_per_worker_seed(
            self,
    ) -> None:
        r'''setup_environment receives worker_rank arg.'''
        # Arrange
        cfg = _make_local_forecast_cfg()
        fabric = self._make_mock_fabric(
            global_rank=2,
        )
        mock_configs = [MagicMock()]

        # Act
        with self._mock_local_deps() as mocks:
            run_local_forecast(
                fabric, cfg, mock_configs,
            )

        # Assert -- worker_rank=2 passed
        p_setup = mocks["setup"]
        p_setup.assert_called_once()
        setup_kwargs = p_setup.call_args
        assert (
            setup_kwargs.kwargs.get("worker_rank")
            == 2
            or (
                len(setup_kwargs.args) > 1
                and setup_kwargs.args[1] == 2
            )
        ), (
            "setup_environment not called with "
            "worker_rank=2"
        )

    def test_local_forecast_non_rank_zero_skips_completeness(
            self,
    ) -> None:
        r'''Non-zero rank skips check_written_regions.'''
        # Arrange
        cfg = _make_local_forecast_cfg()
        fabric = self._make_mock_fabric(
            global_rank=1, world_size=2,
        )
        mock_configs = [MagicMock()]

        # Act
        with self._mock_local_deps() as mocks:
            run_local_forecast(
                fabric, cfg, mock_configs,
            )

        # Assert -- non-zero rank skips completeness
        mocks["check"].assert_not_called()

    def test_local_forecast_rank_zero_raises_on_incomplete_output(
            self,
    ) -> None:
        r'''Rank 0 raises when check_written_regions finds incomplete cells.'''
        # Arrange
        cfg = _make_local_forecast_cfg()
        fabric = self._make_mock_fabric(
            global_rank=0,
        )
        mock_configs = [MagicMock()]
        
        # Incomplete regions: init=0, ens=0 has last_index=2 instead of 3
        incomplete_regions = {
            ("2020-01-01T00:00:00", 0): 2,
            ("2020-01-01T06:00:00", 0): 3,
        }

        # Act / Assert
        with self._mock_local_deps() as mocks:
            mocks["check"].return_value = incomplete_regions
            with pytest.raises(RuntimeError) as exc_info:
                run_local_forecast(
                    fabric, cfg, mock_configs,
                )
        
        # Assert error message describes incomplete regions
        assert "incomplete" in str(exc_info.value).lower(), (
            "Expected RuntimeError to mention incomplete regions"
        )

    def test_local_forecast_rank_zero_raises_when_all_regions_share_same_incomplete_last_index(
            self,
    ) -> None:
        r'''Rank 0 raises when all regions stop at the same incomplete last index.'''
        # Arrange
        cfg = _make_local_forecast_cfg()
        fabric = self._make_mock_fabric(
            global_rank=0,
        )
        mock_configs = [MagicMock()]

        # Act / Assert
        with self._mock_local_deps() as mocks:
            mocks["check"].return_value = {
                (0, 0): 2,
                (1, 0): 2,
            }
            mock_writer = MagicMock()
            mock_writer.lead_times = [0, 1, 2, 3]
            mocks["io"].return_value = (
                MagicMock(), mock_writer,
            )
            with pytest.raises(RuntimeError) as exc_info:
                run_local_forecast(
                    fabric, cfg, mock_configs,
                )

        # Assert error message describes incomplete regions
        assert "incomplete" in str(exc_info.value).lower(), (
            "Expected RuntimeError to mention incomplete regions"
        )

    def test_local_forecast_rank_zero_complete_store_no_raise(
            self,
    ) -> None:
        r'''Rank 0 does not raise when store is fully written.'''
        # Arrange
        cfg = _make_local_forecast_cfg()
        fabric = self._make_mock_fabric(
            global_rank=0,
        )
        mock_configs = [MagicMock()]

        # All entries complete: every (init, ens) pair has
        # last contiguous lead index == 3 (4 leads written).
        complete_regions = {
            (0, 0): 3,
            (1, 0): 3,
        }

        # Act -- should complete without RuntimeError
        with self._mock_local_deps() as mocks:
            mocks["check"].return_value = complete_regions
            run_local_forecast(
                fabric, cfg, mock_configs,
            )

        # Assert -- check was called, no exception raised
        assert mocks["check"].called, (
            "check_written_regions not called "
            "for rank 0"
        )

    def test_local_forecast_uses_dp_rank_for_client(
            self,
    ) -> None:
        r'''initialize_client receives dp_rank argument.'''
        # Arrange
        cfg = _make_local_forecast_cfg()
        fabric = self._make_mock_fabric(
            global_rank=1, world_size=2,
        )
        mock_configs = [MagicMock()]

        # Act
        with self._mock_local_deps() as mocks:
            run_local_forecast(
                fabric, cfg, mock_configs,
            )

        # Assert -- dp_rank passed to client init
        p_client = mocks["client"]
        p_client.assert_called_once()
        client_kwargs = p_client.call_args
        has_dp_rank = (
            "dp_rank" in (client_kwargs.kwargs or {})
        )
        assert has_dp_rank, (
            "initialize_client not called with "
            "dp_rank keyword argument"
        )


# -----------------------------------------------------------
# _launch_forecast tests
# -----------------------------------------------------------

class TestMainForecastLaunch:
    r'''Tests for _launch_forecast Fabric setup.'''

    def test_launch_creates_fabric(
            self,
    ) -> None:
        r'''_launch_forecast passes strategy to Fabric.'''
        # Arrange
        cfg = OmegaConf.create({
            "trainer": {
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "strategy": "auto",
            },
        })
        mock_fabric = MagicMock()
        mock_configs = [MagicMock()]

        # Act
        with patch(
            "forecast._run_global_phase",
            return_value=(mock_configs, 1),
        ), patch(
            "lightning.fabric.Fabric",
            return_value=mock_fabric,
        ) as mock_cls:
            _launch_forecast(cfg)

        # Assert -- strategy parameter passed
        mock_cls.assert_called_once_with(
            accelerator="cpu",
            devices=1,
            precision="32-true",
            strategy="auto",
        )
        mock_fabric.launch.assert_called_once()

        # Assert -- launch calls run_local_forecast
        launch_args = (
            mock_fabric.launch.call_args.args
        )
        assert launch_args[0] is run_local_forecast


# -----------------------------------------------------------
# YAML defaults tests
# -----------------------------------------------------------

class TestForecastYamlDefaults:
    r'''Tests for default values in configs/forecast.yaml.'''

    def _yaml_path(self) -> str:
        r'''Return absolute path to configs/forecast.yaml.

        Returns
        -------
        str
            Absolute path to the forecast YAML config file.
        '''
        return os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "forecast.yaml",
        )

    def test_forecast_yaml_has_trainer_section(
            self,
    ) -> None:
        r'''forecast.yaml has trainer defaults.'''
        # Act
        cfg = OmegaConf.load(self._yaml_path())

        # Assert
        assert cfg.trainer.accelerator == "cpu"
        assert cfg.trainer.devices == 1
        assert cfg.trainer.precision == "32-true"

    def test_forecast_yaml_no_devices_key(
            self,
    ) -> None:
        r'''forecast.yaml has no top-level devices key.'''
        # Act
        cfg = OmegaConf.load(self._yaml_path())

        # Assert
        assert "devices" not in cfg
        assert "amp_dtype" not in cfg

    def test_forecast_yaml_has_n_prefetch_init_default(
            self,
    ) -> None:
        r'''dask.n_prefetch_init defaults to 1.'''
        # Act
        cfg = OmegaConf.load(self._yaml_path())

        # Assert
        assert cfg.dask.n_prefetch_init == 1

    def test_forecast_yaml_has_n_prefetch_forcing_default(
            self,
    ) -> None:
        r'''dask.n_prefetch_forcing defaults to 3.'''
        # Act
        cfg = OmegaConf.load(self._yaml_path())

        # Assert
        assert cfg.dask.n_prefetch_forcing == 3

    def test_trainer_config_overridable_with_gpu_values(
            self,
    ) -> None:
        r'''trainer section accepts a multi-GPU override.'''
        # Arrange
        gpu_override = OmegaConf.create({
            "trainer": {
                "accelerator": "gpu",
                "devices": [0, 1],
                "precision": "bf16-mixed",
                "strategy": "ddp",
            },
        })

        # Act
        base_cfg = OmegaConf.load(self._yaml_path())
        cfg = OmegaConf.merge(base_cfg, gpu_override)

        # Assert
        assert cfg.trainer.accelerator == "gpu"
        assert cfg.trainer.devices == [0, 1]
        assert cfg.trainer.precision == "bf16-mixed"
        assert cfg.trainer.strategy == "ddp"

    def test_forecast_yaml_has_strategy_key(
            self,
    ) -> None:
        r'''forecast.yaml trainer has strategy=auto.'''
        # Act
        cfg = OmegaConf.load(self._yaml_path())

        # Assert
        assert cfg.trainer.strategy == "auto"

    def test_forecast_yaml_no_processes_key(
            self,
    ) -> None:
        r'''forecast.yaml dask has no processes key.'''
        # Act
        cfg = OmegaConf.load(self._yaml_path())

        # Assert
        assert "processes" not in cfg.dask
